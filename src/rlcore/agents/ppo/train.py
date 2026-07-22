"""PPO training: config, epoch/minibatch update, loop, and Hydra entry point.

Total loss per minibatch, matching Stable-Baselines3's composition:

    L = policy_loss + vf_coef * value_loss - ent_coef * entropy

with the clipped surrogate policy loss, MSE value loss against GAE value
targets, and approximate-KL monitoring (optional early epoch stop).
Derivation and analysis: ``docs/algorithms/ppo.md``.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from rlcore.agents.ppo.gae import compute_gae
from rlcore.agents.ppo.loss import (
    approx_kl_divergence,
    clipped_surrogate_loss,
    explained_variance,
    value_mse_loss,
)
from rlcore.agents.ppo.model import ActorCritic
from rlcore.agents.ppo.rollout import RolloutCollector
from rlcore.envs import make_discrete_env_pair
from rlcore.evaluation import EvalStats, evaluate_policy
from rlcore.experiments import write_run_outputs
from rlcore.utils.seeding import seed_everything

if TYPE_CHECKING:
    from rlcore.types import Batch

logger = logging.getLogger(__name__)


@dataclass
class PpoConfig:
    """Hyperparameters and run settings for PPO.

    Defaults follow Stable-Baselines3's PPO defaults (single env), which
    keeps any reference comparison honest.

    Attributes:
        env_id: Gymnasium env id with flat observations and discrete actions.
        seed: Root seed for :func:`rlcore.utils.seeding.seed_everything`.
        total_updates: Maximum number of collect+optimize iterations.
        n_steps: Rollout horizon (env steps per iteration).
        minibatch_size: Minibatch size for the optimization epochs.
        n_epochs: Optimization epochs over each rollout.
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        clip_range: PPO clip epsilon.
        lr: Adam learning rate.
        ent_coef: Entropy bonus coefficient.
        vf_coef: Value-loss coefficient.
        max_grad_norm: Global gradient-norm clip.
        normalize_advantages: Per-minibatch advantage standardization — the
            same *configurable heuristic* caveats as REINFORCE's return
            standardization apply (same-batch statistics; step-size rescaling).
        target_kl: Optional threshold: stop this iteration's remaining epochs
            when the approximate KL exceeds ``1.5 * target_kl`` (monitoring
            itself is always on). ``None`` disables early stopping.
        hidden_sizes: MLP hidden widths for both policy and value networks.
        eval_every: Greedy evaluation every this many iterations (0 disables
            periodic evaluation; a final evaluation always runs).
        eval_episodes: Episodes per evaluation.
        stop_return: Stop early once a periodic greedy evaluation reaches
            this mean return (``None`` disables early stopping).
    """

    env_id: str = "CartPole-v1"
    seed: int = 0
    total_updates: int = 75
    n_steps: int = 2048
    minibatch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    lr: float = 3e-4
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    target_kl: float | None = None
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    eval_every: int = 5
    eval_episodes: int = 20
    stop_return: float | None = 475.0


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Outcome of one PPO training run.

    Attributes:
        model: The trained actor-critic (for further evaluation by callers).
        final_eval: Greedy evaluation after the last update.
        history: Per-iteration metric dicts, in order.
        total_env_steps: Environment steps consumed by training collection.
        stopped_early: Whether ``stop_return`` triggered before the budget.
        wall_time_s: Wall-clock training duration in seconds.
    """

    model: ActorCritic
    final_eval: EvalStats
    history: list[dict[str, float]]
    total_env_steps: int
    stopped_early: bool
    wall_time_s: float


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    data: Batch,
    *,
    n_epochs: int,
    minibatch_size: int,
    clip_range: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
    normalize_advantages: bool,
    target_kl: float | None,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Optimize the model for ``n_epochs`` of minibatches over one rollout.

    Args:
        model: Actor-critic to update.
        optimizer: Optimizer over the model's parameters.
        data: Flattened rollout batch with fields ``obs``, ``action``,
            ``logprob`` (behavior), ``advantage``, ``value_target``.
        n_epochs: Passes over the data.
        minibatch_size: Samples per minibatch.
        clip_range: PPO clip epsilon.
        vf_coef: Value-loss coefficient.
        ent_coef: Entropy bonus coefficient.
        max_grad_norm: Global grad-norm clip.
        normalize_advantages: Standardize advantages per minibatch.
        target_kl: If set, stop remaining epochs when the minibatch
            approximate KL exceeds ``1.5 * target_kl``.
        generator: RNG stream for minibatch shuffling.

    Returns:
        Mean diagnostics over processed minibatches: ``policy_loss``,
        ``value_loss``, ``entropy``, ``approx_kl``, ``clip_fraction``,
        ``grad_norm``, plus ``n_minibatches`` processed and
        ``kl_stopped_epoch`` (-1 when no early stop occurred).
    """
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    kls: list[float] = []
    clip_fractions: list[float] = []
    grad_norms: list[float] = []
    kl_stopped_epoch = -1

    for epoch in range(n_epochs):
        for minibatch in data.minibatches(minibatch_size, generator=generator):
            advantages = minibatch.advantage
            if normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            log_prob, entropy, value = model.evaluate_actions(minibatch.obs, minibatch.action)
            policy_loss, clip_fraction = clipped_surrogate_loss(
                log_prob, minibatch.logprob, advantages, clip_range=clip_range
            )
            v_loss = value_mse_loss(value, minibatch.value_target)
            entropy_mean = entropy.mean()
            loss = policy_loss + vf_coef * v_loss - ent_coef * entropy_mean

            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]  # Tensor.backward is untyped in torch stubs
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            kl = float(approx_kl_divergence(log_prob.detach(), minibatch.logprob).item())
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(v_loss.item()))
            entropies.append(float(entropy_mean.item()))
            kls.append(kl)
            clip_fractions.append(float(clip_fraction.item()))
            grad_norms.append(float(grad_norm.item()))

            if target_kl is not None and kl > 1.5 * target_kl:
                kl_stopped_epoch = epoch
                break
        if kl_stopped_epoch >= 0:
            break

    return {
        "policy_loss": statistics.mean(policy_losses),
        "value_loss": statistics.mean(value_losses),
        "entropy": statistics.mean(entropies),
        "approx_kl": statistics.mean(kls),
        "clip_fraction": statistics.mean(clip_fractions),
        "grad_norm": statistics.mean(grad_norms),
        "n_minibatches": float(len(policy_losses)),
        "kl_stopped_epoch": float(kl_stopped_epoch),
    }


def train(config: PpoConfig, out_dir: Path | None = None) -> TrainResult:
    """Train PPO per ``config`` and return the result.

    Mirrors M1's seeding layout: training env, action space, and evaluation
    env each get independent child seeds of the root; action sampling and
    minibatch shuffling draw from the bundle's explicit torch generator.

    Args:
        config: Run configuration.
        out_dir: If given, writes ``config.yaml``, ``metrics.jsonl``, and
            ``result.json`` there.

    Returns:
        The :class:`TrainResult`, including the final greedy evaluation.
    """
    start = time.perf_counter()
    rng = seed_everything(config.seed)
    envs = make_discrete_env_pair(config.env_id, rng)
    model = ActorCritic(envs.obs_dim, envs.n_actions, hidden_sizes=config.hidden_sizes)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    collector = RolloutCollector(envs.train_env)

    history: list[dict[str, float]] = []
    total_env_steps = 0
    stopped_early = False

    for update in range(1, config.total_updates + 1):
        rollout = collector.collect(model, config.n_steps, generator=rng.torch)
        total_env_steps += len(rollout.batch)

        advantages, value_targets = compute_gae(
            rollout.batch.reward,
            rollout.batch.value,
            rollout.batch.next_value,
            rollout.batch.done,
            gamma=config.gamma,
            lam=config.gae_lambda,
        )
        data = rollout.batch.with_fields(advantage=advantages, value_target=value_targets)

        metrics = ppo_update(
            model,
            optimizer,
            data,
            n_epochs=config.n_epochs,
            minibatch_size=config.minibatch_size,
            clip_range=config.clip_range,
            vf_coef=config.vf_coef,
            ent_coef=config.ent_coef,
            max_grad_norm=config.max_grad_norm,
            normalize_advantages=config.normalize_advantages,
            target_kl=config.target_kl,
            generator=rng.torch,
        )
        train_return = (
            statistics.mean(rollout.completed_returns)
            if rollout.completed_returns
            else float("nan")
        )
        entry: dict[str, float] = {
            "update": float(update),
            "env_steps": float(total_env_steps),
            "train_return_mean": train_return,
            "explained_variance": explained_variance(rollout.batch.value, value_targets),
            **metrics,
        }

        if config.eval_every > 0 and update % config.eval_every == 0:
            stats = evaluate_policy(
                envs.eval_env,
                lambda obs: model.act(obs, deterministic=True)[0],
                episodes=config.eval_episodes,
            )
            entry["eval_return_mean"] = stats.mean_return
            entry["eval_return_std"] = stats.std_return
            logger.info(
                "update %d | steps %d | train return %.1f | eval return %.1f | kl %.4f",
                update,
                total_env_steps,
                train_return,
                stats.mean_return,
                metrics["approx_kl"],
            )
            if config.stop_return is not None and stats.mean_return >= config.stop_return:
                history.append(entry)
                stopped_early = True
                logger.info("stop_return %.1f reached at update %d.", config.stop_return, update)
                break
        history.append(entry)

    final_eval = evaluate_policy(
        envs.eval_env,
        lambda obs: model.act(obs, deterministic=True)[0],
        episodes=config.eval_episodes,
    )
    result = TrainResult(
        model=model,
        final_eval=final_eval,
        history=history,
        total_env_steps=total_env_steps,
        stopped_early=stopped_early,
        wall_time_s=time.perf_counter() - start,
    )
    envs.close()

    if out_dir is not None:
        write_run_outputs(
            out_dir,
            config,
            history,
            {
                "final_eval_return_mean": final_eval.mean_return,
                "final_eval_return_std": final_eval.std_return,
                "final_eval_returns": list(final_eval.episode_returns),
                "total_env_steps": total_env_steps,
                "stopped_early": stopped_early,
                "wall_time_s": result.wall_time_s,
            },
        )
    return result


ConfigStore.instance().store(name="ppo", node=PpoConfig)


@hydra.main(version_base="1.3", config_name="ppo")
def main(cfg: PpoConfig) -> None:
    """Hydra entry point: ``python -m rlcore.agents.ppo.train [key=value ...]``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config = cast("PpoConfig", OmegaConf.to_object(cast("Any", cfg)))
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    result = train(config, out_dir=out_dir)
    logger.info(
        "done: final eval return %.1f +- %.1f | %d env steps | %.1fs | outputs in %s",
        result.final_eval.mean_return,
        result.final_eval.std_return,
        result.total_env_steps,
        result.wall_time_s,
        out_dir,
    )


if __name__ == "__main__":
    main()

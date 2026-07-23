"""REINFORCE training: config, update step, loop, and Hydra entry point.

The loss implemented here is the Monte-Carlo policy gradient with
reward-to-go and optional batch return standardization:

    L(theta) = - mean_t [ log pi_theta(a_t | s_t) * G_t ]

where the mean runs over all steps of all episodes in the update batch and
``G_t`` is treated as a constant (no gradient flows through returns).
Derivation, bias/variance discussion, and known weaknesses:
``docs/algorithms/reinforce.md``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from rlcore.agents.reinforce.policy import CategoricalMlpPolicy
from rlcore.agents.reinforce.returns import discounted_returns
from rlcore.agents.reinforce.rollout import collect_episode
from rlcore.envs import make_discrete_env_pair
from rlcore.evaluation import EvalStats, evaluate_policy
from rlcore.experiments import new_run_id, save_final_model, write_run_outputs, write_run_record
from rlcore.reporting import write_summary
from rlcore.utils.seeding import seed_everything

if TYPE_CHECKING:
    from rlcore.types import Batch

logger = logging.getLogger(__name__)


@dataclass
class ReinforceConfig:
    """Hyperparameters and run settings for REINFORCE.

    Attributes:
        env_id: Gymnasium env id with flat observations and discrete actions.
        seed: Root seed for :func:`rlcore.utils.seeding.seed_everything`.
        updates: Maximum number of gradient updates.
        episodes_per_update: Episodes collected per update (the batch).
        gamma: Discount factor for reward-to-go.
        lr: Adam learning rate.
        hidden_sizes: Policy MLP hidden-layer widths.
        normalize_returns: Same-batch return standardization — a configurable
            variance-reduction heuristic, not an ordinary unbiased baseline:
            same-batch mean subtraction may introduce finite-batch bias, and
            std division rescales the effective step size (derivation doc §4).
        eval_every: Run a greedy evaluation every this many updates
            (0 disables periodic evaluation; a final evaluation always runs).
        eval_episodes: Episodes per evaluation.
        stop_return: Stop early once a periodic greedy evaluation reaches
            this mean return (``None`` disables early stopping).
    """

    env_id: str = "CartPole-v1"
    seed: int = 0
    updates: int = 300
    episodes_per_update: int = 10
    gamma: float = 0.99
    lr: float = 5e-3
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    normalize_returns: bool = True
    eval_every: int = 20
    eval_episodes: int = 20
    stop_return: float | None = 475.0


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Outcome of one training run.

    Attributes:
        run_id: Unique run identifier (rlcore.experiments.new_run_id).
        policy: The trained policy (for further evaluation by callers).
        final_eval: Greedy evaluation after the last update.
        history: Per-update metric dicts, in order.
        total_env_steps: Environment steps consumed by training collection.
        total_episodes: Training episodes collected.
        stopped_early: Whether ``stop_return`` triggered before the budget.
        wall_time_s: Wall-clock training duration in seconds.
    """

    run_id: str
    policy: CategoricalMlpPolicy
    final_eval: EvalStats
    history: list[dict[str, float]]
    total_env_steps: int
    total_episodes: int
    stopped_early: bool
    wall_time_s: float


def reinforce_update(
    policy: CategoricalMlpPolicy,
    optimizer: torch.optim.Optimizer,
    episodes: list[Batch],
    *,
    gamma: float,
    normalize_returns: bool,
) -> dict[str, float]:
    """Perform one REINFORCE gradient step on a batch of episodes.

    Returns are computed per episode (no leakage across boundaries), then
    concatenated; with ``normalize_returns`` they are standardized across the
    whole update batch (a heuristic — see ``ReinforceConfig`` and the
    derivation doc §4 for its caveats). The loss is the mean over all steps
    of ``-log pi(a_t|s_t) * G_t``.

    Args:
        policy: Policy to update.
        optimizer: Optimizer over the policy's parameters.
        episodes: One :class:`Batch` per episode, as produced by
            :func:`rlcore.agents.reinforce.rollout.collect_episode`.
        gamma: Discount factor.
        normalize_returns: Standardize returns within the batch.

    Returns:
        Scalar diagnostics: ``loss``, ``entropy``, ``grad_norm``,
        ``return_mean`` and ``return_std`` (raw, pre-normalization),
        ``train_return_mean`` (mean undiscounted episode return).

    Raises:
        ValueError: If ``episodes`` is empty.
    """
    if not episodes:
        raise ValueError("reinforce_update requires at least one episode.")
    returns = torch.cat([discounted_returns(ep.reward, gamma) for ep in episodes])
    obs = torch.cat([ep.obs for ep in episodes])
    actions = torch.cat([ep.action for ep in episodes])

    return_mean = float(returns.mean().item())
    return_std = float(returns.std().item()) if len(returns) > 1 else 0.0
    if normalize_returns:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    log_prob = policy.action_log_prob(obs, actions)
    loss = -(log_prob * returns.detach()).mean()

    optimizer.zero_grad()
    loss.backward()  # type: ignore[no-untyped-call]  # Tensor.backward is untyped in torch stubs
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float("inf"))
    optimizer.step()

    with torch.no_grad():
        entropy = float(policy.entropy(obs).mean().item())
    return {
        "loss": float(loss.item()),
        "entropy": entropy,
        "grad_norm": float(grad_norm.item()),
        "return_mean": return_mean,
        "return_std": return_std,
        "train_return_mean": float(
            torch.tensor([ep.reward.sum() for ep in episodes]).mean().item()
        ),
    }


def train(config: ReinforceConfig, out_dir: Path | None = None) -> TrainResult:
    """Train REINFORCE per ``config`` and return the result.

    Seeds all streams from ``config.seed`` (training env, evaluation env, and
    action sampling each get their own stream), trains for at most
    ``config.updates`` updates, evaluates greedily every ``config.eval_every``
    updates, and stops early when ``config.stop_return`` is reached.

    Args:
        config: Run configuration.
        out_dir: If given, writes ``config.yaml`` (resolved), ``metrics.jsonl``
            (one history entry per line), and ``result.json`` there.

    Returns:
        The :class:`TrainResult`, including the final greedy evaluation.
    """
    start = time.perf_counter()
    run_id = new_run_id("reinforce", config.env_id, config.seed)
    rng = seed_everything(config.seed)
    envs = make_discrete_env_pair(config.env_id, rng)
    env = envs.train_env
    policy = CategoricalMlpPolicy(envs.obs_dim, envs.n_actions, hidden_sizes=config.hidden_sizes)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr)

    history: list[dict[str, float]] = []
    total_env_steps = 0
    total_episodes = 0
    stopped_early = False

    for update in range(1, config.updates + 1):
        episodes = [
            collect_episode(env, policy, generator=rng.torch)
            for _ in range(config.episodes_per_update)
        ]
        total_env_steps += sum(len(ep) for ep in episodes)
        total_episodes += len(episodes)

        metrics = reinforce_update(
            policy,
            optimizer,
            episodes,
            gamma=config.gamma,
            normalize_returns=config.normalize_returns,
        )
        entry: dict[str, float] = {
            "update": float(update),
            "env_steps": float(total_env_steps),
            **metrics,
        }

        if config.eval_every > 0 and update % config.eval_every == 0:
            stats = evaluate_policy(
                envs.eval_env,
                lambda obs: policy.act(obs, deterministic=True),
                episodes=config.eval_episodes,
            )
            entry["eval_return_mean"] = stats.mean_return
            entry["eval_return_std"] = stats.std_return
            logger.info(
                "update %d | steps %d | train return %.1f | eval return %.1f | entropy %.3f",
                update,
                total_env_steps,
                metrics["train_return_mean"],
                stats.mean_return,
                metrics["entropy"],
            )
            if config.stop_return is not None and stats.mean_return >= config.stop_return:
                history.append(entry)
                stopped_early = True
                logger.info("stop_return %.1f reached at update %d.", config.stop_return, update)
                break
        elif update % 10 == 0:
            logger.info(
                "update %d | steps %d | train return %.1f | entropy %.3f",
                update,
                total_env_steps,
                metrics["train_return_mean"],
                metrics["entropy"],
            )
        history.append(entry)

    final_eval = evaluate_policy(
        envs.eval_env,
        lambda obs: policy.act(obs, deterministic=True),
        episodes=config.eval_episodes,
    )
    result = TrainResult(
        run_id=run_id,
        policy=policy,
        final_eval=final_eval,
        history=history,
        total_env_steps=total_env_steps,
        total_episodes=total_episodes,
        stopped_early=stopped_early,
        wall_time_s=time.perf_counter() - start,
    )
    envs.close()

    if out_dir is not None:
        write_run_record(
            out_dir, run_id=run_id, algo="reinforce", env_id=config.env_id, seed=config.seed
        )
        write_run_outputs(
            out_dir,
            config,
            history,
            {
                "run_id": run_id,
                "final_eval_return_mean": final_eval.mean_return,
                "final_eval_return_std": final_eval.std_return,
                "final_eval_returns": list(final_eval.episode_returns),
                "total_env_steps": total_env_steps,
                "total_episodes": total_episodes,
                "stopped_early": stopped_early,
                "wall_time_s": result.wall_time_s,
            },
        )
        save_final_model(out_dir, algo="reinforce", state_dict=policy.state_dict())
        write_summary(
            out_dir,
            run_id=run_id,
            algo="reinforce",
            env_id=config.env_id,
            seed=config.seed,
            history=history,
            final_eval=final_eval,
            total_env_steps=total_env_steps,
            stopped_early=stopped_early,
            wall_time_s=result.wall_time_s,
        )
    return result


ConfigStore.instance().store(name="reinforce", node=ReinforceConfig)


@hydra.main(version_base="1.3", config_name="reinforce")
def main(cfg: ReinforceConfig) -> None:
    """Hydra entry point: ``python -m rlcore.agents.reinforce.train [key=value ...]``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config = cast("ReinforceConfig", OmegaConf.to_object(cast("Any", cfg)))
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

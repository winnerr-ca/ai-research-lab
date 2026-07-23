"""DQN training: config, TD update, step-based loop, and Hydra entry point.

The loop is step-based (unlike the iteration-based on-policy trainers,
kept local per the M3 audit policy): each env step optionally triggers a
minibatch TD update (``train_freq``) and a periodic hard target-network
synchronization (``target_sync_every``). Logging, evaluation, early
stopping, and checkpointing all happen on ``log_every`` boundaries so
resumed runs align exactly with continuous ones.
Derivation and analysis: ``docs/algorithms/dqn.md``.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from rlcore.agents.dqn.loss import linear_epsilon, q_huber_loss, select_next_values, td_targets
from rlcore.agents.dqn.model import QNetwork, epsilon_greedy_action
from rlcore.checkpoints import (
    check_resume_config,
    load_checkpoint,
    pickle_env,
    save_checkpoint,
    unpickle_env,
)
from rlcore.envs import EnvPair, env_pair_from_envs, make_discrete_env_pair
from rlcore.evaluation import EvalStats, evaluate_policy
from rlcore.experiments import new_run_id, save_final_model, write_run_outputs, write_run_record
from rlcore.replay import ReplayBuffer
from rlcore.reporting import write_summary
from rlcore.utils.seeding import Rng, restore_rng, rng_state, seed_everything

if TYPE_CHECKING:
    from rlcore.tracking import Tracker
    from rlcore.types import Batch

logger = logging.getLogger(__name__)


@dataclass
class DqnConfig:
    """Hyperparameters and run settings for DQN.

    Attributes:
        env_id: Gymnasium env id with flat observations and discrete actions.
        seed: Root seed for :func:`rlcore.utils.seeding.seed_everything`.
        total_steps: Environment-step budget.
        buffer_capacity: Replay-buffer capacity (transitions).
        warmup_steps: Env steps collected (uniform-random actions) before
            any gradient update.
        batch_size: Minibatch size per TD update.
        gamma: Discount factor.
        lr: Adam learning rate.
        train_freq: Do one gradient update every this many env steps.
        target_sync_every: Hard-copy the online network into the target
            network every this many env steps.
        eps_start: Initial exploration epsilon.
        eps_final: Final exploration epsilon.
        eps_fraction: Fraction of ``total_steps`` over which epsilon anneals
            linearly.
        double_q: Use Double-DQN target selection (van Hasselt et al.,
            2016); off reproduces standard DQN exactly.
        max_grad_norm: Global gradient-norm clip.
        hidden_sizes: Q-network hidden widths.
        log_every: Append a metrics entry every this many env steps.
        eval_every: Greedy evaluation every this many env steps; must be a
            multiple of ``log_every`` (0 disables periodic evaluation).
        eval_episodes: Episodes per evaluation.
        stop_return: Stop early once a periodic greedy evaluation reaches
            this mean return (``None`` disables early stopping).
        checkpoint_every: Write ``checkpoint.pt`` every this many env steps;
            must be a multiple of ``log_every`` (0 disables; requires
            ``out_dir``).
    """

    env_id: str = "CartPole-v1"
    seed: int = 0
    total_steps: int = 60_000
    buffer_capacity: int = 50_000
    warmup_steps: int = 1_000
    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 1e-3
    train_freq: int = 1
    target_sync_every: int = 500
    eps_start: float = 1.0
    eps_final: float = 0.05
    eps_fraction: float = 0.2
    double_q: bool = False
    max_grad_norm: float = 10.0
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 128])
    log_every: int = 500
    eval_every: int = 2_500
    eval_episodes: int = 20
    stop_return: float | None = 475.0
    checkpoint_every: int = 0


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Outcome of one DQN training run.

    Attributes:
        run_id: Unique run identifier.
        q_net: The trained online Q-network.
        final_eval: Greedy evaluation after training.
        history: Per-``log_every`` metric dicts, in order.
        total_env_steps: Environment steps consumed.
        stopped_early: Whether ``stop_return`` triggered before the budget.
        wall_time_s: Wall-clock training duration in seconds.
    """

    run_id: str
    q_net: QNetwork
    final_eval: EvalStats
    history: list[dict[str, float]]
    total_env_steps: int
    stopped_early: bool
    wall_time_s: float


def dqn_update(
    q_net: QNetwork,
    target_net: QNetwork,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    *,
    gamma: float,
    double_q: bool,
    max_grad_norm: float,
) -> dict[str, float]:
    """One minibatch TD update on the online network.

    Next-state values come from the target network (Double DQN additionally
    uses the online network for action *selection* only), computed under
    ``no_grad`` — no gradient ever flows through the target.

    Args:
        q_net: Online network (updated).
        target_net: Target network (read-only here).
        optimizer: Optimizer over the online network's parameters.
        batch: Replay sample with ``obs``, ``action``, ``reward``,
            ``next_obs``, ``terminated``.
        gamma: Discount factor.
        double_q: Double-DQN target selection.
        max_grad_norm: Global grad-norm clip.

    Returns:
        Diagnostics: ``loss``, ``q_mean`` (taken-action Q), ``target_mean``,
        ``grad_norm`` (pre-clip).
    """
    with torch.no_grad():
        target_q_next = target_net(batch.next_obs)
        online_q_next = q_net(batch.next_obs) if double_q else None
        next_values = select_next_values(target_q_next, online_q_next, double=double_q)
        targets = td_targets(batch.reward, batch.terminated, next_values, gamma=gamma)

    q_taken = q_net(batch.obs).gather(1, batch.action.unsqueeze(1)).squeeze(1)
    loss = q_huber_loss(q_taken, targets)

    optimizer.zero_grad()
    loss.backward()  # type: ignore[no-untyped-call]  # Tensor.backward is untyped in torch stubs
    grad_norm = torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_grad_norm)
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "q_mean": float(q_taken.mean().item()),
        "target_mean": float(targets.mean().item()),
        "grad_norm": float(grad_norm.item()),
    }


def _validate_schedule(config: DqnConfig) -> None:
    """Reject schedules that would desynchronize log/eval/checkpoint points."""
    if config.eval_every > 0 and config.eval_every % config.log_every != 0:
        raise ValueError(
            f"eval_every ({config.eval_every}) must be a multiple of log_every "
            f"({config.log_every})."
        )
    if config.checkpoint_every > 0 and config.checkpoint_every % config.log_every != 0:
        raise ValueError(
            f"checkpoint_every ({config.checkpoint_every}) must be a multiple of log_every "
            f"({config.log_every})."
        )


def _checkpoint_payload(
    config: DqnConfig,
    *,
    run_id: str,
    step: int,
    history: list[dict[str, float]],
    q_net: QNetwork,
    target_net: QNetwork,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    rng: Rng,
    envs: EnvPair,
    current_obs: np.ndarray,
    episode_return: float,
) -> dict[str, Any]:
    """Assemble the checkpoint payload for DQN's state (incl. replay)."""
    return {
        "run_id": run_id,
        "config": asdict(config),
        "step": step,
        "history": history,
        "model": q_net.state_dict(),
        "target_model": target_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "buffer": buffer.state(),
        "rng": rng_state(rng),
        "current_obs": current_obs,
        "episode_return": episode_return,
        "train_env": pickle_env(envs.train_env),
        "eval_env": pickle_env(envs.eval_env),
    }


def train(
    config: DqnConfig,
    out_dir: Path | None = None,
    resume_from: Path | None = None,
    tracker: Tracker | None = None,
) -> TrainResult:
    """Train DQN per ``config`` and return the result.

    Args:
        config: Run configuration.
        out_dir: If given, writes the standard run directory there.
        resume_from: Optional ``checkpoint.pt``; the config must match the
            checkpoint's except the ``total_steps`` budget.
        tracker: Optional metrics mirror; the local run directory stays the
            source of record.

    Returns:
        The :class:`TrainResult`, including the final greedy evaluation.

    Raises:
        ValueError: On schedule violations, ``checkpoint_every`` without
            ``out_dir``, or resume config mismatches.
    """
    start_time = time.perf_counter()
    _validate_schedule(config)
    if config.checkpoint_every > 0 and out_dir is None:
        raise ValueError("checkpoint_every > 0 requires out_dir (checkpoints live there).")

    if resume_from is not None:
        payload = load_checkpoint(resume_from, expected_algo="dqn")
        check_resume_config(payload["config"], asdict(config), budget_field="total_steps")
        run_id = str(payload["run_id"])
        rng = restore_rng(payload["rng"])
        envs = env_pair_from_envs(
            unpickle_env(payload["train_env"]), unpickle_env(payload["eval_env"])
        )
        q_net = QNetwork(envs.obs_dim, envs.n_actions, hidden_sizes=config.hidden_sizes)
        q_net.load_state_dict(payload["model"])
        target_net = QNetwork(envs.obs_dim, envs.n_actions, hidden_sizes=config.hidden_sizes)
        target_net.load_state_dict(payload["target_model"])
        optimizer = torch.optim.Adam(q_net.parameters(), lr=config.lr)
        optimizer.load_state_dict(payload["optimizer"])
        buffer = ReplayBuffer(config.buffer_capacity, obs_shape=(envs.obs_dim,))
        buffer.restore_state(payload["buffer"])
        history = payload["history"]
        obs = payload["current_obs"]
        episode_return = float(payload["episode_return"])
        start_step = int(payload["step"]) + 1
    else:
        run_id = new_run_id("dqn", config.env_id, config.seed)
        rng = seed_everything(config.seed)
        envs = make_discrete_env_pair(config.env_id, rng)
        q_net = QNetwork(envs.obs_dim, envs.n_actions, hidden_sizes=config.hidden_sizes)
        target_net = QNetwork(envs.obs_dim, envs.n_actions, hidden_sizes=config.hidden_sizes)
        target_net.load_state_dict(q_net.state_dict())
        optimizer = torch.optim.Adam(q_net.parameters(), lr=config.lr)
        buffer = ReplayBuffer(config.buffer_capacity, obs_shape=(envs.obs_dim,))
        history = []
        obs, _ = envs.train_env.reset()
        episode_return = 0.0
        start_step = 1

    for param in target_net.parameters():
        param.requires_grad_(False)

    if tracker is not None:
        tracker.start(
            run_id=run_id,
            algo="dqn",
            env_id=config.env_id,
            seed=config.seed,
            config=asdict(config),
        )

    decay_steps = max(int(config.eps_fraction * config.total_steps), 1)
    env = envs.train_env
    stopped_early = False
    window_losses: list[float] = []
    window_q: list[float] = []
    window_returns: list[float] = []
    step = start_step - 1  # final value if the budget is already consumed

    for step in range(start_step, config.total_steps + 1):
        epsilon = linear_epsilon(
            step, start=config.eps_start, final=config.eps_final, decay_steps=decay_steps
        )
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        if step <= config.warmup_steps:
            action = int(torch.randint(0, envs.n_actions, (1,), generator=rng.torch).item())
        else:
            action = int(
                epsilon_greedy_action(q_net, obs_t, epsilon=epsilon, generator=rng.torch).item()
            )
        next_obs, reward, terminated, truncated, _ = env.step(action)
        # next_obs is the true successor (the final observation at
        # truncation — raw envs do not autoreset), so it is stored as-is.
        buffer.add(np.asarray(obs, dtype=np.float32), action, float(reward), next_obs, terminated)
        episode_return += float(reward)
        if terminated or truncated:
            window_returns.append(episode_return)
            episode_return = 0.0
            obs, _ = env.reset()
        else:
            obs = next_obs

        if step > config.warmup_steps and step % config.train_freq == 0:
            batch = buffer.sample(config.batch_size, generator=rng.torch)
            metrics = dqn_update(
                q_net,
                target_net,
                optimizer,
                batch,
                gamma=config.gamma,
                double_q=config.double_q,
                max_grad_norm=config.max_grad_norm,
            )
            window_losses.append(metrics["loss"])
            window_q.append(metrics["q_mean"])

        if step % config.target_sync_every == 0:
            target_net.load_state_dict(q_net.state_dict())

        if step % config.log_every == 0:
            entry: dict[str, float] = {
                "env_steps": float(step),
                "epsilon": epsilon,
                "buffer_size": float(len(buffer)),
                "loss": statistics.mean(window_losses) if window_losses else float("nan"),
                "q_mean": statistics.mean(window_q) if window_q else float("nan"),
                "train_return_mean": (
                    statistics.mean(window_returns) if window_returns else float("nan")
                ),
            }
            window_losses, window_q, window_returns = [], [], []

            if config.eval_every > 0 and step % config.eval_every == 0:
                stats = evaluate_policy(
                    envs.eval_env,
                    lambda obs: q_net.greedy_action(obs),
                    episodes=config.eval_episodes,
                )
                entry["eval_return_mean"] = stats.mean_return
                entry["eval_return_std"] = stats.std_return
                logger.info(
                    "step %d | eps %.3f | train return %.1f | eval return %.1f | loss %.4f",
                    step,
                    epsilon,
                    entry["train_return_mean"],
                    stats.mean_return,
                    entry["loss"],
                )
                if config.stop_return is not None and stats.mean_return >= config.stop_return:
                    history.append(entry)
                    if tracker is not None:
                        tracker.log_metrics(step, entry)
                    stopped_early = True
                    logger.info("stop_return %.1f reached at step %d.", config.stop_return, step)
                    break
            history.append(entry)
            if tracker is not None:
                tracker.log_metrics(step, entry)
            if config.checkpoint_every > 0 and step % config.checkpoint_every == 0:
                assert out_dir is not None  # validated above
                save_checkpoint(
                    out_dir / "checkpoint.pt",
                    algo="dqn",
                    payload=_checkpoint_payload(
                        config,
                        run_id=run_id,
                        step=step,
                        history=history,
                        q_net=q_net,
                        target_net=target_net,
                        optimizer=optimizer,
                        buffer=buffer,
                        rng=rng,
                        envs=envs,
                        current_obs=np.asarray(obs, dtype=np.float32),
                        episode_return=episode_return,
                    ),
                )

    total_env_steps = step
    final_eval = evaluate_policy(
        envs.eval_env,
        lambda obs: q_net.greedy_action(obs),
        episodes=config.eval_episodes,
    )
    result = TrainResult(
        run_id=run_id,
        q_net=q_net,
        final_eval=final_eval,
        history=history,
        total_env_steps=total_env_steps,
        stopped_early=stopped_early,
        wall_time_s=time.perf_counter() - start_time,
    )
    envs.close()
    if tracker is not None:
        tracker.log_summary(
            {
                "eval_return_mean": final_eval.mean_return,
                "eval_return_std": final_eval.std_return,
                "total_env_steps": total_env_steps,
                "stopped_early": stopped_early,
                "wall_time_s": result.wall_time_s,
            }
        )
        tracker.finish()

    if out_dir is not None:
        write_run_record(out_dir, run_id=run_id, algo="dqn", env_id=config.env_id, seed=config.seed)
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
                "stopped_early": stopped_early,
                "wall_time_s": result.wall_time_s,
            },
        )
        save_final_model(out_dir, algo="dqn", state_dict=q_net.state_dict())
        write_summary(
            out_dir,
            run_id=run_id,
            algo="dqn",
            env_id=config.env_id,
            seed=config.seed,
            history=history,
            final_eval=final_eval,
            total_env_steps=total_env_steps,
            stopped_early=stopped_early,
            wall_time_s=result.wall_time_s,
        )
    return result


ConfigStore.instance().store(name="dqn", node=DqnConfig)


@hydra.main(version_base="1.3", config_name="dqn")
def main(cfg: DqnConfig) -> None:
    """Hydra entry point: ``python -m rlcore.agents.dqn.train [key=value ...]``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config = cast("DqnConfig", OmegaConf.to_object(cast("Any", cfg)))
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    result = train(config, out_dir=out_dir)
    logger.info(
        "done: final eval return %.1f +- %.1f | %d env steps | %.1fs",
        result.final_eval.mean_return,
        result.final_eval.std_return,
        result.total_env_steps,
        result.wall_time_s,
    )


if __name__ == "__main__":
    main()

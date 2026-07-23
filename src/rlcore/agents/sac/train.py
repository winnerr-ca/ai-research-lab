"""SAC training: config, soft updates, step-based loop, Hydra entry point.

Structure mirrors DQN's step-based loop (kept local per the M3 audit
policy): warm-up with uniform-random actions, one soft update per
``train_freq`` env steps (critics, then actor, then temperature, then
Polyak target averaging), logging/eval/checkpoint aligned on ``log_every``
boundaries. Derivation and analysis: ``docs/algorithms/sac.md``.
"""

from __future__ import annotations

import logging
import math
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

from rlcore.agents.sac.loss import (
    actor_loss,
    alpha_loss,
    critic_loss,
    critic_targets,
    polyak_update,
)
from rlcore.agents.sac.model import ContinuousQNetwork, SquashedGaussianActor
from rlcore.checkpoints import (
    check_resume_config,
    load_checkpoint,
    pickle_env,
    save_checkpoint,
    unpickle_env,
)
from rlcore.envs import ContinuousEnvPair, continuous_pair_from_envs, make_continuous_env_pair
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
class SacConfig:
    """Hyperparameters and run settings for SAC.

    Attributes:
        env_id: Gymnasium env id with flat Box observations and a 1-D Box
            action space with finite bounds.
        seed: Root seed.
        total_steps: Environment-step budget.
        buffer_capacity: Replay capacity (transitions).
        warmup_steps: Env steps with uniform-random actions before updates.
        batch_size: Minibatch size per soft update.
        gamma: Discount factor.
        tau: Polyak averaging factor for target critics.
        lr: Adam learning rate (actor, critics, and temperature).
        train_freq: One soft update every this many env steps.
        auto_alpha: Learn the entropy temperature automatically; when off,
            ``fixed_alpha`` is used unchanged.
        fixed_alpha: Temperature used when ``auto_alpha`` is off.
        target_entropy: Target entropy for automatic tuning; ``nan`` means
            the standard heuristic ``-act_dim``.
        hidden_sizes: Hidden widths for actor and critics.
        log_every: Append a metrics entry every this many env steps.
        eval_every: Deterministic evaluation every this many env steps;
            multiple of ``log_every`` (0 disables).
        eval_episodes: Episodes per evaluation.
        stop_return: Early stop once a periodic deterministic evaluation
            reaches this mean return (``None`` disables).
        checkpoint_every: Write ``checkpoint.pt`` every this many env steps;
            multiple of ``log_every`` (0 disables; requires ``out_dir``).
    """

    env_id: str = "Pendulum-v1"
    seed: int = 0
    total_steps: int = 40_000
    buffer_capacity: int = 50_000
    warmup_steps: int = 1_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    train_freq: int = 1
    auto_alpha: bool = True
    fixed_alpha: float = 0.2
    target_entropy: float = float("nan")
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    log_every: int = 500
    eval_every: int = 2_000
    eval_episodes: int = 10
    stop_return: float | None = -180.0
    checkpoint_every: int = 0


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Outcome of one SAC training run.

    Attributes:
        run_id: Unique run identifier.
        actor: The trained policy.
        final_eval: Deterministic evaluation after training.
        history: Per-``log_every`` metric dicts, in order.
        total_env_steps: Environment steps consumed.
        stopped_early: Whether ``stop_return`` triggered before the budget.
        wall_time_s: Wall-clock training duration in seconds.
    """

    run_id: str
    actor: SquashedGaussianActor
    final_eval: EvalStats
    history: list[dict[str, float]]
    total_env_steps: int
    stopped_early: bool
    wall_time_s: float


def _validate_schedule(config: SacConfig) -> None:
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


def sac_update(
    *,
    actor: SquashedGaussianActor,
    q1: ContinuousQNetwork,
    q2: ContinuousQNetwork,
    q1_target: ContinuousQNetwork,
    q2_target: ContinuousQNetwork,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    log_alpha: torch.Tensor,
    alpha_optimizer: torch.optim.Optimizer | None,
    batch: Batch,
    gamma: float,
    tau: float,
    target_entropy: float,
    generator: torch.Generator | None,
) -> dict[str, float]:
    """One full SAC update: critics, actor, temperature, Polyak targets.

    Returns:
        Diagnostics: ``critic_loss``, ``actor_loss``, ``alpha``,
        ``alpha_loss`` (nan in fixed-alpha mode), ``q1_mean``, ``entropy``
        (Monte-Carlo estimate ``-E[log pi]``).
    """
    alpha = float(log_alpha.detach().exp().item())

    with torch.no_grad():
        next_action, next_log_prob = actor.sample(batch.next_obs, generator=generator)
        next_q_min = torch.min(
            q1_target(batch.next_obs, next_action), q2_target(batch.next_obs, next_action)
        )
        targets = critic_targets(
            batch.reward,
            batch.terminated,
            next_q_min,
            next_log_prob,
            gamma=gamma,
            alpha=alpha,
        )

    q1_pred = q1(batch.obs, batch.action)
    q2_pred = q2(batch.obs, batch.action)
    loss_critic = critic_loss(q1_pred, q2_pred, targets)
    critic_optimizer.zero_grad()
    loss_critic.backward()  # type: ignore[no-untyped-call]  # untyped in torch stubs
    critic_optimizer.step()

    new_action, new_log_prob = actor.sample(batch.obs, generator=generator)
    q_min_new = torch.min(q1(batch.obs, new_action), q2(batch.obs, new_action))
    loss_actor = actor_loss(q_min_new, new_log_prob, alpha=alpha)
    actor_optimizer.zero_grad()
    loss_actor.backward()  # type: ignore[no-untyped-call]
    actor_optimizer.step()

    loss_alpha_value = float("nan")
    if alpha_optimizer is not None:
        loss_alpha = alpha_loss(log_alpha, new_log_prob, target_entropy=target_entropy)
        alpha_optimizer.zero_grad()
        loss_alpha.backward()  # type: ignore[no-untyped-call]
        alpha_optimizer.step()
        loss_alpha_value = float(loss_alpha.item())

    polyak_update(q1, q1_target, tau=tau)
    polyak_update(q2, q2_target, tau=tau)

    return {
        "critic_loss": float(loss_critic.item()),
        "actor_loss": float(loss_actor.item()),
        "alpha": alpha,
        "alpha_loss": loss_alpha_value,
        "q1_mean": float(q1_pred.detach().mean().item()),
        "entropy": float(-new_log_prob.detach().mean().item()),
    }


def _checkpoint_payload(
    config: SacConfig,
    *,
    run_id: str,
    step: int,
    history: list[dict[str, float]],
    actor: SquashedGaussianActor,
    q1: ContinuousQNetwork,
    q2: ContinuousQNetwork,
    q1_target: ContinuousQNetwork,
    q2_target: ContinuousQNetwork,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    log_alpha: torch.Tensor,
    alpha_optimizer: torch.optim.Optimizer | None,
    buffer: ReplayBuffer,
    rng: Rng,
    envs: ContinuousEnvPair,
    current_obs: np.ndarray,
    episode_return: float,
) -> dict[str, Any]:
    """Assemble the checkpoint payload for SAC's state (incl. temperature)."""
    return {
        "run_id": run_id,
        "config": asdict(config),
        "step": step,
        "history": history,
        "actor": actor.state_dict(),
        "q1": q1.state_dict(),
        "q2": q2.state_dict(),
        "q1_target": q1_target.state_dict(),
        "q2_target": q2_target.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "log_alpha": log_alpha.detach().clone(),
        "alpha_optimizer": None if alpha_optimizer is None else alpha_optimizer.state_dict(),
        "buffer": buffer.state(),
        "rng": rng_state(rng),
        "current_obs": current_obs,
        "episode_return": episode_return,
        "train_env": pickle_env(envs.train_env),
        "eval_env": pickle_env(envs.eval_env),
    }


def train(
    config: SacConfig,
    out_dir: Path | None = None,
    resume_from: Path | None = None,
    tracker: Tracker | None = None,
) -> TrainResult:
    """Train SAC per ``config`` and return the result.

    Args:
        config: Run configuration.
        out_dir: If given, writes the standard run directory there.
        resume_from: Optional ``checkpoint.pt``; the config must match the
            checkpoint's except the ``total_steps`` budget.
        tracker: Optional metrics mirror.

    Returns:
        The :class:`TrainResult`, including the final deterministic
        evaluation.

    Raises:
        ValueError: On schedule violations, ``checkpoint_every`` without
            ``out_dir``, or resume config mismatches.
    """
    start_time = time.perf_counter()
    _validate_schedule(config)
    if config.checkpoint_every > 0 and out_dir is None:
        raise ValueError("checkpoint_every > 0 requires out_dir (checkpoints live there).")

    def build_networks(
        pair: ContinuousEnvPair,
    ) -> tuple[SquashedGaussianActor, ContinuousQNetwork, ContinuousQNetwork]:
        actor = SquashedGaussianActor(
            pair.obs_dim,
            pair.act_dim,
            action_low=torch.as_tensor(pair.action_low, dtype=torch.float32),
            action_high=torch.as_tensor(pair.action_high, dtype=torch.float32),
            hidden_sizes=config.hidden_sizes,
        )
        q1 = ContinuousQNetwork(pair.obs_dim, pair.act_dim, hidden_sizes=config.hidden_sizes)
        q2 = ContinuousQNetwork(pair.obs_dim, pair.act_dim, hidden_sizes=config.hidden_sizes)
        return actor, q1, q2

    if resume_from is not None:
        payload = load_checkpoint(resume_from, expected_algo="sac")
        check_resume_config(payload["config"], asdict(config), budget_field="total_steps")
        run_id = str(payload["run_id"])
        rng = restore_rng(payload["rng"])
        envs = continuous_pair_from_envs(
            unpickle_env(payload["train_env"]), unpickle_env(payload["eval_env"])
        )
        actor, q1, q2 = build_networks(envs)
        q1_target = ContinuousQNetwork(envs.obs_dim, envs.act_dim, hidden_sizes=config.hidden_sizes)
        q2_target = ContinuousQNetwork(envs.obs_dim, envs.act_dim, hidden_sizes=config.hidden_sizes)
        actor.load_state_dict(payload["actor"])
        q1.load_state_dict(payload["q1"])
        q2.load_state_dict(payload["q2"])
        q1_target.load_state_dict(payload["q1_target"])
        q2_target.load_state_dict(payload["q2_target"])
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.lr)
        actor_optimizer.load_state_dict(payload["actor_optimizer"])
        critic_optimizer = torch.optim.Adam(
            list(q1.parameters()) + list(q2.parameters()), lr=config.lr
        )
        critic_optimizer.load_state_dict(payload["critic_optimizer"])
        log_alpha = payload["log_alpha"].clone().requires_grad_(config.auto_alpha)
        alpha_optimizer = None
        if config.auto_alpha:
            alpha_optimizer = torch.optim.Adam([log_alpha], lr=config.lr)
            if payload["alpha_optimizer"] is not None:
                alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        buffer = ReplayBuffer(
            config.buffer_capacity,
            obs_shape=(envs.obs_dim,),
            action_shape=(envs.act_dim,),
            action_dtype=np.float32,
        )
        buffer.restore_state(payload["buffer"])
        history = payload["history"]
        obs = payload["current_obs"]
        episode_return = float(payload["episode_return"])
        start_step = int(payload["step"]) + 1
    else:
        run_id = new_run_id("sac", config.env_id, config.seed)
        rng = seed_everything(config.seed)
        envs = make_continuous_env_pair(config.env_id, rng)
        actor, q1, q2 = build_networks(envs)
        q1_target = ContinuousQNetwork(envs.obs_dim, envs.act_dim, hidden_sizes=config.hidden_sizes)
        q2_target = ContinuousQNetwork(envs.obs_dim, envs.act_dim, hidden_sizes=config.hidden_sizes)
        q1_target.load_state_dict(q1.state_dict())
        q2_target.load_state_dict(q2.state_dict())
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.lr)
        critic_optimizer = torch.optim.Adam(
            list(q1.parameters()) + list(q2.parameters()), lr=config.lr
        )
        log_alpha = torch.zeros((), requires_grad=config.auto_alpha)
        if not config.auto_alpha:
            with torch.no_grad():
                log_alpha.fill_(math.log(config.fixed_alpha))
        alpha_optimizer = torch.optim.Adam([log_alpha], lr=config.lr) if config.auto_alpha else None
        buffer = ReplayBuffer(
            config.buffer_capacity,
            obs_shape=(envs.obs_dim,),
            action_shape=(envs.act_dim,),
            action_dtype=np.float32,
        )
        history = []
        obs, _ = envs.train_env.reset()
        episode_return = 0.0
        start_step = 1

    for param in list(q1_target.parameters()) + list(q2_target.parameters()):
        param.requires_grad_(False)

    target_entropy = (
        -float(envs.act_dim) if math.isnan(config.target_entropy) else config.target_entropy
    )

    if tracker is not None:
        tracker.start(
            run_id=run_id,
            algo="sac",
            env_id=config.env_id,
            seed=config.seed,
            config=asdict(config),
        )

    env = envs.train_env
    action_low_t = torch.as_tensor(envs.action_low, dtype=torch.float32)
    action_range_t = torch.as_tensor(envs.action_high - envs.action_low, dtype=torch.float32)
    stopped_early = False
    window: dict[str, list[float]] = {}
    window_returns: list[float] = []
    step = start_step - 1

    for step in range(start_step, config.total_steps + 1):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        if step <= config.warmup_steps:
            uniform = torch.rand(envs.act_dim, generator=rng.torch)
            action_np = (action_low_t + uniform * action_range_t).numpy()
        else:
            with torch.no_grad():
                action_t, _ = actor.sample(obs_t, generator=rng.torch)
            action_np = action_t.squeeze(0).numpy()
        next_obs, reward, terminated, truncated, _ = env.step(action_np)
        buffer.add(
            np.asarray(obs, dtype=np.float32),
            action_np.astype(np.float32),
            float(reward),
            next_obs,
            terminated,
        )
        episode_return += float(reward)
        if terminated or truncated:
            window_returns.append(episode_return)
            episode_return = 0.0
            obs, _ = env.reset()
        else:
            obs = next_obs

        if step > config.warmup_steps and step % config.train_freq == 0:
            batch = buffer.sample(config.batch_size, generator=rng.torch)
            metrics = sac_update(
                actor=actor,
                q1=q1,
                q2=q2,
                q1_target=q1_target,
                q2_target=q2_target,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                log_alpha=log_alpha,
                alpha_optimizer=alpha_optimizer,
                batch=batch,
                gamma=config.gamma,
                tau=config.tau,
                target_entropy=target_entropy,
                generator=rng.torch,
            )
            for key, value in metrics.items():
                window.setdefault(key, []).append(value)

        if step % config.log_every == 0:
            entry: dict[str, float] = {
                "env_steps": float(step),
                "buffer_size": float(len(buffer)),
                "train_return_mean": (
                    statistics.mean(window_returns) if window_returns else float("nan")
                ),
            }
            for key in ("critic_loss", "actor_loss", "alpha", "alpha_loss", "q1_mean", "entropy"):
                values = [v for v in window.get(key, []) if not math.isnan(v)]
                entry[key] = statistics.mean(values) if values else float("nan")
            window, window_returns = {}, []

            if config.eval_every > 0 and step % config.eval_every == 0:
                stats = evaluate_policy(
                    envs.eval_env,
                    lambda obs: actor.deterministic_action(obs),
                    episodes=config.eval_episodes,
                )
                entry["eval_return_mean"] = stats.mean_return
                entry["eval_return_std"] = stats.std_return
                logger.info(
                    "step %d | train return %.1f | eval return %.1f | alpha %.3f | entropy %.2f",
                    step,
                    entry["train_return_mean"],
                    stats.mean_return,
                    entry["alpha"],
                    entry["entropy"],
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
                    algo="sac",
                    payload=_checkpoint_payload(
                        config,
                        run_id=run_id,
                        step=step,
                        history=history,
                        actor=actor,
                        q1=q1,
                        q2=q2,
                        q1_target=q1_target,
                        q2_target=q2_target,
                        actor_optimizer=actor_optimizer,
                        critic_optimizer=critic_optimizer,
                        log_alpha=log_alpha,
                        alpha_optimizer=alpha_optimizer,
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
        lambda obs: actor.deterministic_action(obs),
        episodes=config.eval_episodes,
    )
    result = TrainResult(
        run_id=run_id,
        actor=actor,
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
        write_run_record(out_dir, run_id=run_id, algo="sac", env_id=config.env_id, seed=config.seed)
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
        save_final_model(out_dir, algo="sac", state_dict=actor.state_dict())
        write_summary(
            out_dir,
            run_id=run_id,
            algo="sac",
            env_id=config.env_id,
            seed=config.seed,
            history=history,
            final_eval=final_eval,
            total_env_steps=total_env_steps,
            stopped_early=stopped_early,
            wall_time_s=result.wall_time_s,
        )
    return result


ConfigStore.instance().store(name="sac", node=SacConfig)


@hydra.main(version_base="1.3", config_name="sac")
def main(cfg: SacConfig) -> None:
    """Hydra entry point: ``python -m rlcore.agents.sac.train [key=value ...]``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config = cast("SacConfig", OmegaConf.to_object(cast("Any", cfg)))
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

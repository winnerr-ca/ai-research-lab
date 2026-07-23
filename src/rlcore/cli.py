"""User-facing entry points: ``rlcore-train`` and ``rlcore-evaluate``.

``rlcore-train`` is the unified Hydra entry point: the algorithm is a config
group, so one command trains any algorithm with full override support::

    rlcore-train algo=ppo
    rlcore-train algo=reinforce algo.seed=3 algo.lr=0.005

``rlcore-evaluate`` re-evaluates a finished run from its run directory
(``run.json`` + ``config.yaml`` + ``final_model.pt``)::

    rlcore-evaluate outputs/2026-.../ --episodes 50
    rlcore-evaluate outputs/2026-.../ --stochastic --eval-seed 7

Everything here is local: no external service, account, or credentials.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import MISSING, OmegaConf

from rlcore.agents.dqn.model import QNetwork, epsilon_greedy_action
from rlcore.agents.dqn.train import DqnConfig
from rlcore.agents.dqn.train import TrainResult as DqnTrainResult
from rlcore.agents.dqn.train import train as dqn_train
from rlcore.agents.ppo.model import ActorCritic
from rlcore.agents.ppo.train import PpoConfig
from rlcore.agents.ppo.train import TrainResult as PpoTrainResult
from rlcore.agents.ppo.train import train as ppo_train
from rlcore.agents.reinforce.policy import CategoricalMlpPolicy
from rlcore.agents.reinforce.train import ReinforceConfig
from rlcore.agents.reinforce.train import TrainResult as ReinforceTrainResult
from rlcore.agents.reinforce.train import train as reinforce_train
from rlcore.envs import make_discrete_env_pair
from rlcore.evaluation import EvalStats, evaluate_policy
from rlcore.experiments import load_final_model
from rlcore.tracking import Tracker, make_tracker
from rlcore.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


@dataclass
class TrainRunConfig:
    """Root config for ``rlcore-train``: pick an algorithm config group."""

    defaults: list[Any] = field(default_factory=lambda: [{"algo": MISSING}, "_self_"])
    algo: Any = MISSING
    resume_from: str | None = None
    track: str = "none"
    track_project: str = "rlcore"


_cs = ConfigStore.instance()
_cs.store(name="train_run", node=TrainRunConfig)
_cs.store(group="algo", name="reinforce", node=ReinforceConfig)
_cs.store(group="algo", name="ppo", node=PpoConfig)
_cs.store(group="algo", name="dqn", node=DqnConfig)


def dispatch_train(
    algo_config: object,
    out_dir: Path | None,
    resume_from: Path | None = None,
    tracker: Tracker | None = None,
) -> None:
    """Route a resolved algorithm config to its trainer."""
    result: ReinforceTrainResult | PpoTrainResult | DqnTrainResult
    if isinstance(algo_config, ReinforceConfig):
        result = reinforce_train(
            algo_config, out_dir=out_dir, resume_from=resume_from, tracker=tracker
        )
    elif isinstance(algo_config, PpoConfig):
        result = ppo_train(algo_config, out_dir=out_dir, resume_from=resume_from, tracker=tracker)
    elif isinstance(algo_config, DqnConfig):
        result = dqn_train(algo_config, out_dir=out_dir, resume_from=resume_from, tracker=tracker)
    else:
        raise ValueError(
            f"Unknown algorithm config type {type(algo_config).__name__}; "
            "expected ReinforceConfig, PpoConfig, or DqnConfig."
        )
    logger.info(
        "run %s done: final eval return %.1f +- %.1f | %d env steps | %.1fs",
        result.run_id,
        result.final_eval.mean_return,
        result.final_eval.std_return,
        result.total_env_steps,
        result.wall_time_s,
    )


@hydra.main(version_base="1.3", config_name="train_run")
def train_main(cfg: TrainRunConfig) -> None:
    """Hydra entry point for ``rlcore-train``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    resolved = cast("TrainRunConfig", OmegaConf.to_object(cast("Any", cfg)))
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    resume = Path(resolved.resume_from) if resolved.resume_from else None
    tracker = make_tracker(resolved.track, project=resolved.track_project)
    dispatch_train(resolved.algo, out_dir, resume_from=resume, tracker=tracker)


def evaluate_run(
    run_dir: Path,
    *,
    episodes: int = 20,
    deterministic: bool = True,
    eval_seed: int = 12345,
) -> EvalStats:
    """Re-evaluate a finished run from its run directory.

    Rebuilds the network from ``config.yaml`` + ``run.json``, loads
    ``final_model.pt``, and runs fresh evaluation episodes on a newly
    seeded env (independent of every training stream).

    Args:
        run_dir: A run directory written by ``rlcore-train`` (or ``train()``
            with ``out_dir``).
        episodes: Evaluation episodes, >= 1.
        deterministic: Greedy actions; ``False`` samples from the policy.
        eval_seed: Seed for the evaluation env and (stochastic) sampling.

    Returns:
        The evaluation statistics (also appended to ``evaluations.jsonl``
        in the run directory).

    Raises:
        FileNotFoundError: If required run files are missing.
        ValueError: On artifact/algorithm mismatches.
    """
    run_record = json.loads((run_dir / "run.json").read_text())
    algo = run_record["algo"]
    config = OmegaConf.load(run_dir / "config.yaml")
    hidden_sizes = list(config.hidden_sizes)
    env_id = str(config.env_id)

    rng = seed_everything(eval_seed)
    envs = make_discrete_env_pair(env_id, rng)
    state_dict = load_final_model(run_dir / "final_model.pt", expected_algo=algo)

    generator = torch.Generator().manual_seed(eval_seed)
    if algo == "reinforce":
        policy = CategoricalMlpPolicy(envs.obs_dim, envs.n_actions, hidden_sizes=hidden_sizes)
        policy.load_state_dict(state_dict)

        def select_action(obs: torch.Tensor) -> torch.Tensor:
            return policy.act(obs, generator=generator, deterministic=deterministic)

    elif algo == "ppo":
        model = ActorCritic(envs.obs_dim, envs.n_actions, hidden_sizes=hidden_sizes)
        model.load_state_dict(state_dict)

        def select_action(obs: torch.Tensor) -> torch.Tensor:
            return model.act(obs, generator=generator, deterministic=deterministic)[0]

    elif algo == "dqn":
        q_net = QNetwork(envs.obs_dim, envs.n_actions, hidden_sizes=hidden_sizes)
        q_net.load_state_dict(state_dict)
        # Stochastic evaluation for a value-based policy means eps-greedy
        # with a small fixed epsilon (documented; DQN has no action dist).
        eval_epsilon = 0.0 if deterministic else 0.05

        def select_action(obs: torch.Tensor) -> torch.Tensor:
            return epsilon_greedy_action(q_net, obs, epsilon=eval_epsilon, generator=generator)

    else:
        raise ValueError(f"run.json names unknown algorithm {algo!r}.")

    stats = evaluate_policy(envs.eval_env, select_action, episodes=episodes)
    envs.close()

    entry = {
        "run_id": run_record["run_id"],
        "episodes": episodes,
        "deterministic": deterministic,
        "eval_seed": eval_seed,
        "mean_return": stats.mean_return,
        "std_return": stats.std_return,
        "episode_returns": list(stats.episode_returns),
    }
    with (run_dir / "evaluations.jsonl").open("a") as stream:
        stream.write(json.dumps(entry) + "\n")
    return stats


def evaluate_main() -> None:
    """Argparse entry point for ``rlcore-evaluate``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=evaluate_run.__doc__)
    parser.add_argument("run_dir", type=Path, help="run directory written by rlcore-train")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--stochastic", action="store_true", help="sample actions instead of greedy argmax"
    )
    parser.add_argument("--eval-seed", type=int, default=12345)
    args = parser.parse_args()

    stats = evaluate_run(
        args.run_dir,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        eval_seed=args.eval_seed,
    )
    logger.info(
        "%s eval over %d episodes (%s): %.1f +- %.1f (min %.1f, max %.1f)",
        args.run_dir,
        args.episodes,
        "stochastic" if args.stochastic else "greedy",
        stats.mean_return,
        stats.std_return,
        stats.min_return,
        stats.max_return,
    )


if __name__ == "__main__":
    train_main()

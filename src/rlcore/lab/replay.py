"""Trajectory recording for the dashboard's agent view.

Rebuilds a finished run's policy exactly the way ``rlcore-evaluate``
does (:func:`rlcore.cli.evaluate_run`) and plays one greedy episode on a
freshly seeded evaluation env — but records the raw observation at every
step so the browser can *draw* the agent acting (cart position and pole
angle for CartPole, link angles for Acrobot/Pendulum). Nothing here
touches training artifacts: the run directory is read-only, and the
replay writes no files.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import torch
from omegaconf import OmegaConf

from rlcore.agents.dqn.model import QNetwork, epsilon_greedy_action
from rlcore.agents.ppo.model import ActorCritic
from rlcore.agents.reinforce.policy import CategoricalMlpPolicy
from rlcore.agents.sac.model import SquashedGaussianActor
from rlcore.envs import make_continuous_env_pair, make_discrete_env_pair
from rlcore.experiments import load_final_model
from rlcore.utils.seeding import seed_everything

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from torch import Tensor

logger = logging.getLogger(__name__)

#: Hard cap on recorded steps (CartPole's own limit; keeps payloads small).
_MAX_STEPS = 1000


def record_trajectory(run_dir: Path, *, eval_seed: int = 12345) -> dict[str, Any]:
    """Play one greedy episode of a finished run and record every state.

    Args:
        run_dir: A run directory holding ``run.json``, ``config.yaml``,
            and ``final_model.pt``.
        eval_seed: Seed for the evaluation env (independent of training,
            same default as ``rlcore-evaluate``).

    Returns:
        ``{"algo", "env_id", "episode_return", "steps", "truncated",
        "states", "actions", "rewards"}`` — ``states`` holds the raw
        observation vector per step (initial state included).

    Raises:
        FileNotFoundError: If required run files are missing.
        ValueError: On artifact/algorithm mismatches.
    """
    run_record = json.loads((run_dir / "run.json").read_text())
    algo = str(run_record["algo"])
    config = OmegaConf.load(run_dir / "config.yaml")
    hidden_sizes = [int(size) for size in config.hidden_sizes]
    env_id = str(config.env_id)

    rng = seed_everything(eval_seed)
    state_dict = load_final_model(run_dir / "final_model.pt", expected_algo=algo)

    close: Callable[[], None]
    if algo == "sac":
        cont = make_continuous_env_pair(env_id, rng)
        actor = SquashedGaussianActor(
            cont.obs_dim,
            cont.act_dim,
            action_low=torch.as_tensor(cont.action_low),
            action_high=torch.as_tensor(cont.action_high),
            hidden_sizes=hidden_sizes,
        )
        actor.load_state_dict(state_dict)
        select_action: Callable[[Tensor], Tensor] = actor.deterministic_action
        env, close = cont.eval_env, cont.close
    else:
        envs = make_discrete_env_pair(env_id, rng)
        if algo == "reinforce":
            policy = CategoricalMlpPolicy(envs.obs_dim, envs.n_actions, hidden_sizes=hidden_sizes)
            policy.load_state_dict(state_dict)

            def select_action(obs: Tensor) -> Tensor:
                return policy.act(obs, deterministic=True)

        elif algo == "ppo":
            model = ActorCritic(envs.obs_dim, envs.n_actions, hidden_sizes=hidden_sizes)
            model.load_state_dict(state_dict)

            def select_action(obs: Tensor) -> Tensor:
                return model.act(obs, deterministic=True)[0]

        elif algo == "dqn":
            q_net = QNetwork(envs.obs_dim, envs.n_actions, hidden_sizes=hidden_sizes)
            q_net.load_state_dict(state_dict)

            def select_action(obs: Tensor) -> Tensor:
                return epsilon_greedy_action(q_net, obs, epsilon=0.0)

        else:
            raise ValueError(f"run.json names unknown algorithm {algo!r}.")
        env, close = envs.eval_env, envs.close

    # Same stepping contract as rlcore.evaluation.evaluate_policy, plus
    # state recording.
    states: list[list[float]] = []
    actions: list[Any] = []
    rewards: list[float] = []
    obs, _ = env.reset(seed=eval_seed)
    states.append([float(value) for value in obs])
    episode_return = 0.0
    terminated = truncated = False
    with torch.no_grad():
        while not (terminated or truncated) and len(actions) < _MAX_STEPS:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action_t = select_action(obs_t)
            if action_t.is_floating_point():
                env_action: object = action_t.detach().squeeze(0).cpu().numpy()
                actions.append([float(value) for value in action_t.squeeze(0)])
            else:
                env_action = int(action_t.item())
                actions.append(int(action_t.item()))
            obs, reward, terminated, truncated, _ = env.step(env_action)
            states.append([float(value) for value in obs])
            rewards.append(float(reward))
            episode_return += float(reward)
    close()
    return {
        "algo": algo,
        "env_id": env_id,
        "episode_return": episode_return,
        "steps": len(actions),
        "truncated": bool(truncated),
        "states": states,
        "actions": actions,
        "rewards": rewards,
    }

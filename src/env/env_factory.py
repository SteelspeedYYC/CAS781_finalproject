"""
Environment Factory Module

Provides a unified interface for creating Gymnasium environments
used in controlled reinforcement learning experiments.

This abstraction helps ensure consistency across experimental setups
and reduces confounding variables when switching environments.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import gymnasium as gym


SUPPORTED_ENVS = {"frozenlake", "taxi"}


def make_env(
    env_name: str,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    **kwargs: Any,
) -> gym.Env:
    """
    Create and return a Gymnasium environment.

    Supported logical names:
        - "frozenlake"
        - "taxi"

    Parameters:
    
    env_name : str
        Logical environment name used by this project.
    seed : int | None
        Optional random seed for environment reset.
    render_mode : str | None
        Optional Gymnasium render mode.
    **kwargs : Any
        Extra environment-specific configuration.

    Returns:
    
    gym.Env
        Configured Gymnasium environment instance.
    """
    env_key = env_name.strip().lower()

    if env_key not in SUPPORTED_ENVS:
        raise ValueError(
            f"Unsupported environment '{env_name}'. "
            f"Supported environments: {sorted(SUPPORTED_ENVS)}"
        )

    if env_key == "frozenlake":
        env = _make_frozenlake(render_mode=render_mode, **kwargs)
    elif env_key == "taxi":
        env = _make_taxi(render_mode=render_mode, **kwargs)
    else:
        raise ValueError(f"Internal error: environment '{env_name}' not handled.")

    if seed is not None:
        env.reset(seed=seed)

    return env


def get_env_config(env_name: str, **overrides: Any) -> Dict[str, Any]:
    """
    Return the default config dictionary for a supported environment,
    optionally overridden by user-provided values.
    """
    env_key = env_name.strip().lower()

    if env_key == "frozenlake":
        config: Dict[str, Any] = {
            "gym_id": "FrozenLake-v1",
            "map_name": "4x4",
            "is_slippery": False,
        }
    elif env_key == "taxi":
        config = {
            "gym_id": "Taxi-v3",
        }
    else:
        raise ValueError(
            f"Unsupported environment '{env_name}'. "
            f"Supported environments: {sorted(SUPPORTED_ENVS)}"
        )

    config.update(overrides)
    return config


def get_env_spaces(env: gym.Env) -> Tuple[int, int]:
    """
    Return (n_states, n_actions) for discrete-state, discrete-action environments.

    Raises
    ------
    TypeError
        If the environment does not expose discrete observation/action spaces.
    """
    if not hasattr(env.observation_space, "n"):
        raise TypeError(
            "Observation space is not discrete. "
            "This project currently expects a tabular/discrete state space."
        )

    if not hasattr(env.action_space, "n"):
        raise TypeError(
            "Action space is not discrete. "
            "This project currently expects a discrete action space."
        )

    n_states = int(env.observation_space.n)
    n_actions = int(env.action_space.n)
    return n_states, n_actions


def get_env_summary(env_name: str, **kwargs: Any) -> Dict[str, Any]:
    """
    Create an environment temporarily and return a small summary dictionary.
    Useful for logging and quick checks in notebooks.
    """
    env = make_env(env_name, **kwargs)
    n_states, n_actions = get_env_spaces(env)

    summary = {
        "env_name": env_name.strip().lower(),
        "observation_space": str(env.observation_space),
        "action_space": str(env.action_space),
        "n_states": n_states,
        "n_actions": n_actions,
    }

    env.close()
    return summary


def _make_frozenlake(
    render_mode: Optional[str] = None,
    map_name: str = "4x4",
    is_slippery: bool = False,
    **kwargs: Any,
) -> gym.Env:
    """
    Internal factory for FrozenLake-v1.
    """
    return gym.make(
        "FrozenLake-v1",
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode=render_mode,
        **kwargs,
    )


def _make_taxi(
    render_mode: Optional[str] = None,
    **kwargs: Any,
) -> gym.Env:
    """
    Internal factory for Taxi-v3.
    """
    return gym.make(
        "Taxi-v3",
        render_mode=render_mode,
        **kwargs,
    )
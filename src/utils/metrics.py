from __future__ import annotations

from typing import Any, Dict, Iterable, Optional
import numpy as np


def moving_average(values: Iterable[float], window: int = 20) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)

    if window <= 1 or len(values) < window:
        return values

    kernel = np.ones(window, dtype=float) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    return smoothed


def _safe_mean(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def _safe_std(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.std(values))


def _safe_min(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.min(values))


def _safe_max(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.max(values))


def _safe_last(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(values[-1])


def _tail_mean(values: np.ndarray, k: int) -> float:
    if len(values) == 0:
        return float("nan")
    k = min(k, len(values))
    return float(np.mean(values[-k:]))


def _first_success_episode(rewards: np.ndarray, success_threshold: float = 1.0) -> Optional[int]:
    success_indices = np.where(rewards >= success_threshold)[0]
    if len(success_indices) == 0:
        return None
    return int(success_indices[0] + 1)  # 1-based episode index


def _success_rate(rewards: np.ndarray, last_k: int = 100, success_threshold: float = 1.0) -> float:
    if len(rewards) == 0:
        return float("nan")
    k = min(last_k, len(rewards))
    tail = rewards[-k:]
    return float(np.mean(tail >= success_threshold))


def _best_moving_avg(rewards: np.ndarray, window: int = 20) -> float:
    if len(rewards) == 0:
        return float("nan")
    smoothed = moving_average(rewards, window=window)
    if len(smoothed) == 0:
        return float("nan")
    return float(np.max(smoothed))


def _convergence_episode(
    rewards: np.ndarray,
    window: int = 20,
    success_rate_threshold: float = 0.8,
    success_threshold: float = 1.0,
) -> Optional[int]:
    """
    A simple operational definition:
    first episode index such that, over the next 'window' episodes,
    the success rate reaches at least 'success_rate_threshold'.
    """
    if len(rewards) < window:
        return None

    binary_success = (rewards >= success_threshold).astype(float)

    for i in range(len(binary_success) - window + 1):
        local_rate = np.mean(binary_success[i:i + window])
        if local_rate >= success_rate_threshold:
            return int(i + 1)  # 1-based episode index

    return None


def summarize_run(
    training_logs: Dict[str, list],
    eval_logs: Dict[str, float],
    config: Any,
    env_name: str,
    seed: Optional[int] = None,
    success_threshold: float = 1.0,
) -> Dict[str, Any]:
    """
    Summarize one run into a flat dictionary suitable for CSV export.
    """
    rewards = np.asarray(training_logs.get("episode_rewards", []), dtype=float)
    lengths = np.asarray(training_logs.get("episode_lengths", []), dtype=float)
    epsilons = np.asarray(training_logs.get("epsilons", []), dtype=float)

    summary = {
        # run metadata
        "env": env_name,
        "seed": seed,
        "alpha": getattr(config, "alpha", None),
        "gamma": getattr(config, "gamma", None),
        "epsilon_init": getattr(config, "epsilon", None),
        "epsilon_min": getattr(config, "epsilon_min", None),
        "epsilon_decay": getattr(config, "epsilon_decay", None),
        "n_episodes": getattr(config, "n_episodes", None),
        "max_steps_per_episode": getattr(config, "max_steps_per_episode", None),

        # training reward summary
        "train_reward_final": _safe_last(rewards),
        "train_reward_mean": _safe_mean(rewards),
        "train_reward_std": _safe_std(rewards),
        "train_reward_min": _safe_min(rewards),
        "train_reward_max": _safe_max(rewards),

        # training length summary
        "train_length_final": _safe_last(lengths),
        "train_length_mean": _safe_mean(lengths),
        "train_length_std": _safe_std(lengths),
        "train_length_min": _safe_min(lengths),
        "train_length_max": _safe_max(lengths),

        # epsilon summary
        "epsilon_final": _safe_last(epsilons),

        # tail performance
        "train_reward_last_10_mean": _tail_mean(rewards, 10),
        "train_reward_last_50_mean": _tail_mean(rewards, 50),
        "train_reward_last_100_mean": _tail_mean(rewards, 100),
        "train_length_last_10_mean": _tail_mean(lengths, 10),
        "train_length_last_50_mean": _tail_mean(lengths, 50),
        "train_length_last_100_mean": _tail_mean(lengths, 100),

        # success / convergence
        "success_rate_last_100": _success_rate(rewards, last_k=100, success_threshold=success_threshold),
        "first_success_episode": _first_success_episode(rewards, success_threshold=success_threshold),
        "convergence_episode": _convergence_episode(
            rewards,
            window=20,
            success_rate_threshold=0.8,
            success_threshold=success_threshold,
        ),
        "best_moving_avg_reward_20": _best_moving_avg(rewards, window=20),

        # evaluation
        "eval_mean_reward": eval_logs.get("mean_reward", np.nan),
        "eval_std_reward": eval_logs.get("std_reward", np.nan),
        "eval_mean_length": eval_logs.get("mean_length", np.nan),
        "eval_std_length": eval_logs.get("std_length", np.nan),
    }

    return summary
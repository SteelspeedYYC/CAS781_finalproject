"""
helper.py

General utility functions for experiment logging, saving results,
and plotting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def moving_average(values: Iterable[float], window: int = 20) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)

    if window <= 1 or len(values) < window:
        return values

    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    return smoothed


def save_results_csv(df: pd.DataFrame, filepath: str | Path) -> None:
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    df.to_csv(filepath, index=False)


def plot_rewards(
    rewards: Iterable[float],
    title: str = "Training Reward",
    xlabel: str = "Episode",
    ylabel: str = "Reward",
    save_path: Optional[str | Path] = None,
    smooth_window: Optional[int] = None,
) -> None:
    rewards = np.asarray(list(rewards), dtype=float)

    plt.figure(figsize=(8, 5))
    plt.plot(rewards, label="Raw Reward")

    if smooth_window is not None and smooth_window > 1:
        smoothed = moving_average(rewards, window=smooth_window)
        x_vals = np.arange(len(smoothed)) + smooth_window - 1
        plt.plot(x_vals, smoothed, label=f"Moving Avg ({smooth_window})")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()
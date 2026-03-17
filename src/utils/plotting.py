from __future__ import annotations

from typing import Iterable, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .metrics import moving_average
from .logging import ensure_dir


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
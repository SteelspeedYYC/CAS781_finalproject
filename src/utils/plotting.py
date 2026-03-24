from __future__ import annotations

from typing import List, Iterable, Optional
from pathlib import Path

import numpy as np
import pandas as pd
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


def plot_heatmap(
    df: pd.DataFrame,
    x_param: str,
    y_param: str,
    value: str,
    title: str = "Heatmap",
    save_path: str | None = None,
):
    """
    Plot a heatmap for hyperparameter tuning results.

    Parameters:
    - df: summary dataframe (from CSV)
    - x_param: column name for x-axis (e.g., "gamma")
    - y_param: column name for y-axis (e.g., "alpha")
    - value: column to visualize (e.g., "eval_mean_reward")
    """

    # pivot table
    pivot = df.pivot_table(
        index=y_param,
        columns=x_param,
        values=value,
        aggfunc="mean"
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(pivot, aspect="auto")

    # ticks
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)

    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.title(title)

    # color bar
    cbar = plt.colorbar()
    cbar.set_label(value)

    plt.tight_layout()

    if save_path is not None:
        from pathlib import Path
        from .logging import ensure_dir

        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


def plot_heatmap_annotated(
    df: pd.DataFrame,
    x_param: str,
    y_param: str,
    value: str,
    title: str = "Heatmap",
):
    pivot = df.pivot_table(
        index=y_param,
        columns=x_param,
        values=value,
        aggfunc="mean"
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(pivot, aspect="auto")

    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.title(title)

    plt.colorbar(label=value)
    plt.tight_layout()
    plt.show()


def plot_multi_run_curve(
    reward_sequences: List[Iterable[float]],
    title: str = "Multi-run Learning Curve",
    xlabel: str = "Episode",
    ylabel: str = "Reward",
    save_path: Optional[str | Path] = None,
):
    """
    Plot mean + std shaded learning curve over multiple runs.
    """

    min_len = min(len(seq) for seq in reward_sequences)
    rewards = np.array([np.array(seq[:min_len]) for seq in reward_sequences])

    mean = rewards.mean(axis=0)
    std = rewards.std(axis=0)

    x = np.arange(len(mean))

    plt.figure(figsize=(8, 5))

    # mean curve
    plt.plot(x, mean, label="Mean Reward")

    # std shading
    plt.fill_between(x, mean - std, mean + std, alpha=0.3, label="±1 Std")

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


def plot_multi_run_comparison(
    curves_dict,
    title="Multi-run Comparison",
):
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))

    for label, sequences in curves_dict.items():
        min_len = min(len(seq) for seq in sequences)
        data = np.array([seq[:min_len] for seq in sequences])

        mean = data.mean(axis=0)
        std = data.std(axis=0)

        x = np.arange(len(mean))

        plt.plot(x, mean, label=label)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.show()
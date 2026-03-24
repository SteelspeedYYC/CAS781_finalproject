from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_results_csv(df: pd.DataFrame, filepath: str | Path) -> None:
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    df.to_csv(filepath, index=False)


def save_trajectory_csv(
    training_logs: dict,
    filepath: str | Path,
) -> None:
    """
    Save per-episode trajectory data.
    """
    import pandas as pd
    from pathlib import Path

    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    df = pd.DataFrame({
        "episode": list(range(1, len(training_logs["episode_rewards"]) + 1)),
        "reward": training_logs.get("episode_rewards", []),
        "length": training_logs.get("episode_lengths", []),
        "epsilon": training_logs.get("epsilons", []),
    })

    df.to_csv(filepath, index=False)


def records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(records)


def append_record_to_csv(record: Dict[str, Any], filepath: str | Path) -> None:
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    row_df = pd.DataFrame([record])

    if filepath.exists():
        row_df.to_csv(filepath, mode="a", header=False, index=False)
    else:
        row_df.to_csv(filepath, mode="w", header=True, index=False)


def fmt(x):
    return f"{x:.3f}".rstrip("0").rstrip(".")


def make_run_name(config, env_name: str, seed: int) -> str:
    return (
        f"{env_name}"
        f"_a{fmt(config.alpha)}"
        f"_g{fmt(config.gamma)}"
        f"_seed{seed}"
    )


def load_reward_sequences(
    env_name: str,
    alpha: float,
    gamma: float,
    seeds: list[int],
    base_dir: str = "results/raw/trajectories",
):
    sequences = []

    for seed in seeds:
        file_name = f"{env_name}_a{alpha}_g{gamma}_seed{seed}.csv"
        file_path = Path(base_dir) / file_name

        if not file_path.exists():
            print(f"[Warning] Missing file: {file_path}")
            continue

        df = pd.read_csv(file_path)
        sequences.append(df["reward"].values)

    return sequences
"""Script to create test fixtures for LeRobot reader tests.

This script creates minimal v2.1 and v3.0 sample datasets for testing.
Run this script to generate the fixtures:
    python tests/fixtures/create_fixtures.py
"""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def create_v21_sample(output_dir: Path) -> None:
    """Create a minimal LeRobot v2.1 sample dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create directory structure
    meta_dir = output_dir / "meta"
    data_dir = output_dir / "data" / "chunk-0"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create info.json
    info = {
        "codebase_version": "v2.1",
        "robot_type": "test_robot",
        "fps": 30,
        "chunks_size": 1000,
        "total_episodes": 2,
        "total_frames": 10,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [6]},
            "action": {"dtype": "float32", "shape": [6]},
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Create episodes.jsonl
    episodes = [
        {"episode_index": 0, "start_idx": 0, "end_idx": 5, "length": 5, "task": "pick object"},
        {"episode_index": 1, "start_idx": 5, "end_idx": 10, "length": 5, "task": "place object"},
    ]
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")

    # Create episode parquet files
    for ep in episodes:
        ep_idx = ep["episode_index"]
        length = ep["length"]

        # Create data arrays
        states = np.random.randn(length, 6).astype(np.float32)
        actions = np.random.randn(length, 6).astype(np.float32)
        frame_indices = np.arange(length)
        episode_indices = np.full(length, ep_idx)

        # Create pyarrow table
        table = pa.table({
            "frame_index": frame_indices,
            "episode_index": episode_indices,
            "observation.state": [states[i].tolist() for i in range(length)],
            "action": [actions[i].tolist() for i in range(length)],
        })

        # Write parquet
        parquet_path = data_dir / f"episode_{ep_idx:06d}.parquet"
        pq.write_table(table, parquet_path)

    print(f"Created v2.1 sample at: {output_dir}")


def create_v30_sample(output_dir: Path) -> None:
    """Create a minimal LeRobot v3.0 sample dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create directory structure
    meta_dir = output_dir / "meta"
    episodes_meta_dir = meta_dir / "episodes"
    data_dir = output_dir / "data" / "chunk-0"
    meta_dir.mkdir(parents=True, exist_ok=True)
    episodes_meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create info.json
    info = {
        "codebase_version": "v3.0",
        "robot_type": "test_robot",
        "fps": 30,
        "chunks_size": 1000,
        "total_episodes": 2,
        "total_frames": 10,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [6]},
            "action": {"dtype": "float32", "shape": [6]},
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Create episode metadata parquet files and data files
    episodes = [
        {"episode_index": 0, "start_idx": 0, "end_idx": 5, "length": 5, "task": "pick object"},
        {"episode_index": 1, "start_idx": 5, "end_idx": 10, "length": 5, "task": "place object"},
    ]

    for ep in episodes:
        ep_idx = ep["episode_index"]
        length = ep["length"]

        # Create episode metadata parquet (v3.0 style)
        ep_meta_table = pa.table({
            "episode_index": [ep_idx],
            "start_idx": [ep["start_idx"]],
            "end_idx": [ep["end_idx"]],
            "length": [length],
            "task": [ep["task"]],
        })
        ep_meta_path = episodes_meta_dir / f"episode_{ep_idx:06d}.parquet"
        pq.write_table(ep_meta_table, ep_meta_path)

        # Create data arrays
        states = np.random.randn(length, 6).astype(np.float32)
        actions = np.random.randn(length, 6).astype(np.float32)
        frame_indices = np.arange(length)
        episode_indices = np.full(length, ep_idx)

        # Create pyarrow table for data
        data_table = pa.table({
            "frame_index": frame_indices,
            "episode_index": episode_indices,
            "observation.state": [states[i].tolist() for i in range(length)],
            "action": [actions[i].tolist() for i in range(length)],
        })

        # Write data parquet
        parquet_path = data_dir / f"episode_{ep_idx:06d}.parquet"
        pq.write_table(data_table, parquet_path)

    print(f"Created v3.0 sample at: {output_dir}")


def main() -> None:
    """Create all test fixtures."""
    fixtures_dir = Path(__file__).parent

    create_v21_sample(fixtures_dir / "v21_sample")
    create_v30_sample(fixtures_dir / "v30_sample")

    print("\nAll fixtures created successfully!")


if __name__ == "__main__":
    main()

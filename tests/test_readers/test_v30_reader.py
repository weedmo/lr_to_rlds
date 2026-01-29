"""Tests for LeRobot v3.0 reader."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot_to_rlds.core.exceptions import EpisodeProcessingError, SchemaError
from lerobot_to_rlds.core.types import LeRobotVersion
from lerobot_to_rlds.readers.v30_reader import V30Reader


class TestV30Reader:
    """Tests for V30Reader class."""

    @pytest.fixture
    def v30_dataset(self, tmp_path: Path) -> Path:
        """Create a minimal v3.0 dataset for testing."""
        meta_dir = tmp_path / "meta"
        episodes_meta_dir = meta_dir / "episodes"
        data_dir = tmp_path / "data" / "chunk-0"
        meta_dir.mkdir(parents=True)
        episodes_meta_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)

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
            json.dump(info, f)

        # Create episode metadata and data files
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
            pq.write_table(ep_meta_table, episodes_meta_dir / f"episode_{ep_idx:06d}.parquet")

            # Create data parquet
            states = np.random.randn(length, 6).astype(np.float32)
            actions = np.random.randn(length, 6).astype(np.float32)

            data_table = pa.table({
                "frame_index": np.arange(length),
                "episode_index": np.full(length, ep_idx),
                "observation.state": [states[i].tolist() for i in range(length)],
                "action": [actions[i].tolist() for i in range(length)],
            })
            pq.write_table(data_table, data_dir / f"episode_{ep_idx:06d}.parquet")

        return tmp_path

    def test_version_property(self, v30_dataset: Path) -> None:
        """Should return V30 version."""
        reader = V30Reader(v30_dataset)
        assert reader.version == LeRobotVersion.V30

    def test_dataset_name(self, v30_dataset: Path) -> None:
        """Should return directory name as dataset name."""
        reader = V30Reader(v30_dataset)
        assert reader.dataset_name == v30_dataset.name

    def test_load_metadata(self, v30_dataset: Path) -> None:
        """Should load metadata from info.json."""
        reader = V30Reader(v30_dataset)
        metadata = reader.metadata

        assert metadata["codebase_version"] == "v3.0"
        assert metadata["robot_type"] == "test_robot"
        assert metadata["fps"] == 30
        assert metadata["total_episodes"] == 2

    def test_load_metadata_missing_file(self, tmp_path: Path) -> None:
        """Should raise error when info.json is missing."""
        meta_dir = tmp_path / "meta"
        episodes_dir = meta_dir / "episodes"
        meta_dir.mkdir(parents=True)
        episodes_dir.mkdir(parents=True)
        # No info.json

        reader = V30Reader(tmp_path)
        with pytest.raises(SchemaError) as exc_info:
            _ = reader.metadata

        assert "info.json not found" in str(exc_info.value)

    def test_build_feature_specs(self, v30_dataset: Path) -> None:
        """Should build feature specs from metadata."""
        reader = V30Reader(v30_dataset)
        features = reader.features

        assert "observation.state" in features
        assert features["observation.state"].dtype == "float32"
        assert features["observation.state"].shape == (6,)

        assert "action" in features
        assert features["action"].dtype == "float32"

    def test_list_episodes(self, v30_dataset: Path) -> None:
        """Should list all episodes from parquet metadata files."""
        reader = V30Reader(v30_dataset)
        episodes = reader.list_episodes()

        assert len(episodes) == 2

        ep0 = episodes[0]
        assert ep0.episode_index == 0
        assert ep0.length == 5
        assert ep0.task == "pick object"
        assert ep0.data_path.exists()

        ep1 = episodes[1]
        assert ep1.episode_index == 1
        assert ep1.length == 5
        assert ep1.task == "place object"

    def test_list_episodes_missing_dir(self, tmp_path: Path) -> None:
        """Should raise error when episodes directory is missing."""
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir(parents=True)
        with open(meta_dir / "info.json", "w") as f:
            json.dump({"codebase_version": "v3.0"}, f)
        # No episodes directory

        reader = V30Reader(tmp_path)
        with pytest.raises(SchemaError) as exc_info:
            reader.list_episodes()

        assert "Episodes directory not found" in str(exc_info.value)

    def test_list_episodes_sorted_by_index(self, v30_dataset: Path) -> None:
        """Should return episodes sorted by episode index."""
        reader = V30Reader(v30_dataset)
        episodes = reader.list_episodes()

        indices = [ep.episode_index for ep in episodes]
        assert indices == sorted(indices)

    def test_read_episode(self, v30_dataset: Path) -> None:
        """Should read episode with all steps."""
        reader = V30Reader(v30_dataset)
        episodes = reader.list_episodes()
        episode = reader.read_episode(episodes[0])

        assert len(episode) == 5
        assert episode.info.episode_index == 0
        assert episode.info.task == "pick object"

        # Check first step
        first_step = episode.steps[0]
        assert first_step.is_first is True
        assert first_step.is_last is False
        assert "state" in first_step.observation
        assert first_step.observation["state"].shape == (6,)
        assert first_step.action.shape == (6,)
        assert first_step.language_instruction == "pick object"

        # Check last step
        last_step = episode.steps[-1]
        assert last_step.is_first is False
        assert last_step.is_last is True

    def test_read_episode_missing_parquet(self, v30_dataset: Path) -> None:
        """Should raise error when data parquet file is missing."""
        reader = V30Reader(v30_dataset)
        episodes = reader.list_episodes()

        # Delete the data parquet file
        episodes[0].data_path.unlink()

        with pytest.raises(EpisodeProcessingError) as exc_info:
            reader.read_episode(episodes[0])

        assert "Parquet file not found" in str(exc_info.value)

    def test_iter_episodes(self, v30_dataset: Path) -> None:
        """Should iterate over all episodes."""
        reader = V30Reader(v30_dataset)
        episodes = list(reader.iter_episodes())

        assert len(episodes) == 2
        assert episodes[0].info.episode_index == 0
        assert episodes[1].info.episode_index == 1

    def test_episode_count(self, v30_dataset: Path) -> None:
        """Should return correct episode count."""
        reader = V30Reader(v30_dataset)
        assert reader.episode_count == 2

    def test_total_steps(self, v30_dataset: Path) -> None:
        """Should return total steps across all episodes."""
        reader = V30Reader(v30_dataset)
        assert reader.total_steps == 10  # 5 + 5

    def test_chunk_calculation(self, v30_dataset: Path) -> None:
        """Should calculate correct chunk for episode."""
        reader = V30Reader(v30_dataset)

        # With default chunks_size=1000, all should be in chunk 0
        assert reader._get_chunk_for_episode(0) == 0
        assert reader._get_chunk_for_episode(999) == 0
        assert reader._get_chunk_for_episode(1000) == 1

    def test_read_episode_with_reward_and_done(self, tmp_path: Path) -> None:
        """Should read reward and done flags from parquet if present."""
        meta_dir = tmp_path / "meta"
        episodes_meta_dir = meta_dir / "episodes"
        data_dir = tmp_path / "data" / "chunk-0"
        meta_dir.mkdir(parents=True)
        episodes_meta_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)

        with open(meta_dir / "info.json", "w") as f:
            json.dump({"codebase_version": "v3.0", "chunks_size": 1000}, f)

        # Create episode metadata
        ep_meta_table = pa.table({
            "episode_index": [0],
            "start_idx": [0],
            "end_idx": [3],
            "length": [3],
            "task": ["test"],
        })
        pq.write_table(ep_meta_table, episodes_meta_dir / "episode_000000.parquet")

        # Create data parquet with reward and done columns
        table = pa.table({
            "frame_index": [0, 1, 2],
            "episode_index": [0, 0, 0],
            "observation.state": [[0.0] * 6] * 3,
            "action": [[0.0] * 6] * 3,
            "reward": [0.0, 0.5, 1.0],
            "done": [False, False, True],
        })
        pq.write_table(table, data_dir / "episode_000000.parquet")

        reader = V30Reader(tmp_path)
        episode = reader.read_episode(reader.list_episodes()[0])

        assert episode.steps[0].reward == 0.0
        assert episode.steps[1].reward == 0.5
        assert episode.steps[2].reward == 1.0

        assert episode.steps[0].is_terminal is False
        assert episode.steps[1].is_terminal is False
        assert episode.steps[2].is_terminal is True

    def test_handles_empty_episode_metadata(self, tmp_path: Path) -> None:
        """Should handle corrupted/empty episode metadata gracefully."""
        meta_dir = tmp_path / "meta"
        episodes_meta_dir = meta_dir / "episodes"
        data_dir = tmp_path / "data" / "chunk-0"
        meta_dir.mkdir(parents=True)
        episodes_meta_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)

        with open(meta_dir / "info.json", "w") as f:
            json.dump({"codebase_version": "v3.0", "chunks_size": 1000}, f)

        # Create one valid and one empty episode metadata
        valid_meta = pa.table({
            "episode_index": [0],
            "start_idx": [0],
            "end_idx": [3],
            "length": [3],
            "task": ["test"],
        })
        pq.write_table(valid_meta, episodes_meta_dir / "episode_000000.parquet")

        # Empty table (will be skipped)
        empty_meta = pa.table({
            "episode_index": pa.array([], type=pa.int64()),
            "length": pa.array([], type=pa.int64()),
        })
        pq.write_table(empty_meta, episodes_meta_dir / "episode_000001.parquet")

        # Create data for valid episode
        data_table = pa.table({
            "frame_index": [0, 1, 2],
            "episode_index": [0, 0, 0],
            "observation.state": [[0.0] * 6] * 3,
            "action": [[0.0] * 6] * 3,
        })
        pq.write_table(data_table, data_dir / "episode_000000.parquet")

        reader = V30Reader(tmp_path)
        episodes = reader.list_episodes()

        # Should only return the valid episode
        assert len(episodes) == 1
        assert episodes[0].episode_index == 0

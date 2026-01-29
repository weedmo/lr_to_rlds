"""Tests for LeRobot v2.1 reader."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot_to_rlds.core.exceptions import EpisodeProcessingError, SchemaError
from lerobot_to_rlds.core.types import LeRobotVersion
from lerobot_to_rlds.readers.v21_reader import V21Reader


class TestV21Reader:
    """Tests for V21Reader class."""

    @pytest.fixture
    def v21_dataset(self, tmp_path: Path) -> Path:
        """Create a minimal v2.1 dataset for testing."""
        meta_dir = tmp_path / "meta"
        data_dir = tmp_path / "data" / "chunk-0"
        meta_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)

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
            json.dump(info, f)

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
            states = np.random.randn(length, 6).astype(np.float32)
            actions = np.random.randn(length, 6).astype(np.float32)

            table = pa.table({
                "frame_index": np.arange(length),
                "episode_index": np.full(length, ep_idx),
                "observation.state": [states[i].tolist() for i in range(length)],
                "action": [actions[i].tolist() for i in range(length)],
            })
            pq.write_table(table, data_dir / f"episode_{ep_idx:06d}.parquet")

        return tmp_path

    def test_version_property(self, v21_dataset: Path) -> None:
        """Should return V21 version."""
        reader = V21Reader(v21_dataset)
        assert reader.version == LeRobotVersion.V21

    def test_dataset_name(self, v21_dataset: Path) -> None:
        """Should return directory name as dataset name."""
        reader = V21Reader(v21_dataset)
        assert reader.dataset_name == v21_dataset.name

    def test_load_metadata(self, v21_dataset: Path) -> None:
        """Should load metadata from info.json."""
        reader = V21Reader(v21_dataset)
        metadata = reader.metadata

        assert metadata["codebase_version"] == "v2.1"
        assert metadata["robot_type"] == "test_robot"
        assert metadata["fps"] == 30
        assert metadata["total_episodes"] == 2

    def test_load_metadata_missing_file(self, tmp_path: Path) -> None:
        """Should raise error when info.json is missing."""
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir(parents=True)
        # No info.json

        reader = V21Reader(tmp_path)
        with pytest.raises(SchemaError) as exc_info:
            _ = reader.metadata

        assert "info.json not found" in str(exc_info.value)

    def test_build_feature_specs(self, v21_dataset: Path) -> None:
        """Should build feature specs from metadata."""
        reader = V21Reader(v21_dataset)
        features = reader.features

        assert "observation.state" in features
        assert features["observation.state"].dtype == "float32"
        assert features["observation.state"].shape == (6,)

        assert "action" in features
        assert features["action"].dtype == "float32"

    def test_list_episodes(self, v21_dataset: Path) -> None:
        """Should list all episodes."""
        reader = V21Reader(v21_dataset)
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

    def test_list_episodes_missing_jsonl(self, tmp_path: Path) -> None:
        """Should raise error when episodes.jsonl is missing."""
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir(parents=True)
        with open(meta_dir / "info.json", "w") as f:
            json.dump({"codebase_version": "v2.1"}, f)
        # No episodes.jsonl

        reader = V21Reader(tmp_path)
        with pytest.raises(SchemaError) as exc_info:
            reader.list_episodes()

        assert "episodes.jsonl not found" in str(exc_info.value)

    def test_read_episode(self, v21_dataset: Path) -> None:
        """Should read episode with all steps."""
        reader = V21Reader(v21_dataset)
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

    def test_read_episode_missing_parquet(self, v21_dataset: Path) -> None:
        """Should raise error when parquet file is missing."""
        reader = V21Reader(v21_dataset)
        episodes = reader.list_episodes()

        # Delete the parquet file
        episodes[0].data_path.unlink()

        with pytest.raises(EpisodeProcessingError) as exc_info:
            reader.read_episode(episodes[0])

        assert "Parquet file not found" in str(exc_info.value)

    def test_iter_episodes(self, v21_dataset: Path) -> None:
        """Should iterate over all episodes."""
        reader = V21Reader(v21_dataset)
        episodes = list(reader.iter_episodes())

        assert len(episodes) == 2
        assert episodes[0].info.episode_index == 0
        assert episodes[1].info.episode_index == 1

    def test_episode_count(self, v21_dataset: Path) -> None:
        """Should return correct episode count."""
        reader = V21Reader(v21_dataset)
        assert reader.episode_count == 2

    def test_total_steps(self, v21_dataset: Path) -> None:
        """Should return total steps across all episodes."""
        reader = V21Reader(v21_dataset)
        assert reader.total_steps == 10  # 5 + 5

    def test_chunk_calculation(self, v21_dataset: Path) -> None:
        """Should calculate correct chunk for episode."""
        reader = V21Reader(v21_dataset)

        # With default chunks_size=1000, all should be in chunk 0
        assert reader._get_chunk_for_episode(0) == 0
        assert reader._get_chunk_for_episode(999) == 0
        assert reader._get_chunk_for_episode(1000) == 1

    def test_read_episode_with_reward(self, tmp_path: Path) -> None:
        """Should read reward from parquet if present."""
        meta_dir = tmp_path / "meta"
        data_dir = tmp_path / "data" / "chunk-0"
        meta_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)

        with open(meta_dir / "info.json", "w") as f:
            json.dump({"codebase_version": "v2.1", "chunks_size": 1000}, f)

        with open(meta_dir / "episodes.jsonl", "w") as f:
            f.write(json.dumps({"episode_index": 0, "length": 3, "task": "test"}) + "\n")

        # Create parquet with reward column
        table = pa.table({
            "frame_index": [0, 1, 2],
            "episode_index": [0, 0, 0],
            "observation.state": [[0.0] * 6] * 3,
            "action": [[0.0] * 6] * 3,
            "reward": [0.0, 0.5, 1.0],
        })
        pq.write_table(table, data_dir / "episode_000000.parquet")

        reader = V21Reader(tmp_path)
        episode = reader.read_episode(reader.list_episodes()[0])

        assert episode.steps[0].reward == 0.0
        assert episode.steps[1].reward == 0.5
        assert episode.steps[2].reward == 1.0

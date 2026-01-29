"""Tests for LeRobot base reader and factory function."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot_to_rlds.core.types import LeRobotVersion
from lerobot_to_rlds.readers import get_reader
from lerobot_to_rlds.readers.v21_reader import V21Reader
from lerobot_to_rlds.readers.v30_reader import V30Reader


class TestGetReader:
    """Tests for get_reader factory function."""

    @pytest.fixture
    def v21_dataset(self, tmp_path: Path) -> Path:
        """Create a minimal v2.1 dataset."""
        meta_dir = tmp_path / "meta"
        data_dir = tmp_path / "data" / "chunk-0"
        meta_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)

        with open(meta_dir / "info.json", "w") as f:
            json.dump({"codebase_version": "v2.1", "chunks_size": 1000}, f)

        with open(meta_dir / "episodes.jsonl", "w") as f:
            f.write(json.dumps({"episode_index": 0, "length": 3, "task": "test"}) + "\n")

        table = pa.table({
            "frame_index": [0, 1, 2],
            "episode_index": [0, 0, 0],
            "observation.state": [[0.0] * 6] * 3,
            "action": [[0.0] * 6] * 3,
        })
        pq.write_table(table, data_dir / "episode_000000.parquet")

        return tmp_path

    @pytest.fixture
    def v30_dataset(self, tmp_path: Path) -> Path:
        """Create a minimal v3.0 dataset."""
        meta_dir = tmp_path / "meta"
        episodes_meta_dir = meta_dir / "episodes"
        data_dir = tmp_path / "data" / "chunk-0"
        meta_dir.mkdir(parents=True)
        episodes_meta_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)

        with open(meta_dir / "info.json", "w") as f:
            json.dump({"codebase_version": "v3.0", "chunks_size": 1000}, f)

        ep_meta_table = pa.table({
            "episode_index": [0],
            "start_idx": [0],
            "end_idx": [3],
            "length": [3],
            "task": ["test"],
        })
        pq.write_table(ep_meta_table, episodes_meta_dir / "episode_000000.parquet")

        table = pa.table({
            "frame_index": [0, 1, 2],
            "episode_index": [0, 0, 0],
            "observation.state": [[0.0] * 6] * 3,
            "action": [[0.0] * 6] * 3,
        })
        pq.write_table(table, data_dir / "episode_000000.parquet")

        return tmp_path

    def test_get_reader_v21(self, v21_dataset: Path) -> None:
        """Should return V21Reader for v2.1 dataset."""
        reader = get_reader(v21_dataset)

        assert isinstance(reader, V21Reader)
        assert reader.version == LeRobotVersion.V21

    def test_get_reader_v30(self, v30_dataset: Path) -> None:
        """Should return V30Reader for v3.0 dataset."""
        reader = get_reader(v30_dataset)

        assert isinstance(reader, V30Reader)
        assert reader.version == LeRobotVersion.V30

    def test_get_reader_with_string_path(self, v21_dataset: Path) -> None:
        """Should accept string path."""
        reader = get_reader(str(v21_dataset))

        assert isinstance(reader, V21Reader)

    def test_get_reader_invalid_dataset(self, tmp_path: Path) -> None:
        """Should raise error for invalid dataset."""
        from lerobot_to_rlds.core.exceptions import VersionDetectionError

        with pytest.raises(VersionDetectionError):
            get_reader(tmp_path)


class TestStepDataclass:
    """Tests for Step dataclass."""

    def test_step_creation(self) -> None:
        """Should create Step with all fields."""
        from lerobot_to_rlds.readers.base import Step

        step = Step(
            observation={"state": np.array([1.0, 2.0, 3.0])},
            action=np.array([0.1, 0.2]),
            reward=1.0,
            is_first=True,
            is_last=False,
            is_terminal=False,
            language_instruction="pick up the cube",
        )

        assert step.is_first is True
        assert step.is_last is False
        assert step.reward == 1.0
        assert step.language_instruction == "pick up the cube"
        np.testing.assert_array_equal(step.action, [0.1, 0.2])


class TestEpisodeDataclass:
    """Tests for Episode dataclass."""

    def test_episode_len(self) -> None:
        """Should return number of steps."""
        from lerobot_to_rlds.readers.base import Episode, Step
        from lerobot_to_rlds.core.types import EpisodeInfo

        info = EpisodeInfo(
            episode_id="test",
            episode_index=0,
            start_idx=0,
            end_idx=3,
            length=3,
            task="test",
            data_path=Path("/tmp/test.parquet"),
        )

        steps = [
            Step(
                observation={},
                action=np.array([0.0]),
                reward=0.0,
                is_first=(i == 0),
                is_last=(i == 2),
                is_terminal=False,
                language_instruction="",
            )
            for i in range(3)
        ]

        episode = Episode(info=info, steps=steps)

        assert len(episode) == 3

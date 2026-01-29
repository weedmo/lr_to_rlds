"""Tests for LeRobot version detection."""

import pytest
from pathlib import Path

from lerobot_to_rlds.readers.detector import detect_lerobot_version
from lerobot_to_rlds.core.types import LeRobotVersion
from lerobot_to_rlds.core.exceptions import VersionDetectionError


class TestDetectLeRobotVersion:
    """Tests for detect_lerobot_version function."""

    def test_detect_v30_with_parquet(self, tmp_path: Path) -> None:
        """Should detect v3.0 when meta/episodes/*.parquet exists."""
        # Setup v3.0 structure
        meta_dir = tmp_path / "meta"
        episodes_dir = meta_dir / "episodes"
        episodes_dir.mkdir(parents=True)
        (episodes_dir / "episode_000.parquet").touch()

        result = detect_lerobot_version(tmp_path)

        assert result == LeRobotVersion.V30

    def test_detect_v21_with_jsonl(self, tmp_path: Path) -> None:
        """Should detect v2.1 when meta/episodes.jsonl exists."""
        # Setup v2.1 structure
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir(parents=True)
        (meta_dir / "episodes.jsonl").touch()

        result = detect_lerobot_version(tmp_path)

        assert result == LeRobotVersion.V21

    def test_prefer_v30_over_v21(self, tmp_path: Path) -> None:
        """Should prefer v3.0 if both structures exist."""
        # Setup both structures
        meta_dir = tmp_path / "meta"
        episodes_dir = meta_dir / "episodes"
        episodes_dir.mkdir(parents=True)
        (episodes_dir / "episode_000.parquet").touch()
        (meta_dir / "episodes.jsonl").touch()

        result = detect_lerobot_version(tmp_path)

        assert result == LeRobotVersion.V30

    def test_error_when_no_meta_dir(self, tmp_path: Path) -> None:
        """Should raise error when meta directory doesn't exist."""
        with pytest.raises(VersionDetectionError) as exc_info:
            detect_lerobot_version(tmp_path)

        assert "Meta directory not found" in str(exc_info.value)

    def test_error_when_no_episodes(self, tmp_path: Path) -> None:
        """Should raise error when neither v2.1 nor v3.0 structure exists."""
        # Setup empty meta dir
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir(parents=True)

        with pytest.raises(VersionDetectionError) as exc_info:
            detect_lerobot_version(tmp_path)

        assert "Cannot detect LeRobot version" in str(exc_info.value)

    def test_error_when_episodes_dir_empty(self, tmp_path: Path) -> None:
        """Should raise error when episodes dir exists but has no parquet files."""
        # Setup v3.0 structure without parquet files
        meta_dir = tmp_path / "meta"
        episodes_dir = meta_dir / "episodes"
        episodes_dir.mkdir(parents=True)
        # No parquet files

        with pytest.raises(VersionDetectionError):
            detect_lerobot_version(tmp_path)

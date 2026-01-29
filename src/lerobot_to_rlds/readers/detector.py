"""LeRobot version detection."""

from pathlib import Path

from lerobot_to_rlds.core.types import LeRobotVersion
from lerobot_to_rlds.core.exceptions import VersionDetectionError
from lerobot_to_rlds.core.constants import META_DIR, EPISODES_DIR, EPISODES_JSONL


def detect_lerobot_version(dataset_root: Path) -> LeRobotVersion:
    """Detect the LeRobot dataset version.

    Args:
        dataset_root: Path to the LeRobot dataset root directory.

    Returns:
        LeRobotVersion indicating v2.1 or v3.0.

    Raises:
        VersionDetectionError: If version cannot be determined.
    """
    meta_dir = dataset_root / META_DIR

    if not meta_dir.exists():
        raise VersionDetectionError(
            f"Meta directory not found: {meta_dir}. "
            "Is this a valid LeRobot dataset?"
        )

    # Check for v3.0: meta/episodes/ directory with parquet files
    episodes_dir = meta_dir / EPISODES_DIR
    if episodes_dir.is_dir():
        parquet_files = list(episodes_dir.glob("*.parquet"))
        if parquet_files:
            return LeRobotVersion.V30

    # Check for v2.1: meta/episodes.jsonl file
    episodes_jsonl = meta_dir / EPISODES_JSONL
    if episodes_jsonl.exists():
        return LeRobotVersion.V21

    raise VersionDetectionError(
        f"Cannot detect LeRobot version. "
        f"Neither {episodes_dir} (v3.0) nor {episodes_jsonl} (v2.1) found."
    )

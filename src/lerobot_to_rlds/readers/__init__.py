"""LeRobot dataset readers."""

from lerobot_to_rlds.readers.base import Episode, LeRobotReader, Step
from lerobot_to_rlds.readers.detector import detect_lerobot_version
from lerobot_to_rlds.readers.v21_reader import V21Reader
from lerobot_to_rlds.readers.v30_reader import V30Reader

__all__ = [
    "detect_lerobot_version",
    "Episode",
    "LeRobotReader",
    "Step",
    "V21Reader",
    "V30Reader",
]


def get_reader(dataset_root) -> LeRobotReader:
    """Get the appropriate reader for a LeRobot dataset.

    Args:
        dataset_root: Path to the LeRobot dataset root directory.

    Returns:
        A reader instance for the detected dataset version.

    Raises:
        VersionDetectionError: If version cannot be determined.
    """
    from pathlib import Path

    from lerobot_to_rlds.core.types import LeRobotVersion

    dataset_root = Path(dataset_root)
    version = detect_lerobot_version(dataset_root)

    if version == LeRobotVersion.V21:
        return V21Reader(dataset_root)
    elif version == LeRobotVersion.V30:
        return V30Reader(dataset_root)
    else:
        raise ValueError(f"Unsupported LeRobot version: {version}")

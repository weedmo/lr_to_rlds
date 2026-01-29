"""Naming utilities for dataset folder names and output paths."""

import re
from pathlib import Path

# Default output base directory
DEFAULT_OUTPUT_BASE = Path("data")

# Maximum folder name length
MAX_FOLDER_NAME_LENGTH = 64


def sanitize_folder_name(name: str) -> str:
    """Convert a name to a valid folder name.

    - Convert to lowercase
    - Replace special characters (except underscore) with underscore
    - Collapse multiple underscores into one
    - Remove leading/trailing underscores
    - Limit to MAX_FOLDER_NAME_LENGTH characters

    Args:
        name: Input string to sanitize.

    Returns:
        Sanitized folder name.
    """
    # Convert to lowercase
    result = name.lower()

    # Replace special characters with underscore (keep alphanumeric and underscore)
    result = re.sub(r"[^a-z0-9_]", "_", result)

    # Collapse multiple underscores into one
    result = re.sub(r"_+", "_", result)

    # Remove leading/trailing underscores
    result = result.strip("_")

    # Limit length
    if len(result) > MAX_FOLDER_NAME_LENGTH:
        result = result[:MAX_FOLDER_NAME_LENGTH].rstrip("_")

    return result


def get_task_name_from_path(dataset_path: Path) -> str:
    """Extract task name from dataset path.

    Uses the folder name as the task name.

    Args:
        dataset_path: Path to the LeRobot dataset.

    Returns:
        Task name derived from folder name.
    """
    return dataset_path.resolve().name


def get_output_path(
    dataset_path: Path,
    output_name: str | None = None,
    output_base: Path | None = None,
) -> Path:
    """Get the output path for a converted dataset.

    Args:
        dataset_path: Path to the source LeRobot dataset.
        output_name: Optional custom name for the output folder.
                     If not provided, uses sanitized folder name from dataset_path.
        output_base: Base directory for output. Defaults to 'data/'.

    Returns:
        Full path to the output directory (e.g., data/<sanitized_name>/).
    """
    if output_base is None:
        output_base = DEFAULT_OUTPUT_BASE

    if output_name is not None:
        folder_name = sanitize_folder_name(output_name)
    else:
        folder_name = sanitize_folder_name(get_task_name_from_path(dataset_path))

    return output_base / folder_name

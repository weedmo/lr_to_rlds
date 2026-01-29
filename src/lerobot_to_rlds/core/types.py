"""Core type definitions."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class LeRobotVersion(str, Enum):
    """LeRobot dataset version."""

    V21 = "v2.1"
    V30 = "v3.0"


class ConvertMode(str, Enum):
    """Conversion execution mode."""

    SAFE = "safe"
    PARALLEL_HALF = "parallel-half"
    PARALLEL_MAX = "parallel-max"


@dataclass
class FeatureSpec:
    """Specification for a dataset feature."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    source: str  # "parquet", "video", "computed"


@dataclass
class EpisodeInfo:
    """Information about a single episode."""

    episode_id: str
    episode_index: int
    start_idx: int
    end_idx: int
    length: int
    task: str
    data_path: Path
    video_paths: dict[str, Path] = field(default_factory=dict)


@dataclass
class Inventory:
    """Dataset inventory from discovery stage."""

    dataset_name: str
    dataset_root: Path
    lerobot_version: LeRobotVersion
    episodes_count: int
    total_steps: int
    features: dict[str, FeatureSpec]
    metadata: dict[str, Any]


@dataclass
class ProgressRecord:
    """Progress record for checkpointing."""

    episode_id: str
    status: str  # "completed", "failed", "in_progress"
    started_at: str
    completed_at: str | None = None
    steps: int | None = None
    shard: int | None = None
    error: str | None = None
    worker_id: int | None = None


@dataclass
class ValidationResult:
    """Result of validation stage."""

    passed: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts_match: bool = True
    schema_match: bool = True
    data_integrity: bool = True

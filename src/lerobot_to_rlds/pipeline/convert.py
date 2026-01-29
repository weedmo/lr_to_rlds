"""Conversion pipeline for LeRobot to RLDS."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Union

from tqdm import tqdm

from lerobot_to_rlds.core.types import ConvertMode, ProgressRecord
from lerobot_to_rlds.readers import get_reader
from lerobot_to_rlds.readers.base import Episode, LeRobotReader

logger = logging.getLogger(__name__)


@dataclass
class ConversionConfig:
    """Configuration for conversion pipeline."""

    dataset_path: Path
    output_dir: Path
    mode: ConvertMode = ConvertMode.SAFE
    resume: bool = False
    retry_failed: bool = False
    output_format: str = "oxe"  # "oxe" or "legacy"


@dataclass
class ConversionResult:
    """Result of conversion pipeline."""

    success: bool
    episodes_converted: int
    total_steps: int
    errors: list[str]
    output_path: Path | None


class ConversionPipeline:
    """Pipeline for converting LeRobot datasets to RLDS format.

    Supports:
    - SAFE mode: Single-threaded, sequential processing
    - Resume: Continue from checkpoint
    - Progress tracking: JSONL progress file
    - Output formats: "oxe" (OpenVLA compatible) or "legacy"
    """

    def __init__(self, config: ConversionConfig) -> None:
        """Initialize the pipeline.

        Args:
            config: Conversion configuration.
        """
        self.config = config
        self.reader: LeRobotReader | None = None
        self.writer: Union["OXERLDSWriter", "RLDSWriter", None] = None  # type: ignore
        self.progress_file: Path | None = None
        self.completed_episodes: set[str] = set()
        self.failed_episodes: set[str] = set()
        self.errors: list[str] = []

    def run(self) -> ConversionResult:
        """Run the conversion pipeline.

        Returns:
            ConversionResult with status and statistics.
        """
        logger.info(f"Starting conversion: {self.config.dataset_path}")
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info(f"Mode: {self.config.mode.value}")
        logger.info(f"Output format: {self.config.output_format}")

        try:
            # Stage 1: Initialize reader
            self._init_reader()

            # Stage 2: Load progress if resuming
            if self.config.resume:
                self._load_progress()

            # Stage 3: Initialize writer
            self._init_writer()

            # Stage 4: Convert episodes
            episodes_converted = self._convert_episodes()

            logger.info(f"Conversion complete: {episodes_converted} episodes")

            return ConversionResult(
                success=True,
                episodes_converted=episodes_converted,
                total_steps=self.writer.total_steps if self.writer else 0,
                errors=self.errors,
                output_path=self.config.output_dir / self.reader.dataset_name if self.reader else None,
            )

        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            self.errors.append(str(e))
            return ConversionResult(
                success=False,
                episodes_converted=0,
                total_steps=0,
                errors=self.errors,
                output_path=None,
            )

    def _init_reader(self) -> None:
        """Initialize the LeRobot reader."""
        logger.info("Initializing reader...")
        self.reader = get_reader(self.config.dataset_path)
        logger.info(f"Detected LeRobot version: {self.reader.version.value}")
        logger.info(f"Dataset name: {self.reader.dataset_name}")
        logger.info(f"Episodes: {self.reader.episode_count}")

    def _init_writer(self) -> None:
        """Initialize the RLDS writer based on reader metadata and output format."""
        if self.reader is None:
            raise RuntimeError("Reader not initialized")

        logger.info(f"Initializing writer (format={self.config.output_format})...")

        if self.config.output_format == "oxe":
            self._init_oxe_writer()
        else:
            self._init_legacy_writer()

        # Setup progress file
        self.progress_file = self.config.output_dir / "progress.jsonl"

    def _init_oxe_writer(self) -> None:
        """Initialize OXE-compatible writer using rlds submodule."""
        from lerobot_to_rlds.writers.oxe_writer import OXERLDSWriter

        self.writer = OXERLDSWriter(
            output_dir=self.config.output_dir,
            reader=self.reader,  # type: ignore
        )
        logger.info("Using OXE-compatible writer (rlds.tfds.EpisodeWriter)")

    def _init_legacy_writer(self) -> None:
        """Initialize legacy writer with custom serialization."""
        from lerobot_to_rlds.writers.rlds_writer import RLDSWriter

        # Determine dimensions from first episode
        episodes = self.reader.list_episodes()  # type: ignore
        if not episodes:
            raise ValueError("No episodes found in dataset")

        # Read first episode to get shapes
        first_episode = self.reader.read_episode(episodes[0])  # type: ignore
        first_step = first_episode.steps[0]

        # Get state dimension
        state_dim = 0
        if "state" in first_step.observation:
            state_dim = first_step.observation["state"].shape[0]

        # Get action dimension
        action_dim = first_step.action.shape[0] if first_step.action.size > 0 else 0

        # Get image shapes
        image_shapes: dict[str, tuple[int, int, int]] = {}
        for key, value in first_step.observation.items():
            if key.startswith("image_"):
                camera_name = key[6:]  # Remove "image_" prefix
                image_shapes[camera_name] = value.shape  # type: ignore

        logger.info(f"State dimension: {state_dim}")
        logger.info(f"Action dimension: {action_dim}")
        logger.info(f"Cameras: {list(image_shapes.keys())}")

        self.writer = RLDSWriter(
            output_dir=self.config.output_dir,
            dataset_name=self.reader.dataset_name,  # type: ignore
            state_dim=state_dim,
            action_dim=action_dim,
            image_shapes=image_shapes,
            source_version=self.reader.version.value,  # type: ignore
        )
        logger.info("Using legacy writer (custom TFRecord serialization)")

    def _load_progress(self) -> None:
        """Load progress from checkpoint file."""
        if self.progress_file is None or not self.progress_file.exists():
            logger.info("No progress file found, starting fresh")
            return

        logger.info(f"Loading progress from {self.progress_file}")
        with open(self.progress_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("status") == "completed":
                    self.completed_episodes.add(record["episode_id"])
                elif record.get("status") == "failed":
                    self.failed_episodes.add(record["episode_id"])

        logger.info(f"Loaded {len(self.completed_episodes)} completed episodes")
        logger.info(f"Loaded {len(self.failed_episodes)} failed episodes")

    def _convert_episodes(self) -> int:
        """Convert all episodes.

        Returns:
            Number of episodes converted.
        """
        if self.reader is None or self.writer is None:
            raise RuntimeError("Reader or writer not initialized")

        episodes_to_convert = []
        for ep_info in self.reader.list_episodes():
            ep_id = ep_info.episode_id

            # Skip completed
            if ep_id in self.completed_episodes:
                continue

            # Skip failed unless retry is enabled
            if ep_id in self.failed_episodes and not self.config.retry_failed:
                continue

            episodes_to_convert.append(ep_info)

        logger.info(f"Episodes to convert: {len(episodes_to_convert)}")

        # Convert with progress bar
        def episode_generator() -> Iterator[Episode]:
            for ep_info in tqdm(episodes_to_convert, desc="Converting"):
                try:
                    episode = self.reader.read_episode(ep_info)  # type: ignore
                    self._record_progress(ep_info.episode_id, "completed", len(episode.steps))
                    yield episode
                except Exception as e:
                    logger.error(f"Failed to convert {ep_info.episode_id}: {e}")
                    self._record_progress(ep_info.episode_id, "failed", error=str(e))
                    self.errors.append(f"{ep_info.episode_id}: {e}")

        return self.writer.write_episodes(episode_generator())

    def _record_progress(
        self,
        episode_id: str,
        status: str,
        steps: int | None = None,
        error: str | None = None,
    ) -> None:
        """Record progress to checkpoint file."""
        if self.progress_file is None:
            return

        record = ProgressRecord(
            episode_id=episode_id,
            status=status,
            started_at=datetime.now().isoformat(),
            completed_at=datetime.now().isoformat() if status == "completed" else None,
            steps=steps,
            error=error,
        )

        # Ensure parent directory exists
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.progress_file, "a") as f:
            f.write(json.dumps(record.__dict__) + "\n")


def convert_dataset(
    dataset_path: Path,
    output_dir: Path,
    mode: ConvertMode = ConvertMode.SAFE,
    resume: bool = False,
    retry_failed: bool = False,
    output_format: str = "oxe",
) -> ConversionResult:
    """Convert a LeRobot dataset to RLDS format.

    Args:
        dataset_path: Path to LeRobot dataset.
        output_dir: Output directory for RLDS dataset.
        mode: Execution mode (SAFE, PARALLEL_HALF, PARALLEL_MAX).
        resume: Resume from checkpoint.
        retry_failed: Retry previously failed episodes.
        output_format: Output format ("oxe" for OpenVLA compatible, "legacy" for old format).

    Returns:
        ConversionResult with status and statistics.
    """
    config = ConversionConfig(
        dataset_path=dataset_path,
        output_dir=output_dir,
        mode=mode,
        resume=resume,
        retry_failed=retry_failed,
        output_format=output_format,
    )

    pipeline = ConversionPipeline(config)
    return pipeline.run()

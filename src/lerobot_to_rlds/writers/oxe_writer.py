"""OXE/OpenVLA compatible RLDS writer using rlds submodule."""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

# Add project root to path so 'import rlds' finds the vendored rlds/ directory
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from rlds.rlds_types import build_step, build_episode, Episode as RLDSEpisode, Step as RLDSStep
from rlds.tfds import episode_writer

from lerobot_to_rlds.readers.base import Episode, LeRobotReader, Step
from lerobot_to_rlds.writers.feature_mapper import FeatureMapper, OXEFeatureConfig

logger = logging.getLogger(__name__)


@dataclass
class OXEWriteResult:
    """Result of OXE RLDS write operation."""

    success: bool
    output_path: Path
    num_episodes: int
    num_steps: int
    dataset_name: str
    errors: list[str]


class OXERLDSWriter:
    """Writes LeRobot datasets to OXE/OpenVLA compatible RLDS format.

    Uses the official rlds.tfds.EpisodeWriter for proper TFDS serialization.
    Output can be loaded with tfds.load().

    Example:
        ```python
        reader = get_reader("/path/to/lerobot/dataset")
        writer = OXERLDSWriter(
            output_dir=Path("./output"),
            reader=reader,
        )
        result = writer.write()
        if result.success:
            print(f"Wrote {result.num_episodes} episodes to {result.output_path}")
        ```
    """

    def __init__(
        self,
        output_dir: Path,
        reader: LeRobotReader,
        dataset_name: str | None = None,
        max_episodes_per_shard: int = 1000,
    ) -> None:
        """Initialize the writer.

        Args:
            output_dir: Directory to write the RLDS dataset.
            reader: LeRobot reader to get data from.
            dataset_name: Optional dataset name (defaults to reader's dataset name).
            max_episodes_per_shard: Max episodes per TFRecord shard.
        """
        self.output_dir = Path(output_dir)
        self.reader = reader
        self.dataset_name = dataset_name or reader.dataset_name
        self.max_episodes_per_shard = max_episodes_per_shard

        # Build feature config from reader
        self.feature_config = OXEFeatureConfig.from_reader(reader)
        self.feature_mapper = FeatureMapper(self.feature_config)

        self._episode_count = 0
        self._total_steps = 0
        self._errors: list[str] = []

    def write(self) -> OXEWriteResult:
        """Write all episodes to OXE-compatible RLDS format.

        Returns:
            OXEWriteResult with status and statistics.
        """
        logger.info(f"Writing OXE-compatible RLDS to {self.output_dir}")
        logger.info(f"Dataset name: {self.dataset_name}")
        logger.info(f"Feature config: state_dim={self.feature_config.state_dim}, "
                    f"action_dim={self.feature_config.action_dim}, "
                    f"cameras={list(self.feature_config.camera_mapping.values())}")

        try:
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Build dataset config
            ds_config = self.feature_mapper.build_dataset_config(self.dataset_name)

            # Create episode writer
            writer = episode_writer.EpisodeWriter(
                data_directory=str(self.output_dir),
                ds_config=ds_config,
                max_episodes_per_file=self.max_episodes_per_shard,
                split_name="train",
                version="1.0.0",
                overwrite=True,
            )

            # Write all episodes
            for ep_info in self.reader.list_episodes():
                try:
                    episode = self.reader.read_episode(ep_info)
                    rlds_episode = self._convert_episode(episode)
                    writer.add_episode(rlds_episode)
                    self._episode_count += 1
                    self._total_steps += len(episode.steps)

                    if self._episode_count % 100 == 0:
                        logger.info(f"Processed {self._episode_count} episodes...")

                except Exception as e:
                    error_msg = f"Failed to convert episode {ep_info.episode_id}: {e}"
                    logger.error(error_msg)
                    self._errors.append(error_msg)

            # Close writer
            writer.close()

            logger.info(f"Wrote {self._episode_count} episodes, {self._total_steps} steps")

            return OXEWriteResult(
                success=True,
                output_path=self.output_dir / self.dataset_name,
                num_episodes=self._episode_count,
                num_steps=self._total_steps,
                dataset_name=self.dataset_name,
                errors=self._errors,
            )

        except Exception as e:
            logger.error(f"Write failed: {e}")
            self._errors.append(str(e))
            return OXEWriteResult(
                success=False,
                output_path=self.output_dir,
                num_episodes=self._episode_count,
                num_steps=self._total_steps,
                dataset_name=self.dataset_name,
                errors=self._errors,
            )

    def _convert_episode(self, episode: Episode) -> RLDSEpisode:
        """Convert LeRobot Episode to RLDS Episode format.

        Args:
            episode: LeRobot Episode object.

        Returns:
            RLDS Episode dictionary.
        """
        steps = [self._convert_step(step) for step in episode.steps]

        return build_episode(
            steps=steps,
            metadata={},  # Episode-level metadata not supported without config
        )

    def _convert_step(self, step: Step) -> RLDSStep:
        """Convert LeRobot Step to RLDS Step format.

        Args:
            step: LeRobot Step object.

        Returns:
            RLDS Step dictionary.
        """
        # Build observation dict with OXE naming
        observation: dict[str, np.ndarray] = {}

        # Add state
        if "state" in step.observation:
            observation["state"] = step.observation["state"].astype(np.float32)

        # Add images with OXE naming
        for lerobot_key, value in step.observation.items():
            if lerobot_key.startswith("image_"):
                oxe_key = self.feature_mapper.map_camera_name(lerobot_key)
                # Ensure HWC uint8 format
                if value.dtype != np.uint8:
                    value = (value * 255).clip(0, 255).astype(np.uint8)
                observation[oxe_key] = value

        return build_step(
            observation=observation,
            action=step.action.astype(np.float32),
            reward=np.float32(step.reward),
            discount=np.float32(1.0),
            is_terminal=step.is_terminal,
            is_first=step.is_first,
            is_last=step.is_last,
            metadata={
                "language_instruction": step.language_instruction or "",
            },
        )

    def write_episodes(self, episodes: Iterator[Episode]) -> int:
        """Write episodes from an iterator (compatibility with pipeline).

        Args:
            episodes: Iterator of Episode objects.

        Returns:
            Number of episodes written.
        """
        logger.info(f"Writing OXE-compatible RLDS to {self.output_dir}")

        try:
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Build dataset config
            ds_config = self.feature_mapper.build_dataset_config(self.dataset_name)

            # Create episode writer
            writer = episode_writer.EpisodeWriter(
                data_directory=str(self.output_dir),
                ds_config=ds_config,
                max_episodes_per_file=self.max_episodes_per_shard,
                split_name="train",
                version="1.0.0",
                overwrite=True,
            )

            # Write episodes
            for episode in episodes:
                try:
                    rlds_episode = self._convert_episode(episode)
                    writer.add_episode(rlds_episode)
                    self._episode_count += 1
                    self._total_steps += len(episode.steps)
                except Exception as e:
                    logger.error(f"Failed to convert episode: {e}")
                    self._errors.append(str(e))

            writer.close()
            return self._episode_count

        except Exception as e:
            logger.error(f"Write failed: {e}")
            return self._episode_count

    @property
    def episode_count(self) -> int:
        """Return number of episodes written."""
        return self._episode_count

    @property
    def total_steps(self) -> int:
        """Return total steps written."""
        return self._total_steps

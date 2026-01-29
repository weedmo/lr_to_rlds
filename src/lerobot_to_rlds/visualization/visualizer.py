"""Dataset visualizer for LeRobot datasets."""

from dataclasses import dataclass
from pathlib import Path

from lerobot_to_rlds.core.types import EpisodeInfo
from lerobot_to_rlds.readers import get_reader
from lerobot_to_rlds.readers.base import LeRobotReader


@dataclass
class EpisodeSummary:
    """Summary information for an episode."""

    episode_id: str
    episode_index: int
    length: int
    task: str
    has_video: bool
    video_cameras: list[str]


class DatasetVisualizer:
    """Coordinator for visualizing LeRobot datasets.

    Uses the existing reader infrastructure to provide visualization capabilities.
    """

    def __init__(self, dataset_path: Path) -> None:
        """Initialize the visualizer.

        Args:
            dataset_path: Path to the LeRobot dataset.
        """
        self.dataset_path = Path(dataset_path)
        self._reader: LeRobotReader | None = None

    @property
    def reader(self) -> LeRobotReader:
        """Lazy-load the reader."""
        if self._reader is None:
            self._reader = get_reader(self.dataset_path)
        return self._reader

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return self.reader.dataset_name

    @property
    def version(self) -> str:
        """Return the LeRobot version."""
        return self.reader.version.value

    @property
    def episode_count(self) -> int:
        """Return the total number of episodes."""
        return self.reader.episode_count

    @property
    def total_steps(self) -> int:
        """Return the total number of steps."""
        return self.reader.total_steps

    def list_episodes_summary(self) -> list[EpisodeSummary]:
        """Get summary information for all episodes.

        Returns:
            List of EpisodeSummary objects.
        """
        episodes = self.reader.list_episodes()
        summaries = []

        for ep in episodes:
            has_video = bool(ep.video_paths)
            video_cameras = list(ep.video_paths.keys())

            summaries.append(
                EpisodeSummary(
                    episode_id=ep.episode_id,
                    episode_index=ep.episode_index,
                    length=ep.length,
                    task=ep.task,
                    has_video=has_video,
                    video_cameras=video_cameras,
                )
            )

        return summaries

    def get_episode_info(self, episode_index: int) -> EpisodeInfo:
        """Get episode info by index.

        Args:
            episode_index: The episode index.

        Returns:
            EpisodeInfo for the requested episode.

        Raises:
            IndexError: If episode_index is out of range.
        """
        episodes = self.reader.list_episodes()
        for ep in episodes:
            if ep.episode_index == episode_index:
                return ep
        raise IndexError(f"Episode index {episode_index} not found")

    def print_episodes_table(self, max_task_len: int = 50) -> None:
        """Print a formatted table of episodes to console.

        Args:
            max_task_len: Maximum length for task string display.
        """
        summaries = self.list_episodes_summary()

        # Header
        print(f"\nDataset: {self.dataset_name}")
        print(f"Version: {self.version}")
        print(f"Episodes: {self.episode_count}, Total steps: {self.total_steps}")
        print("-" * 80)
        print(f"{'Index':<8} {'Length':<10} {'Video':<8} {'Task'}")
        print("-" * 80)

        for s in summaries:
            task_display = s.task[:max_task_len] + "..." if len(s.task) > max_task_len else s.task
            video_str = ",".join(s.video_cameras) if s.has_video else "None"
            print(f"{s.episode_index:<8} {s.length:<10} {video_str:<8} {task_display}")

        print("-" * 80)

    def print_dataset_info(self) -> None:
        """Print detailed dataset information."""
        print(f"\nDataset Information")
        print("=" * 60)
        print(f"Name:     {self.dataset_name}")
        print(f"Path:     {self.dataset_path}")
        print(f"Version:  {self.version}")
        print(f"Episodes: {self.episode_count}")
        print(f"Steps:    {self.total_steps}")

        # Print features
        print(f"\nFeatures:")
        for name, spec in self.reader.features.items():
            print(f"  {name}: {spec.dtype} {spec.shape} ({spec.source})")

        print("=" * 60)

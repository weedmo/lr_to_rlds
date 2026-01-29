"""Abstract base class for LeRobot dataset readers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from lerobot_to_rlds.core.types import EpisodeInfo, FeatureSpec, LeRobotVersion


@dataclass
class Step:
    """A single step in an episode."""

    observation: dict[str, np.ndarray]
    action: np.ndarray
    reward: float
    is_first: bool
    is_last: bool
    is_terminal: bool
    language_instruction: str


@dataclass
class Episode:
    """An episode containing steps and metadata."""

    info: EpisodeInfo
    steps: list[Step]

    def __len__(self) -> int:
        return len(self.steps)


class LeRobotReader(ABC):
    """Abstract base class for LeRobot dataset readers.

    Subclasses implement reading for specific LeRobot versions.
    """

    def __init__(self, dataset_root: Path) -> None:
        """Initialize the reader.

        Args:
            dataset_root: Path to the LeRobot dataset root directory.
        """
        self.dataset_root = dataset_root
        self._metadata: dict[str, Any] | None = None
        self._features: dict[str, FeatureSpec] | None = None

    @property
    @abstractmethod
    def version(self) -> LeRobotVersion:
        """Return the LeRobot version this reader handles."""
        ...

    @property
    def dataset_name(self) -> str:
        """Return the dataset name (directory name)."""
        return self.dataset_root.name

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the dataset metadata from info.json."""
        if self._metadata is None:
            self._metadata = self._load_metadata()
        return self._metadata

    @property
    def features(self) -> dict[str, FeatureSpec]:
        """Return the feature specifications."""
        if self._features is None:
            self._features = self._build_feature_specs()
        return self._features

    @abstractmethod
    def _load_metadata(self) -> dict[str, Any]:
        """Load dataset metadata from info.json.

        Returns:
            Dictionary containing dataset metadata.
        """
        ...

    @abstractmethod
    def _build_feature_specs(self) -> dict[str, FeatureSpec]:
        """Build feature specifications from metadata.

        Returns:
            Dictionary mapping feature names to FeatureSpec.
        """
        ...

    @abstractmethod
    def list_episodes(self) -> list[EpisodeInfo]:
        """List all episodes in the dataset.

        Returns:
            List of EpisodeInfo for each episode.
        """
        ...

    @abstractmethod
    def read_episode(self, episode_info: EpisodeInfo) -> Episode:
        """Read a single episode with all its steps.

        Args:
            episode_info: Episode information from list_episodes().

        Returns:
            Episode containing all steps and metadata.
        """
        ...

    def iter_episodes(self) -> Iterator[Episode]:
        """Iterate over all episodes in the dataset.

        Yields:
            Episode for each episode in the dataset.
        """
        for episode_info in self.list_episodes():
            yield self.read_episode(episode_info)

    @property
    def episode_count(self) -> int:
        """Return the total number of episodes."""
        return len(self.list_episodes())

    @property
    def total_steps(self) -> int:
        """Return the total number of steps across all episodes."""
        return sum(ep.length for ep in self.list_episodes())

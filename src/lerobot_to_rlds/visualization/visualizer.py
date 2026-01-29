"""Dataset visualizer for RLDS datasets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tensorflow_datasets as tfds


@dataclass
class EpisodeSummary:
    """Summary information for an episode."""

    episode_index: int
    length: int
    language_instruction: str
    has_image: bool
    has_wrist_image: bool


class DatasetVisualizer:
    """Coordinator for visualizing RLDS datasets.

    Uses tfds.load() to read converted RLDS datasets.
    """

    def __init__(self, data_dir: Path, dataset_name: str | None = None) -> None:
        """Initialize the visualizer.

        Args:
            data_dir: Path to the data directory containing the RLDS dataset.
            dataset_name: Name of the dataset. If None, auto-detects from data_dir.
        """
        self.data_dir = Path(data_dir)

        # Auto-detect dataset name if not provided
        if dataset_name is None:
            dataset_name = self._detect_dataset_name()

        self.dataset_name = dataset_name
        self._dataset = None
        self._info = None

    def _detect_dataset_name(self) -> str:
        """Detect dataset name from data directory.

        Returns:
            Dataset name.

        Raises:
            ValueError: If no dataset found.
        """
        # Check if data_dir itself is a dataset
        if (self.data_dir / "dataset_info.json").exists():
            return self.data_dir.name

        # Check subdirectories
        for path in self.data_dir.iterdir():
            if path.is_dir() and (path / "dataset_info.json").exists():
                return path.name

        raise ValueError(f"No RLDS dataset found in {self.data_dir}")

    def _load_dataset(self) -> None:
        """Load the dataset using tfds."""
        if self._dataset is None:
            # Determine the correct data_dir for tfds.load
            if (self.data_dir / "dataset_info.json").exists():
                # data_dir is the dataset itself, parent is the data_dir for tfds
                tfds_data_dir = self.data_dir.parent
            else:
                tfds_data_dir = self.data_dir

            self._dataset = tfds.load(
                self.dataset_name,
                data_dir=str(tfds_data_dir),
                split="train",
            )

    @property
    def dataset(self):
        """Lazy-load the dataset."""
        self._load_dataset()
        return self._dataset

    def _get_dataset_info(self) -> dict[str, Any]:
        """Get dataset info from dataset_info.json."""
        if self._info is not None:
            return self._info

        import json

        # Find dataset_info.json
        if (self.data_dir / "dataset_info.json").exists():
            info_path = self.data_dir / "dataset_info.json"
        else:
            info_path = self.data_dir / self.dataset_name / "dataset_info.json"

        if info_path.exists():
            with open(info_path) as f:
                self._info = json.load(f)
        else:
            self._info = {}

        return self._info

    @property
    def episode_count(self) -> int:
        """Return the total number of episodes."""
        info = self._get_dataset_info()
        splits = info.get("splits", [])
        for split in splits:
            if split.get("name") == "train":
                return split.get("numShards", 0) or split.get("statistics", {}).get("numExamples", 0)
        # Fallback: count episodes
        return sum(1 for _ in self.dataset)

    def list_episodes_summary(self, max_episodes: int = 100) -> list[EpisodeSummary]:
        """Get summary information for episodes.

        Args:
            max_episodes: Maximum number of episodes to scan.

        Returns:
            List of EpisodeSummary objects.
        """
        summaries = []

        for idx, episode in enumerate(self.dataset.take(max_episodes)):
            steps = list(episode["steps"])
            length = len(steps)

            # Get first step to check features
            first_step = steps[0] if steps else None

            has_image = False
            has_wrist_image = False
            language_instruction = ""

            if first_step is not None:
                obs = first_step.get("observation", {})
                has_image = "image" in obs
                has_wrist_image = "wrist_image" in obs

                lang = first_step.get("language_instruction", b"")
                if isinstance(lang, bytes):
                    language_instruction = lang.decode("utf-8", errors="ignore")
                else:
                    language_instruction = str(lang.numpy().decode("utf-8") if hasattr(lang, "numpy") else lang)

            summaries.append(
                EpisodeSummary(
                    episode_index=idx,
                    length=length,
                    language_instruction=language_instruction,
                    has_image=has_image,
                    has_wrist_image=has_wrist_image,
                )
            )

        return summaries

    def get_episode(self, episode_index: int) -> dict:
        """Get a specific episode by index.

        Args:
            episode_index: The episode index.

        Returns:
            Episode data as dict.

        Raises:
            IndexError: If episode_index is out of range.
        """
        for idx, episode in enumerate(self.dataset):
            if idx == episode_index:
                return episode
        raise IndexError(f"Episode index {episode_index} not found")

    def print_episodes_table(self, max_episodes: int = 100, max_task_len: int = 50) -> None:
        """Print a formatted table of episodes to console.

        Args:
            max_episodes: Maximum number of episodes to display.
            max_task_len: Maximum length for task string display.
        """
        summaries = self.list_episodes_summary(max_episodes)

        # Header
        print(f"\nDataset: {self.dataset_name}")
        print(f"Data dir: {self.data_dir}")
        print("-" * 90)
        print(f"{'Index':<8} {'Length':<10} {'Image':<8} {'Wrist':<8} {'Task'}")
        print("-" * 90)

        for s in summaries:
            task_display = s.language_instruction[:max_task_len] + "..." if len(s.language_instruction) > max_task_len else s.language_instruction
            img_str = "Yes" if s.has_image else "No"
            wrist_str = "Yes" if s.has_wrist_image else "No"
            print(f"{s.episode_index:<8} {s.length:<10} {img_str:<8} {wrist_str:<8} {task_display}")

        print("-" * 90)
        if len(summaries) >= max_episodes:
            print(f"Showing first {max_episodes} episodes")

    def print_dataset_info(self) -> None:
        """Print detailed dataset information."""
        info = self._get_dataset_info()

        print(f"\nDataset Information")
        print("=" * 60)
        print(f"Name:     {self.dataset_name}")
        print(f"Path:     {self.data_dir}")

        if info:
            print(f"Version:  {info.get('version', 'unknown')}")
            print(f"Description: {info.get('description', 'N/A')[:100]}")

            # Print features
            features = info.get("features", {})
            if features:
                print(f"\nFeatures:")
                self._print_features(features, indent=2)

            # Print splits
            splits = info.get("splits", [])
            if splits:
                print(f"\nSplits:")
                for split in splits:
                    name = split.get("name", "unknown")
                    num_examples = split.get("statistics", {}).get("numExamples", "?")
                    print(f"  {name}: {num_examples} examples")

        print("=" * 60)

    def _print_features(self, features: dict, indent: int = 0) -> None:
        """Recursively print feature structure."""
        prefix = " " * indent
        for name, spec in features.items():
            if isinstance(spec, dict):
                if "dtype" in spec:
                    shape = spec.get("shape", [])
                    dtype = spec.get("dtype", "unknown")
                    print(f"{prefix}{name}: {dtype} {shape}")
                elif "features" in spec:
                    print(f"{prefix}{name}:")
                    self._print_features(spec["features"], indent + 2)
                else:
                    print(f"{prefix}{name}:")
                    self._print_features(spec, indent + 2)
            else:
                print(f"{prefix}{name}: {spec}")

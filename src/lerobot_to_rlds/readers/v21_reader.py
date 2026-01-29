"""LeRobot v2.1 dataset reader."""

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow.parquet as pq

from lerobot_to_rlds.core.constants import (
    DATA_DIR,
    EPISODES_JSONL,
    INFO_JSON,
    META_DIR,
    VIDEOS_DIR,
)
from lerobot_to_rlds.core.exceptions import EpisodeProcessingError, SchemaError
from lerobot_to_rlds.core.types import EpisodeInfo, FeatureSpec, LeRobotVersion
from lerobot_to_rlds.readers.base import Episode, LeRobotReader, Step


class V21Reader(LeRobotReader):
    """Reader for LeRobot v2.1 datasets.

    v2.1 datasets use:
    - meta/info.json for dataset metadata
    - meta/episodes.jsonl for episode metadata
    - data/chunk-{chunk}/episode_{index}.parquet for episode data
    - videos/chunk-{chunk}/{camera}_episode_{index}.mp4 for video files
    """

    @property
    def version(self) -> LeRobotVersion:
        return LeRobotVersion.V21

    def _load_metadata(self) -> dict[str, Any]:
        """Load dataset metadata from meta/info.json."""
        info_path = self.dataset_root / META_DIR / INFO_JSON
        if not info_path.exists():
            raise SchemaError(f"info.json not found at {info_path}")

        with open(info_path) as f:
            return json.load(f)

    def _build_feature_specs(self) -> dict[str, FeatureSpec]:
        """Build feature specifications from metadata."""
        features: dict[str, FeatureSpec] = {}
        metadata = self.metadata

        # Parse features from info.json
        if "features" in metadata:
            for name, spec in metadata["features"].items():
                dtype = spec.get("dtype", "float32")
                shape = tuple(spec.get("shape", []))
                source = "video" if "video" in name.lower() else "parquet"
                features[name] = FeatureSpec(
                    name=name,
                    dtype=dtype,
                    shape=shape,
                    source=source,
                )

        return features

    def _load_episodes_jsonl(self) -> list[dict[str, Any]]:
        """Load episode metadata from episodes.jsonl."""
        episodes_path = self.dataset_root / META_DIR / EPISODES_JSONL
        if not episodes_path.exists():
            raise SchemaError(f"episodes.jsonl not found at {episodes_path}")

        episodes = []
        with open(episodes_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
        return episodes

    def _get_chunk_for_episode(self, episode_index: int) -> int:
        """Get chunk index for an episode."""
        # Default chunk size is 1000 episodes per chunk
        chunks_size = self.metadata.get("chunks_size", 1000)
        return episode_index // chunks_size

    def _get_parquet_path(self, episode_index: int) -> Path:
        """Get the parquet file path for an episode."""
        chunk = self._get_chunk_for_episode(episode_index)
        # Format: data/chunk-{chunk:03d}/episode_{index:06d}.parquet
        return (
            self.dataset_root
            / DATA_DIR
            / f"chunk-{chunk:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )

    def _get_video_paths(self, episode_index: int) -> dict[str, Path]:
        """Get video file paths for an episode."""
        chunk = self._get_chunk_for_episode(episode_index)
        video_dir = self.dataset_root / VIDEOS_DIR / f"chunk-{chunk:03d}"
        video_paths: dict[str, Path] = {}

        if video_dir.exists():
            # Look for video files matching the episode
            for video_file in video_dir.glob(f"*_episode_{episode_index:06d}.mp4"):
                # Extract camera name from filename
                # Format: {camera}_episode_{index:06d}.mp4
                camera_name = video_file.stem.replace(f"_episode_{episode_index:06d}", "")
                video_paths[camera_name] = video_file

        return video_paths

    def list_episodes(self) -> list[EpisodeInfo]:
        """List all episodes in the dataset."""
        episodes_data = self._load_episodes_jsonl()
        episodes: list[EpisodeInfo] = []

        for ep_data in episodes_data:
            episode_index = ep_data["episode_index"]
            parquet_path = self._get_parquet_path(episode_index)

            # Get task from episode data or metadata
            task = ep_data.get("task", ep_data.get("tasks", ""))
            if isinstance(task, list):
                task = task[0] if task else ""

            episode_info = EpisodeInfo(
                episode_id=f"episode_{episode_index:06d}",
                episode_index=episode_index,
                start_idx=ep_data.get("start_idx", 0),
                end_idx=ep_data.get("end_idx", 0),
                length=ep_data.get("length", ep_data.get("end_idx", 0) - ep_data.get("start_idx", 0)),
                task=task,
                data_path=parquet_path,
                video_paths=self._get_video_paths(episode_index),
            )
            episodes.append(episode_info)

        return episodes

    def read_episode(self, episode_info: EpisodeInfo) -> Episode:
        """Read a single episode with all its steps."""
        # Read parquet data
        parquet_path = episode_info.data_path
        if not parquet_path.exists():
            raise EpisodeProcessingError(
                episode_info.episode_id,
                f"Parquet file not found: {parquet_path}",
            )

        try:
            table = pq.read_table(parquet_path)
            df = table.to_pandas()
        except Exception as e:
            raise EpisodeProcessingError(
                episode_info.episode_id,
                f"Failed to read parquet: {e}",
            ) from e

        # Load video frames if videos exist
        video_frames = self._load_video_frames(episode_info)

        # Build steps
        steps: list[Step] = []
        num_steps = len(df)

        for i in range(num_steps):
            row = df.iloc[i]

            # Build observation dict
            observation = self._build_observation(row, i, video_frames)

            # Get action
            action = self._get_array_from_row(row, "action")

            # Get reward (default 0.0)
            reward = float(row.get("reward", 0.0)) if "reward" in df.columns else 0.0

            # Get terminal flag
            is_terminal = bool(row.get("done", False)) if "done" in df.columns else False

            step = Step(
                observation=observation,
                action=action,
                reward=reward,
                is_first=(i == 0),
                is_last=(i == num_steps - 1),
                is_terminal=is_terminal,
                language_instruction=episode_info.task,
            )
            steps.append(step)

        return Episode(info=episode_info, steps=steps)

    def _build_observation(
        self,
        row: Any,
        step_idx: int,
        video_frames: dict[str, list[np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """Build observation dictionary for a step."""
        observation: dict[str, np.ndarray] = {}

        # Get state from parquet
        if "observation.state" in row.index:
            observation["state"] = self._get_array_from_row(row, "observation.state")

        # Add video frames as images
        for camera_name, frames in video_frames.items():
            if step_idx < len(frames):
                observation[f"image_{camera_name}"] = frames[step_idx]

        return observation

    def _get_array_from_row(self, row: Any, column_prefix: str) -> np.ndarray:
        """Get numpy array from a pandas row, handling nested columns."""
        # Direct column access
        if column_prefix in row.index:
            value = row[column_prefix]
            if isinstance(value, np.ndarray):
                return value
            return np.array(value, dtype=np.float32)

        # Look for nested columns (e.g., "action.0", "action.1", etc.)
        matching_cols = [c for c in row.index if c.startswith(f"{column_prefix}.")]
        if matching_cols:
            values = [row[c] for c in sorted(matching_cols)]
            return np.array(values, dtype=np.float32)

        return np.array([], dtype=np.float32)

    def _load_video_frames(
        self, episode_info: EpisodeInfo
    ) -> dict[str, list[np.ndarray]]:
        """Load video frames for an episode.

        Args:
            episode_info: Episode information with video paths.

        Returns:
            Dictionary mapping camera names to lists of frames (HWC uint8).
        """
        video_frames: dict[str, list[np.ndarray]] = {}

        for camera_name, video_path in episode_info.video_paths.items():
            if not video_path.exists():
                continue

            try:
                frames = self._decode_video(video_path, episode_info.length)
                video_frames[camera_name] = frames
            except Exception as e:
                raise EpisodeProcessingError(
                    episode_info.episode_id,
                    f"Failed to decode video {video_path}: {e}",
                ) from e

        return video_frames

    def _decode_video(self, video_path: Path, expected_frames: int) -> list[np.ndarray]:
        """Decode video file to list of frames.

        Args:
            video_path: Path to video file.
            expected_frames: Expected number of frames.

        Returns:
            List of frames as HWC uint8 numpy arrays.
        """
        try:
            import av
        except ImportError as e:
            raise ImportError(
                "PyAV is required for video decoding. Install with: pip install av"
            ) from e

        frames: list[np.ndarray] = []

        container = av.open(str(video_path))
        try:
            stream = container.streams.video[0]

            for frame in container.decode(stream):
                # Convert to RGB numpy array (HWC uint8)
                rgb_frame = frame.to_ndarray(format="rgb24")
                frames.append(rgb_frame)

                if len(frames) >= expected_frames:
                    break
        finally:
            container.close()

        return frames

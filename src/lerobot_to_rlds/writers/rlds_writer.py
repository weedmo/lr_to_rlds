"""RLDS (TFDS) dataset writer."""

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from lerobot_to_rlds.readers.base import Episode, Step


class RLDSWriter:
    """Writes episodes to RLDS (TFDS) format.

    Creates a TensorFlow Dataset with the standard RLDS schema:
    - episode_metadata: Episode-level information
    - steps: Sequence of (observation, action, reward, ...) tuples
    """

    def __init__(
        self,
        output_dir: Path,
        dataset_name: str,
        state_dim: int,
        action_dim: int,
        image_shapes: dict[str, tuple[int, int, int]],
        source_version: str = "v2.1",
    ) -> None:
        """Initialize the writer.

        Args:
            output_dir: Directory to write the RLDS dataset.
            dataset_name: Name for the output dataset.
            state_dim: Dimension of state vector.
            action_dim: Dimension of action vector.
            image_shapes: Dict mapping camera names to (H, W, C) shapes.
            source_version: Source LeRobot version string.
        """
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.image_shapes = image_shapes
        self.source_version = source_version

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track written episodes
        self._episode_count = 0
        self._total_steps = 0

    def _build_features(self) -> tfds.features.FeaturesDict:
        """Build TFDS feature specification."""
        observation_features: dict[str, Any] = {}

        # Add state feature
        if self.state_dim > 0:
            observation_features["state"] = tfds.features.Tensor(
                shape=(self.state_dim,),
                dtype=np.float32,
            )

        # Add image features
        for camera_name, shape in self.image_shapes.items():
            key = f"image_{camera_name}" if camera_name != "main" else "image"
            observation_features[key] = tfds.features.Image(
                shape=shape,
                dtype=np.uint8,
            )

        step_features = {
            "observation": tfds.features.FeaturesDict(observation_features),
            "action": tfds.features.Tensor(
                shape=(self.action_dim,),
                dtype=np.float32,
            ),
            "reward": tfds.features.Scalar(dtype=np.float32),
            "discount": tfds.features.Scalar(dtype=np.float32),
            "is_first": tfds.features.Scalar(dtype=np.bool_),
            "is_last": tfds.features.Scalar(dtype=np.bool_),
            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
            "language_instruction": tfds.features.Text(),
        }

        return tfds.features.FeaturesDict({
            "episode_metadata": tfds.features.FeaturesDict({
                "episode_id": tfds.features.Scalar(dtype=np.int64),
                "source_dataset_version": tfds.features.Text(),
                "source_episode_index": tfds.features.Scalar(dtype=np.int64),
                "file_path": tfds.features.Text(),
            }),
            "steps": tfds.features.Dataset(step_features),
        })

    def _step_to_dict(self, step: Step) -> dict[str, Any]:
        """Convert a Step to RLDS dictionary format."""
        observation: dict[str, Any] = {}

        # Add state
        if "state" in step.observation:
            state = step.observation["state"]
            # Ensure correct shape
            if state.size > 0:
                observation["state"] = state.astype(np.float32)

        # Add images
        for key, value in step.observation.items():
            if key.startswith("image_"):
                # Already in HWC uint8 format from reader
                observation[key] = value

        return {
            "observation": observation,
            "action": step.action.astype(np.float32),
            "reward": np.float32(step.reward),
            "discount": np.float32(1.0),
            "is_first": step.is_first,
            "is_last": step.is_last,
            "is_terminal": step.is_terminal,
            "language_instruction": step.language_instruction or "",
        }

    def _episode_to_dict(self, episode: Episode) -> dict[str, Any]:
        """Convert an Episode to RLDS dictionary format."""
        steps = [self._step_to_dict(step) for step in episode.steps]

        return {
            "episode_metadata": {
                "episode_id": np.int64(episode.info.episode_index),
                "source_dataset_version": self.source_version,
                "source_episode_index": np.int64(episode.info.episode_index),
                "file_path": str(episode.info.data_path),
            },
            "steps": steps,
        }

    def write_episodes(self, episodes: Iterator[Episode]) -> int:
        """Write episodes to TFDS format.

        Args:
            episodes: Iterator of Episode objects to write.

        Returns:
            Number of episodes written.
        """
        features = self._build_features()

        # Create dataset builder config
        dataset_dir = self.output_dir / self.dataset_name

        # Collect all episodes as examples
        examples = []
        for episode in episodes:
            example = self._episode_to_dict(episode)
            examples.append(example)
            self._episode_count += 1
            self._total_steps += len(episode.steps)

        if not examples:
            return 0

        # Write using TFRecordWriter
        self._write_tfrecords(dataset_dir, examples, features)

        return self._episode_count

    def _write_tfrecords(
        self,
        dataset_dir: Path,
        examples: list[dict[str, Any]],
        features: tfds.features.FeaturesDict,
    ) -> None:
        """Write examples to TFRecord files."""
        dataset_dir.mkdir(parents=True, exist_ok=True)
        version_dir = dataset_dir / "1.0.0"
        version_dir.mkdir(parents=True, exist_ok=True)

        # Write TFRecords
        tfrecord_path = version_dir / f"{self.dataset_name}-train.tfrecord-00000-of-00001"

        with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
            for example in examples:
                serialized = self._serialize_example(example, features)
                writer.write(serialized)

        # Write dataset_info.json
        self._write_dataset_info(version_dir, features, len(examples))

        # Write features.json
        self._write_features_json(version_dir, features)

    def _serialize_example(
        self,
        example: dict[str, Any],
        features: tfds.features.FeaturesDict,
    ) -> bytes:
        """Serialize an example to bytes using tf.train.Example."""
        # Flatten the nested structure for TFRecord
        feature_dict = {}

        # Episode metadata
        metadata = example["episode_metadata"]
        feature_dict["episode_metadata/episode_id"] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[int(metadata["episode_id"])])
        )
        feature_dict["episode_metadata/source_dataset_version"] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[metadata["source_dataset_version"].encode()])
        )
        feature_dict["episode_metadata/source_episode_index"] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[int(metadata["source_episode_index"])])
        )
        feature_dict["episode_metadata/file_path"] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[metadata["file_path"].encode()])
        )

        # Steps
        steps = example["steps"]
        num_steps = len(steps)
        feature_dict["steps/length"] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[num_steps])
        )

        # Serialize each step field as a flat array
        actions = []
        rewards = []
        discounts = []
        is_firsts = []
        is_lasts = []
        is_terminals = []
        language_instructions = []

        for step in steps:
            actions.extend(step["action"].flatten().tolist())
            rewards.append(float(step["reward"]))
            discounts.append(float(step["discount"]))
            is_firsts.append(int(step["is_first"]))
            is_lasts.append(int(step["is_last"]))
            is_terminals.append(int(step["is_terminal"]))
            language_instructions.append(step["language_instruction"].encode())

        feature_dict["steps/action"] = tf.train.Feature(
            float_list=tf.train.FloatList(value=actions)
        )
        feature_dict["steps/reward"] = tf.train.Feature(
            float_list=tf.train.FloatList(value=rewards)
        )
        feature_dict["steps/discount"] = tf.train.Feature(
            float_list=tf.train.FloatList(value=discounts)
        )
        feature_dict["steps/is_first"] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=is_firsts)
        )
        feature_dict["steps/is_last"] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=is_lasts)
        )
        feature_dict["steps/is_terminal"] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=is_terminals)
        )
        feature_dict["steps/language_instruction"] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=language_instructions)
        )

        # Observation state
        if any("state" in step["observation"] for step in steps):
            states = []
            for step in steps:
                if "state" in step["observation"]:
                    states.extend(step["observation"]["state"].flatten().tolist())
            if states:
                feature_dict["steps/observation/state"] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=states)
                )

        # Observation images - encode as raw bytes
        for step_idx, step in enumerate(steps):
            for key, value in step["observation"].items():
                if key.startswith("image_") or key == "image":
                    # Encode image as PNG bytes
                    encoded = tf.io.encode_png(value).numpy()
                    feature_dict[f"steps/{step_idx}/observation/{key}"] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[encoded])
                    )

        tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return tf_example.SerializeToString()

    def _write_dataset_info(
        self,
        version_dir: Path,
        features: tfds.features.FeaturesDict,
        num_examples: int,
    ) -> None:
        """Write dataset_info.json."""
        info = {
            "name": self.dataset_name,
            "version": "1.0.0",
            "description": f"RLDS dataset converted from LeRobot {self.source_version}",
            "splits": {
                "train": {
                    "name": "train",
                    "numExamples": num_examples,
                    "numShards": 1,
                }
            },
            "features": self._features_to_json(features),
            "supervisedKeys": None,
            "citation": "",
            "license": "",
        }

        info_path = version_dir / "dataset_info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    def _write_features_json(
        self,
        version_dir: Path,
        features: tfds.features.FeaturesDict,
    ) -> None:
        """Write features.json for the dataset."""
        features_path = version_dir / "features.json"
        with open(features_path, "w") as f:
            json.dump(self._features_to_json(features), f, indent=2)

    def _features_to_json(self, features: tfds.features.FeaturesDict) -> dict[str, Any]:
        """Convert TFDS features to JSON-serializable dict."""
        result: dict[str, Any] = {}

        for name, feature in features.items():
            if isinstance(feature, tfds.features.FeaturesDict):
                result[name] = {"type": "FeaturesDict", "features": self._features_to_json(feature)}
            elif isinstance(feature, tfds.features.Dataset):
                result[name] = {
                    "type": "Dataset",
                    "features": self._features_to_json(feature.feature),
                }
            elif isinstance(feature, tfds.features.Tensor):
                result[name] = {
                    "type": "Tensor",
                    "shape": list(feature.shape),
                    "dtype": str(feature.dtype),
                }
            elif isinstance(feature, tfds.features.Image):
                result[name] = {
                    "type": "Image",
                    "shape": list(feature.shape),
                    "dtype": "uint8",
                }
            elif isinstance(feature, tfds.features.Scalar):
                result[name] = {
                    "type": "Scalar",
                    "dtype": str(feature.dtype),
                }
            elif isinstance(feature, tfds.features.Text):
                result[name] = {"type": "Text"}
            else:
                result[name] = {"type": str(type(feature).__name__)}

        return result

    @property
    def episode_count(self) -> int:
        """Return the number of episodes written."""
        return self._episode_count

    @property
    def total_steps(self) -> int:
        """Return the total number of steps written."""
        return self._total_steps

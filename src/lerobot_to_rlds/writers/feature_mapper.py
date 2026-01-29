"""Map LeRobot features to OXE/OpenVLA standard schema."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from lerobot_to_rlds.readers.base import LeRobotReader


@dataclass
class OXEFeatureConfig:
    """Configuration for OXE feature mapping."""

    image_shape: tuple[int, int, int]  # (H, W, C)
    state_dim: int
    action_dim: int
    camera_mapping: dict[str, str] = field(default_factory=dict)  # LeRobot name -> OXE name

    @classmethod
    def from_reader(cls, reader: LeRobotReader) -> "OXEFeatureConfig":
        """Create config from LeRobotReader by inspecting first episode."""
        episodes = reader.list_episodes()
        if not episodes:
            raise ValueError("No episodes found in dataset")

        first_episode = reader.read_episode(episodes[0])
        if not first_episode.steps:
            raise ValueError("First episode has no steps")

        first_step = first_episode.steps[0]

        # Get state dimension
        state_dim = 0
        if "state" in first_step.observation:
            state = first_step.observation["state"]
            state_dim = state.shape[0] if state.size > 0 else 0

        # Get action dimension
        action_dim = first_step.action.shape[0] if first_step.action.size > 0 else 0

        # Build camera mapping and get image shape
        camera_mapping: dict[str, str] = {}
        image_shape: tuple[int, int, int] = (256, 256, 3)  # default

        for key, value in first_step.observation.items():
            if key.startswith("image_"):
                camera_name = key[6:]  # Remove "image_" prefix
                oxe_name = FeatureMapper.map_camera_name_static(camera_name)
                camera_mapping[key] = oxe_name
                # Use first image's shape
                if image_shape == (256, 256, 3) and len(value.shape) == 3:
                    image_shape = tuple(value.shape)  # type: ignore

        return cls(
            image_shape=image_shape,
            state_dim=state_dim,
            action_dim=action_dim,
            camera_mapping=camera_mapping,
        )


class FeatureMapper:
    """Maps LeRobot features to OXE-compatible TFDS features."""

    # Standard OXE camera naming conventions
    CAMERA_NAME_MAP = {
        # Top/main camera variations
        "top": "image",
        "main": "image",
        "front": "image",
        "exterior_image_1_left": "image",
        "exterior_image_2_left": "image",
        # Wrist camera variations
        "wrist": "wrist_image",
        "wrist_image_left": "wrist_image",
        "hand": "wrist_image",
        "gripper": "wrist_image",
    }

    def __init__(self, config: OXEFeatureConfig) -> None:
        """Initialize the mapper.

        Args:
            config: OXE feature configuration.
        """
        self.config = config

    @staticmethod
    def map_camera_name_static(lerobot_name: str) -> str:
        """Map LeRobot camera name to OXE standard name (static method)."""
        # Check direct mapping
        if lerobot_name in FeatureMapper.CAMERA_NAME_MAP:
            return FeatureMapper.CAMERA_NAME_MAP[lerobot_name]

        # Check if name contains key patterns
        lower_name = lerobot_name.lower()
        if "wrist" in lower_name or "hand" in lower_name or "gripper" in lower_name:
            return "wrist_image"
        if "top" in lower_name or "front" in lower_name or "main" in lower_name:
            return "image"

        # Default: use as-is but ensure it starts with image_
        if not lerobot_name.startswith("image"):
            return f"image_{lerobot_name}"
        return lerobot_name

    def map_camera_name(self, lerobot_name: str) -> str:
        """Map LeRobot camera name to OXE standard name."""
        # Check configured mapping first
        if lerobot_name in self.config.camera_mapping:
            return self.config.camera_mapping[lerobot_name]
        return self.map_camera_name_static(lerobot_name)

    def build_observation_info(self) -> tfds.features.FeaturesDict:
        """Build observation feature specification."""
        features: dict[str, Any] = {}

        # Add state feature
        if self.config.state_dim > 0:
            features["state"] = tfds.features.Tensor(
                shape=(self.config.state_dim,),
                dtype=np.float32,
            )

        # Add image features based on camera mapping
        added_cameras: set[str] = set()
        for lerobot_name, oxe_name in self.config.camera_mapping.items():
            if oxe_name not in added_cameras:
                features[oxe_name] = tfds.features.Image(
                    shape=self.config.image_shape,
                    dtype=np.uint8,
                    encoding_format="png",
                )
                added_cameras.add(oxe_name)

        # Note: We no longer add a default "image" feature if no cameras exist.
        # State-only datasets are valid RLDS datasets.

        return tfds.features.FeaturesDict(features)

    def build_action_info(self) -> tfds.features.Tensor:
        """Build action feature specification."""
        return tfds.features.Tensor(
            shape=(self.config.action_dim,),
            dtype=np.float32,
        )

    def build_dataset_config(self, name: str) -> "tfds.rlds.rlds_base.DatasetConfig":
        """Build complete TFDS RLDS DatasetConfig.

        Args:
            name: Dataset name.

        Returns:
            DatasetConfig for use with EpisodeWriter.
        """
        return tfds.rlds.rlds_base.DatasetConfig(
            name=name,
            observation_info=self.build_observation_info(),
            action_info=self.build_action_info(),
            reward_info=tf.float32,
            discount_info=tf.float32,
            step_metadata_info={
                "language_instruction": tfds.features.Text(),
            },
        )

# RLDS Schema Specification

## Standard RLDS Episode Structure

RLDS (Reinforcement Learning Datasets) follows a hierarchical structure where each dataset contains episodes, and each episode contains steps.

## Target Schema

```python
{
    "episode_metadata": {
        "episode_id": tf.int64,
        "source_dataset_version": tf.string,  # "v2.1" or "v3.0"
        "source_episode_index": tf.int64,
        "tasks": tf.string,  # JSON array of task names
        "language_instruction": tf.string,  # Optional, empty if not present
        "file_path": tf.string,  # Original data path for traceability
    },
    "steps": [{
        "observation": {
            "image": tf.uint8,  # Shape: [H, W, C], RGB
            # Additional camera images as separate keys:
            # "image_wrist": tf.uint8,
            # "image_side": tf.uint8,
            "state": tf.float32,  # Shape: [state_dim]
        },
        "action": tf.float32,  # Shape: [action_dim]
        "reward": tf.float32,  # Scalar, default: 0.0
        "discount": tf.float32,  # Scalar, default: 1.0
        "is_first": tf.bool,  # True for step[0]
        "is_last": tf.bool,  # True for step[-1]
        "is_terminal": tf.bool,  # True if episode terminated (not truncated)
        "language_instruction": tf.string,  # Per-step instruction (optional)
    }]
}
```

## LeRobot to RLDS Mapping

### Image Observations

| LeRobot Key | LeRobot Format | RLDS Key | RLDS Format | Transformation |
|-------------|----------------|----------|-------------|----------------|
| `observation.images.top` | CHW float32 [0,1] | `observation/image` | HWC uint8 [0,255] | transpose + scale |
| `observation.images.wrist` | CHW float32 [0,1] | `observation/image_wrist` | HWC uint8 [0,255] | transpose + scale |
| `observation.images.*` | CHW float32 [0,1] | `observation/image_*` | HWC uint8 [0,255] | transpose + scale |

### State Observations

| LeRobot Key | LeRobot Format | RLDS Key | RLDS Format | Transformation |
|-------------|----------------|----------|-------------|----------------|
| `observation.state` | float32 [N] | `observation/state` | float32 [N] | direct copy |
| `observation.state.*` | varies | merged into state | float32 [M] | concatenate |

### Actions

| LeRobot Key | LeRobot Format | RLDS Key | RLDS Format | Transformation |
|-------------|----------------|----------|-------------|----------------|
| `action` | float32 [A] | `action` | float32 [A] | direct copy |

### Step Metadata

| RLDS Key | Source | Default Value | Notes |
|----------|--------|---------------|-------|
| `reward` | LeRobot `reward` if exists | `0.0` | Most manipulation datasets lack reward |
| `discount` | N/A | `1.0` | Standard RL convention |
| `is_first` | Computed | `step_idx == 0` | First step in episode |
| `is_last` | Computed | `step_idx == len-1` | Last step in episode |
| `is_terminal` | LeRobot `done` if exists | `False` | Terminal vs truncated |
| `language_instruction` | LeRobot `task` | `""` | Natural language task |

## Image Transformation Details

### CHW to HWC Conversion

```python
def chw_to_hwc(image: np.ndarray) -> np.ndarray:
    """Convert CHW float [0,1] to HWC uint8 [0,255]."""
    # Input: (C, H, W) float32 in [0, 1]
    # Output: (H, W, C) uint8 in [0, 255]

    if image.ndim != 3:
        raise ValueError(f"Expected 3D array, got {image.ndim}D")

    # Transpose CHW -> HWC
    hwc = np.transpose(image, (1, 2, 0))

    # Scale [0,1] -> [0,255] and convert to uint8
    scaled = np.clip(hwc * 255, 0, 255).astype(np.uint8)

    return scaled
```

### Video Frame Extraction

For datasets with video files instead of frame arrays:

```python
def extract_frames(video_path: Path, episode_length: int) -> Iterator[np.ndarray]:
    """Extract frames from video file."""
    import av

    container = av.open(str(video_path))
    stream = container.streams.video[0]

    frame_count = 0
    for frame in container.decode(stream):
        if frame_count >= episode_length:
            break
        # Convert to numpy RGB
        yield frame.to_ndarray(format='rgb24')
        frame_count += 1

    container.close()

    if frame_count != episode_length:
        raise ValueError(f"Expected {episode_length} frames, got {frame_count}")
```

## TFDS Feature Specification

```python
import tensorflow_datasets as tfds

def get_rlds_features(config: DatasetConfig) -> tfds.features.FeaturesDict:
    """Build TFDS feature specification."""

    observation_features = {
        "state": tfds.features.Tensor(
            shape=(config.state_dim,),
            dtype=tf.float32,
        ),
    }

    # Add image features dynamically
    for camera_name, image_shape in config.cameras.items():
        key = f"image_{camera_name}" if camera_name != "main" else "image"
        observation_features[key] = tfds.features.Image(
            shape=image_shape,  # (H, W, C)
            dtype=tf.uint8,
        )

    step_features = {
        "observation": tfds.features.FeaturesDict(observation_features),
        "action": tfds.features.Tensor(
            shape=(config.action_dim,),
            dtype=tf.float32,
        ),
        "reward": tfds.features.Scalar(dtype=tf.float32),
        "discount": tfds.features.Scalar(dtype=tf.float32),
        "is_first": tfds.features.Scalar(dtype=tf.bool),
        "is_last": tfds.features.Scalar(dtype=tf.bool),
        "is_terminal": tfds.features.Scalar(dtype=tf.bool),
        "language_instruction": tfds.features.Text(),
    }

    return tfds.features.FeaturesDict({
        "episode_metadata": tfds.features.FeaturesDict({
            "episode_id": tfds.features.Scalar(dtype=tf.int64),
            "source_dataset_version": tfds.features.Text(),
            "source_episode_index": tfds.features.Scalar(dtype=tf.int64),
            "tasks": tfds.features.Text(),
            "language_instruction": tfds.features.Text(),
            "file_path": tfds.features.Text(),
        }),
        "steps": tfds.features.Dataset(step_features),
    })
```

## Validation Rules

### Schema Validation
1. All required keys must be present
2. Tensor shapes must match specification
3. Dtypes must match specification
4. No NaN or Inf values in numeric tensors

### Content Validation
1. `is_first` must be True only for step[0]
2. `is_last` must be True only for step[-1]
3. Image values must be in [0, 255] for uint8
4. State/action values should be finite (no NaN/Inf)

### Cross-Validation
1. Episode count in output == episode count in input
2. Step count per episode must match
3. Total steps across all episodes must match

## Error Handling

### Missing Features
- If LeRobot dataset lacks a feature, use default value
- Log warning when using defaults
- Include in validation report

### Type Mismatches
- Attempt automatic conversion if safe
- Fail with clear error if conversion is lossy
- Document all conversions in mapping.md

### Invalid Data
- NaN/Inf values: Replace with 0.0, log warning
- Out-of-range images: Clip and log warning
- Missing frames: Fail episode, continue with others

# CLAUDE.md - LeRobot to RLDS Converter

## Project Overview

Python CLI tool to convert LeRobot datasets (v2.1 and v3.0) to RLDS (TFDS) format for use with OpenVLA, OXE, and other reinforcement learning frameworks.

## Quick Start

```bash
# Setup
cd /home/weed/tommoro/lerobot_to_rlds
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Install rlds submodule
pip install -e ./rlds

# Run tests
pytest

# Convert a dataset (OXE/OpenVLA compatible - default)
lerobot-to-rlds convert /path/to/lerobot/dataset --output ./output

# Convert with legacy format
lerobot-to-rlds convert /path/to/lerobot/dataset --output ./output --format legacy
```

## Architecture

```
LeRobot Dataset → [Reader] → [OXE Writer] → RLDS Dataset (tfds.load compatible)
                     ↓
              rlds submodule
              (EpisodeWriter)
```

### Output Formats

| Format | Writer | Compatible With | Command |
|--------|--------|-----------------|---------|
| `oxe` (default) | `OXERLDSWriter` | OpenVLA, OXE, DROID | `--format oxe` |
| `legacy` | `RLDSWriter` | Custom pipelines | `--format legacy` |

## Module Structure

```
lerobot_to_rlds/
├── rlds/                   # git submodule (google-research/rlds)
│   └── rlds/
│       ├── rlds_types.py   # build_step(), build_episode()
│       └── tfds/
│           └── episode_writer.py  # EpisodeWriter
├── src/lerobot_to_rlds/
│   ├── cli.py              # Click CLI entry point
│   ├── core/               # Types, exceptions, constants
│   ├── readers/            # LeRobot v2.1/v3.0 readers
│   │   ├── base.py         # Step, Episode, LeRobotReader
│   │   ├── v21_reader.py   # v2.1 reader
│   │   └── v30_reader.py   # v3.0 reader
│   ├── writers/
│   │   ├── oxe_writer.py   # OXE-compatible (uses rlds submodule)
│   │   ├── feature_mapper.py  # LeRobot → OXE feature mapping
│   │   └── rlds_writer.py  # Legacy custom writer
│   └── pipeline/
│       └── convert.py      # ConversionPipeline
```

## CLI Commands

```bash
# Discover dataset structure
lerobot-to-rlds discover <dataset_path>

# Convert with OXE format (default, OpenVLA compatible)
lerobot-to-rlds convert <dataset_path> --output <output_dir>
lerobot-to-rlds convert <dataset_path> --output <output_dir> --format oxe

# Convert with legacy format
lerobot-to-rlds convert <dataset_path> --output <output_dir> --format legacy

# Resume interrupted conversion
lerobot-to-rlds convert <dataset_path> --output <output_dir> --resume

# Verbose logging
lerobot-to-rlds -v convert <dataset_path> --output <output_dir>
```

## OXE/OpenVLA Compatible Output

The default `oxe` format produces datasets loadable with `tfds.load()`:

```python
import tensorflow_datasets as tfds

# Load converted dataset
ds = tfds.load('dataset_name', data_dir='./output', split='train')

for episode in ds.take(1):
    for step in episode['steps']:
        image = step['observation']['image']
        action = step['action']
        instruction = step['language_instruction']
```

### OXE Schema

```python
{
    'steps': tfds.features.Dataset({
        'observation': {
            'image': Image(H, W, 3),        # Main camera
            'wrist_image': Image(H, W, 3),  # Wrist camera (if available)
            'state': Tensor(N, float32),    # Proprioception
        },
        'action': Tensor(A, float32),
        'reward': Scalar(float32),
        'discount': Scalar(float32),
        'is_first': bool,
        'is_last': bool,
        'is_terminal': bool,
        'language_instruction': Text(),
    }),
}
```

## Conversion Pipeline

```python
from lerobot_to_rlds.pipeline import convert_dataset
from lerobot_to_rlds.core.types import ConvertMode

result = convert_dataset(
    dataset_path=Path("/path/to/lerobot/dataset"),
    output_dir=Path("./output"),
    mode=ConvertMode.SAFE,
    output_format="oxe",  # or "legacy"
)

if result.success:
    print(f"Converted {result.episodes_converted} episodes")
    print(f"Output: {result.output_path}")
```

## Current Status

- ✅ LeRobot v2.1/v3.0 readers
- ✅ OXE-compatible writer (`rlds.tfds.EpisodeWriter`)
- ✅ `tfds.load()` compatible output
- ✅ Resume from checkpoint
- ⏳ PARALLEL modes: Planned
- ⏳ Validation stage: Planned

## Dependencies

### Required
- `tensorflow-datasets >= 4.9.0` - RLDS format
- `tensorflow >= 2.15.0` - TensorFlow backend
- `pyarrow >= 15.0.0` - Parquet reading
- `pandas >= 2.0.0` - DataFrame operations
- `av >= 15.0.0` - Video decoding
- `numpy >= 1.24.0` - Numeric operations
- `click >= 8.1.0` - CLI framework
- `tqdm >= 4.66.0` - Progress bars
- `absl-py` - Required by rlds submodule

### rlds Submodule
```bash
# Install rlds submodule
pip install -e ./rlds
```

## Testing

```bash
# All tests
pytest

# With coverage
pytest --cov=src/lerobot_to_rlds --cov-report=term-missing
```

## Ralph Loop Integration

```bash
cd /home/weed/tommoro/lerobot_to_rlds
claude
> ralph
```

See `.ralph/fix_plan.md` for current development focus.

## Key Files

| File | Purpose |
|------|---------|
| `writers/oxe_writer.py` | OXE-compatible writer using rlds submodule |
| `writers/feature_mapper.py` | LeRobot → OXE feature mapping |
| `pipeline/convert.py` | Conversion orchestrator with format selection |
| `rlds/rlds/tfds/episode_writer.py` | Official RLDS episode writer |

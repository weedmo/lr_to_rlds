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

# Interactive menu
./main.sh

# Convert a dataset (outputs to data/<folder_name>/ by default)
lerobot-to-rlds convert /path/to/lerobot/dataset

# Visualize converted RLDS dataset
lerobot-to-rlds visualize list ./data/my_dataset
```

## Architecture

```
LeRobot Dataset → [Reader] → [OXE Writer] → RLDS Dataset (tfds.load compatible)
                     ↓                              ↓
              rlds submodule                 Visualization
              (EpisodeWriter)                (tfds.load)
```

### Output Formats

| Format | Writer | Compatible With | Command |
|--------|--------|-----------------|---------|
| `oxe` (default) | `OXERLDSWriter` | OpenVLA, OXE, DROID | `--format oxe` |
| `legacy` | `RLDSWriter` | Custom pipelines | `--format legacy` |

## Module Structure

```
lerobot_to_rlds/
├── main.sh                 # Interactive entry point
├── data/                   # Default output directory (RLDS datasets)
├── rlds/                   # git submodule (google-research/rlds)
│   └── rlds/
│       ├── rlds_types.py   # build_step(), build_episode()
│       └── tfds/
│           └── episode_writer.py  # EpisodeWriter
├── src/lerobot_to_rlds/
│   ├── cli.py              # Click CLI entry point
│   ├── core/               # Types, exceptions, constants
│   ├── readers/            # LeRobot v2.1/v3.0 readers (source)
│   │   ├── base.py         # Step, Episode, LeRobotReader
│   │   ├── v21_reader.py   # v2.1 reader
│   │   └── v30_reader.py   # v3.0 reader
│   ├── writers/
│   │   ├── oxe_writer.py   # OXE-compatible (uses rlds submodule)
│   │   ├── feature_mapper.py  # LeRobot → OXE feature mapping
│   │   └── rlds_writer.py  # Legacy custom writer
│   ├── utils/
│   │   ├── logging.py      # Logging utilities
│   │   └── naming.py       # Folder name sanitization, output path
│   ├── visualization/      # RLDS dataset visualization (tfds.load)
│   │   ├── visualizer.py   # DatasetVisualizer, EpisodeSummary
│   │   ├── plotter.py      # EpisodePlotter (matplotlib)
│   │   └── frame_viewer.py # FrameViewer
│   └── pipeline/
│       └── convert.py      # ConversionPipeline
```

## CLI Commands

```bash
# Interactive menu
./main.sh

# List RLDS datasets in data/
lerobot-to-rlds list-datasets
lerobot-to-rlds list-datasets --data-dir /custom/path

# Discover LeRobot source dataset structure
lerobot-to-rlds discover <lerobot_path>

# Convert LeRobot → RLDS (outputs to data/<folder_name>/ by default)
lerobot-to-rlds convert <lerobot_path>
lerobot-to-rlds convert <lerobot_path> -n custom_name      # Custom folder name
lerobot-to-rlds convert <lerobot_path> -o /explicit/path   # Explicit output
lerobot-to-rlds convert <lerobot_path> -f legacy           # Legacy format
lerobot-to-rlds convert <lerobot_path> --resume            # Resume from checkpoint

# Visualize RLDS dataset (converted output)
lerobot-to-rlds visualize list <rlds_path>                 # List episodes
lerobot-to-rlds visualize info <rlds_path>                 # Detailed info
lerobot-to-rlds visualize plot <rlds_path> -e 0 -t both    # Plot state/action
lerobot-to-rlds visualize plot <rlds_path> -e 0 -s plot.png --no-show  # Save plot
lerobot-to-rlds visualize frames <rlds_path> -e 0 -s 50    # View single frame
lerobot-to-rlds visualize frames <rlds_path> -e 0 --steps 0,10,20,30   # Frame grid
lerobot-to-rlds visualize cameras <rlds_path> -e 0         # List cameras
lerobot-to-rlds visualize export-frames <rlds_path> ./output -e 0      # Export PNGs

# Verbose logging
lerobot-to-rlds -v convert <lerobot_path>
```

## main.sh Interactive Menu

```
Main Menu:
  1) List RLDS datasets in data/
  2) Discover LeRobot dataset (source)
  3) Visualize RLDS dataset →
       1) List episodes
       2) Show dataset info
       3) Plot state/action
       4) View frames
       5) List cameras
       6) Export frames as images
  4) Convert LeRobot to RLDS
  h) Help
  q) Quit
```

Features:
- Auto venv activation
- Paths relative to data/ by default
- Task name extraction from folder name

## Workflow

```
1. Source (LeRobot)          2. Convert              3. Output (RLDS)
   /path/to/lerobot/     →   lerobot-to-rlds    →   data/<name>/
   ├── meta/                  convert                ├── dataset_info.json
   │   └── info.json                                 ├── features.json
   └── data/                                         └── *.tfrecord
       └── *.parquet
                                                  4. Visualize
                                                     lerobot-to-rlds visualize list data/<name>/
```

## RLDS Visualization API

```python
from lerobot_to_rlds.visualization import DatasetVisualizer, EpisodePlotter, FrameViewer
from pathlib import Path

# List episodes from converted RLDS dataset
viz = DatasetVisualizer(Path("./data/my_dataset"))
viz.print_episodes_table()
viz.print_dataset_info()

# Plot state/action
plotter = EpisodePlotter(viz)
plotter.plot(episode_index=0, plot_type="both", save_path=Path("plot.png"))

# View frames
viewer = FrameViewer(viz)
viewer.display_frame(episode_index=0, step_idx=50, camera="image")
viewer.display_frame_grid(episode_index=0, step_indices=[0, 10, 20, 30])
viewer.save_frames_as_images(episode_index=0, output_dir=Path("./frames"))
```

## OXE/OpenVLA Compatible Output

The default `oxe` format produces datasets loadable with `tfds.load()`:

```python
import tensorflow_datasets as tfds

# Load converted dataset
ds = tfds.load('my_dataset', data_dir='./data', split='train')

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

## Current Status

- ✅ LeRobot v2.1/v3.0 readers
- ✅ OXE-compatible writer (`rlds.tfds.EpisodeWriter`)
- ✅ `tfds.load()` compatible output
- ✅ Resume from checkpoint
- ✅ RLDS visualization (list, plot, frames)
- ✅ Interactive menu (main.sh)
- ✅ Auto output to data/<task_name>/
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
- `matplotlib >= 3.7.0` - Plotting
- `pillow >= 10.0.0` - Image handling
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

## Key Files

| File | Purpose |
|------|---------|
| `main.sh` | Interactive entry point |
| `cli.py` | Click CLI with visualize subcommands |
| `utils/naming.py` | Folder name sanitization, output path |
| `visualization/visualizer.py` | RLDS DatasetVisualizer (tfds.load) |
| `visualization/plotter.py` | State/action plotting |
| `visualization/frame_viewer.py` | Image frame display |
| `writers/oxe_writer.py` | OXE-compatible writer using rlds submodule |
| `writers/feature_mapper.py` | LeRobot → OXE feature mapping |
| `pipeline/convert.py` | Conversion orchestrator with format selection |
| `rlds/rlds/tfds/episode_writer.py` | Official RLDS episode writer |

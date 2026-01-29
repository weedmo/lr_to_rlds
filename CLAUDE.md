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

# Convert with custom name
lerobot-to-rlds convert /path/to/lerobot/dataset -n my_task_name

# Convert with explicit output path
lerobot-to-rlds convert /path/to/lerobot/dataset -o ./custom/output
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
├── main.sh                 # Interactive entry point
├── data/                   # Default output directory
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
│   ├── utils/
│   │   ├── logging.py      # Logging utilities
│   │   └── naming.py       # Folder name sanitization, output path
│   ├── visualization/      # Dataset visualization
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

# List datasets in data/
lerobot-to-rlds list-datasets
lerobot-to-rlds list-datasets --data-dir /custom/path

# Discover dataset structure
lerobot-to-rlds discover <dataset_path>

# Convert (outputs to data/<folder_name>/ by default)
lerobot-to-rlds convert <dataset_path>
lerobot-to-rlds convert <dataset_path> -n custom_name      # Custom folder name
lerobot-to-rlds convert <dataset_path> -o /explicit/path   # Explicit output
lerobot-to-rlds convert <dataset_path> -f legacy           # Legacy format
lerobot-to-rlds convert <dataset_path> --resume            # Resume from checkpoint

# Visualization commands
lerobot-to-rlds visualize list <path>                      # List episodes
lerobot-to-rlds visualize info <path>                      # Detailed info
lerobot-to-rlds visualize plot <path> -e 0 -t both         # Plot state/action
lerobot-to-rlds visualize plot <path> -e 0 -s plot.png --no-show  # Save plot
lerobot-to-rlds visualize frames <path> -e 0 -s 50         # View single frame
lerobot-to-rlds visualize frames <path> -e 0 --steps 0,10,20,30   # Frame grid
lerobot-to-rlds visualize cameras <path> -e 0              # List cameras
lerobot-to-rlds visualize export-frames <path> ./output -e 0      # Export PNGs

# Verbose logging
lerobot-to-rlds -v convert <dataset_path>
```

## main.sh Interactive Menu

```
Main Menu:
  1) List datasets in data/
  2) Discover dataset structure
  3) Visualize dataset →
       1) List episodes
       2) Show dataset info
       3) Plot state/action
       4) View frames
       5) List cameras
       6) Export frames as images
  4) Convert dataset
  h) Help
  q) Quit
```

Features:
- Auto venv activation
- Paths relative to data/ by default
- Task name extraction from folder name

## OXE/OpenVLA Compatible Output

The default `oxe` format produces datasets loadable with `tfds.load()`:

```python
import tensorflow_datasets as tfds

# Load converted dataset
ds = tfds.load('dataset_name', data_dir='./data', split='train')

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
    output_dir=Path("./data/my_task"),  # or use get_output_path()
    mode=ConvertMode.SAFE,
    output_format="oxe",  # or "legacy"
)

if result.success:
    print(f"Converted {result.episodes_converted} episodes")
    print(f"Output: {result.output_path}")
```

## Visualization API

```python
from lerobot_to_rlds.visualization import DatasetVisualizer, EpisodePlotter, FrameViewer
from lerobot_to_rlds.readers import get_reader

# List episodes
viz = DatasetVisualizer(Path("./dataset"))
viz.print_episodes_table()
viz.print_dataset_info()

# Plot state/action
reader = get_reader(Path("./dataset"))
plotter = EpisodePlotter(reader)
plotter.plot(episode_index=0, plot_type="both", save_path=Path("plot.png"))

# View frames
viewer = FrameViewer(reader)
viewer.display_frame(episode_index=0, step_idx=50)
viewer.display_frame_grid(episode_index=0, step_indices=[0, 10, 20, 30])
viewer.save_frames_as_images(episode_index=0, output_dir=Path("./frames"))
```

## Current Status

- ✅ LeRobot v2.1/v3.0 readers
- ✅ OXE-compatible writer (`rlds.tfds.EpisodeWriter`)
- ✅ `tfds.load()` compatible output
- ✅ Resume from checkpoint
- ✅ Visualization (list, plot, frames)
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
| `main.sh` | Interactive entry point |
| `cli.py` | Click CLI with visualize subcommands |
| `utils/naming.py` | Folder name sanitization, output path |
| `visualization/visualizer.py` | DatasetVisualizer coordinator |
| `visualization/plotter.py` | State/action plotting |
| `visualization/frame_viewer.py` | Video frame display |
| `writers/oxe_writer.py` | OXE-compatible writer using rlds submodule |
| `writers/feature_mapper.py` | LeRobot → OXE feature mapping |
| `pipeline/convert.py` | Conversion orchestrator with format selection |
| `rlds/rlds/tfds/episode_writer.py` | Official RLDS episode writer |

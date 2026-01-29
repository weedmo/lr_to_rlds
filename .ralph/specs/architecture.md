# Architecture Specification: LeRobot → RLDS Converter

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LeRobot Dataset (Input)                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │   v2.1 Format   │  │   v3.0 Format   │  │  Auto-Detect    │          │
│  │ episodes.jsonl  │  │ episodes/*.parq │  │   via info.json │          │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘          │
└───────────┼────────────────────┼────────────────────┼────────────────────┘
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Conversion Pipeline                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Stage A  │→ │ Stage B  │→ │ Stage C  │→ │ Stage D  │→ │ Stage E  │  │
│  │ Discover │  │   Spec   │  │ Convert  │  │ Validate │  │ Publish  │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         RLDS Dataset (Output)                            │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  rlds/<dataset_name>/<version>/                                  │    │
│  │  ├── dataset_info.json                                          │    │
│  │  ├── features.json                                              │    │
│  │  └── data/                                                      │    │
│  │      ├── train-00000-of-00010.tfrecord                          │    │
│  │      └── ...                                                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
src/
├── lerobot_to_rlds/
│   ├── __init__.py
│   ├── cli.py                    # CLI entry point
│   ├── config.py                 # Configuration dataclasses
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── types.py              # Core type definitions
│   │   ├── exceptions.py         # Custom exceptions
│   │   └── constants.py          # Constants and enums
│   │
│   ├── readers/
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract base reader
│   │   ├── v21_reader.py         # LeRobot v2.1 reader
│   │   ├── v30_reader.py         # LeRobot v3.0 reader
│   │   └── detector.py           # Version auto-detection
│   │
│   ├── writers/
│   │   ├── __init__.py
│   │   ├── rlds_writer.py        # TFDS/RLDS writer
│   │   └── shard_manager.py      # Shard file management
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── discover.py           # Stage A: Discovery
│   │   ├── spec.py               # Stage B: Schema mapping
│   │   ├── convert.py            # Stage C: Conversion
│   │   ├── validate.py           # Stage D: Validation
│   │   └── publish.py            # Stage E: Publishing
│   │
│   ├── parallel/
│   │   ├── __init__.py
│   │   ├── worker.py             # Worker process
│   │   ├── pool.py               # Worker pool manager
│   │   └── checkpoint.py         # Progress checkpointing
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py            # Structured logging
│       ├── hashing.py            # Data integrity hashing
│       ├── stats.py              # Statistics computation
│       └── image.py              # Image format conversion
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # Pytest fixtures
│   ├── test_readers/
│   ├── test_writers/
│   ├── test_pipeline/
│   └── fixtures/                 # Test data fixtures
│       ├── v21_sample/
│       └── v30_sample/
│
└── pyproject.toml
```

## Data Flow

### Episode Processing Flow

```
LeRobot Episode
      │
      ▼
┌─────────────────┐
│  Read Episode   │  ← Parquet/Video files
│  - metadata     │
│  - observations │
│  - actions      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Transform Data  │
│ - CHW → HWC     │
│ - float → uint8 │
│ - normalize     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Build Steps    │  ← Add is_first, is_last, reward, discount
│  - step[0..N]   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Write to TFDS   │  → TFRecord shard
│ - serialize     │
│ - index         │
└─────────────────┘
```

## Parallel Processing Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Main Process                                  │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │
│  │ Episode Queue │→ │ Worker Pool   │→ │ Result Queue  │           │
│  │  [ep0..epN]   │  │ (N workers)   │  │ [ok/fail]     │           │
│  └───────────────┘  └───────────────┘  └───────────────┘           │
│         │                  │                   │                    │
│         │           ┌──────┴──────┐           │                    │
│         │           ▼             ▼           │                    │
│         │    ┌──────────┐  ┌──────────┐      │                    │
│         │    │ Worker 0 │  │ Worker 1 │ ... │                    │
│         │    │ ep[i]    │  │ ep[j]    │      │                    │
│         │    └──────────┘  └──────────┘      │                    │
│         │                                     │                    │
│         └─────────────────────────────────────┘                    │
│                           │                                         │
│                           ▼                                         │
│                  ┌─────────────────┐                               │
│                  │ progress.jsonl  │  ← Checkpoint per episode     │
│                  └─────────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Episode-Level Transactions
- Each episode is processed as an atomic unit
- Success/failure is recorded per episode in `progress.jsonl`
- Failed episodes can be retried without reprocessing successful ones

### 2. Version Detection Strategy
```python
def detect_version(dataset_root: Path) -> LeRobotVersion:
    if (dataset_root / "meta" / "episodes").is_dir():
        return LeRobotVersion.V30
    elif (dataset_root / "meta" / "episodes.jsonl").exists():
        return LeRobotVersion.V21
    else:
        raise VersionDetectionError("Cannot detect LeRobot version")
```

### 3. Memory Protection
- Worker count is limited by available RAM
- Each worker processes one episode at a time
- Large videos are streamed, not loaded entirely into memory

### 4. Reproducibility
- Deterministic shard assignment (hash-based)
- Fixed random seeds for any sampling
- Output hash recorded for verification

## Dependencies

### Required
- `tensorflow-datasets >= 4.9.0` - TFDS/RLDS format
- `pyarrow >= 15.0.0` - Parquet reading
- `av >= 15.0.0` - Video decoding
- `numpy >= 1.24.0` - Numeric operations
- `pillow >= 10.0.0` - Image processing
- `tqdm >= 4.66.0` - Progress bars
- `click >= 8.1.0` - CLI framework

### Optional
- `pandas >= 2.0.0` - DataFrames (for analysis)
- `psutil >= 5.9.0` - System resource monitoring

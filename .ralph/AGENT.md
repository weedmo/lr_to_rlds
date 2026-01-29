# Agent Build Instructions: LeRobot → RLDS Converter

## Project Overview
Python CLI tool to convert LeRobot (v2.1/v3.0) datasets to RLDS (TFDS) format.

## Project Setup

```bash
cd /home/weed/tommoro/lerobot_to_rlds

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Or install dependencies manually
pip install -r requirements.txt
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/lerobot_to_rlds --cov-report=term-missing

# Run specific test file
pytest tests/test_readers/test_detector.py -v

# Run with verbose output
pytest -v -s
```

## Build Commands

```bash
# Build package
python -m build

# Type checking
mypy src/lerobot_to_rlds

# Linting
ruff check src/lerobot_to_rlds
ruff format src/lerobot_to_rlds
```

## CLI Usage

```bash
# Discover dataset structure
lerobot-to-rlds discover /path/to/lerobot/dataset --output ./output

# Convert dataset (SAFE mode - default)
lerobot-to-rlds convert /path/to/lerobot/dataset --output ./output

# Convert with parallel processing
lerobot-to-rlds convert /path/to/lerobot/dataset --output ./output --mode parallel-half
lerobot-to-rlds convert /path/to/lerobot/dataset --output ./output --mode parallel-max

# Resume interrupted conversion
lerobot-to-rlds convert /path/to/lerobot/dataset --output ./output --resume

# Validate converted dataset
lerobot-to-rlds validate ./output/rlds/dataset_name

# Full pipeline
lerobot-to-rlds run /path/to/lerobot/dataset --output ./output --mode safe
```

## Directory Structure

```
lerobot_to_rlds/
├── .ralph/
│   ├── PROMPT.md          # Ralph loop instructions
│   ├── AGENT.md           # This file
│   ├── fix_plan.md        # Task tracking
│   └── specs/             # Specifications
│       ├── requirements.md
│       ├── architecture.md
│       ├── rlds_schema.md
│       ├── pipeline_stages.md
│       └── parallel_processing.md
├── src/
│   └── lerobot_to_rlds/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── core/
│       ├── readers/
│       ├── writers/
│       ├── pipeline/
│       ├── parallel/
│       └── utils/
├── tests/
│   ├── conftest.py
│   ├── test_readers/
│   ├── test_writers/
│   ├── test_pipeline/
│   └── fixtures/
├── pyproject.toml
├── requirements.txt
├── CLAUDE.md
└── README.md
```

## Key Learnings

### LeRobot Version Detection
- v3.0: Has `meta/episodes/*.parquet` directory
- v2.1: Has `meta/episodes.jsonl` file
- Always check `meta/info.json` for metadata

### RLDS/TFDS Writing
- Use `tensorflow_datasets.core.SequentialWriter` for large datasets
- Shard size should be ~100MB for optimal performance
- Always include `dataset_info.json` and `features.json`

### Image Format Conversion
- LeRobot: CHW float32 [0, 1]
- RLDS: HWC uint8 [0, 255]
- Use `np.transpose(img, (1, 2, 0))` then `(img * 255).astype(np.uint8)`

### Video Handling
- Use PyAV (`av` package) for video decoding
- Stream frames instead of loading all to memory
- Match frame count with episode length from parquet

### Parallel Processing
- Use `multiprocessing` not `threading` (GIL)
- Episode-level parallelism (not step-level)
- Always checkpoint after each episode

## Feature Development Quality Standards

**CRITICAL**: All new features MUST meet the following mandatory requirements before being considered complete.

### Testing Requirements

- **Minimum Coverage**: 85% code coverage ratio required for all new code
- **Test Pass Rate**: 100% - all tests must pass, no exceptions
- **Test Types Required**:
  - Unit tests for all business logic and services
  - Integration tests for CLI commands
  - End-to-end tests for full conversion pipeline
- **Coverage Validation**:
  ```bash
  pytest --cov=src/lerobot_to_rlds tests/ --cov-report=term-missing --cov-fail-under=85
  ```

### Git Workflow Requirements

Before moving to the next feature, ALL changes must be:

1. **Committed with Clear Messages**:
   ```bash
   git add .
   git commit -m "feat(readers): implement v3.0 parquet reader"
   ```
   - Use conventional commit format: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`
   - Include scope: `feat(readers):`, `fix(cli):`, `test(pipeline):`

2. **Pushed to Remote Repository**:
   ```bash
   git push origin <branch-name>
   ```

### Documentation Requirements

1. **Code Documentation**:
   - All public functions must have docstrings
   - Type hints for all function signatures
   - Inline comments for complex logic

2. **Update CLAUDE.md**: After significant changes

### Feature Completion Checklist

Before marking ANY feature as complete, verify:

- [ ] All tests pass with `pytest`
- [ ] Code coverage meets 85% minimum threshold
- [ ] Type checking passes with `mypy`
- [ ] Linting passes with `ruff`
- [ ] All changes committed with conventional commit messages
- [ ] `.ralph/fix_plan.md` task marked as complete
- [ ] CLAUDE.md updated (if new patterns introduced)

## Environment Variables

```bash
# Logging level
export LEROBOT_TO_RLDS_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# Output directory (default: ./output)
export LEROBOT_TO_RLDS_OUTPUT_DIR=/path/to/output

# Parallel workers (override auto-detection)
export LEROBOT_TO_RLDS_WORKERS=4
```

## Common Issues

### Memory Issues
- Reduce workers: `--mode safe` or `--workers 2`
- Process smaller episodes first
- Check available RAM before starting

### Video Decode Errors
- Ensure PyAV is installed: `pip install av`
- Check video file integrity
- Use `--skip-bad-episodes` to continue on errors

### TFDS Import Errors
- TensorFlow must be installed
- Use `tensorflow-cpu` if no GPU needed
- Check TensorFlow/TFDS version compatibility

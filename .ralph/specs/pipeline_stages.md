# Pipeline Stages Specification

## Stage Overview

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐
│ Stage A │ →  │ Stage B │ →  │ Stage C │ →  │ Stage D  │ →  │ Stage E │
│Discover │    │  Spec   │    │ Convert │    │ Validate │    │ Publish │
└─────────┘    └─────────┘    └─────────┘    └──────────┘    └─────────┘
     │              │              │               │              │
     ▼              ▼              ▼               ▼              ▼
inventory.json  mapping.md   progress.jsonl  validation_    final_report.md
episode_index                 RLDS shards    report.md
  .csv
```

---

## Stage A: Discover (탐색/인벤토리)

### Purpose
Scan input dataset to understand structure, detect version, and create inventory.

### Input
- `dataset_root`: Path to LeRobot dataset

### Output Files
```
output/
├── inventory.json
└── episode_index.csv
```

### inventory.json Schema
```json
{
    "dataset_name": "example_dataset",
    "dataset_root": "/path/to/dataset",
    "lerobot_version": "v3.0",
    "detected_at": "2025-01-28T10:30:00Z",
    "episodes_count": 1000,
    "total_steps": 150000,
    "features": {
        "observation.images.top": {
            "dtype": "float32",
            "shape": [3, 480, 640],
            "source": "video"
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [14],
            "source": "parquet"
        },
        "action": {
            "dtype": "float32",
            "shape": [7],
            "source": "parquet"
        }
    },
    "metadata": {
        "robot_type": "rby1a",
        "fps": 30,
        "tasks": ["pick_object", "place_object"]
    }
}
```

### episode_index.csv Schema
```csv
episode_id,episode_index,start_idx,end_idx,length,task,data_path,video_path
ep_000,0,0,150,150,pick_object,data/chunk-000/episode_000.parquet,videos/chunk-000/episode_000.mp4
ep_001,1,150,300,150,pick_object,data/chunk-000/episode_001.parquet,videos/chunk-000/episode_001.mp4
...
```

### Validation Checks
- [x] `sum(length)` == `total_steps`
- [x] `start_idx` values are monotonically increasing
- [x] No gaps between `end_idx[i]` and `start_idx[i+1]`
- [x] All referenced files exist
- [x] `len(episodes)` == `episodes_count`

### Implementation
```python
class DiscoverStage:
    def run(self, dataset_root: Path) -> DiscoverResult:
        # 1. Detect version
        version = detect_lerobot_version(dataset_root)

        # 2. Load metadata
        info = load_info_json(dataset_root / "meta" / "info.json")

        # 3. Build episode index
        episodes = []
        if version == "v3.0":
            episodes = self._index_v30(dataset_root)
        else:
            episodes = self._index_v21(dataset_root)

        # 4. Extract feature specs
        features = self._extract_features(dataset_root, version)

        # 5. Validate
        self._validate_inventory(episodes, info)

        return DiscoverResult(
            inventory=Inventory(...),
            episode_index=episodes
        )
```

---

## Stage B: Spec (스키마 매핑 확정)

### Purpose
Define exact mapping from LeRobot features to RLDS schema.

### Input
- `inventory.json` from Stage A

### Output Files
```
output/
└── mapping.md
```

### mapping.md Content
```markdown
# Feature Mapping: example_dataset

## Source: LeRobot v3.0

## Observation Mapping

| LeRobot Key | Shape | Dtype | → | RLDS Key | Shape | Dtype | Transform |
|-------------|-------|-------|---|----------|-------|-------|-----------|
| observation.images.top | [3,480,640] | float32 | → | observation/image | [480,640,3] | uint8 | CHW→HWC, scale×255 |
| observation.state | [14] | float32 | → | observation/state | [14] | float32 | none |

## Action Mapping

| LeRobot Key | Shape | Dtype | → | RLDS Key | Shape | Dtype | Transform |
|-------------|-------|-------|---|----------|-------|-------|-----------|
| action | [7] | float32 | → | action | [7] | float32 | none |

## Default Values

| RLDS Key | Value | Reason |
|----------|-------|--------|
| reward | 0.0 | Not present in source |
| discount | 1.0 | Standard default |
| is_terminal | False | Assume truncation, not termination |

## Episode Metadata

| RLDS Key | Source |
|----------|--------|
| episode_id | episode_index |
| source_dataset_version | "v3.0" |
| tasks | info.json → tasks |
| language_instruction | tasks[task_index] |
```

### Validation Checks
- [x] All source features have mappings
- [x] Target schema is valid RLDS
- [x] Transformations are reversible (for debugging)
- [x] No ambiguous mappings

---

## Stage C: Convert (변환 실행)

### Purpose
Read LeRobot data, transform, and write to RLDS format.

### Input
- `inventory.json`
- `mapping.md`
- Original dataset files

### Output Files
```
output/
├── progress.jsonl          # Checkpoint file
└── rlds/
    └── example_dataset/
        └── 1.0.0/
            ├── dataset_info.json
            ├── features.json
            └── example_dataset-train.tfrecord-00000-of-00010
            └── example_dataset-train.tfrecord-00001-of-00010
            └── ...
```

### progress.jsonl Schema (one line per episode)
```json
{"episode_id": "ep_000", "status": "completed", "started_at": "...", "completed_at": "...", "steps": 150, "shard": 0}
{"episode_id": "ep_001", "status": "completed", "started_at": "...", "completed_at": "...", "steps": 150, "shard": 0}
{"episode_id": "ep_002", "status": "failed", "started_at": "...", "error": "Video decode error", "shard": null}
```

### Execution Modes

#### SAFE Mode (Default)
```python
mode = ConvertMode.SAFE
workers = 1
# Sequential processing, maximum reliability
```

#### PARALLEL_HALF Mode
```python
mode = ConvertMode.PARALLEL_HALF
workers = max(1, cpu_count() // 2)
memory_per_worker = "2GB"
# Balanced performance and resource usage
```

#### PARALLEL_MAX Mode
```python
mode = ConvertMode.PARALLEL_MAX
workers = max(1, cpu_count() - 1)
memory_limit = available_ram * 0.8
# Maximum throughput, may use most system resources
```

### Episode Processing Flow
```python
def process_episode(episode: EpisodeInfo, mapping: Mapping) -> EpisodeResult:
    try:
        # 1. Read episode data
        data = read_episode(episode)

        # 2. Transform observations
        steps = []
        for step_idx in range(episode.length):
            step = transform_step(data, step_idx, mapping)
            steps.append(step)

        # 3. Add step metadata
        steps[0]["is_first"] = True
        steps[-1]["is_last"] = True

        # 4. Build episode
        rlds_episode = {
            "episode_metadata": build_metadata(episode),
            "steps": steps
        }

        return EpisodeResult(status="completed", data=rlds_episode)

    except Exception as e:
        return EpisodeResult(status="failed", error=str(e))
```

### Resume Capability
```python
def get_pending_episodes(episode_index: list, progress_file: Path) -> list:
    """Get episodes that haven't been processed yet."""
    completed = set()
    if progress_file.exists():
        for line in progress_file.read_text().splitlines():
            record = json.loads(line)
            if record["status"] == "completed":
                completed.add(record["episode_id"])

    return [ep for ep in episode_index if ep.episode_id not in completed]
```

---

## Stage D: Validate (검증)

### Purpose
Verify conversion correctness and data integrity.

### Input
- Original dataset
- Converted RLDS dataset
- `episode_index.csv`

### Output Files
```
output/
├── validation_report.md
└── diff_summary.json
```

### Validation Checks

#### 1. Count Validation
```python
def validate_counts(original: Inventory, converted: RLDSDataset) -> bool:
    assert converted.num_episodes == original.episodes_count
    assert converted.total_steps == original.total_steps
    return True
```

#### 2. Per-Episode Validation
```python
def validate_episode(original_ep: Episode, converted_ep: RLDSEpisode) -> bool:
    assert len(converted_ep.steps) == original_ep.length
    return True
```

#### 3. Schema Validation
```python
def validate_schema(converted: RLDSDataset, expected_features: dict) -> bool:
    for key, spec in expected_features.items():
        assert key in converted.features
        assert converted.features[key].shape == spec.shape
        assert converted.features[key].dtype == spec.dtype
    return True
```

#### 4. Data Integrity Validation
```python
def validate_data_integrity(converted: RLDSDataset) -> ValidationResult:
    issues = []
    for episode in converted:
        for step in episode["steps"]:
            # Check for NaN/Inf
            if np.any(np.isnan(step["observation"]["state"])):
                issues.append(f"NaN in state: episode {episode.id}")
            if np.any(np.isnan(step["action"])):
                issues.append(f"NaN in action: episode {episode.id}")

            # Check image range
            for key in step["observation"]:
                if "image" in key:
                    img = step["observation"][key]
                    if img.min() < 0 or img.max() > 255:
                        issues.append(f"Image out of range: {key}")

    return ValidationResult(passed=len(issues) == 0, issues=issues)
```

#### 5. Sample Comparison (Optional)
```python
def sample_comparison(original: Dataset, converted: RLDSDataset, n_samples: int = 50):
    """Visual/numerical comparison of random samples."""
    samples = random.sample(range(original.episodes_count), n_samples)
    comparisons = []

    for ep_idx in samples:
        orig_ep = original.get_episode(ep_idx)
        conv_ep = converted.get_episode(ep_idx)

        # Compare first, middle, last steps
        for step_idx in [0, len(orig_ep)//2, len(orig_ep)-1]:
            comparison = compare_step(orig_ep[step_idx], conv_ep[step_idx])
            comparisons.append(comparison)

    return comparisons
```

### validation_report.md Content
```markdown
# Validation Report: example_dataset

## Summary
- **Status**: ✅ PASSED
- **Validated at**: 2025-01-28T12:00:00Z
- **Duration**: 5m 32s

## Count Validation
| Metric | Original | Converted | Match |
|--------|----------|-----------|-------|
| Episodes | 1000 | 1000 | ✅ |
| Total Steps | 150000 | 150000 | ✅ |

## Schema Validation
| Feature | Expected Shape | Actual Shape | Match |
|---------|---------------|--------------|-------|
| observation/image | [480,640,3] | [480,640,3] | ✅ |
| observation/state | [14] | [14] | ✅ |
| action | [7] | [7] | ✅ |

## Data Integrity
- NaN values: 0
- Inf values: 0
- Out-of-range images: 0

## Sample Comparison
- Samples validated: 50 episodes × 3 steps
- Visual match: 100%
- Numerical tolerance: 1e-6
```

---

## Stage E: Publish (배포/공유)

### Purpose
Package and prepare dataset for distribution.

### Input
- Converted RLDS dataset
- All reports and logs

### Output Files
```
output/
├── final_report.md
├── rlds/
│   └── example_dataset/
│       └── 1.0.0/
│           └── ... (tfrecord files)
└── logs/
    ├── convert.log
    └── errors.log
```

### final_report.md Content
```markdown
# Conversion Report: example_dataset

## Dataset Information
- **Source**: /path/to/lerobot/dataset
- **Source Version**: LeRobot v3.0
- **Target**: RLDS/TFDS format
- **Converted at**: 2025-01-28T14:00:00Z

## Conversion Statistics
| Metric | Value |
|--------|-------|
| Episodes | 1000 |
| Total Steps | 150,000 |
| Shards | 10 |
| Total Size | 45.2 GB |
| Conversion Time | 2h 15m |
| Mode | PARALLEL_HALF (8 workers) |

## Validation
- ✅ Count validation passed
- ✅ Schema validation passed
- ✅ Data integrity passed
- ✅ Sample comparison passed (50 episodes)

## Output Structure
\`\`\`
rlds/example_dataset/1.0.0/
├── dataset_info.json
├── features.json
└── example_dataset-train.tfrecord-{00000..00009}-of-00010
\`\`\`

## Usage
\`\`\`python
import tensorflow_datasets as tfds

dataset = tfds.load('example_dataset', data_dir='output/rlds')
for episode in dataset['train']:
    print(episode['episode_metadata']['episode_id'])
    for step in episode['steps']:
        print(step['observation']['image'].shape)
\`\`\`

## Checksums
| File | SHA256 |
|------|--------|
| dataset_info.json | abc123... |
| tfrecord-00000 | def456... |
| ... | ... |
```

### Packaging
```python
def package_output(output_dir: Path, config: PublishConfig):
    # 1. Generate checksums
    checksums = generate_checksums(output_dir / "rlds")

    # 2. Collect all reports
    reports = collect_reports(output_dir)

    # 3. Generate final report
    final_report = generate_final_report(
        checksums=checksums,
        reports=reports,
        config=config
    )

    # 4. Write final report
    (output_dir / "final_report.md").write_text(final_report)

    # 5. Optional: Create archive
    if config.create_archive:
        create_tar_gz(output_dir, config.archive_name)
```

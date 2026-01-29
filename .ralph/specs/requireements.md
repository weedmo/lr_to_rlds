# Spec: LeRobot to RLDS Conversion Pipeline

## 1. Goal
- Convert LeRobot (v2.1/v3.0) datasets to RLDS (TFDS) format.
- Ensure 100% data integrity, reproducibility, and high performance via parallel processing.

## 2. Input/Output
- **Input**: LeRobot root (detect v3.0 via `meta/episodes/*.parquet`, else v2.1 via `meta/episodes.jsonl`).
- **Output**: TFDS shards (`rlds/<dataset_name>/<version>/...`).

## 3. Pipeline Stages
- **Stage A (Discover)**: Generate `inventory.json` and `episode_index.csv`.
- **Stage B (Spec)**: Define `mapping.md` for feature mapping (CHW -> HWC, etc.).
- **Stage C (Convert)**: Episode-level transaction. Modes: SAFE, PARALLEL_HALF, PARALLEL_MAX.
- **Stage D (Validate)**: Compare episode count, step count, and tensor shapes.
- **Stage E (Publish)**: Final report and tagging.

## 4. Constraints
- Lossless conversion: No silent drops allowed.
- Memory protection: Minimum 2GB RAM per worker in parallel mode.
- Resume-ability: Use `progress.jsonl` for checkpointing.

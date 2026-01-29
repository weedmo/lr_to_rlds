"""Constants and default values."""

# File names
INFO_JSON = "info.json"
EPISODES_JSONL = "episodes.jsonl"
EPISODES_DIR = "episodes"
META_DIR = "meta"
DATA_DIR = "data"
VIDEOS_DIR = "videos"

# Output file names
INVENTORY_JSON = "inventory.json"
EPISODE_INDEX_CSV = "episode_index.csv"
MAPPING_MD = "mapping.md"
PROGRESS_JSONL = "progress.jsonl"
VALIDATION_REPORT_MD = "validation_report.md"
DIFF_SUMMARY_JSON = "diff_summary.json"
FINAL_REPORT_MD = "final_report.md"

# Logging
LOG_DIR = "logs"
CONVERT_LOG = "convert.log"
ERRORS_LOG = "errors.log"

# RLDS defaults
DEFAULT_REWARD = 0.0
DEFAULT_DISCOUNT = 1.0
DEFAULT_IS_TERMINAL = False

# Memory limits
MIN_MEMORY_PER_WORKER_GB = 2.0
MEMORY_HEADROOM_PERCENT = 0.2

# Shard settings
DEFAULT_SHARD_SIZE_MB = 100
MAX_SHARD_SIZE_MB = 500

# Retry settings
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.0

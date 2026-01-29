# Parallel Processing Specification

## Overview

The converter supports three execution modes to balance safety, performance, and resource usage.

## Execution Modes

### SAFE Mode (Default)

```python
class SafeMode:
    workers: int = 1
    description: str = "Sequential processing, maximum reliability"

    # No parallelism
    # Best for: debugging, small datasets, unreliable systems
```

**Characteristics:**
- Single worker, sequential episode processing
- Maximum observability and debuggability
- No race conditions possible
- Recommended for first-time conversions

### PARALLEL_HALF Mode

```python
class ParallelHalfMode:
    @property
    def workers(self) -> int:
        return max(1, os.cpu_count() // 2)

    memory_per_worker: str = "2GB"
    description: str = "Balanced performance and resource usage"

    # Uses ~50% of CPU cores
    # Best for: production use while allowing other work
```

**Characteristics:**
- Uses half of available CPU cores
- Leaves headroom for system tasks
- Good for running alongside other processes
- Recommended for most production use cases

### PARALLEL_MAX Mode

```python
class ParallelMaxMode:
    @property
    def workers(self) -> int:
        return max(1, os.cpu_count() - 1)

    @property
    def memory_limit(self) -> int:
        return int(psutil.virtual_memory().available * 0.8)

    description: str = "Maximum throughput, high resource usage"

    # Uses ~90% of CPU cores
    # Best for: dedicated conversion machines
```

**Characteristics:**
- Uses maximum available CPU cores (minus one for system)
- May make system less responsive
- Highest throughput possible
- Recommended only for dedicated conversion jobs

## Worker Pool Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Main Process                                   │
│                                                                          │
│  ┌──────────────────┐                    ┌──────────────────┐           │
│  │   Episode Queue  │                    │   Result Queue   │           │
│  │  ┌─────────────┐ │                    │  ┌─────────────┐ │           │
│  │  │ ep_000      │ │                    │  │ ep_000: OK  │ │           │
│  │  │ ep_001      │ │                    │  │ ep_001: OK  │ │           │
│  │  │ ep_002      │ │                    │  │ ep_003: FAIL│ │           │
│  │  │ ...         │ │                    │  │ ...         │ │           │
│  │  └─────────────┘ │                    │  └─────────────┘ │           │
│  └────────┬─────────┘                    └────────▲─────────┘           │
│           │                                       │                      │
│           │         ┌─────────────────┐          │                      │
│           └────────►│  Worker Pool    │──────────┘                      │
│                     │                 │                                  │
│                     │  ┌───┐ ┌───┐   │                                  │
│                     │  │W0 │ │W1 │   │                                  │
│                     │  └───┘ └───┘   │                                  │
│                     │  ┌───┐ ┌───┐   │                                  │
│                     │  │W2 │ │W3 │   │                                  │
│                     │  └───┘ └───┘   │                                  │
│                     └─────────────────┘                                  │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      progress.jsonl                               │   │
│  │  {"episode_id": "ep_000", "status": "completed", ...}            │   │
│  │  {"episode_id": "ep_001", "status": "completed", ...}            │   │
│  │  {"episode_id": "ep_003", "status": "failed", "error": "..."}    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Implementation

### Worker Process

```python
import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional

@dataclass
class WorkerTask:
    episode_id: str
    episode_index: int
    data_path: Path
    video_path: Optional[Path]

@dataclass
class WorkerResult:
    episode_id: str
    status: str  # "completed" | "failed"
    steps: int
    error: Optional[str] = None
    duration_ms: int = 0

def worker_process(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    mapping: dict,
    output_dir: Path,
    worker_id: int
):
    """Worker process that converts episodes."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ignore Ctrl+C in workers

    while True:
        try:
            task = task_queue.get(timeout=1.0)
            if task is None:  # Poison pill
                break

            start_time = time.time()

            try:
                # Process episode
                result = convert_episode(task, mapping, output_dir)
                duration_ms = int((time.time() - start_time) * 1000)

                result_queue.put(WorkerResult(
                    episode_id=task.episode_id,
                    status="completed",
                    steps=result.steps,
                    duration_ms=duration_ms
                ))

            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                result_queue.put(WorkerResult(
                    episode_id=task.episode_id,
                    status="failed",
                    steps=0,
                    error=str(e),
                    duration_ms=duration_ms
                ))

        except queue.Empty:
            continue
```

### Worker Pool Manager

```python
class WorkerPool:
    def __init__(self, mode: ConvertMode, mapping: dict, output_dir: Path):
        self.mode = mode
        self.mapping = mapping
        self.output_dir = output_dir
        self.workers: list[mp.Process] = []
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()

    def start(self):
        """Start worker processes."""
        num_workers = self._get_num_workers()

        for i in range(num_workers):
            p = mp.Process(
                target=worker_process,
                args=(
                    self.task_queue,
                    self.result_queue,
                    self.mapping,
                    self.output_dir,
                    i
                )
            )
            p.start()
            self.workers.append(p)

        logger.info(f"Started {num_workers} workers")

    def _get_num_workers(self) -> int:
        """Calculate number of workers based on mode."""
        cpu = os.cpu_count() or 4

        if self.mode == ConvertMode.SAFE:
            return 1
        elif self.mode == ConvertMode.PARALLEL_HALF:
            return max(1, cpu // 2)
        elif self.mode == ConvertMode.PARALLEL_MAX:
            return max(1, cpu - 1)
        else:
            return 1

    def submit(self, task: WorkerTask):
        """Submit a task to the pool."""
        self.task_queue.put(task)

    def get_result(self, timeout: float = None) -> Optional[WorkerResult]:
        """Get a result from completed workers."""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def shutdown(self):
        """Gracefully shutdown all workers."""
        # Send poison pills
        for _ in self.workers:
            self.task_queue.put(None)

        # Wait for workers to finish
        for p in self.workers:
            p.join(timeout=30)
            if p.is_alive():
                logger.warning(f"Worker {p.pid} did not terminate, killing")
                p.terminate()

        self.workers.clear()
```

## Memory Protection

### Per-Worker Memory Estimation

```python
def estimate_episode_memory(episode: EpisodeInfo) -> int:
    """Estimate memory required to process an episode."""
    # Base overhead
    memory = 100 * 1024 * 1024  # 100MB base

    # Image memory (assume all frames in memory worst case)
    for camera, shape in episode.cameras.items():
        h, w, c = shape
        bytes_per_frame = h * w * c * 4  # float32
        memory += bytes_per_frame * episode.length

    # State/action memory
    state_bytes = episode.state_dim * 4 * episode.length
    action_bytes = episode.action_dim * 4 * episode.length
    memory += state_bytes + action_bytes

    return memory

def calculate_safe_workers(mode: ConvertMode, max_episode_memory: int) -> int:
    """Calculate number of workers that fit in available memory."""
    import psutil

    available = psutil.virtual_memory().available

    if mode == ConvertMode.SAFE:
        return 1

    # Leave 20% headroom
    usable = int(available * 0.8)

    # Minimum 2GB per worker
    min_per_worker = 2 * 1024 * 1024 * 1024

    memory_per_worker = max(min_per_worker, max_episode_memory)
    max_workers_by_memory = usable // memory_per_worker

    # Also consider CPU
    cpu_count = os.cpu_count() or 4
    if mode == ConvertMode.PARALLEL_HALF:
        max_workers_by_cpu = cpu_count // 2
    else:
        max_workers_by_cpu = cpu_count - 1

    return max(1, min(max_workers_by_memory, max_workers_by_cpu))
```

### Memory Monitoring

```python
import psutil

class MemoryMonitor:
    def __init__(self, threshold_percent: float = 90.0):
        self.threshold_percent = threshold_percent
        self.warning_issued = False

    def check(self) -> bool:
        """Check if memory usage is acceptable."""
        mem = psutil.virtual_memory()
        usage_percent = mem.percent

        if usage_percent > self.threshold_percent:
            if not self.warning_issued:
                logger.warning(
                    f"Memory usage at {usage_percent:.1f}%, "
                    f"threshold is {self.threshold_percent:.1f}%"
                )
                self.warning_issued = True
            return False

        self.warning_issued = False
        return True

    def get_stats(self) -> dict:
        """Get current memory statistics."""
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
            "used_percent": mem.percent
        }
```

## Checkpointing

### Progress File Format

```python
@dataclass
class ProgressRecord:
    episode_id: str
    status: str  # "completed" | "failed" | "in_progress"
    started_at: str  # ISO timestamp
    completed_at: Optional[str]
    steps: Optional[int]
    shard: Optional[int]
    error: Optional[str]
    worker_id: Optional[int]

def write_progress(record: ProgressRecord, progress_file: Path):
    """Append progress record to checkpoint file."""
    with open(progress_file, "a") as f:
        f.write(json.dumps(asdict(record)) + "\n")
        f.flush()
        os.fsync(f.fileno())  # Ensure write to disk
```

### Resume Logic

```python
def load_progress(progress_file: Path) -> dict[str, ProgressRecord]:
    """Load progress from checkpoint file."""
    progress = {}

    if not progress_file.exists():
        return progress

    for line in progress_file.read_text().splitlines():
        if not line.strip():
            continue
        record = ProgressRecord(**json.loads(line))
        progress[record.episode_id] = record

    return progress

def get_pending_episodes(
    episode_index: list[EpisodeInfo],
    progress: dict[str, ProgressRecord],
    retry_failed: bool = False
) -> list[EpisodeInfo]:
    """Get episodes that need processing."""
    pending = []

    for episode in episode_index:
        if episode.episode_id not in progress:
            pending.append(episode)
        elif retry_failed and progress[episode.episode_id].status == "failed":
            pending.append(episode)

    return pending
```

## Error Handling

### Failure Modes

| Failure | Recovery | Action |
|---------|----------|--------|
| Worker crash | Automatic | Restart worker, retry episode |
| OOM | Reduce workers | Log warning, reduce worker count |
| Video decode error | Skip episode | Log error, mark episode failed |
| Network timeout | Retry | Exponential backoff, max 3 retries |
| Disk full | Abort | Stop all workers, raise error |

### Graceful Degradation

```python
class AdaptiveWorkerPool(WorkerPool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oom_count = 0
        self.min_workers = 1

    def handle_worker_crash(self, worker_id: int, episode_id: str):
        """Handle a worker crash."""
        logger.error(f"Worker {worker_id} crashed processing {episode_id}")

        # Check if it was OOM
        if self._check_oom():
            self.oom_count += 1
            if self.oom_count >= 3:
                # Reduce worker count
                new_count = max(self.min_workers, len(self.workers) - 1)
                logger.warning(f"Multiple OOMs, reducing workers to {new_count}")
                self._resize_pool(new_count)

        # Requeue the failed episode
        self._requeue_episode(episode_id)

        # Restart a worker
        self._start_worker(worker_id)
```

## Performance Metrics

```python
@dataclass
class ConversionMetrics:
    total_episodes: int
    completed_episodes: int
    failed_episodes: int
    total_steps: int
    total_duration_s: float
    episodes_per_second: float
    steps_per_second: float
    peak_memory_gb: float
    avg_worker_utilization: float

def calculate_metrics(
    progress: dict[str, ProgressRecord],
    start_time: float,
    memory_samples: list[float]
) -> ConversionMetrics:
    """Calculate conversion performance metrics."""
    completed = [p for p in progress.values() if p.status == "completed"]
    failed = [p for p in progress.values() if p.status == "failed"]

    total_steps = sum(p.steps or 0 for p in completed)
    duration = time.time() - start_time

    return ConversionMetrics(
        total_episodes=len(progress),
        completed_episodes=len(completed),
        failed_episodes=len(failed),
        total_steps=total_steps,
        total_duration_s=duration,
        episodes_per_second=len(completed) / duration if duration > 0 else 0,
        steps_per_second=total_steps / duration if duration > 0 else 0,
        peak_memory_gb=max(memory_samples) if memory_samples else 0,
        avg_worker_utilization=0.0  # TODO: implement
    )
```

# Ralph Fix Plan: LeRobot → RLDS Converter

## Current Focus
**Loop 2**: OXE/OpenVLA 호환 RLDS 출력 (rlds 서브모듈 활용)

---

## 🔥 URGENT: OXE/OpenVLA 호환성 수정

### 문제점
현재 `writers/rlds_writer.py`는 custom TFRecord 직렬화를 사용하여:
- `tfds.load()`로 로드 불가
- OpenVLA/OXE 파이프라인과 비호환
- Steps가 flat array로 저장됨 (nested dataset 아님)

### 해결책: rlds 서브모듈 활용

rlds 서브모듈 (`/rlds/`)이 제공하는 공식 API 사용:
- `rlds.tfds.EpisodeWriter` - TFDS 호환 에피소드 작성
- `rlds.build_step()` / `rlds.build_episode()` - 표준 빌더
- `tfds.rlds.rlds_base.DatasetConfig` - 데이터셋 설정

### Phase 5.1: OXE 호환 Writer 구현 ✅

- [x] `writers/oxe_writer.py` 생성 - rlds.tfds.EpisodeWriter 활용
- [x] `writers/feature_mapper.py` 생성 - LeRobot→OXE 피처 매핑
- [x] DatasetConfig 생성 로직 구현
- [x] 기존 `rlds_writer.py` 유지 (legacy로 사용)
- [x] `pipeline/convert.py` 수정 - OXE writer 사용
- [x] CLI에 `--format oxe|legacy` 옵션 추가
- [ ] OpenVLA로 로드 테스트

### OXE Writer 핵심 구현

```python
# writers/oxe_writer.py
from rlds import rlds_types
from rlds.tfds import episode_writer
import tensorflow_datasets as tfds

class OXERLDSWriter:
    def __init__(self, output_dir, dataset_name, feature_config):
        self.config = tfds.rlds.rlds_base.DatasetConfig(
            name=dataset_name,
            observation_info=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=(H, W, 3)),
                'state': tfds.features.Tensor(shape=(N,), dtype=tf.float32),
            }),
            action_info=tfds.features.Tensor(shape=(A,), dtype=tf.float32),
            reward_info=tfds.features.Scalar(dtype=tf.float32),
            discount_info=tfds.features.Scalar(dtype=tf.float32),
            step_metadata_info={
                'language_instruction': tfds.features.Text(),
            },
        )
        self.writer = episode_writer.EpisodeWriter(
            data_directory=str(output_dir),
            ds_config=self.config,
        )

    def write_episode(self, episode: Episode):
        steps = [
            rlds_types.build_step(
                observation={'image': step.observation['image_*'], 'state': ...},
                action=step.action,
                reward=step.reward,
                discount=1.0,
                is_terminal=step.is_terminal,
                is_first=step.is_first,
                is_last=step.is_last,
                metadata={'language_instruction': step.language_instruction},
            )
            for step in episode.steps
        ]
        rlds_episode = rlds_types.build_episode(steps, metadata={...})
        self.writer.add_episode(rlds_episode)

    def close(self):
        self.writer.close()
```

### OpenVLA 호환 스키마

```python
# OpenVLA가 기대하는 RLDS 스키마
{
    'steps': tfds.features.Dataset({
        'observation': {
            'image': Image(256, 256, 3),       # 메인 카메라
            'wrist_image': Image(256, 256, 3), # 손목 카메라 (optional)
            'state': Tensor(shape=(N,)),       # proprioception
        },
        'action': Tensor(shape=(7,)),          # 7-DoF 액션
        'reward': Scalar(float32),
        'discount': Scalar(float32),
        'is_first': bool,
        'is_last': bool,
        'is_terminal': bool,
        'language_instruction': Text(),
    }),
}
```

### 검증 방법

```bash
# 1. tfds.load() 테스트
python -c "
import tensorflow_datasets as tfds
ds = tfds.load('dataset_name', data_dir='output/', split='train')
for ep in ds.take(1):
    print(ep.keys())
    for step in ep['steps'].take(1):
        print(step.keys())
"

# 2. OpenVLA 데이터 로더 테스트
# (OpenVLA fine-tuning 스크립트로 로드 확인)
```

---

## High Priority (Loop 1-2)

### Phase 1: Project Setup ✅
- [x] Create Python package structure (`src/lerobot_to_rlds/`)
- [x] Setup `pyproject.toml` with dependencies
- [x] Create core types and exceptions (`core/types.py`, `core/exceptions.py`)
- [x] Implement structured logging (`utils/logging.py`)

### Phase 2: LeRobot Readers ✅
- [x] Implement version detector (`readers/detector.py`)
- [x] Write detector unit tests
- [x] Implement base reader abstract class (`readers/base.py`)
- [x] Implement v2.1 reader (`readers/v21_reader.py`)
- [x] Implement v3.0 reader (`readers/v30_reader.py`)
- [x] Write reader unit tests (42 tests, all passing)

### Phase 3-4: Discover/Spec Stages (SKIP for now)
> Focus on conversion first, add discovery later

### Phase 5: Stage C - Convert ✅ (Legacy) → 🔄 (OXE 호환 수정 중)
- [x] Implement image transformation (CHW→HWC) - handled in V21Reader
- [x] Implement step builder with is_first/is_last
- [x] ~~Implement RLDS writer~~ (legacy - custom serialization)
- [x] Implement progress checkpointing (`progress.jsonl`)
- [ ] **Implement OXE-compatible writer using rlds submodule**
- [ ] Write convert stage tests

### Phase 6-8: Validate/Publish/CLI
- [x] CLI skeleton working
- [ ] Add `--format` option for oxe/legacy
- [ ] Implement validation with tfds.load() check

---

## 파일 변경 계획

| 파일 | 작업 |
|------|------|
| `writers/oxe_writer.py` | **신규** - rlds.tfds.EpisodeWriter 활용 |
| `writers/feature_mapper.py` | **신규** - 피처 매핑 |
| `writers/rlds_writer.py` | → `legacy_writer.py` rename |
| `writers/__init__.py` | 수정 - exports 추가 |
| `pipeline/convert.py` | 수정 - OXE writer 사용 |
| `cli.py` | 수정 - `--format` 옵션 |
| `pyproject.toml` | rlds 서브모듈 경로 추가 |

---

## rlds 서브모듈 통합

```bash
# 서브모듈 위치
rlds/                          # git submodule
├── rlds/
│   ├── rlds_types.py         # build_step(), build_episode()
│   └── tfds/
│       ├── episode_writer.py # EpisodeWriter 클래스
│       └── config_generator.py
```

### pyproject.toml 수정

```toml
[tool.setuptools.packages.find]
where = ["src", "rlds"]  # rlds 서브모듈 포함

# 또는 rlds를 editable로 설치
# pip install -e ./rlds
```

---

## Medium Priority (Loop 3-4)

### Phase 9: Parallel Processing
- [ ] Implement worker process
- [ ] Implement PARALLEL_HALF mode
- [ ] Add memory monitoring

### Phase 10: Resume Capability
- [ ] Implement progress loading
- [ ] Implement `--resume` flag

---

## Completed
- [x] Project initialization
- [x] **Phase 1: Project Setup**
- [x] **Phase 2: LeRobot Readers** (42 tests passing)
- [x] CLI skeleton with commands
- [x] Legacy RLDS writer (custom serialization - 비호환)

---

## Notes

### 핵심 참조
- [OpenVLA GitHub](https://github.com/openvla/openvla) - 타겟 플랫폼
- [google-research/rlds](https://github.com/google-research/rlds) - 서브모듈 소스
- [kpertsch/rlds_dataset_builder](https://github.com/kpertsch/rlds_dataset_builder) - OXE 예제

### Test Dataset
- LeRobot: `/home/tommoro/data_collection/habilis_dataset_manager/data/curation/habilis_beta_v4`

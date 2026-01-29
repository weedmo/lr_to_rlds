# Ralph Loop 실행 가이드

## 개요

Ralph Loop는 Claude가 자율적으로 개발 작업을 수행하는 반복 실행 패턴입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                      Ralph Loop Cycle                            │
│                                                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│   │  Study   │ →  │   Plan   │ →  │Implement │ →  │  Test    │ │
│   │  specs/  │    │fix_plan  │    │  code    │    │& verify  │ │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│        ↑                                               │        │
│        └───────────────────────────────────────────────┘        │
│                         (repeat)                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 실행 방법

### 방법 1: 단일 프롬프트 실행

터미널에서 Claude Code를 실행하고 다음 프롬프트를 입력:

```
ralph
```

또는 더 명시적으로:

```
Ralph Loop를 실행해줘. .ralph/fix_plan.md에서 가장 높은 우선순위 작업을 구현해.
```

### 방법 2: 연속 실행 (권장)

```bash
# Claude Code 실행
cd /home/weed/tommoro/lerobot_to_rlds
claude

# 프롬프트 입력
> ralph loop 시작
```

Claude가 작업을 완료하면 상태 리포트와 함께 종료됩니다.
다음 loop를 실행하려면 다시 `ralph` 입력.

### 방법 3: 자동 연속 실행 (headless)

```bash
# 단일 loop 실행
claude -p "ralph loop 실행. fix_plan.md의 최우선 작업 1개를 구현하고 테스트해."

# 여러 loop 연속 (주의: 비용 발생)
for i in {1..5}; do
  echo "=== Ralph Loop $i ==="
  claude -p "ralph loop 실행"
  sleep 2
done
```

---

## Ralph 프롬프트 예시

### 기본 실행
```
ralph
```

### 특정 Phase 지정
```
ralph loop를 실행해. Phase 2의 v21_reader.py를 구현해줘.
```

### 테스트 중심
```
ralph loop 실행. 구현보다 테스트 작성에 집중해줘.
```

### 디버깅/수정
```
ralph loop 실행. 이전에 실패한 테스트를 수정해줘.
```

---

## Loop 종료 조건

Ralph는 각 loop 끝에 상태를 보고합니다:

```
---RALPH_STATUS---
STATUS: IN_PROGRESS | COMPLETE | BLOCKED
TASKS_COMPLETED_THIS_LOOP: 3
FILES_MODIFIED: 5
TESTS_STATUS: PASSING | FAILING | NOT_RUN
WORK_TYPE: IMPLEMENTATION | TESTING | DOCUMENTATION
EXIT_SIGNAL: false | true
RECOMMENDATION: 다음 loop에서 Phase 3 진행 권장
---
```

- `STATUS: COMPLETE` - 모든 작업 완료
- `STATUS: BLOCKED` - 사용자 입력 필요
- `EXIT_SIGNAL: true` - loop 중단 권장

---

## 현재 프로젝트 상태 확인

```bash
# fix_plan.md 확인
cat .ralph/fix_plan.md

# 현재 진행 상황
grep -E "^\s*-\s*\[x\]" .ralph/fix_plan.md | wc -l  # 완료된 작업
grep -E "^\s*-\s*\[ \]" .ralph/fix_plan.md | wc -l  # 남은 작업

# 테스트 실행
python3 -m pytest tests/ -v
```

---

## 파일 역할

| 파일 | 역할 | Ralph 동작 |
|------|------|-----------|
| `.ralph/PROMPT.md` | Loop 실행 지침 | 매 loop 시작 시 읽음 |
| `.ralph/fix_plan.md` | 작업 목록/우선순위 | 작업 선택, 완료 시 업데이트 |
| `.ralph/AGENT.md` | 빌드/테스트 명령어 | 테스트 실행 시 참조 |
| `.ralph/specs/*.md` | 상세 명세 | 구현 전 참조 |
| `CLAUDE.md` | 프로젝트 개요 | 컨텍스트 이해 |

---

## Loop 전략 (이 프로젝트)

### Loop 1-2: 기본 인프라 (SAFE 모드)
```
목표: 작은 데이터셋 1개 변환 성공
작업: Phase 1-8 (readers, pipeline stages, CLI)
검증: pytest 통과, TFDS 로드 성공
```

### Loop 3-4: 병렬 처리
```
목표: PARALLEL_HALF 모드 동작
작업: Phase 9-11 (workers, resume, robustness)
검증: SAFE 모드와 동일 출력, 속도 향상
```

### Loop 5+: 최적화/문서화
```
목표: Production-ready
작업: Phase 12-14
검증: 대규모 데이터셋 변환 성공
```

---

## 주의사항

1. **한 번에 하나의 작업**: Ralph는 각 loop에서 하나의 주요 작업에 집중
2. **테스트 우선**: 구현 후 반드시 테스트 실행
3. **Checkpoint**: 작업 완료 시 fix_plan.md 업데이트
4. **Git 커밋**: 중요 변경 후 커밋 (Ralph에게 요청 가능)

---

## 트러블슈팅

### Ralph가 잘못된 파일을 수정할 때
```
잠깐, .ralph/fix_plan.md를 다시 읽고 현재 우선순위를 확인해줘.
```

### 테스트 실패 시
```
테스트 실패 원인을 분석하고 수정해줘. 새 기능 추가하지 말고.
```

### Loop가 너무 오래 걸릴 때
```
현재 작업을 중단하고 상태를 보고해줘.
```

---

## 빠른 시작

```bash
cd /home/weed/tommoro/lerobot_to_rlds
claude

# Claude Code 프롬프트에서:
> ralph
```

이것만으로 Ralph가 자동으로:
1. `.ralph/specs/` 읽기
2. `.ralph/fix_plan.md`에서 최우선 작업 선택
3. 구현
4. 테스트
5. 문서 업데이트
6. 상태 보고

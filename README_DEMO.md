# 교통 신호등 제어 시연 가이드

학습된 모델을 사용하여 교통 상황 시뮬레이션을 ASCII 아트로 시각화하는 시연 스크립트입니다.

## 📋 사전 준비

### 1. 시연용 가상 데이터 생성

모든 시나리오의 데이터를 한 번에 생성:
```bash
python generate_demo_data.py
```

특정 시나리오만 생성:
```bash
python generate_demo_data.py --scenario normal --steps 500 --seed 42
```

**옵션:**
- `--scenario`: 시나리오 선택 (normal, morning_rush, evening_rush, congestion, night)
- `--steps`: 생성할 스텝 수 (기본: 500)
- `--seed`: 랜덤 시드 (기본: 42)

생성된 데이터는 `./demo_data/` 디렉토리에 저장됩니다.

## 🎬 시연 실행

### 기본 사용법

**단일 시나리오 시연:**
```bash
python traffic_demo.py --scenario normal
```

**고정 신호와 성능 비교 (나란히 비교):**
```bash
# 강화학습 모델과 고정 신호를 나란히 비교
python traffic_demo.py --scenario normal --compare

# 고정 신호 주기 변경 (기본: 30초)
python traffic_demo.py --scenario normal --compare --baseline-cycle 20
```

**모든 시나리오 자동 테스트:**
```bash
# 시각화 없이 빠른 테스트 (권장)
python traffic_demo.py --scenario all --no-visualize

# 시각화와 함께 모든 시나리오 테스트
python traffic_demo.py --scenario all
```

### 옵션 설명

```bash
```

**주요 옵션:**
- `--scenario`: 시나리오 선택 (필수)
  - `normal`: 균등한 교통량
  - `morning_rush`: 북/남 집중 (출근 시간)
  - `evening_rush`: 동/서 집중 (퇴근 시간)
  - `congestion`: 극심한 혼잡
  - `night`: 한산한 시간

- `--agent-type`: 알고리즘 선택
  - `dqn`: DQN 모델 사용
  - `ddqn`: Double DQN 모델 사용 (기본: dqn)

- `--speed`: 시뮬레이션 속도
  - 값이 작을수록 빠름 (예: 0.1 = 빠름, 1.0 = 보통, 2.0 = 느림)
  - 기본: 1.0초 (보기 좋은 속도)

- `--steps`: 최대 스텝 수
  - 지정하지 않으면 저장된 데이터 전체 실행

- `--no-visualize`: 시각화 없이 빠른 테스트
  - 모든 시나리오 테스트 시 유용
  - 결과만 요약해서 출력

- `--compare`: 고정 신호와 성능 비교 모드
  - 강화학습 모델과 고정 신호를 나란히 비교
  - 실시간으로 성능 차이 확인 가능

- `--baseline-cycle`: 고정 신호 주기 (초)
  - 기본값: 30초
  - 비교 모드에서만 사용

- `--model`: 모델 파일 경로
  - 지정하지 않으면 자동으로 찾음:
    - `./models/optimized/{agent_type}_{scenario}/agent_{scenario}_optimized.pt`

- `--scenario all`: 모든 시나리오 자동 테스트
  - normal, morning_rush, evening_rush, congestion, night 순차 실행
  - 결과 요약 테이블 자동 출력

## 📊 시연 화면 설명

시연 중 화면에는 다음 정보가 표시됩니다:

**일반 모드:**
```
======================================================================
  교통 신호등 제어 시뮬레이션 - Step   50 | 신호 지속:  15초
======================================================================

        [G] 북쪽: OOOOOO........ ( 6대)

        │
        │
서쪽: OOO........... ( 3대) ────┼──── 동쪽: OOOOOOOOOO..... (10대)
        [R]
        │
        │
        [G] 남쪽: OO............. ( 2대)

----------------------------------------------------------------------
  총 대기 차량:  21대  |  북/남:  8대  |  동/서: 13대
  선택된 행동: 신호 유지
  Q-value: 유지= -25.32  변경= -30.15
======================================================================
```

**비교 모드 (`--compare` 옵션):**
```
================================================================================================================================================
  교통 신호등 제어 성능 비교 - Step   50
================================================================================================================================================

강화학습 모델 (RL)                                                          │ 고정 신호 (Baseline)
------------------------------------------------------------------------------------------------------------------------------------------------
[G] 북: OOOOOO.. ( 6대)                                                    │ [R] 북: OOOOOOOO ( 8대)

서: OOO....... ( 3대) ──┼── 동: OOOOOOOO.. (10대)                        │ 서: OOOO...... ( 4대) ──┼── 동: OOOOOOOO.. (10대)
                                                                           │ 
[R]                                                                        │ [G]
                                                                           │ 
[G] 남: OO...... ( 2대)                                                    │ [R] 남: OOOO.... ( 4대)

------------------------------------------------------------------------------------------------------------------------------------------------
총 대기 차량:   21대                                                       │ 총 대기 차량:   26대
개선율:     19.2% 감소                                                     │ 
행동: 신호 유지                                                             │ 행동: 고정 주기
Q-value: 유지= -25.32 변경= -30.15                                         │ 
신호 지속:   15초                                                           │ 신호 지속:   25초
================================================================================================================================================
```

**표시 내용:**
- `[G]`: 초록불 (Green)
- `[R]`: 빨간불 (Red)
- `[Y]`: 노란불 (Yellow)
- `O`: 대기 중인 차량
- `.`: 빈 공간
- 숫자: 각 방향의 대기 차량 수

**통계 정보:**
- 총 대기 차량 수
- 방향별 대기 차량 수
- 선택된 행동 (신호 유지/변경)
- Q-value (각 행동의 예상 가치)

## 🎯 시연 예시

### 1. 모든 시나리오 빠른 테스트 (권장)
```bash
# 시각화 없이 빠르게 모든 시나리오 테스트
python traffic_demo.py --scenario all --no-visualize --agent-type dqn

# Double DQN으로 모든 시나리오 테스트
python traffic_demo.py --scenario all --no-visualize --agent-type ddqn
```

### 2. 고정 신호와 성능 비교 (권장)
```bash
# 평시 교통량에서 고정 신호와 비교
python traffic_demo.py --scenario normal --compare --agent-type dqn

# 출근 시간 혼잡 상황 비교
python traffic_demo.py --scenario morning_rush --compare --agent-type ddqn --speed 0.3

# 고정 신호 주기 변경하여 비교
python traffic_demo.py --scenario normal --compare --baseline-cycle 20
```

### 3. 단일 시나리오 시각화 시연
```bash
# 평시 교통량 시나리오
python traffic_demo.py --scenario normal --agent-type dqn --speed 0.3

# 출근 시간 혼잡 상황
python traffic_demo.py --scenario morning_rush --agent-type ddqn --speed 0.5

# 극심한 혼잡 상황 (빠른 속도)
python traffic_demo.py --scenario congestion --agent-type ddqn --speed 0.2 --steps 300
```

### 4. 모든 시나리오 시각화 시연
```bash
# 모든 시나리오를 순차적으로 시각화하며 테스트
python traffic_demo.py --scenario all --agent-type dqn --speed 0.3
```

## 💡 팁

1. **모든 시나리오 테스트**: `--scenario all --no-visualize`로 빠르게 모든 시나리오를 테스트하고 결과를 비교할 수 있습니다.

2. **시연 속도 조절**: `--speed` 값을 조절하여 시연 속도를 변경할 수 있습니다.
   - 빠른 시연: `--speed 0.3`
   - 보통 속도: `--speed 1.0` (기본값)
   - 느린 시연: `--speed 2.0`

3. **중단**: 시연 중 `Ctrl+C`로 언제든지 중단할 수 있습니다.

4. **데이터 재생성**: 다른 패턴을 보고 싶다면 `--seed` 값을 변경하여 데이터를 재생성하세요.

5. **모델 비교**: 같은 시나리오에서 `--agent-type`을 변경하여 DQN과 Double DQN의 차이를 비교할 수 있습니다.

6. **빠른 테스트**: 시각화 없이 빠르게 테스트하려면 `--no-visualize` 옵션을 사용하세요.

## 📁 파일 구조

```
.
├── generate_demo_data.py      # 가상 데이터 생성 스크립트
├── traffic_demo.py             # 시연 스크립트
├── demo_data/                  # 생성된 가상 데이터
│   ├── normal_traffic_data.json
│   ├── morning_rush_traffic_data.json
│   └── ...
└── models/                     # 학습된 모델
    └── optimized/
        ├── dqn_normal/
        ├── ddqn_normal/
        └── ...
```

## ⚠️ 주의사항

1. 시연을 실행하기 전에 반드시 `generate_demo_data.py`를 실행하여 가상 데이터를 생성해야 합니다.

2. 모델 파일이 존재하는지 확인하세요. 모델이 없으면 먼저 학습을 실행해야 합니다.

3. 터미널 창 크기를 충분히 크게 설정하는 것을 권장합니다 (최소 80x24).

4. Windows에서는 이모지가 제대로 표시되지 않을 수 있으므로 ASCII 문자를 사용합니다.


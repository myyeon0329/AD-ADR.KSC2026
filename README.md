# AD-ADR: Adaptive Dynamics-aware Domain Randomization

ANYmal-C 험지(rough terrain) 보행 학습에서 마찰 계수의 도메인 랜덤화를 동역학 예측 오차로 조절하는 AD-ADR의 구현 코드입니다.
NVIDIA Isaac Lab(RSL-RL, PPO)의 ANYmal-C rough 환경 위에서 UDR(Baseline), AD-ADR, ADR(Mehta et al.)을 비교합니다.

## 파일 구성

Isaac Lab 저장소에 아래 위치로 복사합니다.

| 파일 | 복사 위치 | 설명 |
|---|---|---|
| `source/jetbot_dr/__init__.py` | `source/isaaclab_tasks/isaaclab_tasks/manager_based/jetbot_dr/` | 태스크 등록 |
| `source/jetbot_dr/jetbot_env_cfg2.py` | 위와 동일 | UDR(Baseline) 환경 |
| `source/jetbot_dr/jetbot_env_cfg1.py` | 위와 동일 | **AD-ADR 환경** (동역학 예측기 + 마찰 적응 샘플링) |
| `source/jetbot_dr/jetbot_env_cfg1_v3_backup.py` | 위와 동일 | 논문 실험에 쓴 AD-ADR 샘플링 함수의 백업본 |
| `source/jetbot_dr/jetbot_env_cfg_adr_full.py` | 위와 동일 | ADR 환경 (판별기 + SVPG) |
| `source/jetbot_dr/adr_filtered_wrapper.py` | 위와 동일 | ADR 래퍼 |
| `scripts/train.py`, `scripts/train_adr.py` | `scripts/reinforcement_learning/rsl_rl/` | 학습 (UDR·AD-ADR / ADR) |
| `scripts/eval_ood_friction.py` | 위와 동일 | 마찰 조건별 평가 |

태스크 이름의 `Jetbot`은 초기 개발 때 쓰던 이름이며, 실제 로봇은 ANYmal-C입니다.

| 조건 | Task ID | 환경 수 |
|---|---|---|
| UDR (Baseline) | `Isaac-Jetbot-Baseline-v0` | 4,096 |
| AD-ADR | `Isaac-Jetbot-ADADR-v0` | 4,096 |
| ADR | `Isaac-Jetbot-ADRFull-v0` | 8,192 (학습용 4,096 + 기준 4,096) |

## 환경

- Isaac Lab: `<TODO: 버전 또는 커밋>`, Python 3.11, GPU: RTX 5070
- 학습: Isaac Lab 기본 `AnymalCRoughPPORunnerCfg`(RSL-RL PPO) 그대로, 1,500 iteration
- 환경 설정: Isaac Lab `LocomotionVelocityRoughEnvCfg` + `ANYMAL_C_CFG` 기반

## 학습

PowerShell, Isaac Lab 폴더에서 실행합니다. 시드는 `--seed`로 지정합니다.

```powershell
# UDR
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Jetbot-Baseline-v0 --headless --seed 1

# AD-ADR  (논문 설정: sigmoid 모드)
$env:ADADR_ALPHA_MODE = "sigmoid"
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Jetbot-ADADR-v0 --headless --seed 1

# ADR
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train_adr.py --task Isaac-Jetbot-ADRFull-v0 --headless --num_envs 8192 --seed 1
```

`ADADR_ALPHA_MODE`는 통제 실험용 환경 변수입니다(`uniform`, `rank`, `shuffle`, 그 외 값은 논문의 sigmoid 방식).
논문의 AD-ADR 결과는 `sigmoid` 방식입니다.

## 평가

`<TODO: eval_ood_friction.py 실행 명령 — 인자는 `--help` 출력 참고>`

평가 조건: 마찰 정적 μ 고정(동적 0.9μ), 마찰 랜덤화·외력·관측 잡음 비활성, 900개 환경 × 5 에피소드, 지형 레벨 균등 배정.
성공률(SR)은 base 접촉 없이 time-out으로 종료한 에피소드의 비율입니다.

## AD-ADR 핵심 (`jetbot_env_cfg1.py`)

1. 환경별 동역학 예측기가 다음 상태를 예측하고, 예측 오차(MSE)의 지수이동평균(0.9/0.1)으로 불확실성 U를 구합니다.
2. `U_norm = sigmoid(20·U − 2)`, `α = max(1 − 0.9·U_norm, 0.1)`
3. 에피소드 리셋 시 `x ~ Beta(α, α)`, 정적 마찰 `μ = 0.4 + 0.8·x` (범위 [0.4, 1.2]), 동적 마찰 `0.9·μ`.

예측기: 최근 3스텝의 상태·행동 이력(72차원) → 인코더(32차원 잠재) → 잠재 변환 → 디코더 → 다음 상태 변화량.
Adam(lr 1e-3)으로 매 스텝 온라인 학습하며, 병렬 환경이 하나의 가중치를 공유합니다. `<TODO: 상태 12차원의 정의 확인>`

## 실험 설정 요약

- 시드: `<TODO: 시드 목록과 개수 확인>`, 조건별 1,500 iteration
- 보고 지표: 커리큘럼 내, OOD(μ=0.25, μ=1.6) 성공률의 시드 평균과 표준편차
- 계산 비용 측정: 100 iteration 벤치마크 3회 반복(`bench_resources.py`)

## 라이선스

Isaac Lab 기반 코드는 해당 소스 헤더의 BSD-3-Clause 라이선스를 따릅니다.

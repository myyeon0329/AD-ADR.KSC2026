# AD-ADR: Adaptive Dynamics-aware Domain Randomization

ANYmal-C 험지(rough terrain) 보행 학습에서 마찰 계수의 도메인 랜덤화를 **동역학 예측 오차**로 조절하는 AD-ADR의 구현입니다.
NVIDIA Isaac Lab(RSL-RL, PPO)의 ANYmal-C rough 환경 위에서 동작하며, 판별기 없이 동역학 예측기와 Beta 분포만 사용합니다.

## 설치

Isaac Lab 저장소(커밋 `b0542fe2d`, Python 3.11)에 아래 위치로 파일을 복사합니다.

| 이 저장소의 파일 | 복사 위치 |
|---|---|
| `source/jetbot_dr/*.py` | `source/isaaclab_tasks/isaaclab_tasks/manager_based/jetbot_dr/` |
| `scripts/train.py`, `scripts/eval_ood_friction.py`, `scripts/eval_terrain_levels.py` | `scripts/reinforcement_learning/rsl_rl/` |

복사 후 태스크 `Isaac-Jetbot-ADADR-v0`가 등록됩니다. 태스크 이름의 `Jetbot`은 초기 개발 때 쓰던 이름이며, 실제 로봇은 ANYmal-C입니다.

## AD-ADR 학습

Isaac Lab 폴더에서 실행합니다(1,500 iteration, 4,096개 환경, Isaac Lab 기본 PPO 설정).

```powershell
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Jetbot-ADADR-v0 --headless --num_envs 4096 --max_iterations 1500
```

학습 결과는 `logs/rsl_rl/anymal_c_rough/<실행 폴더>/`에 저장됩니다. 학습 중 실행 폴더에 α와 불확실성 U의 통계 로그(`alpha_log.txt`, `u_log3.txt`)가 함께 기록됩니다.

## 평가

`--load_run`에 학습 실행 폴더 이름을 넣습니다. 평가는 900개 환경, 에피소드 5회입니다.

```powershell
# 학습 분포 내(커리큘럼)
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\eval_terrain_levels.py --headless --task Isaac-Jetbot-ADADR-v0 --load_run <실행 폴더명> --num_envs 900 --num_episodes 5

# 마찰 조건 평가(분포 밖): --friction 0.25 또는 1.6
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\eval_ood_friction.py --headless --task Isaac-Jetbot-ADADR-v0 --load_run <실행 폴더명> --friction 0.25 --num_envs 900 --num_episodes 5
```

- 마찰 평가에서는 정적 마찰 μ를 고정(동적 0.9μ)하고, 마찰 랜덤화·외력(push)·관측 잡음과 지형 커리큘럼을 끕니다. 지형 난이도 레벨은 환경에 균등하게 배정합니다.
- 성공률(SR)은 base 접촉 없이 time-out으로 종료한 에피소드의 비율이며, 지형 레벨별 값과 전체 값을 출력합니다.

## 동작 방식 (`source/jetbot_dr/jetbot_env_cfg1.py`)

1. 환경별 동역학 예측기가 다음 상태를 예측하고, 예측 오차(MSE)의 지수이동평균(0.9/0.1)으로 불확실성 U를 구합니다.
2. `U_norm = sigmoid(20·U − 2)`, `α = max(1 − 0.9·U_norm, 0.1)`
3. 에피소드 리셋 시 `x ~ Beta(α, α)`로 정적 마찰 `μ = 0.4 + 0.8·x`(범위 [0.4, 1.2])를 뽑고, 동적 마찰은 `0.9·μ`로 둡니다.

**동역학 예측기**: 입력은 기본 자세 대비 관절 위치(12차원)와 행동(12차원)의 최근 3스텝 이력(72차원)입니다. 인코더(72→128→32) → 잠재 변환(32→64→32, 잔차) → 디코더(32→64→12)로 다음 상태 변화량을 예측해 현재 상태에 더합니다. Adam(lr 1e-3)으로 매 스텝 온라인 학습하며, 병렬 환경이 하나의 가중치를 공유합니다. 리셋된 환경의 이력은 0으로 초기화합니다.

환경 변수 `ADADR_ALPHA_MODE`는 통제 실험용입니다(`uniform`, `rank`, `shuffle`). 설정하지 않으면 위의 AD-ADR(sigmoid 방식)이 실행됩니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `source/jetbot_dr/jetbot_env_cfg1.py` | **AD-ADR 환경** (동역학 예측기 + 마찰 적응 샘플링) |
| `source/jetbot_dr/__init__.py` | 태스크 등록 |
| `source/jetbot_dr/jetbot_env_cfg1_v3_backup.py` | AD-ADR 샘플링 함수의 백업본 (참고용) |
| `scripts/train.py` | 학습 |
| `scripts/eval_terrain_levels.py`, `scripts/eval_ood_friction.py` | 평가 |

비교 실험용 파일(AD-ADR 실행에는 필요하지 않지만 `__init__.py`가 불러오므로 함께 복사합니다):
`jetbot_env_cfg2.py`(UDR 환경, `Isaac-Jetbot-Baseline-v0`), `jetbot_env_cfg_adr_full.py`·`svpg_sampler.py`·`adr_filtered_wrapper.py`(ADR 환경, `Isaac-Jetbot-ADRFull-v0`), `scripts/train_adr.py`(ADR 학습), `scripts/run_eval_sweep.py`(일괄 평가), `scripts/bench_resources.py`(학습 시간·GPU 메모리 측정).

## 라이선스

Isaac Lab 기반 코드는 해당 소스 헤더의 BSD-3-Clause 라이선스를 따릅니다.

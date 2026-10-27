"""
run_eval_sweep.py
==================
학습이 27개 시드 x 3방법 전부 끝난 뒤에 실행.
seed 1 때 쓰신 평가 명령어 그대로(num_envs=900, num_episodes=5), --load_run만
각 시드/방법에 맞게 자동으로 채워서 27 x 3 x 3(커리큘럼/OOD0.25/OOD1.6) 전부 실행.
각 결과는 results_seedN\ 폴더에 seed1 때처럼 .txt로 저장되고,
동시에 results_all.csv에도 숫자만 뽑아 누적 저장됨.

사용법:
    python run_eval_sweep.py
"""

import csv
import os
import re
import subprocess
import sys

PYTHON_EXE = sys.executable

SEEDS = [s for s in range(2, 29) if s not in (1, 42, 100)]  # 27개
assert len(SEEDS) == 27

LOG_ROOT = r"logs\rsl_rl\anymal_c_rough"

METHODS = {
    "baseline": {"task": "Isaac-Jetbot-Baseline-v0", "run_prefix": "Baseline_seed"},
    "adadr": {"task": "Isaac-Jetbot-ADADR-v0", "run_prefix": "ADADR_v3_seed"},
    "adrfull": {"task": "Isaac-Jetbot-ADRFull-v0", "run_prefix": "ADR_full_seed"},
}

RESULTS_CSV = "results_all.csv"


def find_load_run(run_name):
    """logs\\rsl_rl\\anymal_c_rough 아래에서 run_name으로 끝나는 폴더를 찾아
    그 폴더 이름을 반환 (seed 1 때 <BASELINE> 자리에 직접 채워 넣으셨던 것과 동일한 역할)."""
    if not os.path.isdir(LOG_ROOT):
        return None
    candidates = [d for d in os.listdir(LOG_ROOT) if d.endswith(run_name)]
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1]  # 혹시 여러 개면 가장 최근 것


def run_and_save(cmd, out_path):
    print("[RUN]", " ".join(cmd))
    with open(out_path, "w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    with open(out_path, encoding="utf-8", errors="replace") as f:
        return proc.returncode, f.read()


def parse_rate(text, ood):
    if ood:
        m = re.search(r"Overall OOD success rate.*?:\s*([\d.]+)%", text)
    else:
        m = re.search(r"Overall success rate:\s*([\d.]+)%", text)
    return float(m.group(1)) if m else None


def append_csv(method, seed, condition, rate):
    is_new = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["method", "seed", "condition", "success_rate"])
        w.writerow([method, seed, condition, rate])


for seed in SEEDS:
    out_dir = f"results_seed{seed}"
    os.makedirs(out_dir, exist_ok=True)

    for method, cfg in METHODS.items():
        run_name = f"{cfg['run_prefix']}{seed}"
        load_run = find_load_run(run_name)

        if load_run is None:
            print(f"[경고] {run_name}에 대한 체크포인트 폴더를 못 찾았습니다. 건너뜁니다.")
            continue

        # --- 커리큘럼 (eval_terrain_levels.py, seed 1 때와 동일한 num_envs/num_episodes) ---
        cmd = [
            PYTHON_EXE, r"scripts\reinforcement_learning\rsl_rl\eval_terrain_levels.py",
            "--task", cfg["task"],
            "--num_envs", "900",
            "--num_episodes", "5",
            "--load_run", load_run,
        ]
        out_path = os.path.join(out_dir, f"{method}_terrain.txt")
        rc, text = run_and_save(cmd, out_path)
        rate = parse_rate(text, ood=False)
        append_csv(method, seed, "curriculum", rate)
        print(f"  {method} seed{seed} curriculum = {rate}%")

        # --- OOD 0.25 ---
        cmd = [
            PYTHON_EXE, r"scripts\reinforcement_learning\rsl_rl\eval_ood_friction.py",
            "--task", cfg["task"],
            "--num_envs", "900",
            "--num_episodes", "5",
            "--friction", "0.25",
            "--load_run", load_run,
        ]
        out_path = os.path.join(out_dir, f"{method}_ood025.txt")
        rc, text = run_and_save(cmd, out_path)
        rate = parse_rate(text, ood=True)
        append_csv(method, seed, "ood_0.25", rate)
        print(f"  {method} seed{seed} ood_0.25 = {rate}%")

        # --- OOD 1.6 ---
        cmd = [
            PYTHON_EXE, r"scripts\reinforcement_learning\rsl_rl\eval_ood_friction.py",
            "--task", cfg["task"],
            "--num_envs", "900",
            "--num_episodes", "5",
            "--friction", "1.6",
            "--load_run", load_run,
        ]
        out_path = os.path.join(out_dir, f"{method}_ood16.txt")
        rc, text = run_and_save(cmd, out_path)
        rate = parse_rate(text, ood=True)
        append_csv(method, seed, "ood_1.6", rate)
        print(f"  {method} seed{seed} ood_1.6 = {rate}%")

print(f"\n27 x 3 x 3 평가 전체 완료. 숫자는 {RESULTS_CSV}, 원본 로그는 results_seedN\\ 폴더에 있습니다.")

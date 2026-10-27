"""
UDR / AD-ADR / ADR 학습 자원량 벤치마크.

실행 위치: Isaac Lab 루트 폴더 (Isaac Lab 가상환경 활성화 상태)
실행:      python bench_resources.py
옵션:      --iters 100 --reps 3 --cooldown 40 --seed 9001

측정 항목 (조건당 reps회, 조건을 번갈아 실행해 열 드리프트 영향 분산):
  - 정상상태 iteration 시간: rsl_rl 콘솔의 "Iteration time"을 모아 앞 warmup개 제외 후 중앙값
  - 처리량(steps/s): rsl_rl 콘솔의 "Computation: N steps/s" 중앙값
  - GPU 메모리 최댓값(MiB): nvidia-smi 1회/0.5초 폴링, 실행 전 유휴 메모리를 뺀 증가분도 기록
  - GPU 사용률 평균(%), 전력 평균(W)
  - 프로세스 전체 시간, train 스크립트가 출력하는 "Training time"

결과: bench_out/results.csv, bench_out/summary.txt, 각 실행 원본 로그 bench_out/<조건>_rep<k>.log
주의: 학습 로그 폴더가 seed 9001 이름으로 logs/rsl_rl 아래 생깁니다. 끝나고 지우세요.
"""
import argparse
import csv
import os
import re
import statistics as st
import subprocess
import sys
import threading
import time

CONDITIONS = [
    # (이름, 스크립트, 태스크)
    ("UDR",    r"scripts\reinforcement_learning\rsl_rl\train.py",     "Isaac-Jetbot-Baseline-v0"),
    ("AD-ADR", r"scripts\reinforcement_learning\rsl_rl\train.py",     "Isaac-Jetbot-ADADR-v0"),
    ("ADR",    r"scripts\reinforcement_learning\rsl_rl\train_adr.py", "Isaac-Jetbot-ADRFull-v0"),
]

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
RE_ITER = re.compile(r"Iteration time:\s*([\d.]+)\s*s")
RE_COMP = re.compile(r"Computation:\s*(\d+)\s*steps/s")
RE_TRAIN = re.compile(r"Training time:\s*([\d.]+)\s*seconds")

SMI = ["nvidia-smi", "-i", "0",
       "--query-gpu=memory.used,utilization.gpu,power.draw",
       "--format=csv,noheader,nounits"]


def smi_sample():
    try:
        out = subprocess.run(SMI, capture_output=True, text=True, timeout=5).stdout.strip()
        mem, util, pw = [x.strip() for x in out.split(",")]
        f = lambda s: float(s) if s not in ("[N/A]", "N/A", "") else float("nan")
        return f(mem), f(util), f(pw)
    except Exception:
        return float("nan"), float("nan"), float("nan")


class Monitor(threading.Thread):
    def __init__(self, period=0.5):
        super().__init__(daemon=True)
        self.period = period
        self.stop_flag = threading.Event()
        self.samples = []

    def run(self):
        while not self.stop_flag.is_set():
            self.samples.append(smi_sample())
            time.sleep(self.period)


def nanmedian(xs):
    xs = [x for x in xs if x == x]
    return st.median(xs) if xs else float("nan")


def nanmean(xs):
    xs = [x for x in xs if x == x]
    return st.mean(xs) if xs else float("nan")


def run_one(name, script, task, iters, seed, outdir, rep, warmup):
    # 유휴 메모리 기준선
    idle = nanmedian([smi_sample()[0] for _ in range(4)])
    cmd = ["cmd.exe", "/c", "isaaclab.bat", "-p", script,
           "--task", task, "--headless",
           "--max_iterations", str(iters), "--seed", str(seed)]
    if name == "ADR":
        cmd += ["--num_envs", "8192"]
    log_path = os.path.join(outdir, f"{name}_rep{rep}.log")
    mon = Monitor()
    t0 = time.time()
    mon.start()
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            lf.write(line)
        p.wait()
    wall = time.time() - t0
    mon.stop_flag.set()
    mon.join(timeout=3)

    text = ANSI.sub("", open(log_path, encoding="utf-8", errors="replace").read())
    it_times = [float(x) for x in RE_ITER.findall(text)]
    comps = [float(x) for x in RE_COMP.findall(text)]
    tr = RE_TRAIN.findall(text)

    steady = it_times[warmup:] if len(it_times) > warmup else it_times
    mems = [s[0] for s in mon.samples]
    utils = [s[1] for s in mon.samples]
    pws = [s[2] for s in mon.samples]
    peak = max([m for m in mems if m == m], default=float("nan"))
    # 학습 구간만(메모리가 유휴보다 충분히 큰 구간) 사용률/전력 평균
    active = [(u, w) for (m, u, w) in mon.samples if m == m and m > idle + 500]
    return {
        "cond": name, "rep": rep, "rc": p.returncode,
        "n_iter_lines": len(it_times),
        "iter_time_median_s": nanmedian(steady),
        "iter_time_mean_s": nanmean(steady),
        "steps_per_s_median": nanmedian(comps[warmup:] if len(comps) > warmup else comps),
        "train_time_s": float(tr[-1]) if tr else float("nan"),
        "wall_s": wall,
        "gpu_idle_MiB": idle,
        "gpu_peak_MiB": peak,
        "gpu_peak_delta_MiB": peak - idle if peak == peak else float("nan"),
        "gpu_util_mean": nanmean([a[0] for a in active]),
        "gpu_power_mean_W": nanmean([a[1] for a in active]),
    }


def fmt(xs):
    xs = [x for x in xs if x == x]
    if not xs:
        return "n/a"
    if len(xs) == 1:
        return f"{xs[0]:.3f}"
    return f"{st.mean(xs):.3f} ± {st.stdev(xs):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=10, help="iteration 시간에서 제외할 앞부분 개수")
    ap.add_argument("--cooldown", type=int, default=40)
    ap.add_argument("--seed", type=int, default=9001)
    ap.add_argument("--outdir", default="bench_out")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    print("GPU 유휴 상태:", smi_sample(), "(메모리MiB, 사용률%, 전력W)")
    print("다른 GPU 작업(브라우저 영상, 다른 학습)을 끄고 노트북이면 전원을 연결하세요.")
    rows = []
    for rep in range(1, a.reps + 1):
        for name, script, task in [c for c in CONDITIONS if c[0] in os.environ.get("ONLY", "UDR,AD-ADR,ADR").split(",")]:
            print(f"\n=== rep {rep}/{a.reps}  {name}  ({task}) ===", flush=True)
            r = run_one(name, script, task, a.iters, a.seed, a.outdir, rep, a.warmup)
            rows.append(r)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
            if r["rc"] != 0 or r["n_iter_lines"] == 0:
                print(f"!! 실패 또는 iteration 로그 없음: {a.outdir}\\{name}_rep{rep}.log 를 확인하세요.")
            time.sleep(a.cooldown)

    keys = list(rows[0].keys())
    with open(os.path.join(a.outdir, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    metrics = ["iter_time_median_s", "steps_per_s_median", "gpu_peak_MiB",
               "gpu_peak_delta_MiB", "gpu_util_mean", "gpu_power_mean_W", "train_time_s"]
    lines = [f"iters={a.iters} reps={a.reps} warmup={a.warmup} seed={a.seed}", ""]
    byc = {}
    for name, _, _ in CONDITIONS:
        sub = [r for r in rows if r["cond"] == name and r["rc"] == 0]
        byc[name] = sub
        lines.append(f"[{name}] n={len(sub)}")
        for m in metrics:
            lines.append(f"  {m:22s} {fmt([r[m] for r in sub])}")
        lines.append("")

    def med(name, m):
        return nanmedian([r[m] for r in byc.get(name, [])])

    lines.append("비율 (중앙값 기준)")
    for m in ["iter_time_median_s", "gpu_peak_MiB", "gpu_peak_delta_MiB"]:
        u, ad, adr = med("UDR", m), med("AD-ADR", m), med("ADR", m)
        lines.append(f"  {m}: AD-ADR/UDR={ad/u:.3f}  ADR/AD-ADR={adr/ad:.3f}  ADR/UDR={adr/u:.3f}")
    txt = "\n".join(lines)
    open(os.path.join(a.outdir, "summary.txt"), "w", encoding="utf-8").write(txt)
    print("\n" + txt)


if __name__ == "__main__":
    sys.exit(main())
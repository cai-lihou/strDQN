import argparse
import os
import queue
import subprocess
import sys
from dataclasses import dataclass

import pandas as pd

from result_io import default_result_dir, parse_float_list, parse_int_list


def token(value):
    return str(value).replace(".", "p").replace("-", "m")


@dataclass
class Job:
    label: str
    cmd: list
    path: str = None
    budget: int = None
    duration: float = None
    seed: int = None
    extra_filters: dict = None
    uses_gpu: bool = True


def run_command(cmd, dry_run=False, gpu=None):
    prefix = f"[gpu={gpu}] " if gpu is not None else ""
    print(prefix + " ".join(cmd))
    if not dry_run:
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        subprocess.run(cmd, check=True, env=env)


def parse_gpu_list(gpu_arg, gpus_arg, parallel_jobs):
    raw = gpus_arg if gpus_arg else gpu_arg
    gpus = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not gpus:
        gpus = ["0"]
    if parallel_jobs is not None:
        if parallel_jobs < 1:
            raise ValueError("--parallel-jobs must be >= 1")
        gpus = gpus[:parallel_jobs]
    return gpus


def replace_gpu_arg(cmd, gpu):
    cmd = list(cmd)
    if "--gpu" in cmd:
        idx = cmd.index("--gpu")
        if idx + 1 < len(cmd):
            cmd[idx + 1] = str(gpu)
    return cmd


def result_path(result_dir, dataset, suffix):
    return os.path.join(result_dir, f"result_{dataset}{suffix}.xlsx")


def ablation_result_path(result_dir, dataset, mode, suffix):
    return os.path.join(result_dir, f"result_{dataset}_{mode}{suffix}.xlsx")


def read_excel_if_exists(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_excel(path)
    except Exception as exc:
        print(f"[resume] Cannot read {path}: {exc}. This target will be rerun.")
        return pd.DataFrame()


def numeric_match(series, value, tol=1e-9):
    values = pd.to_numeric(series, errors="coerce")
    return (values - float(value)).abs() <= tol


def dqn_combo_completed(path, budget, duration, seed, extra_filters=None):
    df = read_excel_if_exists(path)
    required = {"MAX_BUDGET", "ACTIVATION_DURATION_PCT", "RANDOM_SEED", "Model", "ActionType"}
    if df.empty or not required.issubset(df.columns):
        return False

    mask = (
        numeric_match(df["MAX_BUDGET"], budget) &
        numeric_match(df["ACTIVATION_DURATION_PCT"], duration) &
        numeric_match(df["RANDOM_SEED"], seed) &
        (df["Model"].astype(str) == "final") &
        (df["ActionType"].astype(str).str.upper() == "DONE")
    )
    for col, expected in (extra_filters or {}).items():
        if col not in df.columns:
            return False
        mask &= numeric_match(df[col], expected)
    return bool(mask.any())


def dqn_job(label, path, budget, duration, seed, cmd, extra_filters=None):
    return Job(
        label=label,
        cmd=cmd,
        path=path,
        budget=budget,
        duration=duration,
        seed=seed,
        extra_filters=extra_filters,
        uses_gpu=True,
    )


def job_completed(job, args):
    if not args.resume or job.path is None:
        return False
    return dqn_combo_completed(job.path, job.budget, job.duration, job.seed, job.extra_filters)


def run_job(job, args, gpu):
    if job_completed(job, args):
        print(f"[skip] {job.label}: seed={job.seed}, budget={job.budget}, duration={job.duration}")
        return
    cmd = replace_gpu_arg(job.cmd, gpu) if job.uses_gpu else list(job.cmd)
    run_command(cmd, args.dry_run, gpu=gpu if job.uses_gpu else None)


def run_jobs_parallel(jobs, gpus, args):
    if not jobs:
        return
    if args.dry_run and len(gpus) > 1:
        print(f"[parallel] dry-run {len(jobs)} jobs on GPUs: {','.join(gpus)}")
        for idx, job in enumerate(jobs):
            run_job(job, args, gpus[idx % len(gpus)] if job.uses_gpu else None)
        return
    if len(gpus) == 1:
        for job in jobs:
            run_job(job, args, gpus[0] if job.uses_gpu else None)
        return

    print(f"[parallel] running {len(jobs)} jobs on GPUs: {','.join(gpus)}")
    job_queue = queue.Queue()
    for job in jobs:
        job_queue.put(job)

    import concurrent.futures

    def worker(gpu):
        while True:
            try:
                job = job_queue.get_nowait()
            except queue.Empty:
                return
            try:
                run_job(job, args, gpu if job.uses_gpu else None)
            finally:
                job_queue.task_done()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(worker, gpu) for gpu in gpus]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def interleave_job_groups(groups):
    active = [list(group) for group in groups if group]
    ordered = []
    while active:
        next_active = []
        for group in active:
            ordered.append(group.pop(0))
            if group:
                next_active.append(group)
        active = next_active
    return ordered


def main():
    parser = argparse.ArgumentParser(description="Run the minimal required StrDQN experiment suite")
    parser.add_argument("--dataset", default="thiers_2012")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--gpus", default=None,
                        help="Comma-separated GPU IDs for parallel experiment jobs, e.g. 0,1.")
    parser.add_argument("--parallel-jobs", type=int, default=None,
                        help="Maximum parallel GPU jobs. Defaults to the number of IDs in --gpus/--gpu.")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--budgets", default="10,20,30,50")
    parser.add_argument("--durations", default="0.001,0.005,0.01")
    parser.add_argument("--ablation-budgets", default="30")
    parser.add_argument("--ablation-durations", default="0.005")
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--suffix", default="_minimal")
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--parts", default="main,ablation,sensitivity,fair_eval,aggregate,plot")
    parser.add_argument("--ablation-modes", default="no_wait,no_action_bias,unified,no_wait_compensation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                        help="Skip completed experiment combinations. Enabled by default.")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Run every requested combination even if result rows already exist.")
    parser.add_argument("--pwait-values", default="0.1,0.2,0.3,0.5")
    parser.add_argument("--lambda-values", default="0,0.001,0.01,0.05")
    parser.add_argument("--sensitivity-budget", type=int, default=30)
    parser.add_argument("--sensitivity-duration", default="0.005")
    parser.add_argument("--fair-eval-rounds", type=int, default=20)
    parser.add_argument("--static-weight-mode", choices=["unweighted", "weighted", "capped"], default="weighted",
                        help="Static baseline graph mode: unweighted, weighted, or capped.")
    parser.add_argument("--weight-cap", type=int, default=3,
                        help="Static baseline cap used when --static-weight-mode capped.")
    args = parser.parse_args()
    if not args.result_dir:
        args.result_dir = default_result_dir(args.dataset)

    seeds = parse_int_list(args.seeds, list(range(10)))
    budgets = parse_int_list(args.budgets, [10, 20, 30, 50])
    durations = parse_float_list(args.durations, [0.001, 0.005, 0.01])
    ablation_budgets = parse_int_list(args.ablation_budgets, [30])
    ablation_durations = parse_float_list(args.ablation_durations, [0.005])
    sensitivity_durations = parse_float_list(args.sensitivity_duration, [0.005])
    ablation_modes = [mode.strip() for mode in args.ablation_modes.split(",") if mode.strip()]
    parts = {part.strip() for part in args.parts.split(",") if part.strip()}
    py = sys.executable
    gpus = parse_gpu_list(args.gpu, args.gpus, args.parallel_jobs)
    main_jobs = []
    ablation_jobs = []
    baseline_jobs = []
    sensitivity_jobs = []

    if "main" in parts:
        path = result_path(args.result_dir, args.dataset, args.suffix)
        for seed in seeds:
            for budget in budgets:
                for duration in durations:
                    main_jobs.append(dqn_job("main", path, budget, duration, seed, [
                        py, "train_strdqn.py",
                        "--dataset", args.dataset,
                        "--gpu", args.gpu,
                        "--seed", str(seed),
                        "--eval-seed", str(seed),
                        "--budgets", str(budget),
                        "--durations", str(duration),
                        "--episodes", str(args.episodes),
                        "--result-suffix", args.suffix,
                        "--result-dir", args.result_dir,
                    ]))

    if "ablation" in parts:
        for seed in seeds:
            for mode in ablation_modes:
                path = ablation_result_path(args.result_dir, args.dataset, mode, args.suffix)
                for budget in ablation_budgets:
                    for duration in ablation_durations:
                        ablation_jobs.append(dqn_job(f"ablation/{mode}", path, budget, duration, seed, [
                            py, "train_ablation.py",
                            "--dataset", args.dataset,
                            "--gpu", args.gpu,
                            "--ablation", mode,
                            "--seed", str(seed),
                            "--eval-seed", str(seed),
                            "--budgets", str(budget),
                            "--durations", str(duration),
                            "--episodes", str(args.episodes),
                            "--result-suffix", args.suffix,
                            "--result-dir", args.result_dir,
                        ]))

    if "baseline" in parts:
        baseline_cmd = [
            py, "run_baselines.py",
            "--dataset", args.dataset,
            "--seeds", args.seeds,
            "--budgets", args.budgets,
            "--durations", args.durations,
            "--result-suffix", args.suffix,
            "--result-dir", args.result_dir,
            "--static-weight-mode", args.static_weight_mode,
            "--weight-cap", str(args.weight_cap),
        ]
        if args.resume:
            baseline_cmd.append("--resume")
        baseline_jobs.append(Job(label="baseline", cmd=baseline_cmd, uses_gpu=True))

    if "sensitivity" in parts:
        pwait_values = parse_float_list(args.pwait_values, [0.1, 0.2, 0.3, 0.5])
        lambda_values = parse_float_list(args.lambda_values, [0.0, 0.001, 0.01, 0.05])
        for seed in seeds:
            for value in pwait_values:
                suffix = f"_sens_pwait_{token(value)}{args.suffix}"
                path = result_path(args.result_dir, args.dataset, suffix)
                for duration in sensitivity_durations:
                    sensitivity_jobs.append(dqn_job("sensitivity/P_WAIT", path, args.sensitivity_budget, duration, seed, [
                        py, "train_strdqn.py",
                        "--dataset", args.dataset,
                        "--gpu", args.gpu,
                        "--seed", str(seed),
                        "--eval-seed", str(seed),
                        "--budgets", str(args.sensitivity_budget),
                        "--durations", str(duration),
                        "--episodes", str(args.episodes),
                        "--force-wait-prob", str(value),
                        "--result-suffix", suffix,
                        "--result-dir", args.result_dir,
                    ], {"FORCE_WAIT_PROB": value}))
            for value in lambda_values:
                suffix = f"_sens_lambda_{token(value)}{args.suffix}"
                path = result_path(args.result_dir, args.dataset, suffix)
                for duration in sensitivity_durations:
                    sensitivity_jobs.append(dqn_job("sensitivity/lambda", path, args.sensitivity_budget, duration, seed, [
                        py, "train_strdqn.py",
                        "--dataset", args.dataset,
                        "--gpu", args.gpu,
                        "--seed", str(seed),
                        "--eval-seed", str(seed),
                        "--budgets", str(args.sensitivity_budget),
                        "--durations", str(duration),
                        "--episodes", str(args.episodes),
                        "--wait-reward-coef", str(value),
                        "--result-suffix", suffix,
                        "--result-dir", args.result_dir,
                    ], {"WAIT_REWARD_COEF": value}))
            suffix = f"_sens_delta{args.suffix}"
            path = result_path(args.result_dir, args.dataset, suffix)
            for duration in durations:
                sensitivity_jobs.append(dqn_job("sensitivity/Delta", path, args.sensitivity_budget, duration, seed, [
                    py, "train_strdqn.py",
                    "--dataset", args.dataset,
                    "--gpu", args.gpu,
                    "--seed", str(seed),
                    "--eval-seed", str(seed),
                    "--budgets", str(args.sensitivity_budget),
                    "--durations", str(duration),
                    "--episodes", str(args.episodes),
                    "--result-suffix", suffix,
                    "--result-dir", args.result_dir,
                ]))

    train_jobs = interleave_job_groups([main_jobs, ablation_jobs, baseline_jobs, sensitivity_jobs])
    run_jobs_parallel(train_jobs, gpus, args)

    if "fair_eval" in parts:
        fair_eval_cmd = [
            py, "fair_evaluate.py",
            "--dataset", args.dataset,
            "--suffix", args.suffix,
            "--result-dir", args.result_dir,
            "--rounds", str(args.fair_eval_rounds),
        ]
        if args.resume:
            fair_eval_cmd.append("--resume")
        else:
            fair_eval_cmd.append("--no-resume")
        run_command(fair_eval_cmd, args.dry_run)

    if "aggregate" in parts:
        run_command([py, "aggregate_results.py", "--dataset", args.dataset, "--suffix", args.suffix,
                     "--result-dir", args.result_dir],
                    args.dry_run)

    if "plot" in parts:
        run_command([py, "plot_results.py", "--dataset", args.dataset, "--result-dir", args.result_dir],
                    args.dry_run)


if __name__ == "__main__":
    main()

import argparse
import os
import subprocess
import sys

import pandas as pd

from result_utils import default_result_dir, parse_float_list, parse_int_list


def token(value):
    return str(value).replace(".", "p").replace("-", "m")


def run_command(cmd, dry_run=False):
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


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


def maybe_run_dqn(label, path, budget, duration, seed, cmd, args, extra_filters=None):
    if args.resume and dqn_combo_completed(path, budget, duration, seed, extra_filters):
        print(f"[skip] {label}: seed={seed}, budget={budget}, duration={duration}")
        return
    run_command(cmd, args.dry_run)


def main():
    parser = argparse.ArgumentParser(description="Run the minimal required StrDQN experiment suite")
    parser.add_argument("--dataset", default="thiers_2012")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--budgets", default="10,20,30,50")
    parser.add_argument("--durations", default="0.001,0.005,0.01")
    parser.add_argument("--ablation-budgets", default="30")
    parser.add_argument("--ablation-durations", default="0.005")
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--suffix", default="_minimal")
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--parts", default="main,ablation,baseline,sensitivity,fair_eval,aggregate,plot")
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

    if "main" in parts:
        path = result_path(args.result_dir, args.dataset, args.suffix)
        for seed in seeds:
            for budget in budgets:
                for duration in durations:
                    maybe_run_dqn("main", path, budget, duration, seed, [
                        py, "ceshi1.py",
                        "--dataset", args.dataset,
                        "--gpu", args.gpu,
                        "--seed", str(seed),
                        "--eval-seed", str(seed),
                        "--budgets", str(budget),
                        "--durations", str(duration),
                        "--episodes", str(args.episodes),
                        "--result-suffix", args.suffix,
                        "--result-dir", args.result_dir,
                    ], args)

    if "ablation" in parts:
        for seed in seeds:
            for mode in ablation_modes:
                path = ablation_result_path(args.result_dir, args.dataset, mode, args.suffix)
                for budget in ablation_budgets:
                    for duration in ablation_durations:
                        maybe_run_dqn(f"ablation/{mode}", path, budget, duration, seed, [
                            py, "ablation.py",
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
                        ], args)

    if "baseline" in parts:
        baseline_cmd = [
            py, "static_peak_baseline.py",
            "--dataset", args.dataset,
            "--seeds", args.seeds,
            "--budgets", args.budgets,
            "--durations", args.durations,
            "--result-suffix", args.suffix,
            "--result-dir", args.result_dir,
        ]
        if args.resume:
            baseline_cmd.append("--resume")
        run_command(baseline_cmd, args.dry_run)

    if "sensitivity" in parts:
        pwait_values = parse_float_list(args.pwait_values, [0.1, 0.2, 0.3, 0.5])
        lambda_values = parse_float_list(args.lambda_values, [0.0, 0.001, 0.01, 0.05])
        for seed in seeds:
            for value in pwait_values:
                suffix = f"_sens_pwait_{token(value)}{args.suffix}"
                path = result_path(args.result_dir, args.dataset, suffix)
                for duration in sensitivity_durations:
                    maybe_run_dqn("sensitivity/P_WAIT", path, args.sensitivity_budget, duration, seed, [
                        py, "ceshi1.py",
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
                    ], args, {"FORCE_WAIT_PROB": value})
            for value in lambda_values:
                suffix = f"_sens_lambda_{token(value)}{args.suffix}"
                path = result_path(args.result_dir, args.dataset, suffix)
                for duration in sensitivity_durations:
                    maybe_run_dqn("sensitivity/lambda", path, args.sensitivity_budget, duration, seed, [
                        py, "ceshi1.py",
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
                    ], args, {"WAIT_REWARD_COEF": value})
            suffix = f"_sens_delta{args.suffix}"
            path = result_path(args.result_dir, args.dataset, suffix)
            for duration in durations:
                maybe_run_dqn("sensitivity/Delta", path, args.sensitivity_budget, duration, seed, [
                    py, "ceshi1.py",
                    "--dataset", args.dataset,
                    "--gpu", args.gpu,
                    "--seed", str(seed),
                    "--eval-seed", str(seed),
                    "--budgets", str(args.sensitivity_budget),
                    "--durations", str(duration),
                    "--episodes", str(args.episodes),
                    "--result-suffix", suffix,
                    "--result-dir", args.result_dir,
                ], args)

    if "fair_eval" in parts:
        fair_eval_cmd = [
            py, "fair_eval_results.py",
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
        run_command([py, "aggregate_minimal_results.py", "--dataset", args.dataset, "--suffix", args.suffix,
                     "--result-dir", args.result_dir],
                    args.dry_run)

    if "plot" in parts:
        run_command([py, "plot_minimal_results.py", "--dataset", args.dataset, "--result-dir", args.result_dir],
                    args.dry_run)


if __name__ == "__main__":
    main()

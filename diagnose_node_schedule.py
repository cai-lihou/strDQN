import argparse
import os
from datetime import datetime

import pandas as pd

from IC import t2EICModel
from fair_eval_results import (
    load_test_edges,
    parse_seed_schedule,
    read_baseline_schedules,
    read_dqn_schedules,
    safe_float,
    safe_int,
)
from result_utils import default_result_dir, delta_metadata, format_mean_std


def parse_optional_ints(value):
    if value is None or str(value).strip() == "":
        return None
    return {int(float(item.strip())) for item in str(value).split(",") if item.strip()}


def parse_optional_floats(value):
    if value is None or str(value).strip() == "":
        return None
    return {float(item.strip()) for item in str(value).split(",") if item.strip()}


def should_keep(record, budgets, durations, seeds):
    if budgets is not None and safe_int(record.get("Budget")) not in budgets:
        return False
    if durations is not None:
        duration = safe_float(record.get("Duration"))
        if not any(abs(duration - item) < 1e-12 for item in durations):
            return False
    if seeds is not None and safe_int(record.get("RANDOM_SEED"), 0) not in seeds:
        return False
    return True


def build_records(dataset, suffix, result_dir, budgets, durations, seeds, baseline_name):
    main_path = os.path.join(result_dir, f"result_{dataset}{suffix}.xlsx")
    full_records = read_dqn_schedules(main_path, "Full StrDQN", "main")
    full_records = [record for record in full_records if should_keep(record, budgets, durations, seeds)]

    test_edges = load_test_edges(dataset)
    if not test_edges:
        raise RuntimeError(f"No test edges found for dataset {dataset}")
    test_start = float(test_edges[0][2])

    records = []
    for record in full_records:
        original = dict(record)
        original["Diagnosis"] = "original_time"
        original["Algorithm"] = "Full StrDQN original-time"
        records.append(original)

        original_seeds = parse_seed_schedule(record.get("Seeds"))
        start_time_seeds = [(node, test_start) for node, _ in original_seeds]
        node_only = dict(record)
        node_only["Diagnosis"] = "node_only_test_start"
        node_only["Algorithm"] = "Full StrDQN node-only test-start"
        node_only["Seeds"] = str(start_time_seeds)
        node_only["SeedCount"] = len(start_time_seeds)
        node_only["ScheduledTime"] = test_start
        records.append(node_only)

    baseline_path = os.path.join(result_dir, f"{dataset}_static_peak{suffix}.xlsx")
    baseline_records = read_baseline_schedules(baseline_path)
    for record in baseline_records:
        if record.get("Algorithm") != baseline_name:
            continue
        if not should_keep(record, budgets, durations, seeds):
            continue
        baseline = dict(record)
        baseline["Diagnosis"] = "baseline"
        baseline["Algorithm"] = baseline_name
        records.append(baseline)

    return records, test_edges, test_start


def evaluate_records(records, test_edges, rounds):
    rows = []
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for idx, record in enumerate(records, start=1):
        seeds = parse_seed_schedule(record.get("Seeds"))
        duration = safe_float(record.get("Duration"))
        random_seed = safe_int(record.get("RANDOM_SEED"), 0)
        activation_prob = safe_float(record.get("ACTIVATION_PROB"), 0.5)

        model = t2EICModel(
            test_edges,
            activation_prob=activation_prob,
            activation_duration_pct=duration,
            random_state=random_seed,
        )
        spread = model.simulate(seeds, num_rounds=rounds, use_cache=False)
        row = dict(record)
        row.update(delta_metadata(duration, test_edges))
        row["Fair_Spread"] = spread
        row["Spread"] = spread
        row["FAIR_EVAL_ROUNDS"] = int(rounds)
        row["DIAGNOSIS_TIME"] = run_time
        rows.append(row)
        print(
            f"[{idx}/{len(records)}] {row['Algorithm']} "
            f"B={row['Budget']} D={row['Duration']} seed={row['RANDOM_SEED']} -> {spread:.4f}"
        )
    return pd.DataFrame(rows)


def summarize(detail_df):
    if detail_df.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["Algorithm", "Diagnosis", "Budget", "Duration"]
    for key, group in detail_df.groupby(group_cols, dropna=False):
        values = pd.to_numeric(group["Fair_Spread"], errors="coerce").dropna()
        row = dict(zip(group_cols, key))
        row["mean"] = values.mean()
        row["std"] = values.std(ddof=1) if len(values) > 1 else 0.0
        row["n_runs"] = len(values)
        row["mean_std"] = format_mean_std(row["mean"], row["std"])
        if "DELTA_MINUTES" in group.columns:
            row["DELTA_MINUTES"] = pd.to_numeric(group["DELTA_MINUTES"], errors="coerce").dropna().mean()
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparison(detail_df, baseline_name):
    if detail_df.empty:
        return pd.DataFrame()
    base_cols = ["Budget", "Duration", "RANDOM_SEED"]
    pivot = detail_df.pivot_table(
        index=base_cols,
        columns="Algorithm",
        values="Fair_Spread",
        aggfunc="last",
    ).reset_index()
    original_col = "Full StrDQN original-time"
    node_col = "Full StrDQN node-only test-start"

    if node_col in pivot.columns and baseline_name in pivot.columns:
        pivot["NodeOnly_minus_Baseline"] = pivot[node_col] - pivot[baseline_name]
    if node_col in pivot.columns and original_col in pivot.columns:
        pivot["NodeOnly_minus_Original"] = pivot[node_col] - pivot[original_col]
    if original_col in pivot.columns and baseline_name in pivot.columns:
        pivot["Original_minus_Baseline"] = pivot[original_col] - pivot[baseline_name]

    verdicts = []
    for _, row in pivot.iterrows():
        node_minus_base = row.get("NodeOnly_minus_Baseline")
        node_minus_original = row.get("NodeOnly_minus_Original")
        if pd.notna(node_minus_base) and node_minus_base < 0:
            verdict = "node_selection_weaker_than_baseline"
        elif pd.notna(node_minus_original) and node_minus_original > 0:
            verdict = "schedule_time_hurts"
        elif pd.notna(node_minus_original) and node_minus_original < 0:
            verdict = "schedule_time_helps"
        else:
            verdict = "inconclusive"
        verdicts.append(verdict)
    pivot["DiagnosisVerdict"] = verdicts
    return pivot


def write_outputs(detail_df, summary_df, paired_df, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with pd.ExcelWriter(out_path) as writer:
        detail_df.to_excel(writer, sheet_name="detail", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        paired_df.to_excel(writer, sheet_name="paired", index=False)
    print(f"Saved node/schedule diagnosis to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose whether Full StrDQN loses because of node selection or seed scheduling."
    )
    parser.add_argument("--dataset", default="thiers_2012")
    parser.add_argument("--suffix", default="_minimal")
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--baseline", default="Degree")
    parser.add_argument("--budgets", default="")
    parser.add_argument("--durations", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if not args.result_dir:
        args.result_dir = default_result_dir(args.dataset)

    budgets = parse_optional_ints(args.budgets)
    durations = parse_optional_floats(args.durations)
    seeds = parse_optional_ints(args.seeds)
    out_path = args.out or os.path.join(
        args.result_dir,
        f"{args.dataset}_node_schedule_diagnosis{args.suffix}.xlsx",
    )

    records, test_edges, test_start = build_records(
        args.dataset,
        args.suffix,
        args.result_dir,
        budgets,
        durations,
        seeds,
        args.baseline,
    )
    if not records:
        print("No schedules found. Check --result-dir, --suffix, and baseline file.")
        return

    print(f"Test-start timestamp: {test_start}")
    detail_df = evaluate_records(records, test_edges, args.rounds)
    summary_df = summarize(detail_df)
    paired_df = paired_comparison(detail_df, args.baseline)
    write_outputs(detail_df, summary_df, paired_df, out_path)

    if not summary_df.empty:
        print("\nSummary:")
        print(summary_df[["Algorithm", "Budget", "Duration", "mean_std", "n_runs"]].to_string(index=False))


if __name__ == "__main__":
    main()

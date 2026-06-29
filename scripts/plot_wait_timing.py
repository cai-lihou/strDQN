import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    plt = None
    MATPLOTLIB_IMPORT_ERROR = exc
else:
    MATPLOTLIB_IMPORT_ERROR = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fair_evaluate import (
    load_test_edges,
    parse_seed_schedule,
    read_dqn_schedules,
    read_excel,
    safe_float,
    safe_int,
)
from src.common.result_io import default_result_dir


ALGORITHMS = {
    "full": {
        "label": "Full StrDQN",
        "file_template": "result_{dataset}{suffix}.xlsx",
        "experiment": "main",
    },
    "no_wait": {
        "label": "w/o WAIT",
        "file_template": "result_{dataset}_no_wait{suffix}.xlsx",
        "experiment": "ablation",
    },
}


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


def load_raw_edges(dataset):
    path = os.path.join("processed", f"{dataset}_main.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return sorted(data["temporal_edges"], key=lambda x: x[2])


def collect_records(dataset, suffix, result_dir, budgets, durations, seeds):
    records = []
    for spec in ALGORITHMS.values():
        path = os.path.join(result_dir, spec["file_template"].format(dataset=dataset, suffix=suffix))
        label = spec["label"]
        for record in read_dqn_schedules(path, label, spec["experiment"]):
            if should_keep(record, budgets, durations, seeds):
                records.append(record)
    return records


def load_fair_spread_map(dataset, suffix, result_dir):
    path = os.path.join(result_dir, f"{dataset}_fair_eval{suffix}.xlsx")
    df = read_excel(path)
    required = {"Experiment", "Algorithm", "Budget", "Duration", "RANDOM_SEED", "Fair_Spread"}
    if df.empty or not required.issubset(df.columns):
        return {}

    if "FAIR_EVAL_ROUNDS" in df.columns:
        rounds = pd.to_numeric(df["FAIR_EVAL_ROUNDS"], errors="coerce")
        if rounds.notna().any():
            df = df[rounds == rounds.max()].copy()

    df = df[
        (
            (df["Experiment"].astype(str) == "main") &
            (df["Algorithm"].astype(str) == "Full StrDQN")
        ) |
        (
            (df["Experiment"].astype(str) == "ablation") &
            (df["Algorithm"].astype(str) == "w/o WAIT")
        )
    ].copy()
    df = df.drop_duplicates(
        subset=["Experiment", "Algorithm", "Budget", "Duration", "RANDOM_SEED"],
        keep="last",
    )

    out = {}
    for _, row in df.iterrows():
        key = (
            str(row["Algorithm"]),
            safe_int(row["Budget"]),
            safe_float(row["Duration"]),
            safe_int(row["RANDOM_SEED"], 0),
        )
        out[key] = safe_float(row["Fair_Spread"])
    return out


def records_to_events(records, test_start, spread_map):
    rows = []
    for record in records:
        algorithm = str(record.get("Algorithm"))
        budget = safe_int(record.get("Budget"))
        duration = safe_float(record.get("Duration"))
        random_seed = safe_int(record.get("RANDOM_SEED"), 0)
        fair_spread = spread_map.get((algorithm, budget, duration, random_seed))

        schedule = parse_seed_schedule(record.get("Seeds"))
        for rank, (node, timestamp) in enumerate(schedule, start=1):
            timestamp = float(timestamp)
            rows.append({
                "Algorithm": algorithm,
                "Budget": budget,
                "Duration": duration,
                "RANDOM_SEED": random_seed,
                "SeedRank": rank,
                "Node": node,
                "Time": timestamp,
                "RelativeMinutes": (timestamp - test_start) / 60.0,
                "RelativeHours": (timestamp - test_start) / 3600.0,
                "IsAtTestStart": abs(timestamp - test_start) < 1e-9,
                "Fair_Spread": fair_spread,
                "SeedCount": len(schedule),
                "SourceFile": record.get("SourceFile"),
            })
    return pd.DataFrame(rows)


def summarize_seed_timing(events_df, test_start):
    if events_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    group_cols = ["Algorithm", "Budget", "Duration", "RANDOM_SEED"]
    for key, group in events_df.groupby(group_cols, dropna=False):
        times = pd.to_numeric(group["Time"], errors="coerce").dropna()
        rel_minutes = pd.to_numeric(group["RelativeMinutes"], errors="coerce").dropna()
        row = dict(zip(group_cols, key))
        row["SeedCount"] = len(group)
        row["UniqueNodeCount"] = group["Node"].nunique()
        row["Fair_Spread"] = pd.to_numeric(group["Fair_Spread"], errors="coerce").dropna().iloc[0] if pd.to_numeric(group["Fair_Spread"], errors="coerce").notna().any() else np.nan
        row["FirstSeedTime"] = times.min() if not times.empty else np.nan
        row["LastSeedTime"] = times.max() if not times.empty else np.nan
        row["FirstSeedMinutes"] = rel_minutes.min() if not rel_minutes.empty else np.nan
        row["LastSeedMinutes"] = rel_minutes.max() if not rel_minutes.empty else np.nan
        row["MeanSeedMinutes"] = rel_minutes.mean() if not rel_minutes.empty else np.nan
        row["MedianSeedMinutes"] = rel_minutes.median() if not rel_minutes.empty else np.nan
        row["SeedTimeSpanMinutes"] = row["LastSeedMinutes"] - row["FirstSeedMinutes"]
        row["AtStartCount"] = int(group["IsAtTestStart"].sum())
        row["WaitedCount"] = int((group["Time"] > test_start + 1e-9).sum())
        row["WaitedRatio"] = row["WaitedCount"] / float(len(group)) if len(group) else np.nan
        rows.append(row)

    summary = pd.DataFrame(rows)
    paired = paired_timing_summary(summary)
    return summary, paired


def paired_timing_summary(summary_df):
    if summary_df.empty:
        return pd.DataFrame()
    index_cols = ["Budget", "Duration", "RANDOM_SEED"]
    value_cols = [
        "Fair_Spread",
        "SeedCount",
        "LastSeedMinutes",
        "MeanSeedMinutes",
        "MedianSeedMinutes",
        "WaitedCount",
        "WaitedRatio",
    ]
    pivot = summary_df.pivot_table(
        index=index_cols,
        columns="Algorithm",
        values=value_cols,
        aggfunc="last",
    )
    pivot.columns = [f"{metric}_{algorithm}" for metric, algorithm in pivot.columns]
    pivot = pivot.reset_index()

    full = "Full StrDQN"
    no_wait = "w/o WAIT"
    for metric in value_cols:
        a = f"{metric}_{full}"
        b = f"{metric}_{no_wait}"
        if a in pivot.columns and b in pivot.columns:
            pivot[f"{metric}_Full_minus_NoWait"] = pivot[a] - pivot[b]
    return pivot


def activity_histogram(test_edges, bin_minutes):
    times = np.array([float(edge[2]) for edge in test_edges], dtype=float)
    if len(times) == 0:
        return np.array([]), np.array([]), 0.0
    start = float(times.min())
    end = float(times.max())
    bin_seconds = max(float(bin_minutes) * 60.0, 1.0)
    bins = np.arange(start, end + bin_seconds, bin_seconds)
    if len(bins) < 2:
        bins = np.array([start, start + bin_seconds])
    counts, edges = np.histogram(times, bins=bins)
    centers = ((edges[:-1] + edges[1:]) / 2.0 - start) / 3600.0
    width_hours = bin_seconds / 3600.0
    return centers, counts, width_hours


def plot_timing(events_df, test_edges, out_dir, dataset, bin_minutes, highlight_seeds):
    if plt is None:
        print("Plot step skipped: matplotlib is not installed.")
        print(f"Original import error: {MATPLOTLIB_IMPORT_ERROR}")
        return
    if events_df.empty:
        return

    os.makedirs(out_dir, exist_ok=True)
    centers, counts, width_hours = activity_histogram(test_edges, bin_minutes)
    colors = {"Full StrDQN": "#4C78A8", "w/o WAIT": "#F58518"}
    markers = {"Full StrDQN": "o", "w/o WAIT": "x"}
    offsets = {"Full StrDQN": -0.15, "w/o WAIT": 0.15}

    for (budget, duration), group in events_df.groupby(["Budget", "Duration"], dropna=False):
        seeds = sorted(group["RANDOM_SEED"].dropna().astype(int).unique())
        if not seeds:
            continue
        seed_to_y = {seed: idx for idx, seed in enumerate(seeds)}

        fig, (ax_activity, ax_select) = plt.subplots(
            2,
            1,
            figsize=(12, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.6]},
        )

        if len(centers) > 0:
            ax_activity.bar(centers, counts, width=width_hours, color="#9AA0A6", alpha=0.65)
        ax_activity.set_ylabel("Edges / bin")
        ax_activity.set_title(f"{dataset}: seed timing diagnosis, k={budget}, Delta={duration}")
        ax_activity.grid(axis="y", alpha=0.25)

        for algorithm, sub in group.groupby("Algorithm", sort=False):
            y = sub["RANDOM_SEED"].astype(int).map(seed_to_y).astype(float) + offsets.get(algorithm, 0.0)
            sizes = np.where(sub["RANDOM_SEED"].astype(int).isin(highlight_seeds), 70, 36)
            ax_select.scatter(
                sub["RelativeHours"],
                y,
                label=algorithm,
                alpha=0.8,
                s=sizes,
                marker=markers.get(algorithm, "o"),
                color=colors.get(algorithm),
            )

        ax_select.axvline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
        ax_select.set_yticks(list(seed_to_y.values()))
        ax_select.set_yticklabels([str(seed) for seed in seeds])
        ax_select.set_xlabel("Hours after test start")
        ax_select.set_ylabel("RANDOM_SEED")
        ax_select.grid(axis="x", alpha=0.25)
        ax_select.legend(loc="upper right")

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{dataset}_wait_timing_B{budget}_D{duration}.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved {out_path}")


def write_excel(events_df, summary_df, paired_df, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with pd.ExcelWriter(out_path) as writer:
        events_df.to_excel(writer, sheet_name="select_times", index=False)
        summary_df.to_excel(writer, sheet_name="seed_summary", index=False)
        paired_df.to_excel(writer, sheet_name="paired_summary", index=False)
    print(f"Saved wait timing diagnosis to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot and summarize Full StrDQN vs w/o WAIT seed timing.")
    parser.add_argument("--dataset", default="thiers_2012")
    parser.add_argument("--suffix", default="_minimal")
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--budgets", default="")
    parser.add_argument("--durations", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--highlight-seeds", default="0,1")
    parser.add_argument("--bin-minutes", type=float, default=30.0)
    parser.add_argument("--out", default="")
    parser.add_argument("--plot-dir", default="")
    args = parser.parse_args()
    if not args.result_dir:
        args.result_dir = default_result_dir(args.dataset)

    budgets = parse_optional_ints(args.budgets)
    durations = parse_optional_floats(args.durations)
    seeds = parse_optional_ints(args.seeds)
    highlight_seeds = parse_optional_ints(args.highlight_seeds) or set()

    test_edges = load_test_edges(args.dataset)
    if not test_edges:
        raise RuntimeError(f"No test edges found for dataset {args.dataset}")
    test_start = float(test_edges[0][2])

    records = collect_records(args.dataset, args.suffix, args.result_dir, budgets, durations, seeds)
    if not records:
        print("No Full StrDQN / w/o WAIT schedules found. Check --result-dir and --suffix.")
        return

    spread_map = load_fair_spread_map(args.dataset, args.suffix, args.result_dir)
    events_df = records_to_events(records, test_start, spread_map)
    summary_df, paired_df = summarize_seed_timing(events_df, test_start)

    out_path = args.out or os.path.join(
        args.result_dir,
        f"{args.dataset}_wait_timing_diagnosis{args.suffix}.xlsx",
    )
    plot_dir = args.plot_dir or os.path.join(args.result_dir, "diagnostics")

    write_excel(events_df, summary_df, paired_df, out_path)
    plot_timing(events_df, test_edges, plot_dir, args.dataset, args.bin_minutes, highlight_seeds)

    if not paired_df.empty:
        display_cols = [
            col for col in [
                "Budget",
                "Duration",
                "RANDOM_SEED",
                "Fair_Spread_Full_minus_NoWait",
                "LastSeedMinutes_Full_minus_NoWait",
                "WaitedCount_Full_minus_NoWait",
            ] if col in paired_df.columns
        ]
        print("\nPaired timing summary:")
        print(paired_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()

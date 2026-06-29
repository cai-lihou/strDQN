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
    from matplotlib.ticker import MaxNLocator
except ModuleNotFoundError as exc:
    plt = None
    MaxNLocator = None
    MATPLOTLIB_IMPORT_ERROR = exc
else:
    MATPLOTLIB_IMPORT_ERROR = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.result_io import default_result_dir


def parse_optional_ints(value):
    if value is None or str(value).strip() == "":
        return None
    return {int(float(item.strip())) for item in str(value).split(",") if item.strip()}


def parse_int_list(value):
    if value is None or str(value).strip() == "":
        raise ValueError("Expected at least one integer value.")
    return [int(float(item.strip())) for item in str(value).split(",") if item.strip()]


def parse_float_list(value):
    if value is None or str(value).strip() == "":
        raise ValueError("Expected at least one float value.")
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def nearly_equal(series, value, tol=1e-12):
    values = pd.to_numeric(series, errors="coerce")
    return (values - float(value)).abs() <= tol


def load_temporal_edges(dataset):
    path = os.path.join("processed", f"{dataset}_main.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data not found: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    edges = sorted(data["temporal_edges"], key=lambda x: x[2])
    if not edges:
        raise RuntimeError(f"No temporal edges found for dataset {dataset}")
    split_idx = int(len(edges) * 0.7)
    return edges, edges[:split_idx], edges[split_idx:]


def load_seed_events(result_path, budget, duration, seeds):
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Result file not found: {result_path}")
    df = pd.read_excel(result_path)
    required = {"Model", "ActionType", "MAX_BUDGET", "ACTIVATION_DURATION_PCT", "Time"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"{result_path} is missing required columns: {sorted(missing)}")

    mask = (
        (df["Model"].astype(str) == "final") &
        (df["ActionType"].astype(str) == "SELECT") &
        nearly_equal(df["MAX_BUDGET"], budget) &
        nearly_equal(df["ACTIVATION_DURATION_PCT"], duration)
    )
    if seeds is not None:
        if "RANDOM_SEED" not in df.columns:
            raise RuntimeError("--seeds was provided, but RANDOM_SEED is missing from the result file.")
        mask &= df["RANDOM_SEED"].apply(lambda x: int(float(x)) if pd.notna(x) else None).isin(seeds)

    events = df.loc[mask].copy()
    events["Time"] = pd.to_numeric(events["Time"], errors="coerce")
    events = events.dropna(subset=["Time"])
    return events


def make_bins(test_edges, bin_minutes):
    test_times = np.array([float(edge[2]) for edge in test_edges], dtype=float)
    test_start = float(test_times.min())
    test_end = float(test_times.max())
    bin_seconds = max(float(bin_minutes) * 60.0, 1.0)
    bins = np.arange(test_start, test_end + bin_seconds, bin_seconds)
    if len(bins) < 2:
        bins = np.array([test_start, test_start + bin_seconds])
    return bins, test_start, test_end, bin_seconds


def activity_counts(test_edges, bins):
    times = np.array([float(edge[2]) for edge in test_edges], dtype=float)
    counts, _ = np.histogram(times, bins=bins)
    return counts


def seed_counts(seed_events, bins):
    times = seed_events["Time"].to_numpy(dtype=float)
    counts, _ = np.histogram(times, bins=bins)
    return counts


def output_name(dataset, budget, duration):
    duration_token = str(duration).replace(".", "p").replace("-", "m")
    return f"{dataset}_seed_activity_B{budget}_D{duration_token}.png"


def resolve_axis_config(test_edges, bin_minutes_arg, time_unit_arg):
    test_times = np.array([float(edge[2]) for edge in test_edges], dtype=float)
    span_seconds = max(float(test_times.max() - test_times.min()), 1.0)
    span_hours = span_seconds / 3600.0

    if str(bin_minutes_arg).strip().lower() == "auto":
        if span_hours <= 8:
            bin_minutes = 5.0
        elif span_hours <= 24:
            bin_minutes = 15.0
        elif span_hours <= 72:
            bin_minutes = 30.0
        else:
            bin_minutes = 60.0
    else:
        bin_minutes = float(bin_minutes_arg)

    if str(time_unit_arg).strip().lower() == "auto":
        if span_hours <= 8:
            unit_name = "minutes"
            unit_seconds = 60.0
        elif span_hours >= 24 * 10:
            unit_name = "days"
            unit_seconds = 86400.0
        else:
            unit_name = "hours"
            unit_seconds = 3600.0
    else:
        units = {
            "minutes": ("minutes", 60.0),
            "minute": ("minutes", 60.0),
            "min": ("minutes", 60.0),
            "hours": ("hours", 3600.0),
            "hour": ("hours", 3600.0),
            "h": ("hours", 3600.0),
            "days": ("days", 86400.0),
            "day": ("days", 86400.0),
            "d": ("days", 86400.0),
        }
        key = str(time_unit_arg).strip().lower()
        if key not in units:
            raise ValueError("--time-unit must be auto, minutes, hours, or days.")
        unit_name, unit_seconds = units[key]

    return bin_minutes, unit_name, unit_seconds


def plot_alignment(
    dataset,
    budget,
    duration,
    test_edges,
    seed_events,
    out_path,
    bin_minutes_arg,
    time_unit_arg,
):
    if plt is None:
        print("Plot step skipped: matplotlib is not installed.")
        print(f"Original import error: {MATPLOTLIB_IMPORT_ERROR}")
        return

    bin_minutes, unit_name, unit_seconds = resolve_axis_config(test_edges, bin_minutes_arg, time_unit_arg)
    bins, test_start, test_end, bin_seconds = make_bins(test_edges, bin_minutes)
    activity = activity_counts(test_edges, bins)
    seeds = seed_counts(seed_events, bins)
    centers = ((bins[:-1] + bins[1:]) / 2.0 - test_start) / unit_seconds
    width = bin_seconds / unit_seconds

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax_activity = plt.subplots(figsize=(12, 5.8))
    ax_seed = ax_activity.twinx()

    activity_color = "#4C78A8"
    seed_color = "#D62728"
    ax_activity.fill_between(centers, activity, step="mid", color=activity_color, alpha=0.25)
    ax_activity.plot(centers, activity, color=activity_color, linewidth=2.0, label="Network activity")

    nonzero = seeds > 0
    ax_seed.bar(
        centers[nonzero],
        seeds[nonzero],
        width=width * 0.7,
        color=seed_color,
        alpha=0.65,
        label="Seed activations",
    )
    ax_seed.scatter(
        centers[nonzero],
        seeds[nonzero],
        color=seed_color,
        s=35,
        zorder=3,
    )

    x_max = max((test_end - test_start) / unit_seconds, width)
    ax_activity.set_xlim(0.0, x_max)
    ax_activity.set_ylim(0, max(1, activity.max()) * 1.12)
    ax_seed.set_ylim(0, max(1, seeds.max()) * 1.25)

    ax_activity.set_xlabel(f"Time since test start ({unit_name})")
    ax_activity.set_ylabel(f"Interactions per {bin_minutes:g} min")
    ax_seed.set_ylabel(f"Seed activations per {bin_minutes:g} min")
    ax_activity.set_title(f"{dataset}: seed activation vs. network activity (k={budget}, Delta={duration})")
    ax_activity.grid(True, axis="y", alpha=0.25)
    if MaxNLocator is not None:
        ax_activity.xaxis.set_major_locator(MaxNLocator(nbins=10, min_n_ticks=5))

    handles1, labels1 = ax_activity.get_legend_handles_labels()
    handles2, labels2 = ax_seed.get_legend_handles_labels()
    ax_activity.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved {out_path}")
    print(f"Adaptive plot config: x-axis={unit_name}, bin={bin_minutes:g} min")
    print(f"Test edges used: {len(test_edges)}")
    print(f"SELECT rows used: {len(seed_events)}")
    print(f"Seed count in plot bins: {int(seeds.sum())}")
    if int(seeds.sum()) != len(seed_events):
        print("Warning: some SELECT times fell outside the test-edge plotting range.")


def main():
    parser = argparse.ArgumentParser(
        description="Plot StrDQN seed activation timing against temporal network activity."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--suffix", default="_minimal")
    parser.add_argument(
        "--budget",
        default="50",
        help="Seed budget or comma-separated budgets, e.g. 50 or 10,20,30,50.",
    )
    parser.add_argument(
        "--duration",
        default="0.01",
        help="Influence-window percentage or comma-separated values, e.g. 0.01 or 0.001,0.005,0.01.",
    )
    parser.add_argument(
        "--bin-minutes",
        default="auto",
        help="Bin size in minutes, or 'auto'. Default: auto.",
    )
    parser.add_argument(
        "--time-unit",
        default="auto",
        help="X-axis unit: auto, minutes, hours, or days. Default: auto.",
    )
    parser.add_argument("--seeds", default="")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    result_dir = args.result_dir or default_result_dir(args.dataset)
    out_dir = args.out_dir or os.path.join(result_dir, "activity_alignment_plots")
    result_path = os.path.join(result_dir, f"result_{args.dataset}{args.suffix}.xlsx")

    _, _, test_edges = load_temporal_edges(args.dataset)
    budgets = parse_int_list(args.budget)
    durations = parse_float_list(args.duration)
    selected_seeds = parse_optional_ints(args.seeds)

    generated = 0
    skipped = 0
    for budget in budgets:
        for duration in durations:
            seed_events = load_seed_events(
                result_path,
                budget=budget,
                duration=duration,
                seeds=selected_seeds,
            )
            if seed_events.empty:
                skipped += 1
                print("No matching Full StrDQN SELECT rows found.")
                print(f"Checked: {result_path}")
                print(f"Filters: Model=final, ActionType=SELECT, Budget={budget}, Duration={duration}")
                continue

            out_path = os.path.join(out_dir, output_name(args.dataset, budget, duration))
            plot_alignment(
                args.dataset,
                budget,
                duration,
                test_edges,
                seed_events,
                out_path,
                args.bin_minutes,
                args.time_unit,
            )
            generated += 1

    print(f"Finished. Generated plots: {generated}; skipped combinations: {skipped}.")


if __name__ == "__main__":
    main()

import argparse
import os

import pandas as pd

from result_utils import default_result_dir

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    plt = None
    MATPLOTLIB_IMPORT_ERROR = exc
else:
    MATPLOTLIB_IMPORT_ERROR = None

if plt is not None:
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    })


MAIN_ALGORITHMS = [
    "Full StrDQN",
    "Degree",
    "PageRank",
    "CELF",
    "Dynamic RIS",
    "IncInf",
    "S2V-DQN",
    "FINDER",
]


def read_excel_if_exists(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_excel(path)


def plot_main_group(group, dataset, out_dir, duration, filename, title_suffix):
    if group.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo, sub in group.groupby("Algorithm", sort=False):
        sub = sub.sort_values("Budget")
        ax.errorbar(sub["Budget"], sub["mean"], yerr=sub["std"], marker="o", linewidth=2.4,
                    capsize=4, label=algo)
    delta_min = group["DELTA_MINUTES"].dropna().mean() if "DELTA_MINUTES" in group else None
    suffix = f" ({delta_min:.2f} min)" if delta_min is not None and pd.notna(delta_min) else ""
    ax.set_title(f"{dataset}: {title_suffix}, Delta={duration}{suffix}")
    ax.set_xlabel("Budget k")
    ax.set_ylabel("Influence spread")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=300)
    plt.close(fig)


def plot_main(df, dataset, out_dir):
    if df.empty:
        return
    main_set = set(MAIN_ALGORITHMS)
    for duration, group in df.groupby("Duration"):
        core = group[group["Algorithm"].isin(MAIN_ALGORITHMS)].copy()
        if not core.empty:
            core["Algorithm"] = pd.Categorical(core["Algorithm"], categories=MAIN_ALGORITHMS, ordered=True)
            core = core.sort_values(["Algorithm", "Budget"])
            plot_main_group(
                core,
                dataset,
                out_dir,
                duration,
                f"{dataset}_minimal_main_D{duration}.png",
                "influence spread",
            )

        other = group[(~group["Algorithm"].isin(main_set)) | (group["Algorithm"] == "Full StrDQN")].copy()
        if not other.empty:
            other_order = ["Full StrDQN"] + sorted(
                algo for algo in other["Algorithm"].dropna().astype(str).unique()
                if algo != "Full StrDQN"
            )
            other["Algorithm"] = pd.Categorical(other["Algorithm"], categories=other_order, ordered=True)
            other = other.sort_values(["Algorithm", "Budget"])
            plot_main_group(
                other,
                dataset,
                out_dir,
                duration,
                f"{dataset}_minimal_main_other_D{duration}.png",
                "other baselines vs Full StrDQN",
            )


def plot_ablation(df, dataset, out_dir):
    if df.empty:
        return
    order = [
        "Full StrDQN",
        "w/o WAIT",
        "w/o action-biased exploration",
        "Coupled-DQN",
        "w/o WAIT compensation",
    ]
    ablation_algorithms = set(order[1:])
    ablation_rows = df[df["Algorithm"].isin(ablation_algorithms)].copy()
    if ablation_rows.empty:
        print("No ablation variants found. Only Full StrDQN rows are available.")
        return

    for (budget, duration), ablation_group in ablation_rows.groupby(["Budget", "Duration"]):
        full_group = df[
            (df["Algorithm"] == "Full StrDQN") &
            (df["Budget"] == budget) &
            (df["Duration"] == duration)
        ]
        subset = pd.concat([full_group, ablation_group], ignore_index=True)
        if subset.empty:
            continue
        missing = [name for name in order if name not in set(subset["Algorithm"])]
        if missing:
            print(f"Ablation plot B={budget}, Delta={duration}: missing {', '.join(missing)}")
        subset["Algorithm"] = pd.Categorical(subset["Algorithm"], categories=order, ordered=True)
        subset = subset.sort_values("Algorithm")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(subset["Algorithm"], subset["mean"], yerr=subset["std"], capsize=5)
        ax.set_title(f"{dataset}: ablation study, k={budget}, Delta={duration}")
        ax.set_ylabel("Influence spread")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{dataset}_minimal_ablation_B{budget}_D{duration}.png"), dpi=300)
        fig.savefig(os.path.join(out_dir, f"{dataset}_minimal_ablation_D{duration}.png"), dpi=300)
        plt.close(fig)


def plot_sensitivity(df, dataset, out_dir):
    if df.empty:
        return
    for sensitivity, group in df.groupby("Sensitivity"):
        fig, ax = plt.subplots(figsize=(9, 6))
        for duration, sub in group.groupby("Duration"):
            sub = sub.sort_values("SensitivityValue")
            ax.errorbar(sub["SensitivityValue"], sub["mean"], yerr=sub["std"], marker="o",
                        linewidth=2.4, capsize=4, label=f"Delta={duration}")
        ax.set_title(f"{dataset}: sensitivity to {sensitivity}")
        ax.set_xlabel(sensitivity)
        ax.set_ylabel("Influence spread")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{dataset}_minimal_sensitivity_{sensitivity}.png"), dpi=300)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot minimal StrDQN results")
    parser.add_argument("--dataset", default="thiers_2012")
    parser.add_argument("--result-dir", default=None)
    args = parser.parse_args()
    if not args.result_dir:
        args.result_dir = default_result_dir(args.dataset)

    if MATPLOTLIB_IMPORT_ERROR is not None:
        print("Plot step skipped: matplotlib is not installed in this Python environment.")
        print("Install it, then rerun this command:")
        print("  python -m pip install matplotlib")
        print(f"Original import error: {MATPLOTLIB_IMPORT_ERROR}")
        return

    out_dir = os.path.join(args.result_dir, "minimal_plots")
    os.makedirs(out_dir, exist_ok=True)

    main_df = read_excel_if_exists(os.path.join(args.result_dir, f"{args.dataset}_minimal_main_results.xlsx"))
    ablation_df = read_excel_if_exists(os.path.join(args.result_dir, f"{args.dataset}_minimal_ablation_results.xlsx"))
    sensitivity_df = read_excel_if_exists(os.path.join(args.result_dir, f"{args.dataset}_minimal_sensitivity_results.xlsx"))

    plot_main(main_df, args.dataset, out_dir)
    plot_ablation(ablation_df, args.dataset, out_dir)
    plot_sensitivity(sensitivity_df, args.dataset, out_dir)
    print(f"Saved minimal plots to {out_dir}")


if __name__ == "__main__":
    main()

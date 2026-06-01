import argparse
import glob
import os

import numpy as np
import pandas as pd

from result_utils import default_result_dir, format_mean_std, keep_latest_run

try:
    from scipy import stats
except Exception:
    stats = None


def read_dqn_summary(path, algorithm, experiment="main"):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_excel(path)
    required = {"MAX_BUDGET", "ACTIVATION_DURATION_PCT", "Model", "ActionType", "TotalReward"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    df = keep_latest_run(df, ["MAX_BUDGET", "ACTIVATION_DURATION_PCT", "Model", "RANDOM_SEED"])
    df = df[(df["ActionType"] == "DONE") & (df["Model"] == "final")].copy()
    if df.empty:
        return df
    df["Algorithm"] = algorithm
    df["Experiment"] = experiment
    df["Budget"] = df["MAX_BUDGET"]
    df["Duration"] = df["ACTIVATION_DURATION_PCT"]
    df["Spread"] = df["TotalReward"]
    if "RANDOM_SEED" not in df.columns:
        df["RANDOM_SEED"] = df.groupby(["Budget", "Duration"]).cumcount()
    return df


def read_static_peak(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_excel(path)
    if "Spread" not in df.columns and "Fair_Spread" in df.columns:
        df["Spread"] = df["Fair_Spread"]
    if "Budget" not in df.columns and "MAX_BUDGET" in df.columns:
        df["Budget"] = df["MAX_BUDGET"]
    if "Duration" not in df.columns and "ACTIVATION_DURATION_PCT" in df.columns:
        df["Duration"] = df["ACTIVATION_DURATION_PCT"]
    dedupe_cols = ["Algorithm", "Budget", "Duration", "RANDOM_SEED"]
    if all(col in df.columns for col in dedupe_cols):
        df = df.drop_duplicates(subset=dedupe_cols, keep="last")
    df["Experiment"] = "static_peak"
    return df


def read_fair_eval(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_excel(path)
    required = {"Experiment", "Algorithm", "Budget", "Duration", "RANDOM_SEED", "Fair_Spread"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()
    df = df.copy()
    if "FAIR_EVAL_ROUNDS" in df.columns:
        rounds = pd.to_numeric(df["FAIR_EVAL_ROUNDS"], errors="coerce")
        if rounds.notna().any():
            df = df[rounds == rounds.max()].copy()
    df["Spread"] = df["Fair_Spread"]
    dedupe_cols = ["Experiment", "Algorithm", "Budget", "Duration", "RANDOM_SEED"]
    if "Sensitivity" in df.columns or "SensitivityValue" in df.columns:
        for col in ["Sensitivity", "SensitivityValue"]:
            if col not in df.columns:
                df[col] = ""
        dedupe_cols.extend(["Sensitivity", "SensitivityValue"])
    if all(col in df.columns for col in dedupe_cols):
        df = df.drop_duplicates(subset=dedupe_cols, keep="last")
    return df


ABLATION_FILES = {
    "no_wait": "w/o WAIT",
    "no_action_bias": "w/o action-biased exploration",
    "unified": "Coupled-DQN",
    "no_wait_compensation": "w/o WAIT compensation",
}


def read_ablation_results(dataset, suffix, result_dir):
    frames = []
    for mode, label in ABLATION_FILES.items():
        path = os.path.join(result_dir, f"result_{dataset}_{mode}{suffix}.xlsx")
        df = read_dqn_summary(path, label, "ablation")
        if not df.empty:
            df["AblationMode"] = mode
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize(df, group_cols):
    if df.empty:
        return df
    rows = []
    for key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        values = pd.to_numeric(group["Spread"], errors="coerce").dropna()
        row["mean"] = values.mean()
        row["std"] = values.std(ddof=1) if len(values) > 1 else 0.0
        row["n_runs"] = len(values)
        row["mean_std"] = format_mean_std(row["mean"], row["std"])
        if "DELTA_MINUTES" in group.columns:
            row["DELTA_MINUTES"] = pd.to_numeric(group["DELTA_MINUTES"], errors="coerce").dropna().mean()
        if "DELTA_HOURS" in group.columns:
            row["DELTA_HOURS"] = pd.to_numeric(group["DELTA_HOURS"], errors="coerce").dropna().mean()
        rows.append(row)
    return pd.DataFrame(rows)


def add_significance(summary_df, raw_df, baseline_name="Full StrDQN"):
    if raw_df.empty or summary_df.empty or stats is None:
        summary_df["paired_t_p"] = np.nan
        summary_df["wilcoxon_p"] = np.nan
        return summary_df
    out = summary_df.copy()
    out["paired_t_p"] = np.nan
    out["wilcoxon_p"] = np.nan
    for idx, row in out.iterrows():
        algo = row.get("Algorithm")
        if algo == baseline_name:
            continue
        mask_base = (
            (raw_df["Algorithm"] == baseline_name) &
            (raw_df["Budget"] == row["Budget"]) &
            (np.isclose(raw_df["Duration"], row["Duration"]))
        )
        mask_algo = (
            (raw_df["Algorithm"] == algo) &
            (raw_df["Budget"] == row["Budget"]) &
            (np.isclose(raw_df["Duration"], row["Duration"]))
        )
        base = raw_df[mask_base][["RANDOM_SEED", "Spread"]].rename(columns={"Spread": "base"})
        other = raw_df[mask_algo][["RANDOM_SEED", "Spread"]].rename(columns={"Spread": "other"})
        paired = pd.merge(base, other, on="RANDOM_SEED")
        if len(paired) < 2:
            continue
        try:
            out.at[idx, "paired_t_p"] = stats.ttest_rel(paired["base"], paired["other"]).pvalue
        except Exception:
            pass
        try:
            out.at[idx, "wilcoxon_p"] = stats.wilcoxon(paired["base"], paired["other"]).pvalue
        except Exception:
            pass
    return out


def read_sensitivity_files(dataset, suffix, result_dir):
    rows = []
    pattern = os.path.join(result_dir, f"result_{dataset}_sens_*{suffix}.xlsx")
    for path in glob.glob(pattern):
        df = read_dqn_summary(path, "Full StrDQN", experiment="sensitivity")
        if df.empty:
            continue
        name = os.path.basename(path)
        if "_pwait_" in name:
            df["Sensitivity"] = "P_WAIT"
            df["SensitivityValue"] = df["FORCE_WAIT_PROB"]
        elif "_lambda_" in name:
            df["Sensitivity"] = "lambda"
            df["SensitivityValue"] = df["WAIT_REWARD_COEF"]
        elif "_delta" in name:
            df["Sensitivity"] = "Delta"
            df["SensitivityValue"] = df["Duration"]
        else:
            df["Sensitivity"] = "unknown"
            df["SensitivityValue"] = np.nan
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Aggregate minimal StrDQN experiment results")
    parser.add_argument("--dataset", default="thiers_2012")
    parser.add_argument("--suffix", default="_minimal")
    parser.add_argument("--result-dir", default=None)
    args = parser.parse_args()
    if not args.result_dir:
        args.result_dir = default_result_dir(args.dataset)

    full_path = os.path.join(args.result_dir, f"result_{args.dataset}{args.suffix}.xlsx")
    static_path = os.path.join(args.result_dir, f"{args.dataset}_static_peak{args.suffix}.xlsx")
    fair_path = os.path.join(args.result_dir, f"{args.dataset}_fair_eval{args.suffix}.xlsx")

    fair_eval = read_fair_eval(fair_path)
    if not fair_eval.empty:
        print(f"Using fair evaluation results from {fair_path}")
        main_raw = fair_eval[fair_eval["Experiment"] == "main"].copy()
        full_for_ablation = main_raw[main_raw["Algorithm"] == "Full StrDQN"].copy()
        ablation_variants = fair_eval[fair_eval["Experiment"] == "ablation"].copy()
        ablation_raw = pd.concat([full_for_ablation, ablation_variants], ignore_index=True)
        sensitivity_raw = fair_eval[fair_eval["Experiment"] == "sensitivity"].copy()
        if sensitivity_raw.empty:
            print("Fair evaluation has no sensitivity rows; falling back to original sensitivity values.")
            sensitivity_raw = read_sensitivity_files(args.dataset, args.suffix, args.result_dir)
    else:
        print("Fair evaluation file not found or empty; falling back to original result values.")
        full = read_dqn_summary(full_path, "Full StrDQN", "main")
        ablations = read_ablation_results(args.dataset, args.suffix, args.result_dir)
        static_peak = read_static_peak(static_path)
        main_raw = pd.concat([full, static_peak], ignore_index=True)
        ablation_raw = pd.concat([full, ablations], ignore_index=True)
        sensitivity_raw = read_sensitivity_files(args.dataset, args.suffix, args.result_dir)

    main_summary = summarize(main_raw, ["Algorithm", "Budget", "Duration"])
    main_summary = add_significance(main_summary, main_raw)
    ablation_summary = summarize(ablation_raw, ["Algorithm", "Budget", "Duration"])
    ablation_summary = add_significance(ablation_summary, ablation_raw)
    sensitivity_summary = summarize(sensitivity_raw, ["Sensitivity", "SensitivityValue", "Budget", "Duration"])

    main_out = os.path.join(args.result_dir, f"{args.dataset}_minimal_main_results.xlsx")
    ablation_out = os.path.join(args.result_dir, f"{args.dataset}_minimal_ablation_results.xlsx")
    sensitivity_out = os.path.join(args.result_dir, f"{args.dataset}_minimal_sensitivity_results.xlsx")

    main_summary.to_excel(main_out, index=False)
    ablation_summary.to_excel(ablation_out, index=False)
    sensitivity_summary.to_excel(sensitivity_out, index=False)
    print(f"Saved {main_out}")
    print(f"Saved {ablation_out}")
    print(f"Saved {sensitivity_out}")


if __name__ == "__main__":
    main()

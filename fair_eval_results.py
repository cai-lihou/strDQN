import argparse
import ast
import glob
import os
import pickle
from datetime import datetime

import pandas as pd

from IC import t2EICModel
from result_utils import default_result_dir, delta_metadata, keep_latest_run


ABLATION_FILES = {
    "no_wait": "w/o WAIT",
    "no_action_bias": "w/o action-biased exploration",
    "unified": "Coupled-DQN",
    "no_wait_compensation": "w/o WAIT compensation",
}

KEY_COLS = [
    "Experiment",
    "Algorithm",
    "Budget",
    "Duration",
    "RANDOM_SEED",
    "FAIR_EVAL_ROUNDS",
    "Sensitivity",
    "SensitivityValue",
]


def read_excel(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_excel(path)
    except Exception as exc:
        print(f"Cannot read {path}: {exc}")
        return pd.DataFrame()


def load_test_edges(dataset):
    data_path = os.path.join("processed", f"{dataset}_main.pkl")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    all_edges = sorted(data["temporal_edges"], key=lambda x: x[2])
    split_idx = int(len(all_edges) * 0.7)
    return all_edges[split_idx:]


def safe_float(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=None):
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def parse_seed_schedule(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            value = ast.literal_eval(text)
        except Exception:
            return []

    if isinstance(value, tuple) and len(value) == 2:
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []

    seeds = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        node = safe_int(item[0])
        timestamp = safe_float(item[1])
        if node is None or timestamp is None:
            continue
        seeds.append((node, timestamp))
    return sorted(seeds, key=lambda x: x[1])


def schedule_from_select_rows(group):
    rows = group.sort_values("Time")
    seeds = []
    for _, row in rows.iterrows():
        seed_id = str(row.get("SeedID", "")).strip()
        if seed_id.upper() in {"", "NONE", "NAN", "SUMMARY"}:
            continue
        node = safe_int(seed_id)
        timestamp = safe_float(row.get("Time"))
        if node is None or timestamp is None:
            continue
        seeds.append((node, timestamp))
    return seeds


def first_value(group, col, default=None):
    if col not in group.columns:
        return default
    values = group[col].dropna()
    if values.empty:
        return default
    return values.iloc[0]


def first_done_value(path, col):
    df = read_excel(path)
    if df.empty or col not in df.columns:
        return None
    if "Model" in df.columns and "ActionType" in df.columns:
        done = df[
            (df["Model"].astype(str) == "final") &
            (df["ActionType"].astype(str).str.upper() == "DONE")
        ]
        if not done.empty:
            return first_value(done, col)
    return first_value(df, col)


def read_dqn_schedules(path, algorithm, experiment, ablation_mode=None, sensitivity=None, sensitivity_value=None):
    df = read_excel(path)
    required = {"MAX_BUDGET", "ACTIVATION_DURATION_PCT", "Model", "ActionType", "SeedID", "Time"}
    if df.empty or not required.issubset(df.columns):
        return []

    if "RANDOM_SEED" not in df.columns:
        df["RANDOM_SEED"] = 0

    df = keep_latest_run(df, ["MAX_BUDGET", "ACTIVATION_DURATION_PCT", "Model", "RANDOM_SEED"])
    df = df[
        (df["Model"].astype(str) == "final") &
        (df["ActionType"].astype(str).str.upper() == "SELECT")
    ].copy()
    if df.empty:
        return []

    records = []
    group_cols = ["MAX_BUDGET", "ACTIVATION_DURATION_PCT", "RANDOM_SEED"]
    for (budget, duration, random_seed), group in df.groupby(group_cols, dropna=False):
        seeds = schedule_from_select_rows(group)
        if not seeds:
            continue
        record = {
            "Experiment": experiment,
            "Algorithm": algorithm,
            "Budget": safe_int(budget),
            "Duration": safe_float(duration),
            "MAX_BUDGET": safe_int(budget),
            "ACTIVATION_DURATION_PCT": safe_float(duration),
            "RANDOM_SEED": safe_int(random_seed, 0),
            "EVAL_SEED": safe_int(first_value(group, "EVAL_SEED"), safe_int(random_seed, 0)),
            "ACTIVATION_PROB": safe_float(first_value(group, "ACTIVATION_PROB"), 0.5),
            "Seeds": str(seeds),
            "SeedCount": len(seeds),
            "SourceFile": os.path.basename(path),
            "SourceKind": "dqn_select_rows",
        }
        if ablation_mode:
            record["AblationMode"] = ablation_mode
        if sensitivity is not None:
            record["Sensitivity"] = sensitivity
            record["SensitivityValue"] = sensitivity_value
        records.append(record)
    return records


def read_baseline_schedules(path):
    df = read_excel(path)
    required = {"Algorithm", "Seeds", "RANDOM_SEED"}
    if df.empty or not required.issubset(df.columns):
        return []

    if "Budget" not in df.columns and "MAX_BUDGET" in df.columns:
        df["Budget"] = df["MAX_BUDGET"]
    if "Duration" not in df.columns and "ACTIVATION_DURATION_PCT" in df.columns:
        df["Duration"] = df["ACTIVATION_DURATION_PCT"]
    if "Budget" not in df.columns or "Duration" not in df.columns:
        return []

    dedupe_cols = ["Algorithm", "Budget", "Duration", "RANDOM_SEED"]
    df = df.drop_duplicates(subset=dedupe_cols, keep="last")

    records = []
    for _, row in df.iterrows():
        seeds = parse_seed_schedule(row.get("Seeds"))
        if not seeds:
            continue
        budget = safe_int(row.get("Budget"))
        duration = safe_float(row.get("Duration"))
        random_seed = safe_int(row.get("RANDOM_SEED"), 0)
        records.append({
            "Experiment": "main",
            "Algorithm": str(row.get("Algorithm")),
            "Budget": budget,
            "Duration": duration,
            "MAX_BUDGET": budget,
            "ACTIVATION_DURATION_PCT": duration,
            "RANDOM_SEED": random_seed,
            "EVAL_SEED": random_seed,
            "ACTIVATION_PROB": safe_float(row.get("ACTIVATION_PROB"), 0.5),
            "Seeds": str(seeds),
            "SeedCount": len(seeds),
            "SourceFile": os.path.basename(path),
            "SourceKind": "baseline_seeds",
        })
    return records


def completed_keys(path, rounds):
    df = read_excel(path)
    if df.empty:
        return set()
    for col in KEY_COLS:
        if col not in df.columns:
            df[col] = ""
    df = df[pd.to_numeric(df["FAIR_EVAL_ROUNDS"], errors="coerce") == int(rounds)]
    keys = set()
    for _, row in df.iterrows():
        keys.add((
            str(row["Experiment"]),
            str(row["Algorithm"]),
            safe_int(row["Budget"]),
            safe_float(row["Duration"]),
            safe_int(row["RANDOM_SEED"], 0),
            safe_int(row["FAIR_EVAL_ROUNDS"], rounds),
            str(row.get("Sensitivity", "")),
            safe_float(row.get("SensitivityValue")),
        ))
    return keys


def record_key(record, rounds):
    return (
        str(record["Experiment"]),
        str(record["Algorithm"]),
        safe_int(record["Budget"]),
        safe_float(record["Duration"]),
        safe_int(record["RANDOM_SEED"], 0),
        int(rounds),
        str(record.get("Sensitivity", "")),
        safe_float(record.get("SensitivityValue")),
    )


def evaluate_records(records, test_edges, rounds, out_path, resume):
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    done = completed_keys(out_path, rounds) if resume else set()
    rows = []

    for idx, record in enumerate(records, start=1):
        key = record_key(record, rounds)
        if key in done:
            print(f"[skip] fair_eval {key[0]}/{key[1]} B={key[2]} D={key[3]} seed={key[4]}")
            continue

        seeds = parse_seed_schedule(record["Seeds"])
        duration = safe_float(record["Duration"])
        random_seed = safe_int(record["RANDOM_SEED"], 0)
        activation_prob = safe_float(record.get("ACTIVATION_PROB"), 0.5)
        model = t2EICModel(
            test_edges,
            activation_prob=activation_prob,
            activation_duration_pct=duration,
            random_state=random_seed,
        )
        spread = model.simulate(seeds, num_rounds=rounds, use_cache=False)
        meta = delta_metadata(duration, test_edges)
        row = dict(record)
        row.update(meta)
        row["Fair_Spread"] = spread
        row["Spread"] = spread
        row["FAIR_EVAL_ROUNDS"] = int(rounds)
        row["FAIR_EVAL_TIME"] = run_time
        rows.append(row)
        print(f"[{idx}/{len(records)}] fair_eval {record['Algorithm']} B={record['Budget']} "
              f"D={record['Duration']} seed={record['RANDOM_SEED']} -> {spread:.4f}")

    if resume and os.path.exists(out_path):
        old_df = read_excel(out_path)
        out_df = pd.concat([old_df, pd.DataFrame(rows)], ignore_index=True) if rows else old_df
    else:
        out_df = pd.DataFrame(rows)

    if not out_df.empty:
        for col in KEY_COLS:
            if col not in out_df.columns:
                out_df[col] = ""
        out_df = out_df.drop_duplicates(subset=KEY_COLS, keep="last")
    out_df.to_excel(out_path, index=False)
    print(f"Saved fair evaluation to {out_path} ({len(rows)} new rows)")


def collect_records(dataset, suffix, result_dir):
    records = []
    main_path = os.path.join(result_dir, f"result_{dataset}{suffix}.xlsx")
    records.extend(read_dqn_schedules(main_path, "Full StrDQN", "main"))

    baseline_path = os.path.join(result_dir, f"{dataset}_static_peak{suffix}.xlsx")
    records.extend(read_baseline_schedules(baseline_path))

    for mode, label in ABLATION_FILES.items():
        path = os.path.join(result_dir, f"result_{dataset}_{mode}{suffix}.xlsx")
        records.extend(read_dqn_schedules(path, label, "ablation", ablation_mode=mode))

    pattern = os.path.join(result_dir, f"result_{dataset}_sens_*{suffix}.xlsx")
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        if "_pwait_" in name:
            sensitivity = "P_WAIT"
            sensitivity_value = safe_float(first_done_value(path, "FORCE_WAIT_PROB"))
        elif "_lambda_" in name:
            sensitivity = "lambda"
            sensitivity_value = safe_float(first_done_value(path, "WAIT_REWARD_COEF"))
        elif "_delta" in name:
            sensitivity = "Delta"
            sensitivity_value = None
        else:
            sensitivity = "unknown"
            sensitivity_value = None

        sens_records = read_dqn_schedules(
            path,
            "Full StrDQN",
            "sensitivity",
            sensitivity=sensitivity,
            sensitivity_value=sensitivity_value,
        )
        if sensitivity == "Delta":
            for record in sens_records:
                record["SensitivityValue"] = record["Duration"]
        records.extend(sens_records)

    return records


def main():
    parser = argparse.ArgumentParser(description="Fairly re-evaluate saved seed schedules with strict TW-IC")
    parser.add_argument("--dataset", default="thiers_2012")
    parser.add_argument("--suffix", default="_minimal")
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()
    if not args.result_dir:
        args.result_dir = default_result_dir(args.dataset)

    os.makedirs(args.result_dir, exist_ok=True)
    out_path = os.path.join(args.result_dir, f"{args.dataset}_fair_eval{args.suffix}.xlsx")
    test_edges = load_test_edges(args.dataset)
    records = collect_records(args.dataset, args.suffix, args.result_dir)
    if not records:
        print("No seed schedules found for fair evaluation.")
        return
    evaluate_records(records, test_edges, args.rounds, out_path, args.resume)


if __name__ == "__main__":
    main()

import math
import os
import time

import pandas as pd


RUN_ID_COL = "RUN_ID"
RUN_TIME_COL = "RUN_TIME"
RUN_ORDER_COL = "__RUN_ORDER__"


class FileLock:
    def __init__(self, path, timeout=3600, poll_interval=0.5, stale_after=86400):
        self.lock_path = f"{path}.lock"
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.stale_after = stale_after
        self.fd = None

    def __enter__(self):
        start = time.time()
        while True:
            try:
                self.fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(self.fd, f"pid={os.getpid()} time={time.time()}\n".encode("utf-8"))
                return self
            except FileExistsError:
                if self._is_stale():
                    try:
                        os.remove(self.lock_path)
                        continue
                    except OSError:
                        pass
                if time.time() - start > self.timeout:
                    raise TimeoutError(f"Timed out waiting for lock: {self.lock_path}")
                time.sleep(self.poll_interval)

    def __exit__(self, exc_type, exc, tb):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            os.remove(self.lock_path)
        except FileNotFoundError:
            pass

    def _is_stale(self):
        try:
            return time.time() - os.path.getmtime(self.lock_path) > self.stale_after
        except OSError:
            return False


def default_result_dir(dataset):
    return f"./result_new_{dataset}"


def safe_filename_token(value):
    text = str(value or "")
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in text)


def append_excel_locked(path, new_df):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with FileLock(path):
        if os.path.exists(path):
            out_df = pd.concat([pd.read_excel(path), new_df], ignore_index=True)
        else:
            out_df = new_df
        out_df.to_excel(path, index=False)


def write_excel_locked(path, df):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with FileLock(path):
        df.to_excel(path, index=False)


def attach_run_metadata(df, run_id, run_time):
    df = df.copy()
    df[RUN_ID_COL] = run_id
    df[RUN_TIME_COL] = run_time
    return df


def parse_int_list(value, default):
    if value is None or str(value).strip() == "":
        return list(default)
    return [int(float(item.strip())) for item in str(value).split(",") if item.strip()]


def parse_float_list(value, default):
    if value is None or str(value).strip() == "":
        return list(default)
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def delta_metadata(duration_pct, temporal_edges):
    if temporal_edges:
        start_ts = float(temporal_edges[0][2])
        end_ts = float(temporal_edges[-1][2])
        total_seconds = max(0.0, end_ts - start_ts)
    else:
        total_seconds = 0.0
    delta_seconds = total_seconds * float(duration_pct)
    return {
        "DELTA_PCT": float(duration_pct) * 100.0,
        "DELTA_SECONDS": delta_seconds,
        "DELTA_MINUTES": delta_seconds / 60.0,
        "DELTA_HOURS": delta_seconds / 3600.0,
    }


def format_mean_std(mean_value, std_value, precision=2):
    if mean_value is None or (isinstance(mean_value, float) and math.isnan(mean_value)):
        return ""
    if std_value is None or (isinstance(std_value, float) and math.isnan(std_value)):
        std_value = 0.0
    return f"{mean_value:.{precision}f} \u00b1 {std_value:.{precision}f}"


def assign_run_order(df, group_cols):
    """Add a monotonic run order per experiment group.

    New result files use RUN_ID.  For older appended Excel files without RUN_ID,
    infer runs by treating DONE/SUMMARY rows as the end of one run.
    """
    df = df.copy()
    missing = [col for col in group_cols if col not in df.columns]
    if missing or df.empty:
        df[RUN_ORDER_COL] = 0
        return df

    if RUN_ID_COL in df.columns and df[RUN_ID_COL].notna().any():
        df[RUN_ORDER_COL] = -1
        for _, group in df.groupby(group_cols, sort=False, dropna=False):
            run_ids = group[RUN_ID_COL].fillna("__legacy__").astype(str)
            order_map = {run_id: order for order, run_id in enumerate(pd.unique(run_ids))}
            df.loc[group.index, RUN_ORDER_COL] = run_ids.map(order_map).astype(int).values
        return df

    df[RUN_ORDER_COL] = 0
    for _, group in df.groupby(group_cols, sort=False, dropna=False):
        run_order = 0
        for idx, row in group.iterrows():
            df.at[idx, RUN_ORDER_COL] = run_order
            action = str(row.get("ActionType", "")).upper()
            seed_id = str(row.get("SeedID", "")).upper()
            if action == "DONE" or seed_id == "SUMMARY":
                run_order += 1
    return df


def keep_latest_run(df, group_cols):
    if df.empty:
        return df.copy()
    ordered = assign_run_order(df, group_cols)
    latest = ordered.groupby(group_cols, sort=False, dropna=False)[RUN_ORDER_COL].transform("max")
    return ordered[ordered[RUN_ORDER_COL] == latest].drop(columns=[RUN_ORDER_COL])

import argparse
import gc
import math
import os
import pickle
import random
import time
from collections import defaultdict
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

from IC import t2EICModel
from result_utils import default_result_dir, delta_metadata, parse_float_list, parse_int_list

BASELINE_ALGORITHMS = {
    "Degree",
    "Degree + peak-time",
    "PageRank",
    "PageRank + peak-time",
    "CELF",
    "CELF + peak-time",
    "TIM",
    "TIM + peak-time",
    "IMM",
    "IMM + peak-time",
    "Dynamic CI",
    "Dynamic RIS",
    "IncInf",
    "FINDER",
    "S2V-DQN",
}

try:
    from finder_pytorch import train_and_select as finder_train_and_select
except Exception:
    finder_train_and_select = None

try:
    from s2v_im_baseline import IMEnv as S2VEnv
    from s2v_im_baseline import QFunction as S2VQFunction
    from s2v_im_baseline import DEVICE as S2V_DEVICE
    from s2v_im_baseline import LEARNING_RATE as S2V_LR
    from s2v_im_baseline import GAMMA as S2V_GAMMA
    from s2v_im_baseline import NUM_EPISODES as S2V_EPISODES
except Exception:
    S2VEnv = None
    S2VQFunction = None
    S2V_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S2V_LR = 1e-3
    S2V_GAMMA = 0.99
    S2V_EPISODES = 50


def build_static_graph(temporal_edges, num_nodes, activation_prob):
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    for u, v, _ in temporal_edges:
        if graph.has_edge(u, v):
            graph[u][v]["weight"] += 1
        else:
            graph.add_edge(u, v, weight=1)
    for _, _, data in graph.edges(data=True):
        weight = data.get("weight", 1)
        data["prob"] = 1.0 - (1.0 - activation_prob) ** weight
    return graph


def run_degree(graph, budget):
    return [node for node, _ in sorted(graph.degree(weight="weight"), key=lambda x: x[1], reverse=True)[:budget]]


def run_pagerank(graph, budget):
    scores = nx.pagerank(graph, weight="weight", max_iter=200)
    return [node for node, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:budget]]


def static_ic_spread(graph, seeds, rounds):
    total = 0
    for _ in range(rounds):
        active = set(seeds)
        frontier = set(seeds)
        while frontier:
            new_frontier = set()
            for u in frontier:
                for v in graph.neighbors(u):
                    if v not in active and random.random() < graph[u][v].get("prob", 0.0):
                        new_frontier.add(v)
            active.update(new_frontier)
            frontier = new_frontier
        total += len(active)
    return total / max(1, rounds)


def run_celf(graph, budget, rounds):
    candidates = [node for node, _ in sorted(graph.degree(weight="weight"), key=lambda x: x[1], reverse=True)[:200]]
    seeds = []
    current = 0.0
    for _ in range(budget):
        best_node = None
        best_gain = -float("inf")
        for node in candidates:
            if node in seeds:
                continue
            spread = static_ic_spread(graph, seeds + [node], rounds)
            gain = spread - current
            if gain > best_gain:
                best_gain = gain
                best_node = node
        if best_node is None:
            break
        seeds.append(best_node)
        current += max(0.0, best_gain)
    return seeds


class RRSetSolver:
    def __init__(self, graph, num_nodes):
        self.num_nodes = num_nodes
        self.rev_adj = defaultdict(list)
        for u, v, data in graph.edges(data=True):
            prob = data.get("prob", 0.0)
            if prob > 0:
                self.rev_adj[u].append((v, prob))
                self.rev_adj[v].append((u, prob))

    def generate_rr_set(self):
        start = random.randint(0, self.num_nodes - 1)
        rr_set = {start}
        queue = [start]
        while queue:
            current = queue.pop(0)
            for prev, prob in self.rev_adj[current]:
                if prev not in rr_set and random.random() < prob:
                    rr_set.add(prev)
                    queue.append(prev)
        return rr_set

    def select(self, budget, theta):
        rr_sets = [self.generate_rr_set() for _ in range(theta)]
        seeds = []
        uncovered = rr_sets
        for _ in range(budget):
            freq = defaultdict(int)
            for rr in uncovered:
                for node in rr:
                    freq[node] += 1
            if not freq:
                break
            best = max(freq, key=freq.get)
            seeds.append(best)
            uncovered = [rr for rr in uncovered if best not in rr]
        return seeds


def run_tim(graph, budget, num_nodes):
    solver = RRSetSolver(graph, num_nodes)
    theta = max(5000, num_nodes * 2)
    return solver.select(budget, theta)


def run_imm(graph, budget, num_nodes, epsilon=0.5):
    solver = RRSetSolver(graph, num_nodes)
    alpha = math.sqrt(math.log(max(num_nodes, 2)) + math.log(2))
    beta = math.sqrt((1 - 1 / math.e) * math.log(max(num_nodes, 2)))
    theta = int(2 * num_nodes * ((1 - 1 / math.e) * alpha + beta) ** 2 / max(epsilon ** 2, 1e-9) / 10)
    theta = min(max(theta, 5000), 100000)
    return solver.select(budget, theta)


def estimate_peak_time_from_train(train_edges, test_edges, bin_size_seconds):
    if not test_edges:
        return 0.0
    if not train_edges:
        return float(test_edges[0][2])
    train_start = float(train_edges[0][2])
    train_end = float(train_edges[-1][2])
    test_start = float(test_edges[0][2])
    test_end = float(test_edges[-1][2])
    train_duration = max(1.0, train_end - train_start)
    test_duration = max(1.0, test_end - test_start)
    rel_times = np.array([float(edge[2]) - train_start for edge in train_edges])
    bins = np.arange(0.0, train_duration + bin_size_seconds, bin_size_seconds)
    if len(bins) < 2:
        bins = np.array([0.0, train_duration])
    counts, edges = np.histogram(rel_times, bins=bins)
    peak_idx = int(np.argmax(counts)) if len(counts) else 0
    peak_center = (edges[peak_idx] + edges[peak_idx + 1]) / 2.0
    peak_fraction = min(max(peak_center / train_duration, 0.0), 1.0)
    return test_start + peak_fraction * test_duration


def calculate_dynamic_degrees(num_nodes, temporal_edges):
    time_neighbors = defaultdict(lambda: defaultdict(set))
    for u, v, t in temporal_edges:
        time_neighbors[t][u].add(v)
        time_neighbors[t][v].add(u)
    timestamps = sorted(time_neighbors.keys())
    dynamic_degree = np.zeros(num_nodes)
    temporal_neighbors = defaultdict(set)
    for node in range(num_nodes):
        for idx, ts in enumerate(timestamps):
            current_neighbors = time_neighbors[ts][node]
            if not current_neighbors:
                continue
            temporal_neighbors[node].update(current_neighbors)
            prev_neighbors = set() if idx == 0 else time_neighbors[timestamps[idx - 1]][node]
            union = prev_neighbors.union(current_neighbors)
            diff = prev_neighbors.difference(current_neighbors)
            if union:
                dynamic_degree[node] += (len(diff) / len(union)) * len(current_neighbors)
    return dynamic_degree, temporal_neighbors


def dynamic_ci(num_nodes, temporal_edges, dynamic_degree, budget, l_param=20):
    dball = defaultdict(set)
    sorted_edges = sorted(temporal_edges, key=lambda x: x[2])
    for root in range(num_nodes):
        earliest = {root: -1}
        for u, v, ts in sorted_edges:
            for src, dst in ((u, v), (v, u)):
                if src in earliest:
                    if earliest[src] == -1:
                        earliest[src] = ts
                    duration = ts - earliest[src]
                    if duration <= l_param:
                        if dst not in earliest:
                            earliest[dst] = ts
                        if duration == l_param:
                            dball[root].add(dst)
    score = np.zeros(num_nodes)
    for node in range(num_nodes):
        score[node] = max(0, dynamic_degree[node] - 1) * sum(max(0, dynamic_degree[u] - 1) for u in dball[node])
    return [int(x) for x in np.argsort(score)[-budget:][::-1]]


def dynamic_ris(num_nodes, temporal_edges, budget, activation_prob, theta=500, d_param=20):
    rr_sets = []
    for _ in range(theta):
        root = random.randint(0, num_nodes - 1)
        sampled = [edge for edge in temporal_edges if random.random() < activation_prob]
        sampled.sort(key=lambda x: x[2], reverse=True)
        rr = {root}
        latest = {root: float("inf")}
        for src, dst, ts in sampled:
            for n1, n2 in ((src, dst), (dst, src)):
                if n2 in rr:
                    if latest[n2] == float("inf"):
                        latest[n2] = ts
                    if latest[n2] - ts <= d_param:
                        rr.add(n1)
                        if n1 not in latest or ts > latest[n1]:
                            latest[n1] = ts
        rr_sets.append(rr)
    seeds = []
    uncovered = rr_sets
    for _ in range(budget):
        freq = defaultdict(int)
        for rr in uncovered:
            for node in rr:
                freq[node] += 1
        if not freq:
            break
        best = max(freq, key=freq.get)
        seeds.append(int(best))
        uncovered = [rr for rr in uncovered if best not in rr]
    return seeds


def dynamic_incinf(num_nodes, train_edges, budget, activation_prob, num_snapshots=5):
    sorted_edges = sorted(train_edges, key=lambda x: x[2])
    chunk_size = max(1, len(sorted_edges) // max(1, num_snapshots))
    current_seeds = []
    global_adj = defaultdict(set)

    def local_gain(node, existing):
        return sum(1 for neighbor in global_adj[node] if neighbor not in existing)

    for snap in range(num_snapshots):
        start = snap * chunk_size
        end = len(sorted_edges) if snap == num_snapshots - 1 else min(len(sorted_edges), (snap + 1) * chunk_size)
        delta_edges = sorted_edges[start:end]
        delta_degree = defaultdict(int)
        for u, v, _ in delta_edges:
            global_adj[u].add(v)
            global_adj[v].add(u)
            delta_degree[u] += 1
            delta_degree[v] += 1

        if snap == 0:
            scores = {node: len(global_adj[node]) for node in range(num_nodes)}
            selected = set()
            for _ in range(budget):
                candidate = max(scores, key=scores.get)
                current_seeds.append(candidate)
                selected.add(candidate)
                scores.pop(candidate, None)
                for neighbor in global_adj[candidate]:
                    if neighbor in scores and neighbor not in selected:
                        scores[neighbor] -= activation_prob
        else:
            candidates = sorted(delta_degree.keys(), key=lambda x: delta_degree[x], reverse=True)[:budget * 2]
            candidates = [node for node in candidates if node not in current_seeds]
            for candidate in candidates:
                if not current_seeds:
                    break
                seed_gains = [(local_gain(seed, set(current_seeds) - {seed}), seed) for seed in current_seeds]
                weakest_gain, weakest = min(seed_gains, key=lambda x: x[0])
                candidate_gain = local_gain(candidate, set(current_seeds) - {weakest})
                if candidate_gain > weakest_gain:
                    current_seeds.remove(weakest)
                    current_seeds.append(candidate)

    while len(current_seeds) < budget:
        candidate = random.randint(0, num_nodes - 1)
        if candidate not in current_seeds:
            current_seeds.append(candidate)
    return [int(node) for node in current_seeds[:budget]]


def read_existing_results(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_excel(path)
    except Exception as exc:
        print(f"[resume] Cannot read {path}: {exc}. Baselines will be recomputed.")
        return pd.DataFrame()


def numeric_match(series, value, tol=1e-9):
    values = pd.to_numeric(series, errors="coerce")
    return (values - float(value)).abs() <= tol


def baseline_combo_completed(existing_df, random_seed, budget, duration, expected_algorithms):
    required = {"Algorithm", "Budget", "Duration", "RANDOM_SEED"}
    if existing_df.empty or not required.issubset(existing_df.columns):
        return False
    mask = (
        numeric_match(existing_df["RANDOM_SEED"], random_seed) &
        numeric_match(existing_df["Budget"], budget) &
        numeric_match(existing_df["Duration"], duration)
    )
    present = set(existing_df.loc[mask, "Algorithm"].astype(str))
    return expected_algorithms.issubset(present)


def write_baseline_results(path, existing_df, new_rows, resume):
    new_df = pd.DataFrame(new_rows)
    if resume and not existing_df.empty:
        out_df = pd.concat([existing_df, new_df], ignore_index=True) if not new_df.empty else existing_df.copy()
    else:
        out_df = new_df

    dedupe_cols = ["Algorithm", "Budget", "Duration", "RANDOM_SEED"]
    if not out_df.empty and all(col in out_df.columns for col in dedupe_cols):
        out_df = out_df.drop_duplicates(subset=dedupe_cols, keep="last")
    out_df.to_excel(path, index=False)


def run_s2v_dqn(graph, num_nodes, budget):
    if S2VEnv is None or S2VQFunction is None:
        return [], 0.0
    env = S2VEnv(graph, num_nodes, budget)
    q_net = S2VQFunction(node_dim=2, hidden_dim=64).to(S2V_DEVICE)
    optimizer = optim.Adam(q_net.parameters(), lr=S2V_LR)
    epsilon = 1.0
    start_time = time.time()
    for _ in range(S2V_EPISODES):
        state = env.reset()
        done = False
        while not done:
            valid_mask = state[:, 0] == 0
            if random.random() < epsilon:
                candidates = torch.nonzero(valid_mask).flatten().cpu().numpy()
                action = int(random.choice(candidates))
            else:
                with torch.no_grad():
                    q_vals = q_net(state, env.edge_index, env.edge_w).flatten()
                    q_vals[~valid_mask] = -1e9
                    action = int(q_vals.argmax().item())
            next_state, reward, done = env.step(action)
            with torch.no_grad():
                q_next = q_net(next_state, env.edge_index, env.edge_w).flatten()
                q_next[next_state[:, 0] == 1] = -1e9
                target = reward + S2V_GAMMA * q_next.max() * (0 if done else 1)
            pred = q_net(state, env.edge_index, env.edge_w).flatten()[action]
            target = target.clone().detach() if torch.is_tensor(target) else torch.tensor(target, device=S2V_DEVICE)
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = next_state
        epsilon = max(0.05, epsilon * 0.95)

    state = env.reset()
    seeds = []
    with torch.no_grad():
        for _ in range(budget):
            q_vals = q_net(state, env.edge_index, env.edge_w).flatten()
            valid_mask = state[:, 0] == 0
            q_vals[~valid_mask] = -1e9
            action = int(q_vals.argmax().item())
            seeds.append(action)
            state, _, _ = env.step(action)
    return seeds, time.time() - start_time


def main():
    parser = argparse.ArgumentParser(description="Baseline suite with static, peak-time, temporal, and RL baselines")
    parser.add_argument("--dataset", default="thiers_2012")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--budgets", default="10,20,30,50")
    parser.add_argument("--durations", default="0.001,0.005,0.01")
    parser.add_argument("--activation-prob", type=float, default=0.5)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--celf-rounds", type=int, default=20)
    parser.add_argument("--bin-size-seconds", type=float, default=3600.0)
    parser.add_argument("--result-suffix", default="_minimal")
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Append only missing baseline seed/budget/duration combinations.")
    args = parser.parse_args()
    if not args.result_dir:
        args.result_dir = default_result_dir(args.dataset)

    data_path = f"./processed/{args.dataset}_main.pkl"
    out_path = os.path.join(args.result_dir, f"{args.dataset}_static_peak{args.result_suffix}.xlsx")
    os.makedirs(args.result_dir, exist_ok=True)
    existing_df = read_existing_results(out_path) if args.resume else pd.DataFrame()

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    temporal_edges = sorted(data["temporal_edges"], key=lambda x: x[2])
    split_idx = int(len(temporal_edges) * 0.7)
    train_edges = temporal_edges[:split_idx]
    test_edges = temporal_edges[split_idx:]
    num_nodes = data["num_nodes"]
    graph = build_static_graph(train_edges, num_nodes, args.activation_prob)
    peak_time = estimate_peak_time_from_train(train_edges, test_edges, args.bin_size_seconds)
    test_start_time = float(test_edges[0][2]) if test_edges else 0.0
    dynamic_degree, _ = calculate_dynamic_degrees(num_nodes, train_edges)

    budgets = parse_int_list(args.budgets, [10, 20, 30, 50])
    durations = parse_float_list(args.durations, [0.001, 0.005, 0.01])
    seeds = parse_int_list(args.seeds, list(range(10)))
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []

    static_algorithms = {
        "Degree": lambda k: run_degree(graph, k),
        "PageRank": lambda k: run_pagerank(graph, k),
        "CELF": lambda k: run_celf(graph, k, args.celf_rounds),
        "TIM": lambda k: run_tim(graph, k, num_nodes),
        "IMM": lambda k: run_imm(graph, k, num_nodes),
    }
    expected_algorithms = set(BASELINE_ALGORITHMS)
    if finder_train_and_select is None:
        expected_algorithms.discard("FINDER")

    for random_seed in seeds:
        random.seed(random_seed)
        np.random.seed(random_seed)
        for budget in budgets:
            if args.resume and all(baseline_combo_completed(existing_df, random_seed, budget, duration,
                                                            expected_algorithms)
                                   for duration in durations):
                print(f"[skip] baseline: seed={random_seed}, budget={budget}, all durations complete")
                continue

            selected = {}
            timings = {}
            for name, selector in static_algorithms.items():
                start = time.time()
                selected[name] = selector(budget)[:budget]
                timings[name] = time.time() - start
                selected[f"{name} + peak-time"] = selected[name]
                timings[f"{name} + peak-time"] = timings[name]

            temporal_start = time.time()
            selected["Dynamic CI"] = dynamic_ci(num_nodes, train_edges, dynamic_degree, budget)
            timings["Dynamic CI"] = time.time() - temporal_start
            temporal_start = time.time()
            selected["Dynamic RIS"] = dynamic_ris(num_nodes, train_edges, budget, args.activation_prob)
            timings["Dynamic RIS"] = time.time() - temporal_start
            temporal_start = time.time()
            selected["IncInf"] = dynamic_incinf(num_nodes, train_edges, budget, args.activation_prob)
            timings["IncInf"] = time.time() - temporal_start
            if finder_train_and_select is not None:
                selected["FINDER"], timings["FINDER"] = finder_train_and_select(graph, budget)
            selected["S2V-DQN"], timings["S2V-DQN"] = run_s2v_dqn(graph, num_nodes, budget)

            for duration in durations:
                if args.resume and baseline_combo_completed(existing_df, random_seed, budget, duration,
                                                            expected_algorithms):
                    print(f"[skip] baseline: seed={random_seed}, budget={budget}, duration={duration}")
                    continue
                eval_model = t2EICModel(test_edges, activation_prob=args.activation_prob,
                                        activation_duration_pct=duration, random_state=random_seed)
                meta = delta_metadata(duration, test_edges)
                for name, nodes in selected.items():
                    schedule_time = peak_time if name.endswith("+ peak-time") else test_start_time
                    schedule_name = "peak-time" if name.endswith("+ peak-time") else "test-start"
                    scheduled = [(int(node), float(schedule_time)) for node in nodes]
                    spread = eval_model.simulate(scheduled, num_rounds=args.rounds, use_cache=False)
                    rows.append({
                        "Algorithm": name,
                        "MAX_BUDGET": budget,
                        "Budget": budget,
                        "ACTIVATION_DURATION_PCT": duration,
                        "Duration": duration,
                        "Spread": spread,
                        "Fair_Spread": spread,
                        "Seeds": str(scheduled),
                        "ScheduledTime": schedule_time,
                        "Schedule": schedule_name,
                        "RANDOM_SEED": random_seed,
                        "ACTIVATION_PROB": args.activation_prob,
                        "InferenceTime": timings[name],
                        "RUN_ID": f"{args.dataset}_static_peak_seed{random_seed}",
                        "RUN_TIME": run_time,
                        **meta,
                    })
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_baseline_results(out_path, existing_df, rows, args.resume)
    if rows:
        print(f"Saved baseline suite to {out_path} ({len(rows)} new rows)")
    else:
        print(f"No missing baseline rows. Kept {out_path}")


if __name__ == "__main__":
    main()

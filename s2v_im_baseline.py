import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import networkx as nx
import random
import pandas as pd
from collections import defaultdict
from result_utils import default_result_dir

# ==============================================================================
# 0. 閰嶇疆涓庝緷璧?
# ==============================================================================
try:
    from IC import t2EICModel
except ImportError:
    print("Warning: IC.py not found; TW-IC evaluation is unavailable.")
    t2EICModel = None

# --- 瀹為獙璁剧疆 (蹇呴』涓?DQN 涓ユ牸涓€鑷? ---
DATA_NAME = 'thiers_2012'
DATA_PATH = f'./processed/{DATA_NAME}_main.pkl'
RESULT_FILE = os.path.join(default_result_dir(DATA_NAME), f'{DATA_NAME}_s2v_fair.xlsx')

TRAIN_SPLIT_RATIO = 0.7  # 70% 璁粌, 30% 娴嬭瘯
BUDGETS = [10, 20, 30, 50]
DURATIONS = [0.001, 0.005, 0.01]

# --- 绠楁硶鍙傛暟 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTIVATION_PROB = 0.5
MC_ROUNDS_TRAIN = 5
TW_IC_ROUNDS_TEST = 20  # [淇敼] 澧炲姞鍒?20 浠ュ噺灏戞柟宸?
LEARNING_RATE = 1e-3
GAMMA = 0.99
NUM_EPISODES = 50
BATCH_SIZE = 32


# ==============================================================================
# 1. 鏍稿績宸ュ叿鍑芥暟
# ==============================================================================
def scatter_add(src, index, dim=0, dim_size=None):
    if dim_size is None:
        dim_size = index.max() + 1
    size = list(src.size())
    size[dim] = dim_size
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    target_index = index.view(-1, 1).expand_as(src)
    return out.scatter_add_(dim, target_index, src)


def build_static_graph_from_temporal(temporal_edges, num_nodes):
    """浠庢椂搴忚竟鏋勫缓闈欐€佹鐜囧浘 (浠呯敤浜?S2V 璁粌)"""
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edge_cnt = defaultdict(int)

    for u, v, _ in temporal_edges:
        a, b = sorted((u, v))
        edge_cnt[(a, b)] += 1

    for (u, v), count in edge_cnt.items():
        prob = 1.0 - (1.0 - ACTIVATION_PROB) ** count
        G.add_edge(u, v, prob=prob)

    return G


# ==============================================================================
# 2. S2V-DQN 缃戠粶缁撴瀯
# ==============================================================================
class S2VLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(S2VLayer, self).__init__()
        self.lin_node = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_neig = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_edge = nn.Linear(1, out_dim, bias=False)

    def forward(self, h, edge_index, edge_w):
        h_self = self.lin_node(h)
        w_emb = F.relu(self.lin_edge(edge_w))
        src, dst = edge_index[0], edge_index[1]
        aggr_h = scatter_add(h[src], dst, dim=0, dim_size=h.size(0))
        aggr_h = self.lin_neig(aggr_h)
        aggr_w = scatter_add(w_emb, dst, dim=0, dim_size=h.size(0))
        return F.relu(h_self + aggr_h + aggr_w)


class QFunction(nn.Module):
    def __init__(self, node_dim, hidden_dim, T=3):
        super(QFunction, self).__init__()
        self.T = T
        self.s2v_layers = nn.ModuleList()
        self.s2v_layers.append(S2VLayer(node_dim, hidden_dim))
        for _ in range(T - 1):
            self.s2v_layers.append(S2VLayer(hidden_dim, hidden_dim))
        self.lin_1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.lin_2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_w):
        h = x
        for i in range(self.T):
            h = self.s2v_layers[i](h, edge_index, edge_w)
        g_emb = torch.sum(h, dim=0, keepdim=True).repeat(h.size(0), 1)
        cat_emb = torch.cat([h, g_emb], dim=1)
        q_val = self.lin_2(F.relu(self.lin_1(cat_emb)))
        return q_val


# ==============================================================================
# 3. 鐜绫?(淇敼涓烘帴鏀?NetworkX 鍥惧璞?
# ==============================================================================
class IMEnv:
    def __init__(self, G_static, num_nodes, budget):
        self.G_static = G_static
        self.num_nodes = num_nodes
        self.budget = budget
        self.setup_graph_tensors()
        self.reset()

    def setup_graph_tensors(self):
        u_list, v_list, w_list = [], [], []
        for u, v, data in self.G_static.edges(data=True):
            prob = data.get('prob', 0.0)
            u_list.extend([u, v])
            v_list.extend([v, u])
            w_list.extend([prob, prob])

        self.edge_index = torch.tensor([u_list, v_list], dtype=torch.long).to(DEVICE)
        self.edge_w = torch.tensor(w_list, dtype=torch.float32).view(-1, 1).to(DEVICE)

        degrees = np.array([val for _, val in self.G_static.degree()])
        max_deg = degrees.max() + 1e-5
        self.norm_degrees = torch.tensor(degrees / max_deg, dtype=torch.float32).view(-1, 1).to(DEVICE)

    def reset(self):
        self.selected = set()
        self.current_step = 0
        self.x_state = torch.zeros(self.num_nodes, 1).to(DEVICE)
        self.features = torch.cat([self.x_state, self.norm_degrees], dim=1)
        return self.features

    def step(self, action_node):
        self.selected.add(action_node)
        self.x_state[action_node] = 1.0
        self.features = torch.cat([self.x_state, self.norm_degrees], dim=1)
        reward = 0
        # 绠€鍗曠殑涓€闃堕偦灞呭鍔变綔涓?Proxy
        if action_node in self.G_static:
            for v in self.G_static[action_node]:
                if v not in self.selected:
                    reward += self.G_static[action_node][v]['prob']
        self.current_step += 1
        done = (self.current_step >= self.budget)
        return self.features, reward, done


# ==============================================================================
# 4. 涓荤▼搴?
# ==============================================================================
def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    # 1. 鍔犺浇骞跺垏鍒嗘暟鎹?
    print(f"Loading and splitting data (Ratio {TRAIN_SPLIT_RATIO})...")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    all_edges = data['temporal_edges']
    num_nodes = data['num_nodes']

    split_idx = int(len(all_edges) * TRAIN_SPLIT_RATIO)
    train_edges = all_edges[:split_idx]
    test_edges = all_edges[split_idx:]
    test_start_time = test_edges[0][2]

    print(f"Train Edges: {len(train_edges)} | Test Edges: {len(test_edges)}")

    # 2. 鏋勫缓璁粌鐢ㄧ殑闈欐€佸浘 (鍩轰簬鍓?70%)
    G_train = build_static_graph_from_temporal(train_edges, num_nodes)

    # 鍒濆鍖栫幆澧?
    # 娉ㄦ剰: 鐜鍙煡閬?G_train锛屽畬鍏ㄧ湅涓嶅埌娴嬭瘯闆?
    env = IMEnv(G_train, num_nodes, budget=10)

    all_results = []
    print(f"\n=== S2V-DQN Experiment Start (Fair Split) ===")

    for k in BUDGETS:
        print(f"\n[Training Phase] Budget k={k}")
        env.budget = k

        q_net = QFunction(node_dim=2, hidden_dim=64).to(DEVICE)
        optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)

        # --- 璁粌寰幆 ---
        epsilon = 1.0
        for i_ep in range(NUM_EPISODES):
            state = env.reset()
            done = False
            while not done:
                valid_mask = (state[:, 0] == 0)
                if random.random() < epsilon:
                    candidates = torch.nonzero(valid_mask).flatten().cpu().numpy()
                    action = random.choice(candidates)
                else:
                    with torch.no_grad():
                        q_vals = q_net(state, env.edge_index, env.edge_w).flatten()
                        q_vals[~valid_mask] = -1e9
                        action = q_vals.argmax().item()

                next_state, reward, done = env.step(action)

                with torch.no_grad():
                    q_next = q_net(next_state, env.edge_index, env.edge_w).flatten()
                    q_next[next_state[:, 0] == 1] = -1e9
                    target = reward + GAMMA * q_next.max() * (0 if done else 1)

                q_pred = q_net(state, env.edge_index, env.edge_w).flatten()[action]
                loss = F.mse_loss(q_pred, torch.tensor(target).to(DEVICE))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                state = next_state
            epsilon = max(0.05, epsilon * 0.95)

        # --- 閫夌偣闃舵 (鍩轰簬璁粌濂界殑妯″瀷鍦?G_train 涓婇€夌偣) ---
        print(f"  > Selecting {k} seeds...")
        state = env.reset()
        seeds = []
        with torch.no_grad():
            for _ in range(k):
                q_vals = q_net(state, env.edge_index, env.edge_w).flatten()
                valid_mask = (state[:, 0] == 0)
                q_vals[~valid_mask] = -1e9
                action = q_vals.argmax().item()
                seeds.append(action)
                state, _, _ = env.step(action)

        # === 璇勪及闃舵 (鍦?Test Set 涓婅繘琛? ===
        if t2EICModel:
            for duration in DURATIONS:
                print(f"    [Evaluation] Duration={duration}...", end="")

                # 璇勪及妯″瀷鍙姞杞芥祴璇曢泦
                eval_model = t2EICModel(
                    temporal_edges=test_edges,
                    activation_prob=ACTIVATION_PROB,
                    activation_duration_pct=duration
                )

                # 绉嶅瓙缁熶竴鍦ㄦ祴璇曢泦璧峰鏃跺埢婵€娲?
                seeds_with_time = [(u, test_start_time) for u in seeds]

                spread = eval_model.simulate(seeds_with_time, num_rounds=TW_IC_ROUNDS_TEST)
                print(f" Spread={spread:.2f}")

                all_results.append({
                    "Budget": k,
                    "Duration": duration,
                    "Algorithm": "S2V-DQN",
                    "Spread": spread,
                    "Seeds": str(seeds)
                })

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_excel(RESULT_FILE, index=False)
        print(f"\nSaved to: {RESULT_FILE}")


if __name__ == "__main__":
    main()

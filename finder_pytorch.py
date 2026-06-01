import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import pandas as pd
import time
from collections import defaultdict
from result_utils import default_result_dir

# ==============================================================================
# 0. 配置区域
# ==============================================================================
DATA_NAME = 'thiers_2012'
DATA_PATH = f'./processed/{DATA_NAME}_main.pkl'
RESULT_FILE = os.path.join(default_result_dir(DATA_NAME), f'{DATA_NAME}_finder_fair.xlsx')

TRAIN_SPLIT_RATIO = 0.7
BUDGETS = [10, 20, 30, 50]
DURATIONS = [0.001, 0.005, 0.01]
ACTIVATION_PROB = 0.5
TW_IC_ROUNDS = 20  # [修改] 增加评估轮数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from IC import t2EICModel
except ImportError:
    print("Error: 未找到 IC.py")
    exit()


# ==============================================================================
# 1. 核心工具函数
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
    """从时序数据构建无向加权图 (FINDER通常用于无向图)"""
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for u, v, _ in temporal_edges:
        if G.has_edge(u, v):
            G[u][v]['weight'] += 1
        else:
            G.add_edge(u, v, weight=1)
    return G


# ==============================================================================
# 2. FINDER 模型定义
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


class FINDER_Net(nn.Module):
    def __init__(self, node_dim, hidden_dim, T=3):
        super(FINDER_Net, self).__init__()
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
        g_emb = torch.max(h, dim=0, keepdim=True)[0].repeat(h.size(0), 1)
        cat_emb = torch.cat([h, g_emb], dim=1)
        return self.lin_2(F.relu(self.lin_1(cat_emb)))


# ==============================================================================
# 3. 训练与选点逻辑
# ==============================================================================
def train_and_select(G_static, k_budget):
    num_nodes = G_static.number_of_nodes()
    u_list, v_list, w_list = [], [], []
    for u, v, data in G_static.edges(data=True):
        w = data.get('weight', 1.0)
        norm_w = min(w, 10.0) / 10.0
        u_list.extend([u, v])
        v_list.extend([v, u])
        w_list.extend([norm_w, norm_w])

    edge_index = torch.tensor([u_list, v_list], dtype=torch.long).to(DEVICE)
    edge_w = torch.tensor(w_list, dtype=torch.float32).view(-1, 1).to(DEVICE)

    degrees = np.array([d for n, d in G_static.degree(weight='weight')])
    norm_degrees = torch.tensor(degrees / (degrees.max() + 1e-5), dtype=torch.float32).view(-1, 1).to(DEVICE)

    model = FINDER_Net(node_dim=2, hidden_dim=64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 预训练
    model.train()
    current_state = torch.zeros(num_nodes, 1).to(DEVICE)
    input_feat = torch.cat([current_state, norm_degrees], dim=1)

    for _ in range(100):
        optimizer.zero_grad()
        pred = model(input_feat, edge_index, edge_w)
        loss = F.mse_loss(pred, norm_degrees)
        loss.backward()
        optimizer.step()

    # 选点
    model.eval()
    seeds = []
    current_state = torch.zeros(num_nodes, 1).to(DEVICE)

    start_time = time.time()
    with torch.no_grad():
        for _ in range(k_budget):
            input_feat = torch.cat([current_state, norm_degrees], dim=1)
            q_vals = model(input_feat, edge_index, edge_w).flatten()
            if seeds: q_vals[seeds] = -1e9
            best_node = q_vals.argmax().item()
            seeds.append(best_node)
            current_state[best_node] = 1.0

    return seeds, time.time() - start_time


# ==============================================================================
# 4. 主流程
# ==============================================================================
def main():
    if not os.path.exists(DATA_PATH):
        print("Error: Data file not found.")
        return

    # 1. 加载并切分数据
    print(f"Loading and splitting data (Ratio {TRAIN_SPLIT_RATIO})...")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    all_edges = data['temporal_edges']
    num_nodes = data['num_nodes']
    split_idx = int(len(all_edges) * TRAIN_SPLIT_RATIO)
    train_edges = all_edges[:split_idx]
    test_edges = all_edges[split_idx:]
    test_start_time = test_edges[0][2]

    # 2. 构建训练图
    G_train = build_static_graph_from_temporal(train_edges, num_nodes)

    all_results = []
    print(f"\n=== RL-FINDER Experiment Start (Fair Split) ===")

    for k in BUDGETS:
        print(f"\n[Selection Phase] Budget k={k}")
        seeds, t_inf = train_and_select(G_train, k)
        print(f"  > Seeds: {seeds}")

        for duration in DURATIONS:
            print(f"    [Evaluation] Duration={duration}...", end="")

            # 评估器只加载测试集
            eval_model = t2EICModel(
                temporal_edges=test_edges,
                activation_prob=ACTIVATION_PROB,
                activation_duration_pct=duration
            )

            # 种子在测试集起始时刻激活
            seeds_input = [(u, test_start_time) for u in seeds]
            spread = eval_model.simulate(seeds_input, num_rounds=TW_IC_ROUNDS)
            print(f" Spread={spread:.2f}")

            all_results.append({
                "Budget": k,
                "Duration": duration,
                "Algorithm": "RL-FINDER",
                "Spread": spread,
                "InferenceTime": t_inf,
                "Seeds": str(seeds)
            })

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_excel(RESULT_FILE, index=False)
        print(f"\nSaved to: {RESULT_FILE}")


if __name__ == "__main__":
    main()

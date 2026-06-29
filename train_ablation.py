import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import logging
import os
import pickle
import sys
import gc
import contextlib
import io
from collections import deque
import pandas as pd
import argparse
from datetime import datetime
from result_io import (
    append_excel_locked,
    attach_run_metadata,
    default_result_dir,
    delta_metadata,
    parse_float_list,
    parse_int_list,
    safe_filename_token,
)

# 引入原TGAN项目依赖
try:
    from strict_tw_ic import t2EICModel
    from module import TGAN
    from graph import NeighborFinder
except ImportError:
    print("Warning: Custom modules not found. Code requires strict_tw_ic, module, graph files.")

# ==============================================================================
# 1. 全局配置与参数解析
# ==============================================================================
parser = argparse.ArgumentParser(description='Run Streaming DQN with Ablation')
parser.add_argument('--dataset', type=str, default='thiers_2012', help='数据集名称')
parser.add_argument('--gpu', type=str, default='0', help='指定 GPU ID')
parser.add_argument('--ablation', type=str, default='none',
                    choices=['none', 'no_wait', 'unified', 'no_action_bias', 'no_wait_compensation'],
                    help='消融实验模式: none(完整), no_wait(无等待), unified(统一网络), no_action_bias(无动作偏置)')

parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--eval-seed', type=int, default=2024, help='common random seed for evaluation')
parser.add_argument('--budgets', type=str, default='10,20,30,50', help='comma-separated budgets')
parser.add_argument('--durations', type=str, default='0.001,0.005,0.01', help='comma-separated Delta fractions')
parser.add_argument('--episodes', type=int, default=None, help='override training episodes')
parser.add_argument('--force-wait-prob', type=float, default=None, help='WAIT exploration probability')
parser.add_argument('--wait-reward-coef', type=float, default=None, help='WAIT compensation coefficient')
parser.add_argument('--wait-reward-cap', type=float, default=None, help='WAIT compensation cap')
parser.add_argument('--activation-prob', type=float, default=None, help='TW-IC activation probability')
parser.add_argument('--result-suffix', type=str, default='', help='suffix appended to result file name')
parser.add_argument('--result-dir', type=str, default=None, help='directory for result Excel files')
args, _ = parser.parse_known_args()

# 全局变量赋值
DATA_NAME = args.dataset
ABLATION_MODE = args.ablation
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

print(f"Starting task: dataset={DATA_NAME}, mode={ABLATION_MODE}, GPU={args.gpu}")

# === 路径配置 (自动添加消融后缀) ===
suffix_ablation = "" if ABLATION_MODE == 'none' else f"_{ABLATION_MODE}"
PROCESSED_DATA_PATH = f'./processed/{DATA_NAME}_main.pkl'
RESULT_DIR = args.result_dir or default_result_dir(DATA_NAME)
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
# 结果文件加上后缀，防止覆盖
RESULT_FILE = os.path.join(RESULT_DIR, f"result_{DATA_NAME}{suffix_ablation}{args.result_suffix}.xlsx")

# TGAN模型路径通常不变，因为是预训练好的
TGAN_MODEL_PATH = f'./saved_models/-attn-prod-{DATA_NAME}.pth'

# 默认参数（会在循环中被覆盖）
TIME_WINDOW_DURATION = 20
MAX_BUDGET = 20
ACTIVATION_DURATION_PCT = 0.01
SCALE_FACTOR = 1.0
ACTIVATION_PROB = args.activation_prob if args.activation_prob is not None else 0.5

# 训练参数
EPISODES = args.episodes if args.episodes is not None else 1500
BATCH_SIZE = 32
LR = 1e-4
GAMMA = 0.999
HIDDEN_DIM = 128
UPDATE_FREQ = 5

# 经验池
MEMORY_CAPACITY = 20000
N_STEP = 3
FORCE_WAIT_PROB = 0.2
WAIT_COMPENSATION_COEF = 0.01
WAIT_COMPENSATION_CAP = 1.0
if args.force_wait_prob is not None:
    FORCE_WAIT_PROB = args.force_wait_prob
if args.wait_reward_coef is not None:
    WAIT_COMPENSATION_COEF = args.wait_reward_coef
if args.wait_reward_cap is not None:
    WAIT_COMPENSATION_CAP = args.wait_reward_cap
RANDOM_SEED = args.seed
EVAL_SEED = args.eval_seed

# 探索
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 1500

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with contextlib.redirect_stdout(io.StringIO()):
    import train_strdqn as main_components

MainStrictStreamingGraphEnv = main_components.StrictStreamingGraphEnv
MainTGANInferenceWrapper = main_components.TGANInferenceWrapper


def sync_main_component_globals():
    main_components.TIME_WINDOW_DURATION = TIME_WINDOW_DURATION
    main_components.MAX_BUDGET = MAX_BUDGET
    main_components.SCALE_FACTOR = SCALE_FACTOR
    main_components.WAIT_COMPENSATION_COEF = WAIT_COMPENSATION_COEF
    main_components.WAIT_COMPENSATION_CAP = WAIT_COMPENSATION_CAP
    main_components.ACTIVATION_PROB = ACTIVATION_PROB


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


# ==============================================================================
# 2. 基础组件 (保持不变)
# ==============================================================================
class TGANInferenceWrapper:
    def __init__(self, data, model_path, node_dim, time_dim, device):
        self.device = device
        self.num_nodes = data['num_nodes']
        empty_adj_list = [[] for _ in range(self.num_nodes + 1)]
        self.ngh_finder = NeighborFinder(empty_adj_list, uniform=False)
        self.tgan = TGAN(self.ngh_finder, data['n_feat'], data['e_feat'], num_layers=2,
                         use_time='postime', agg_method='attn', attn_mode='prod',
                         seq_len=10, n_head=2, drop_out=0.0, node_dim=node_dim, time_dim=time_dim)

        if os.path.exists(model_path):
            try:
                self.tgan.load_state_dict(torch.load(model_path, map_location=device), strict=False)
            except Exception as e:
                print(f"TGAN loading warning (strict=False used): {e}")
        self.tgan.to(device)
        self.tgan.eval()

    def update_graph(self, new_edges):
        for src, dst, idx, ts in new_edges:
            self.ngh_finder.adj_list[src].append((dst, idx, ts))
            self.ngh_finder.adj_list[dst].append((src, idx, ts))

    def get_embeddings(self, current_timestamp):
        all_node_ids = np.arange(self.num_nodes)
        batch_size = 5000
        embeddings_list = []
        with torch.no_grad():
            for i in range(0, self.num_nodes, batch_size):
                batch_nodes = all_node_ids[i: i + batch_size]
                batch_times = np.full(len(batch_nodes), current_timestamp, dtype=np.float64)
                batch_emb = self.tgan.tem_conv(batch_nodes, batch_times, self.tgan.num_layers, 5)
                embeddings_list.append(batch_emb)
        if not embeddings_list: return torch.empty(0, self.tgan.feat_dim).to(self.device)
        return torch.cat(embeddings_list, dim=0)


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def add(self, p, data):
        tree_idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(tree_idx, p)
        self.write += 1
        if self.write >= self.capacity: self.write = 0
        if self.n_entries < self.capacity: self.n_entries += 1

    def total_p(self):
        return self.tree[0]

    def get(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.abs_err_upper = 1.0
        self.epsilon = 0.01

    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def push(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0: max_p = self.abs_err_upper
        self.tree.add(max_p, transition)

    def sample(self, batch_size):
        batch, idxs, priorities = [], [], []
        segment = self.tree.total_p() / batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            while data is None or data == 0:
                s = random.uniform(0, self.tree.total_p())
                idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)
        sampling_probabilities = np.array(priorities) / self.tree.total_p()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities + 1e-10, -self.beta)
        is_weights /= is_weights.max()
        return batch, idxs, torch.FloatTensor(is_weights).to(device)

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = self._get_priority(error)
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries


# ==============================================================================
# 4. Environment (保持不变)
# ==============================================================================
class StreamingGraphEnv:
    def __init__(self, data, tgan_wrapper, propagation_model_cls,
                 activation_prob, activation_duration_pct, wait_reward=0,
                 enable_crn=False, crn_seed=42, mc_rounds=100, global_min_node_id=None):
        self.data = data
        self.tgan = tgan_wrapper
        self.temporal_edges = data['temporal_edges']
        self.num_nodes = data['num_nodes']
        self.prop_model = propagation_model_cls(
            temporal_edges=data['temporal_edges'],
            activation_prob=activation_prob,
            activation_duration_pct=activation_duration_pct
        )
        self.wait_reward = wait_reward
        self.enable_crn = enable_crn
        self.crn_seed = crn_seed
        self.mc_rounds = mc_rounds
        self.cached_crn_spread = None

        self.selected_node_ids = set()
        self.current_window_active_nodes = set()

        if global_min_node_id is not None:
            self.min_node_id = global_min_node_id
        else:
            all_src = [e[0] for e in self.temporal_edges]
            all_dst = [e[1] for e in self.temporal_edges]
            self.min_node_id = min(min(all_src), min(all_dst))

        self.edge_cursor = 0
        if len(self.temporal_edges) > 0:
            self.current_time = self.temporal_edges[0][2]
            self.start_timestamp = self.temporal_edges[0][2]
            self.end_timestamp = self.temporal_edges[-1][2]
        else:
            self.current_time = 0
            self.start_timestamp = 0
            self.end_timestamp = 1

        self.budget_left = MAX_BUDGET
        self.seeds = []
        self.total_duration = self.end_timestamp - self.start_timestamp
        self.cached_embeddings = None

    def reset(self, random_start=True):
        self.budget_left = MAX_BUDGET
        self.seeds = []
        self.selected_node_ids = set()
        self.current_window_active_nodes = set()
        self.cached_crn_spread = None

        safe_margin = 1000
        max_start_idx = max(0, len(self.temporal_edges) - safe_margin)

        if random_start and max_start_idx > 0:
            self.edge_cursor = random.randint(0, max_start_idx)
        else:
            self.edge_cursor = 0

        if self.edge_cursor < len(self.temporal_edges):
            self.current_time = self.temporal_edges[self.edge_cursor][2]
        else:
            self.current_time = self.start_timestamp

        empty_adj = [[] for _ in range(self.num_nodes + 1)]
        self.tgan.ngh_finder.adj_list = empty_adj
        if hasattr(self.tgan.ngh_finder, 'init_off_set'):
            self.tgan.ngh_finder.init_off_set(empty_adj)

        self.cached_embeddings = None
        self._advance_time()
        return self._get_state()

    def _advance_time(self):
        if self.edge_cursor >= len(self.temporal_edges): return False

        start_idx = self.edge_cursor
        start_timestamp = self.temporal_edges[start_idx][2]
        cutoff_time = start_timestamp + TIME_WINDOW_DURATION
        end_idx = start_idx
        while end_idx < len(self.temporal_edges):
            if self.temporal_edges[end_idx][2] < cutoff_time:
                end_idx += 1
            else:
                break

        new_batch = []
        batch_max_ts = start_timestamp
        emb_matrix_size = self.tgan.tgan.node_raw_embed.weight.shape[0]
        self.current_window_active_nodes.clear()

        for i in range(start_idx, end_idx):
            u_raw, v_raw, ts = self.temporal_edges[i]
            u, v = u_raw - self.min_node_id, v_raw - self.min_node_id
            if 0 <= u < emb_matrix_size and 0 <= v < emb_matrix_size:
                new_batch.append((u, v, i + 1, ts))
                batch_max_ts = ts
                self.current_window_active_nodes.add(u)
                self.current_window_active_nodes.add(v)

        if new_batch:
            self.tgan.update_graph(new_batch)
            self.current_time = batch_max_ts
            self.cached_embeddings = None
        else:
            self.current_time = cutoff_time
            self.cached_embeddings = None

        self.edge_cursor = end_idx
        return True

    def _get_state(self):
        if self.cached_embeddings is None:
            self.cached_embeddings = self.tgan.get_embeddings(self.current_time)
        embeddings = self.cached_embeddings
        mask = torch.ones(self.num_nodes + 1, dtype=torch.bool, device=self.tgan.device)
        if self.current_window_active_nodes: mask[list(self.current_window_active_nodes)] = False
        if self.selected_node_ids: mask[list(self.selected_node_ids)] = True
        mask[self.num_nodes] = False
        budget_ratio = self.budget_left / MAX_BUDGET
        duration_val = max(1.0, self.total_duration)
        progress = min(max((self.current_time - self.start_timestamp) / duration_val, 0.0), 1.0)
        global_feat = torch.tensor([progress, budget_ratio], dtype=torch.float32, device=self.tgan.device)
        return embeddings, mask, global_feat

    def step(self, action):
        done = False
        reward = 0.0
        if action == self.num_nodes:
            prev_time = self.current_time
            if not self._advance_time(): done = True
            actual_time_skipped = self.current_time - prev_time
            steps_ratio = max(1.0, actual_time_skipped / TIME_WINDOW_DURATION)
            steps_ratio = min(steps_ratio, 50.0)
            reward = self.wait_reward * steps_ratio
        else:
            node_idx = int(action)
            if node_idx in self.selected_node_ids:
                reward = -1.0
            else:
                self.budget_left -= 1
                if self.enable_crn:
                    current_seeds_arg = [(s[0] + self.min_node_id, s[1]) for s in self.seeds]
                    new_node_arg = (node_idx + self.min_node_id, self.current_time)
                    if self.cached_crn_spread is None:
                        seed_everything(self.crn_seed)
                        if not self.seeds:
                            self.cached_crn_spread = 0.0
                        else:
                            self.cached_crn_spread = self.prop_model.simulate(current_seeds_arg,
                                                                              num_rounds=self.mc_rounds)
                    seed_everything(self.crn_seed)
                    new_seeds_input = current_seeds_arg + [new_node_arg]
                    spread_new = self.prop_model.simulate(new_seeds_input, num_rounds=self.mc_rounds)
                    gain = spread_new - self.cached_crn_spread
                    reward = max(0.0, gain) / SCALE_FACTOR
                    self.seeds.append((node_idx, self.current_time))
                    self.selected_node_ids.add(node_idx)
                    self.cached_crn_spread = spread_new
                else:
                    gain = self.prop_model.calculate_marginal_gain(
                        [(s[0] + self.min_node_id, s[1]) for s in self.seeds],
                        node_idx + self.min_node_id, self.current_time)
                    reward = max(0.0, gain) / SCALE_FACTOR
                    self.seeds.append((node_idx, self.current_time))
                    self.selected_node_ids.add(node_idx)
                if self.budget_left <= 0: done = True
        next_emb, next_mask, next_global = self._get_state()
        if self.edge_cursor >= len(self.temporal_edges): done = True
        return (next_emb, next_mask, next_global), reward, done, {}


def get_n_step_info(n_step_buffer, gamma):
    n_step_reward = sum([(gamma ** i) * n_step_buffer[i][4] for i in range(len(n_step_buffer))])
    last = n_step_buffer[-1]
    return n_step_reward, last[5], last[6], last[7], last[8]


# ==============================================================================
# 🌟 [消融核心] 新增模型架构
# ==============================================================================
class ScoringDQN(nn.Module):
    """标准的解耦架构 (Decoupled)"""

    def __init__(self, emb_dim, hidden_dim, global_dim=2):
        super(ScoringDQN, self).__init__()
        self.node_scorer = nn.Sequential(nn.Linear(emb_dim + global_dim, hidden_dim), nn.ReLU(),
                                         nn.Linear(hidden_dim, 1))
        self.wait_scorer = nn.Sequential(nn.Linear(emb_dim + global_dim, hidden_dim), nn.ReLU(),
                                         nn.Linear(hidden_dim, 1))

    def forward(self, node_embeddings, global_feat):
        if node_embeddings.dim() == 2: node_embeddings, global_feat = node_embeddings.unsqueeze(
            0), global_feat.unsqueeze(0)
        batch, num, _ = node_embeddings.shape
        node_in = torch.cat([node_embeddings, global_feat.unsqueeze(1).expand(-1, num, -1)], dim=2)
        wait_in = torch.cat([torch.max(node_embeddings, dim=1)[0], global_feat], dim=1)
        return torch.cat([self.node_scorer(node_in).squeeze(-1), self.wait_scorer(wait_in)], dim=1)


class UnifiedScoringDQN(nn.Module):
    """[新增] 消融实验模型：统一网络架构 (Coupled / Unified)"""

    def __init__(self, emb_dim, hidden_dim, global_dim=2):
        super(UnifiedScoringDQN, self).__init__()
        # 共享特征提取骨干网络
        self.shared_backbone = nn.Sequential(
            nn.Linear(emb_dim + global_dim, hidden_dim),
            nn.ReLU()
        )
        # 共享单一输出层
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, node_embeddings, global_feat):
        if node_embeddings.dim() == 2:
            node_embeddings, global_feat = node_embeddings.unsqueeze(0), global_feat.unsqueeze(0)
        batch, num, _ = node_embeddings.shape

        # N 个节点的特征拼装 (batch, num, emb_dim + global_dim)
        node_in = torch.cat([node_embeddings, global_feat.unsqueeze(1).expand(-1, num, -1)], dim=2)

        # WAIT 动作的特征拼装 (batch, 1, emb_dim + global_dim) -> 使用全局池化代表全图状态
        wait_in = torch.cat([torch.max(node_embeddings, dim=1)[0], global_feat], dim=1).unsqueeze(1)

        # 强行合并动作空间 (batch, num + 1, emb_dim + global_dim)
        combined_in = torch.cat([node_in, wait_in], dim=1)

        # 走同一个网络提取特征并打分
        hidden = self.shared_backbone(combined_in)
        out = self.output_layer(hidden).squeeze(-1)  # 输出 (batch, num + 1)

        return out


# ==============================================================================
# 5. Training Loop
# ==============================================================================
def train_streaming(current_act_duration, train_data, global_min_node_id, ablation_mode='none'):
    sync_main_component_globals()
    MODEL_INTERNAL_DIM = 128
    AUGMENTED_DIM = MODEL_INTERNAL_DIM + 5
    tgan_wrapper = MainTGANInferenceWrapper(train_data, TGAN_MODEL_PATH, node_dim=MODEL_INTERNAL_DIM,
                                            time_dim=MODEL_INTERNAL_DIM,
                                            device=device)

    train_wait_reward = 0.0 if ablation_mode == 'no_wait_compensation' else WAIT_COMPENSATION_COEF
    train_wait_cap = 0.0 if ablation_mode == 'no_wait_compensation' else WAIT_COMPENSATION_CAP
    env = MainStrictStreamingGraphEnv(train_data, tgan_wrapper, t2EICModel, activation_prob=ACTIVATION_PROB,
                                      activation_duration_pct=current_act_duration,
                                      wait_reward=train_wait_reward,
                                      wait_reward_cap=train_wait_cap,
                                      global_min_node_id=global_min_node_id)

    # === [消融核心] 动态选择网络架构 ===
    if ablation_mode == 'unified':
        policy_net = UnifiedScoringDQN(AUGMENTED_DIM, HIDDEN_DIM).to(device)
        target_net = UnifiedScoringDQN(AUGMENTED_DIM, HIDDEN_DIM).to(device)
    else:
        policy_net = ScoringDQN(AUGMENTED_DIM, HIDDEN_DIM).to(device)
        target_net = ScoringDQN(AUGMENTED_DIM, HIDDEN_DIM).to(device)

    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)

    # 仅使用普通的优先经验回放 (PER)
    memory = PrioritizedReplayBuffer(MEMORY_CAPACITY, alpha=0.6, beta=0.4)

    best_reward = -float('inf')
    global_step = 0
    global_max_wait_count = 0
    epsilon = EPSILON_START

    for episode in range(EPISODES):
        emb, mask, global_feat = env.reset(random_start=True)
        total_reward = 0
        wait_count = 0
        n_step_buffer = deque(maxlen=N_STEP)

        while True:
            # === [消融核心 1] 动作选择 ===
            # w/o WAIT: 强制不等待，除非没得选
            can_force_no_wait = ablation_mode == 'no_wait' and not torch.all(mask[:env.num_nodes])

            if random.random() < epsilon:
                # [新增消融]: w/o Action-Bias 纯随机，不加偏置
                if ablation_mode == 'no_action_bias':
                    valid_all = torch.where(~mask)[0]  # 这包含了可用的节点和恒定为可用的WAIT
                    action = valid_all[torch.randint(0, len(valid_all), (1,), device=device).item()].item() if len(
                        valid_all) > 0 else env.num_nodes
                else:
                    # 原本的 Action-bias 逻辑
                    current_force_wait_prob = 0 if can_force_no_wait else FORCE_WAIT_PROB

                    if random.random() < current_force_wait_prob:
                        action = env.num_nodes
                    else:
                        valid = torch.where(~mask)[0]
                        valid = valid[valid != env.num_nodes]  # 排除WAIT动作
                        action = valid[torch.randint(0, len(valid), (1,), device=device).item()].item() if len(
                            valid) > 0 else env.num_nodes
            else:
                with torch.no_grad():
                    q = policy_net(emb, global_feat)
                    q[0, mask] = -float('inf')

                    # 如果是 no_wait 模式，且有有效节点可选，则屏蔽 WAIT 动作
                    if can_force_no_wait:
                        q[0, env.num_nodes] = -float('inf')

                    action = q.argmax().item()

            if action == env.num_nodes: wait_count += 1
            (next_emb, next_mask, next_global), reward, done, _ = env.step(action)

            transition_full = (emb.detach().cpu(), mask.detach().cpu(), global_feat.detach().cpu(),
                               action, reward,
                               next_emb.detach().cpu(), next_mask.detach().cpu(), next_global.detach().cpu(), done)
            n_step_buffer.append(transition_full)

            if len(n_step_buffer) == N_STEP:
                nr, ne, nm, ng, nd = get_n_step_info(n_step_buffer, GAMMA)
                s, m, g, a = n_step_buffer[0][0], n_step_buffer[0][1], n_step_buffer[0][2], n_step_buffer[0][3]
                memory.push((s, m, g, a, nr, ne, nm, ng, nd))

            emb, mask, global_feat = next_emb, next_mask, next_global
            total_reward += reward

            if len(memory) > BATCH_SIZE:
                batch, idxs, is_weights = memory.sample(BATCH_SIZE)

                b_emb, b_mask, b_global, b_action, b_reward, b_n_emb, b_n_mask, b_n_global, b_done = zip(*batch)
                b_emb = torch.stack(b_emb).to(device)
                b_mask = torch.stack(b_mask).to(device)
                b_global = torch.stack(b_global).to(device)
                b_action = torch.tensor(b_action, device=device).unsqueeze(1)
                b_reward = torch.tensor(b_reward, device=device).unsqueeze(1)
                b_n_emb = torch.stack(b_n_emb).to(device)
                b_n_mask = torch.stack(b_n_mask).to(device)
                b_n_global = torch.stack(b_n_global).to(device)
                b_done = torch.tensor(b_done, dtype=torch.float, device=device).unsqueeze(1)

                curr_q = policy_net(b_emb, b_global).gather(1, b_action)
                with torch.no_grad():
                    next_q = target_net(b_n_emb, b_n_global).masked_fill(b_n_mask, -1e9)
                    target_q = b_reward + (GAMMA ** N_STEP) * next_q.max(1, keepdim=True)[0] * (1 - b_done)

                loss = (is_weights * nn.functional.mse_loss(curr_q, target_q, reduction='none').squeeze()).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                memory.update_priorities(idxs, torch.abs(curr_q - target_q).squeeze().detach().cpu().numpy())
                global_step += 1
                if global_step % UPDATE_FREQ == 0: target_net.load_state_dict(policy_net.state_dict())

            if done:
                while len(n_step_buffer) > 0:
                    nr, ne, nm, ng, nd = get_n_step_info(n_step_buffer, GAMMA)
                    s, m, g, a = n_step_buffer[0][0], n_step_buffer[0][1], n_step_buffer[0][2], n_step_buffer[0][3]
                    memory.push((s, m, g, a, nr, ne, nm, ng, nd))
                    n_step_buffer.popleft()
                if wait_count > global_max_wait_count: global_max_wait_count = wait_count
                break

        if total_reward > best_reward:
            best_reward = total_reward
            if not os.path.exists('./saved_models'): os.makedirs('./saved_models')
            torch.save(policy_net.state_dict(), DQN_BEST_MODEL_PATH)

        epsilon = max(EPSILON_END, epsilon - (EPSILON_START - EPSILON_END) / EPSILON_DECAY_STEPS)

        if episode % 10 == 0:
            start_hour = (env.current_time - env.start_timestamp) / 3600.0
            logger.info(
                f"[{ablation_mode}] Ep {episode}: Reward={total_reward:.2f} | Wait={wait_count} | Eps={epsilon:.3f}")

    torch.save(policy_net.state_dict(), DQN_MODEL_SAVE_PATH)
    return global_max_wait_count


# ==============================================================================
# 6. Testing Function
# ==============================================================================
def test_streaming(model_type, current_act_duration, global_max_wait, test_data, test_budget, global_min_node_id,
                   ablation_mode='none', history_edges=None):
    sync_main_component_globals()
    MODEL_INTERNAL_DIM = 128
    AUGMENTED_DIM = MODEL_INTERNAL_DIM + 5
    model_path = DQN_BEST_MODEL_PATH if model_type == 'best' and os.path.exists(
        DQN_BEST_MODEL_PATH) else DQN_MODEL_SAVE_PATH

    tgan_wrapper = MainTGANInferenceWrapper(test_data, TGAN_MODEL_PATH, node_dim=MODEL_INTERNAL_DIM,
                                            time_dim=MODEL_INTERNAL_DIM,
                                            device=device)

    env = MainStrictStreamingGraphEnv(
        test_data,
        tgan_wrapper,
        t2EICModel,
        ACTIVATION_PROB,
        current_act_duration,
        wait_reward=0.0,
        enable_crn=True,
        crn_seed=EVAL_SEED,
        mc_rounds=100,
        global_min_node_id=global_min_node_id,
        history_edges=history_edges
    )

    env.budget_left = test_budget

    # === [消融核心] 动态选择网络架构 ===
    if ablation_mode == 'unified':
        policy_net = UnifiedScoringDQN(AUGMENTED_DIM, HIDDEN_DIM).to(device)
    else:
        policy_net = ScoringDQN(AUGMENTED_DIM, HIDDEN_DIM).to(device)

    if os.path.exists(model_path):
        try:
            policy_net.load_state_dict(torch.load(model_path, map_location=device))
        except:
            policy_net.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    else:
        print(f"Model not found: {model_path}")
        return []
    policy_net.eval()

    emb, mask, global_feat = env.reset(random_start=False)
    env.budget_left = test_budget

    logs = []
    total_reward = 0
    step_count = 0
    wait_count = 0

    while True:
        with torch.no_grad():
            q = policy_net(emb, global_feat)
            q[0, mask] = -float('inf')

            # === [消融核心 1] 动作选择 - 测试时也屏蔽WAIT ===
            can_force_no_wait = ablation_mode == 'no_wait' and not torch.all(mask[:env.num_nodes])
            if can_force_no_wait:
                q[0, env.num_nodes] = -float('inf')

            action = q.argmax().item()

        if action == env.num_nodes:
            wait_count += 1
            action_type = "WAIT"
            node_id_str = "None"
        else:
            action_type = "SELECT"
            node_id_str = str(action)

        (next_emb, next_mask, next_global), reward, done, _ = env.step(action)
        step_count += 1
        total_reward += reward

        if action_type == "SELECT":
            logs.append({
                'MAX_BUDGET': test_budget,
                'ACTIVATION_DURATION_PCT': current_act_duration,
                'GlobalMaxWait': global_max_wait,
                'Model': model_type,
                'SeedID': node_id_str,
                'ActionType': action_type,
                'Time': round(env.current_time, 2),
                'StepReward': round(reward * SCALE_FACTOR, 4),
                'TotalReward': round(total_reward * SCALE_FACTOR, 4),
                'TotalWaitCount': wait_count
            })

        emb, mask, global_feat = next_emb, next_mask, next_global
        if done: break

    summary_log = {
        'MAX_BUDGET': test_budget,
        'ACTIVATION_DURATION_PCT': current_act_duration,
        'GlobalMaxWait': global_max_wait,
        'Model': model_type,
        'SeedID': "SUMMARY",
        'ActionType': "DONE",
        'Time': round(env.current_time, 2),
        'StepReward': 0,
        'TotalReward': round(total_reward * SCALE_FACTOR, 4),
        'TotalWaitCount': wait_count
    }
    logs.append(summary_log)
    return logs


# ==============================================================================
# 7. Main Loop
# ==============================================================================
if __name__ == "__main__":
    RUN_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    RUN_ID = datetime.now().strftime(f"{DATA_NAME}_{ABLATION_MODE}_seed{RANDOM_SEED}_%Y%m%d_%H%M%S")
    seed_everything(RANDOM_SEED)

    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"Data not found: {PROCESSED_DATA_PATH}")
        sys.exit(1)

    with open(PROCESSED_DATA_PATH, 'rb') as f:
        SHARED_DATA = pickle.load(f)

    all_edges = sorted(SHARED_DATA['temporal_edges'], key=lambda x: x[2])
    all_src = [e[0] for e in all_edges]
    all_dst = [e[1] for e in all_edges]
    GLOBAL_MIN_NODE_ID = min(min(all_src), min(all_dst))

    split_idx = int(len(all_edges) * 0.7)
    train_edges = all_edges[:split_idx]
    test_edges = all_edges[split_idx:]

    TRAIN_DATA = SHARED_DATA.copy()
    TRAIN_DATA['temporal_edges'] = train_edges

    TEST_DATA = SHARED_DATA.copy()
    TEST_DATA['temporal_edges'] = test_edges

    print(f"Total Edges: {len(all_edges)}")
    print(f"Train Edges: {len(train_edges)} (0 - {split_idx})")
    print(f"Test Edges: {len(test_edges)} ({split_idx} - end)")
    print(f"Global Min Node ID: {GLOBAL_MIN_NODE_ID}")

    if 'n_feat' not in SHARED_DATA and 'feat_dim' in SHARED_DATA:
        pass

    scenarios = [
        (10, 0.001),  # 1. 高难度/短时效 (证明 Waiting 的价值)
        (30, 0.005),  # 2. 标准/中等 (基准性能)
        (50, 0.01)  # 3. 富裕/长时效 (证明 Elite 的价值)
    ]

    budget_list = parse_int_list(args.budgets, [10, 20, 30, 50])
    duration_list = parse_float_list(args.durations, [0.001, 0.005, 0.01])
    scenarios = [(b, d) for b in budget_list for d in duration_list]
    total_experiments = len(scenarios)
    current_exp = 0

    for b, d in scenarios:
        MAX_BUDGET = b

        current_exp += 1
        print(f"\n[{current_exp}/{total_experiments}] STARTING RUN ({ABLATION_MODE}): Budget={b}, Duration={d}")

        # 模型路径加上后缀
        run_suffix = safe_filename_token(args.result_suffix)
        suffix = f"_B{b}_D{d}_S{RANDOM_SEED}{suffix_ablation}"
        if run_suffix:
            suffix = f"{suffix}_{run_suffix}"
        DQN_MODEL_SAVE_PATH = f'./saved_models/dqn_final_{DATA_NAME}{suffix}.pth'
        DQN_BEST_MODEL_PATH = f'./saved_models/dqn_best_{DATA_NAME}{suffix}.pth'

        print(f" -> Model saving to: {DQN_BEST_MODEL_PATH}")

        try:
            # 训练
            max_wait = train_streaming(d, TRAIN_DATA, GLOBAL_MIN_NODE_ID, ablation_mode=ABLATION_MODE)

            # 测试 (Best & Final)
            logs = []
            logs.extend(test_streaming('best', d, max_wait, TEST_DATA, test_budget=b,
                                       global_min_node_id=GLOBAL_MIN_NODE_ID, ablation_mode=ABLATION_MODE,
                                       history_edges=train_edges))
            logs.extend(test_streaming('final', d, max_wait, TEST_DATA, test_budget=b,
                                       global_min_node_id=GLOBAL_MIN_NODE_ID, ablation_mode=ABLATION_MODE,
                                       history_edges=train_edges))

            new_df = pd.DataFrame(logs)
            new_df = attach_run_metadata(new_df, RUN_ID, RUN_TIME)
            for key, value in delta_metadata(d, test_edges).items():
                new_df[key] = value
            new_df['RANDOM_SEED'] = RANDOM_SEED
            new_df['EVAL_SEED'] = EVAL_SEED
            new_df['FORCE_WAIT_PROB'] = FORCE_WAIT_PROB
            new_df['WAIT_REWARD_COEF'] = 0.0 if ABLATION_MODE == 'no_wait_compensation' else WAIT_COMPENSATION_COEF
            new_df['WAIT_REWARD_CAP'] = 0.0 if ABLATION_MODE == 'no_wait_compensation' else WAIT_COMPENSATION_CAP
            new_df['ACTIVATION_PROB'] = ACTIVATION_PROB
            cols = ['MAX_BUDGET', 'ACTIVATION_DURATION_PCT',
                    'GlobalMaxWait', 'Model', 'SeedID', 'ActionType', 'Time',
                    'StepReward', 'TotalReward', 'TotalWaitCount',
                    'DELTA_PCT', 'DELTA_SECONDS', 'DELTA_MINUTES', 'DELTA_HOURS',
                    'RANDOM_SEED', 'EVAL_SEED', 'FORCE_WAIT_PROB',
                    'WAIT_REWARD_COEF', 'WAIT_REWARD_CAP', 'ACTIVATION_PROB',
                    'RUN_ID', 'RUN_TIME']

            for c in cols:
                if c not in new_df.columns: new_df[c] = None
            new_df = new_df[cols]
            new_df['Ablation'] = ABLATION_MODE

            if os.path.exists(RESULT_FILE):
                # 读取旧文件，追加新数据
                old_df = pd.DataFrame()
                # 简单去重：如果同参数同模型的记录已存在，可以选择覆盖或追加。这里直接追加。
                append_excel_locked(RESULT_FILE, new_df)
            else:
                append_excel_locked(RESULT_FILE, new_df)

            print(f" -> Saved logs for B={b}, D={d}")

            del logs, new_df
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error in experiment B={b}, D={d}: {e}")
            import traceback

            traceback.print_exc()
            continue

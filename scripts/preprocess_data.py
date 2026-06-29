import numpy as np
import pandas as pd
import pickle
import os
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def split_edge_line(line):
    if ',' in line:
        return [item.strip() for item in line.split(',')]
    return line.split()


def parse_numeric_prefix(parts):
    if len(parts) < 3:
        raise ValueError("Temporal edge rows need at least three columns.")
    return [float(parts[0]), float(parts[1]), float(parts[2])]


def infer_temporal_columns(raw_rows):
    """Infer which of the first three columns is time.

    This handles both u,v,t and SocioPatterns-style t,u,v rows.  Small
    SocioPatterns timestamps can otherwise be mistaken for node IDs.
    """
    if not raw_rows:
        raise ValueError("No numeric rows available for column inference.")

    arr = np.asarray(raw_rows, dtype=float)
    candidates = []
    for ts_col in range(3):
        node_cols = [col for col in range(3) if col != ts_col]
        ts_values = arr[:, ts_col]
        if len(ts_values) > 1:
            monotonic_ratio = float(np.mean(np.diff(ts_values) >= 0))
        else:
            monotonic_ratio = 1.0
        node_values = np.concatenate([arr[:, node_cols[0]], arr[:, node_cols[1]]])
        candidates.append({
            "ts_col": ts_col,
            "node_cols": node_cols,
            "monotonic_ratio": monotonic_ratio,
            "node_count": len(np.unique(node_values.astype(int))),
            "ts_count": len(np.unique(ts_values)),
        })

    max_monotonic = max(item["monotonic_ratio"] for item in candidates)
    plausible = [
        item for item in candidates
        if item["monotonic_ratio"] >= max_monotonic - 0.01
    ]
    return min(plausible, key=lambda item: (item["node_count"], -item["ts_count"]))


def parse_temporal_edge(parts, column_info=None):
    values = parse_numeric_prefix(parts)
    if column_info is None:
        column_info = infer_temporal_columns([values])
    u_col, v_col = column_info["node_cols"]
    ts_col = column_info["ts_col"]
    return int(values[u_col]), int(values[v_col]), float(values[ts_col])


# ==========================================
# 1. 特征生成函数 (融合 process1 的丰富特征工程)
# ==========================================

def generate_node_features(df, num_nodes, feat_dim=172):
    """基于图结构生成丰富的节点特征，已适配连续映射后的节点ID"""
    print(f"正在生成节点特征: 共 {num_nodes} 个节点, {feat_dim} 维...")
    src_l = df['u'].values
    dst_l = df['i'].values
    ts_l = df['ts'].values

    # 初始化特征矩阵
    node_features = np.zeros((num_nodes, feat_dim))

    # 1. 计算基础度数
    degree = Counter()
    in_degree = Counter()
    out_degree = Counter()
    for src, dst in zip(src_l, dst_l):
        degree[src] += 1
        degree[dst] += 1
        out_degree[src] += 1
        in_degree[dst] += 1

    # 2. 计算时间特征
    first_time = {}
    last_time = {}
    appear_count = Counter()

    for src, dst, ts in zip(src_l, dst_l, ts_l):
        for node in [src, dst]:
            if node not in first_time:
                first_time[node] = ts
            last_time[node] = ts
            appear_count[node] += 1

    # 3. 计算邻居集合
    neighbors = defaultdict(set)
    for src, dst in zip(src_l, dst_l):
        neighbors[src].add(dst)
        neighbors[dst].add(src)

    # 4. 填充 16 维核心手工特征
    for node_idx in range(num_nodes):
        # 此时 node_idx 已经是完美连续的 0 ~ num_nodes-1

        # --- 度数特征 (0-4) ---
        node_features[node_idx, 0] = degree.get(node_idx, 0)
        node_features[node_idx, 1] = in_degree.get(node_idx, 0)
        node_features[node_idx, 2] = out_degree.get(node_idx, 0)
        node_features[node_idx, 3] = np.log1p(degree.get(node_idx, 0))
        if degree.get(node_idx, 0) > 0:
            node_features[node_idx, 4] = in_degree.get(node_idx, 0) / degree[node_idx]

        # --- 时间特征 (5-9) ---
        if node_idx in first_time:
            node_features[node_idx, 5] = first_time[node_idx]
            node_features[node_idx, 6] = last_time[node_idx]
            node_features[node_idx, 7] = last_time[node_idx] - first_time[node_idx]
            node_features[node_idx, 8] = appear_count[node_idx]
            if node_features[node_idx, 7] > 0:
                node_features[node_idx, 9] = appear_count[node_idx] / node_features[node_idx, 7]

        # --- 邻居特征 (10-14) ---
        if node_idx in neighbors:
            node_neighbors = neighbors[node_idx]
            node_features[node_idx, 10] = len(node_neighbors)
            neighbor_degrees = [degree.get(n, 0) for n in node_neighbors]
            if neighbor_degrees:
                node_features[node_idx, 11] = np.mean(neighbor_degrees)
                node_features[node_idx, 12] = np.max(neighbor_degrees)
                node_features[node_idx, 13] = np.min(neighbor_degrees)
                node_features[node_idx, 14] = np.std(neighbor_degrees)

        # --- 聚类系数 (15) ---
        if node_idx in neighbors:
            node_neighbors = list(neighbors[node_idx])
            k = len(node_neighbors)
            if k > 1:
                edges_between = 0
                for i, n1 in enumerate(node_neighbors):
                    for n2 in node_neighbors[i + 1:]:
                        if n2 in neighbors.get(n1, set()):
                            edges_between += 1
                possible = k * (k - 1) / 2
                if possible > 0:
                    node_features[node_idx, 15] = edges_between / possible

    # 5. 循环复制核心特征填满 172 维 (抛弃随机噪声)
    for i in range(16, feat_dim):
        base_idx = i % 16
        node_features[:, i] = node_features[:, base_idx]

    # 标准化
    scaler = StandardScaler()
    node_features = scaler.fit_transform(node_features)

    return node_features


def generate_edge_features(df, feat_dim=172):
    """基于边的属性生成丰富的边特征"""
    num_edges = len(df)
    print(f"正在生成边特征: 共 {num_edges} 条边, {feat_dim} 维...")

    # +1 是因为后续在图网络中，索引 0 通常保留作为 padding(占位符)
    edge_features = np.zeros((num_edges + 1, feat_dim))

    # 计算节点度数
    degree = Counter()
    for u, i in zip(df['u'].values, df['i'].values):
        degree[u] += 1
        degree[i] += 1

    # 填充 22 维核心手工特征
    for i in range(len(df)):
        edge_idx = int(df.iloc[i]['idx'])  # 获取当前边的时间排序编号
        src = df.iloc[i]['u']
        dst = df.iloc[i]['i']
        ts = df.iloc[i]['ts']

        src_deg = degree[src]
        dst_deg = degree[dst]

        # --- 度数组合特征 (0-9) ---
        edge_features[edge_idx, 0] = src_deg
        edge_features[edge_idx, 1] = dst_deg
        edge_features[edge_idx, 2] = src_deg + dst_deg
        edge_features[edge_idx, 3] = src_deg * dst_deg
        edge_features[edge_idx, 4] = abs(src_deg - dst_deg)
        edge_features[edge_idx, 5] = max(src_deg, dst_deg)
        edge_features[edge_idx, 6] = min(src_deg, dst_deg)
        edge_features[edge_idx, 7] = np.log1p(src_deg)
        edge_features[edge_idx, 8] = np.log1p(dst_deg)
        if dst_deg > 0:
            edge_features[edge_idx, 9] = src_deg / dst_deg

        # --- 周期性时间特征 (10-15) ---
        edge_features[edge_idx, 10] = ts
        edge_features[edge_idx, 11] = np.sin(ts)
        edge_features[edge_idx, 12] = np.cos(ts)
        edge_features[edge_idx, 13] = np.sin(2 * ts)
        edge_features[edge_idx, 14] = np.cos(2 * ts)
        edge_features[edge_idx, 15] = np.log1p(max(0, ts))  # 加 max 防御负数引发报错

        # --- 节点ID组合特征 (16-21) ---
        edge_features[edge_idx, 16] = src
        edge_features[edge_idx, 17] = dst
        edge_features[edge_idx, 18] = src + dst
        edge_features[edge_idx, 19] = abs(src - dst)
        edge_features[edge_idx, 20] = max(src, dst)
        edge_features[edge_idx, 21] = min(src, dst)

        # 重复填充补满维度
        for j in range(22, feat_dim):
            base_idx = j % 22
            edge_features[edge_idx, j] = edge_features[edge_idx, base_idx]

    # 标准化
    scaler = StandardScaler()
    edge_features = scaler.fit_transform(edge_features)

    return edge_features


# ==========================================
# 2. 主处理函数 (融合 process2 的鲁棒读取与字典映射)
# ==========================================

def preprocess_and_convert(data_name, data_dir='./processed', feat_dim=172):
    print(f"=== 开始处理数据集: {data_name} ===")

    # 1. 智能查找与读取文件
    file_path = None
    for ext in ['.csv', '.txt', '.dat']:
        temp_path = os.path.join(data_dir, f'{data_name}{ext}')
        if os.path.exists(temp_path):
            file_path = temp_path
            break

    if not file_path:
        print(f"错误: 在 {data_dir} 中找不到 {data_name} 的数据文件")
        return

    raw_rows = []
    skipped_header = False
    print(f"读取文件: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line: continue

            # 智能分词兼容 csv 和 dat
            parts = split_edge_line(line)

            if len(parts) < 3:
                print(f"警告：第{idx}行格式错误（不足3列），跳过")
                continue

            try:
                # 兼容部分带小数点的节点ID
                raw_rows.append(parse_numeric_prefix(parts))
                continue
                label = 1  # 统一设为正样本交互

                u_list.append(u)
                i_list.append(i)
                ts_list.append(ts)
                label_list.append(label)
            except ValueError:
                if idx == 0:
                    print("检测到表头，自动跳过...")
                continue

    if not u_list:
        print("错误: 没有读取到有效数据。")
        return

    # 2. 安全完美的节点ID重映射 (基于字典)
    all_nodes = np.unique(u_list + i_list)
    num_nodes = len(all_nodes)
    min_node_id = min(all_nodes)

    node_map = {old_id: new_id for new_id, old_id in enumerate(all_nodes)}
    u_mapped = [node_map[u] for u in u_list]
    i_mapped = [node_map[i] for i in i_list]

    # 3. 构建DataFrame并强制时间排序
    df = pd.DataFrame({
        'u': u_mapped,
        'i': i_mapped,
        'ts': ts_list,
        'label': label_list
    })

    # 【极其重要的一步】强制按时间戳升序排列，并重新分配边ID (1 到 N)
    df = df.sort_values(by='ts').reset_index(drop=True)
    df['idx'] = df.index + 1

    # 4. (保留 process1 特色) 20秒窗口边数统计
    ts_edge_count = df.groupby('ts').size().sort_index()
    avg_edges_per_20s = ts_edge_count.mean()
    median_edges_per_20s = ts_edge_count.median()

    print(f"\n=== 时序窗口密度统计 ===")
    print(f"有效时间窗口数：{len(ts_edge_count)}")
    print(f"单窗口平均边数：{avg_edges_per_20s:.2f}")
    print(f"单窗口边数中位数：{median_edges_per_20s:.2f}")
    print("=========================\n")

    # 5. 生成基于图结构的特征
    node_feat = generate_node_features(df, num_nodes, feat_dim)
    edge_feat = generate_edge_features(df, feat_dim)

    # 6. 构建邻接表
    src = df['u'].values
    dst = df['i'].values
    ts = df['ts'].values
    labels = df['label'].values
    num_edges = len(df)

    adj_list = [[] for _ in range(num_nodes)]
    for idx, (u, v, t) in enumerate(zip(src, dst, ts)):
        # 注意这里我们用的边序号是从 1 开始的，和 edge_features 对齐
        adj_list[u].append((v, idx + 1, t))
        adj_list[v].append((u, idx + 1, t))

    temporal_edges = list(zip(src, dst, ts))

    # 7. 打包数据并保存
    data = {
        'src': src,
        'dst': dst,
        'ts': ts,
        'label': labels,
        'n_feat': node_feat,
        'e_feat': edge_feat,
        'adj_list': adj_list,
        'temporal_edges': temporal_edges,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'feat_dim': feat_dim,
        'min_node_id': min_node_id,
        'ts_edge_stats': {
            'avg_edges_per_20s': avg_edges_per_20s,
            'median_edges_per_20s': median_edges_per_20s,
            'window_count': len(ts_edge_count)
        }
    }

    os.makedirs(data_dir, exist_ok=True)

    output_pkl = os.path.join(data_dir, f'{data_name}_main.pkl')
    with open(output_pkl, 'wb') as f:
        pickle.dump(data, f)

    output_csv = os.path.join(data_dir, f'ml_{data_name}.csv')
    df.to_csv(output_csv, index=False)
    np.save(os.path.join(data_dir, f'ml_{data_name}.npy'), edge_feat)
    np.save(os.path.join(data_dir, f'ml_{data_name}_node.npy'), node_feat)

    print(f"\n=== 预处理与融合完成 ===")
    print(f"真实节点数: {num_nodes} (已重映射为 0~{num_nodes - 1})")
    print(f"总边数: {num_edges}")
    print(f"时间跨度: {np.min(ts):.2f} ~ {np.max(ts):.2f}")
    print("=========================\n")


if __name__ == "__main__":
    # 你可以在这里修改为其他数据集名称
    DATA_NAME = 'thiers_2012'
    preprocess_and_convert(DATA_NAME, data_dir='./processed')

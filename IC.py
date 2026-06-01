# ==============================================================================
# 4. 传播模型 (pEIC, tEIC, t1EIC)
# ==============================================================================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import logging
import math
import heapq
import os
import pickle
import bisect
from collections import defaultdict
from typing import List, Tuple, Hashable, Optional, Set, Dict, Union, Any
import warnings


class pEICModel:
    """持久演化独立级联模型（pEIC）传播模拟器（支持种子节点独立激活时间）"""

    def __init__(self, temporal_edges: List[Tuple[Hashable, Hashable, float]],
                 activation_prob: float = 0.2,
                 random_state: Optional[int] = None):
        """
        初始化时序传播模型

        参数:
            temporal_edges: 时序边列表，格式为 [(u, v, t), ...]，其中u、v为节点ID（可哈希类型），t为时间戳
            activation_prob: 边的默认激活概率（0~1之间）
            random_state: 随机种子，确保模拟结果可复现
        """
        self.activation_prob = activation_prob
        self.edges = defaultdict(list)  # 存储时序边：(u, v) -> [(t1, p1), (t2, p2), ...]
        self.adj = defaultdict(list)  # 预建邻接表：u -> [v1, v2, ...]（去重邻居）
        self.all_nodes: Set[Hashable] = set()  # 网络中所有节点
        self.influence_cache = dict()  # 影响力缓存：(sorted_seeds, num_rounds) -> 平均影响力
        self.rng = random.Random(random_state)  # 随机数生成器（支持可复现）

        # 解析时序边并构建数据结构
        for edge in temporal_edges:
            if len(edge) != 3:
                raise ValueError(f"时序边必须为三元组 (u, v, t)，实际为 {edge}")
            u, v, t = edge
            # 验证节点和时间类型
            if not (isinstance(u, Hashable) and isinstance(v, Hashable)):
                raise TypeError(f"节点ID必须是可哈希类型（如int、str），实际u={type(u)}, v={type(v)}")
            if not isinstance(t, (int, float)):
                raise TypeError(f"时间戳必须是数值类型，实际t={type(t)}")

            # 存储边信息（默认使用类的激活概率）
            for src, dst in ((u, v), (v, u)):
                self.edges[(src, dst)].append((t, self.activation_prob))
            self.all_nodes.add(u)
            self.all_nodes.add(v)
            # 构建邻接表（去重邻居）
            if v not in self.adj[u]:
                self.adj[u].append(v)

        # 对每条边的时序事件按时间戳排序（确保二分查找有效）
        for edge_key in self.edges:
            self.edges[edge_key].sort(key=lambda x: x[0])  # 按时间升序

    def _find_earliest_t_uv(self, u: Hashable, v: Hashable, t: float) -> Optional[Tuple[float, float]]:
        """
        查找节点对 (u, v) 在时间 t 之后的最早连接事件（时间戳+激活概率）

        参数:
            u: 源节点
            v: 目标节点
            t: 参考时间（仅查找 >= t 的事件）

        返回:
            若存在则返回 (t_uv, prob)，否则返回 None
        """
        if (u, v) not in self.edges:
            return None  # 边 (u, v) 不存在

        edges_list = self.edges[(u, v)]
        if not edges_list:
            return None  # 边 (u, v) 无事件

        # 提前判断：若最后一个事件时间仍 < t，则无有效事件
        if edges_list[-1][0] < t:
            return None

        # 二分查找第一个 >= t 的事件
        left, right = 0, len(edges_list)
        while left < right:
            mid = (left + right) // 2
            if edges_list[mid][0] >= t:
                right = mid
            else:
                left = mid + 1

        return edges_list[left] if left < len(edges_list) else None

    def _ensure_iterable(self, nodes: Union[Hashable, Tuple, List, None]) -> List:
        """
        确保输入的节点/种子是可迭代类型，兼容：
        - 单个节点（如 5 或 "A"）
        - 单个带时间的种子（如 (5, 10.0)）
        - 节点列表（如 [5, 6]）
        - 带时间的种子列表（如 [(5, 10.0), (6, 20.0)]）
        """
        if nodes is None:
            return []
        # 处理单个带时间的种子（元组）
        if isinstance(nodes, tuple) and (len(nodes) == 1 or len(nodes) == 2):
            return [nodes]
        # 处理单个节点（非元组）
        if isinstance(nodes, Hashable) and not isinstance(nodes, (list, tuple)):
            return [nodes]
        # 尝试迭代并转为列表
        try:
            iter(nodes)
            return list(nodes)
        except TypeError:
            return [nodes]

    def _single_simulation(self, seeds_with_time: List[Tuple[Hashable, float]]) -> int:
        """
        单轮传播模拟（核心逻辑）

        参数:
            seeds_with_time: 带激活时间的种子列表，格式为 [(node, t_activation), ...]

        返回:
            单轮模拟中被激活的节点总数
        """
        # 初始化激活时间字典：记录所有节点的激活时间（初始为 None，未激活）
        activation_time = {node: None for node in self.all_nodes}
        # 临时存储本轮新增的节点（避免修改全局 self.all_nodes）
        temp_nodes: Set[Hashable] = set()

        # 1. 初始化种子节点的激活时间
        for node, t_activation in seeds_with_time:
            if node in activation_time:
                activation_time[node] = t_activation  # 种子的专属激活时间
            else:
                # 若种子不在原始网络中，临时添加（仅本轮有效）
                activation_time[node] = t_activation
                temp_nodes.add(node)

        # 2. 初始化传播事件堆和已处理边对
        processed_pairs = set()  # 记录 (u, v) 避免重复添加事件
        event_heap = []  # 优先队列：(事件时间, 源节点u, 目标节点v, 激活概率)

        # 为每个种子节点生成初始传播事件（基于其激活时间）
        for u, t_u in seeds_with_time:
            # 从邻接表获取u的所有邻居（高效）
            neighbors = self.adj.get(u, [])
            for v in neighbors:
                if (u, v) in processed_pairs:
                    continue  # 跳过已处理的边

                # 查找u激活后（t_u之后）与v的最早连接事件
                res = self._find_earliest_t_uv(u, v, t_u)
                if res is not None:
                    t_uv, prob = res  # t_uv 是 u 激活后与 v 的最早连接时间
                    heapq.heappush(event_heap, (t_uv, u, v, prob))  # 按时间入堆
                    processed_pairs.add((u, v))  # 标记为已处理

        # 3. 按时间顺序处理传播事件（核心循环）
        event_count = 0
        while event_heap:
            event_count += 1
            if event_count % 1000 == 0:
                logger.debug(f"处理事件 {event_count}，堆大小: {len(event_heap)}")

            # 弹出最早的事件（时间戳最小）
            t_uv, u, v, prob = heapq.heappop(event_heap)

            # 检查目标节点v是否已被更早激活（若已激活则跳过）
            if activation_time[v] is not None and activation_time[v] <= t_uv:
                continue

            # 以概率 prob 激活 v（激活时间为 t_uv）
            if self.rng.random() < prob:
                activation_time[v] = t_uv

                # 为v的邻居生成传播事件（基于v的激活时间 t_uv）
                v_neighbors = self.adj.get(v, [])
                for w in v_neighbors:
                    if (v, w) in processed_pairs:
                        continue  # 跳过已处理的边

                    # 查找v激活后（t_uv之后）与w的最早连接事件
                    res = self._find_earliest_t_uv(v, w, t_uv)
                    if res is not None:
                        t_vw, next_prob = res
                        heapq.heappush(event_heap, (t_vw, v, w, next_prob))
                        processed_pairs.add((v, w))

        # 4. 统计激活节点总数（包含临时节点）
        all_activated_nodes = list(activation_time.keys()) + list(temp_nodes - set(activation_time.keys()))
        return sum(1 for node in all_activated_nodes if activation_time.get(node) is not None)

    def simulate(self,
                 seed_nodes: Union[Hashable, List[Hashable], Tuple[Hashable, float], List[Tuple[Hashable, float]]],
                 num_rounds: int = 5,
                 use_cache: bool = True) -> float:
        """
        多轮传播模拟（返回平均激活节点数）
        """
        if num_rounds <= 0:
            return 0.0

        # 处理种子格式：统一转为 [(node, t_activation), ...]
        processed_seeds: List[Tuple[Hashable, float]] = []
        for seed in self._ensure_iterable(seed_nodes):
            if isinstance(seed, tuple) and len(seed) == 2:
                # 带时间的种子：(node, t_activation)
                node, t_activation = seed
                if not isinstance(t_activation, (int, float)):
                    raise TypeError(f"种子激活时间必须是数值类型，实际为 {type(t_activation)}")
                processed_seeds.append((node, t_activation))
            else:
                # 不带时间的种子：默认激活时间为 0.0
                node = seed
                processed_seeds.append((node, 0.0))

        # 缓存处理：键为排序后的种子（确保唯一性）+ 轮数
        cache_key = None
        if use_cache:
            # 按节点ID和时间排序，确保相同种子集顺序不同时缓存一致
            sorted_seeds = tuple(sorted(processed_seeds, key=lambda x: (x[0], x[1])))
            cache_key = (sorted_seeds, num_rounds)
            if cache_key in self.influence_cache:
                return self.influence_cache[cache_key]

        # 多轮模拟取平均值
        total_activated = 0
        for _ in range(num_rounds):
            total_activated += self._single_simulation(processed_seeds)
        avg_influence = total_activated / num_rounds

        # 更新缓存
        if use_cache and cache_key is not None:
            self.influence_cache[cache_key] = avg_influence

        return avg_influence

    def calculate_marginal_gain(self, current_seeds: List[Tuple[Hashable, float]],
                                candidate_node: Hashable,
                                candidate_t: float = 0.0,
                                num_rounds: int = 5) -> float:
        """
        计算候选节点的边际影响力增益（添加候选节点后影响力的提升）
        """
        # 去重当前种子集（避免重复节点）
        current_seeds = list({(n, t) for n, t in current_seeds})  # 集合去重元组
        # 检查候选节点是否已在当前种子中（按节点ID判断）
        if any(n == candidate_node for n, t in current_seeds):
            return 0.0

        # 计算当前种子集的影响力
        influence_without = self.simulate(current_seeds, num_rounds)
        # 计算添加候选节点后的影响力
        seeds_with_candidate = current_seeds + [(candidate_node, candidate_t)]
        influence_with = self.simulate(seeds_with_candidate, num_rounds)

        return influence_with - influence_without


class tEICModel:
    """
    瞬态演化独立级联模型 (tEIC) - One-Shot Step-by-Step 版
    """

    def __init__(self, temporal_edges: List[Tuple[Hashable, Hashable, float]],
                 activation_prob: float = 0.2,
                 random_state: Optional[int] = 42):
        self.activation_prob = activation_prob
        self.rng = random.Random(random_state)
        self.influence_cache = dict()

        # 1. 构建时间桶 {timestamp: [(u, v, prob), ...]}
        self.time_bucket = defaultdict(list)
        for u, v, t in temporal_edges:
            t = float(t)
            self.time_bucket[t].append((u, v, self.activation_prob))

        # 2. 时间戳排序 (确保是严格的时间流)
        self.sorted_timestamps = sorted(self.time_bucket.keys())

    def _ensure_iterable(self, nodes):
        if nodes is None: return []
        if isinstance(nodes, tuple) and len(nodes) == 2 and isinstance(nodes[1], (int, float)):
            return [nodes]
        if isinstance(nodes, Hashable) and not isinstance(nodes, (list, tuple, set)):
            return [nodes]
        return list(nodes)

    def _single_simulation(self, seeds_with_time: List[Tuple[Hashable, float]]) -> int:
        """
        单次模拟：接力传播逻辑
        """
        # --- 1. 预处理种子计划 ---
        # 将种子按激活时间归类，方便在对应时间点直接提取
        seed_schedule = defaultdict(list)
        for node, t_act in seeds_with_time:
            seed_schedule[t_act].append(node)

        # --- 2. 状态初始化 ---
        # infected_history: 记录所有“得过病”的节点 (用于计算总分，并防止重复感染)
        # 初始化时包含所有种子
        infected_history = set(node for node, _ in seeds_with_time)

        # spreaders_next_round: 这里的节点将在 "下一轮" 具备传染力
        # 初始为空，因为第一批种子会在循环中从 seed_schedule 提取
        spreaders_next_round = set()

        # --- 3. 时间轴遍历 ---
        for current_ts in self.sorted_timestamps:
            # A. 确定当前时刻的传播者 (Current Spreaders)
            # 来源1: 上一轮被感染的节点 (接力棒)
            current_spreaders = spreaders_next_round

            # 来源2: 当前时刻觉醒的种子
            if current_ts in seed_schedule:
                # 注意：种子加入传播大军，不需要检查是否已感染（种子本身就是源头）
                current_spreaders.update(seed_schedule[current_ts])

            # 如果当前没有任何传播者，且未来也没有种子了，理论上可以提前结束
            # 但为了代码简单，我们继续跑完或者只跳过当前处理
            if not current_spreaders:
                spreaders_next_round = set()  # 这一轮没人传，下一轮自然也没接力棒
                continue

            # B. 获取当前时刻的边
            edges = self.time_bucket.get(current_ts, [])

            # C. 开始传播
            newly_infected_this_round = set()

            for u, v, prob in edges:
                # 只有 "当前传播者" 才有资格感染别人
                if u in current_spreaders:
                    # 只有 "从未感染过的人" 才能被感染
                    if v not in infected_history and v not in newly_infected_this_round:
                        if self.rng.random() < prob:
                            newly_infected_this_round.add(v)

            # D. 状态更新
            # 1. 将新感染者加入历史记录 (防止未来再次被感染)
            infected_history.update(newly_infected_this_round)

            # 2. 传递接力棒：本轮的新感染者，将在 "下一轮" 成为传播者
            spreaders_next_round = newly_infected_this_round

        return len(infected_history)

    def simulate(self, seed_nodes, num_rounds=5, use_cache=True) -> float:
        if num_rounds <= 0: return 0.0

        # 种子格式化
        processed_seeds = []
        for s in self._ensure_iterable(seed_nodes):
            if isinstance(s, tuple) and len(s) == 2:
                processed_seeds.append((s[0], float(s[1])))
            else:
                # 默认时间设为最早时间，或者根据业务需求设为 0
                processed_seeds.append((s, self.sorted_timestamps[0] if self.sorted_timestamps else 0.0))

        # 缓存逻辑
        cache_key = None
        if use_cache:
            sorted_seeds = tuple(sorted(processed_seeds, key=lambda x: str(x[0]) + str(x[1])))
            cache_key = (sorted_seeds, num_rounds)
            if cache_key in self.influence_cache: return self.influence_cache[cache_key]

        total = 0
        for _ in range(num_rounds):
            total += self._single_simulation(processed_seeds)

        avg = total / num_rounds
        if use_cache and cache_key: self.influence_cache[cache_key] = avg
        return avg

    def calculate_marginal_gain(self, current_seeds, candidate_node, candidate_t=0.0, num_rounds=5):
        # 检查重复
        for node, _ in current_seeds:
            if node == candidate_node: return 0.0

        base = self.simulate(current_seeds, num_rounds)
        new_seeds = current_seeds + [(candidate_node, float(candidate_t))]
        new_score = self.simulate(new_seeds, num_rounds)

        return new_score - base


class t1EICModel:
    """瞬态演化独立级联模型（支持时间段+重复激活）"""

    def __init__(self, temporal_edges, activation_prob=0.2, activation_duration_pct=0.1, random_state=42):
        self.activation_prob = activation_prob
        self.activation_duration_pct = activation_duration_pct
        self.edges = defaultdict(list)
        self.all_nodes = set()
        self.rng = random.Random(random_state)
        self.influence_cache = dict()  # Added cache support

        for u, v, t in temporal_edges:
            t = float(t)
            for src, dst in ((u, v), (v, u)):
                self.edges[(src, dst)].append((t, self.activation_prob))
            self.all_nodes.add(u);
            self.all_nodes.add(v)

        for k in self.edges: self.edges[k].sort(key=lambda x: x[0])

        self.timestamp_to_edges = defaultdict(list)
        for (u, v), events in self.edges.items():
            for t, p in events: self.timestamp_to_edges[t].append((u, v, p))

        self.sorted_timestamps = sorted(self.timestamp_to_edges.keys())
        if self.sorted_timestamps:
            self.min_ts, self.max_ts = self.sorted_timestamps[0], self.sorted_timestamps[-1]
            self.total_time = self.max_ts - self.min_ts
        else:
            self.min_ts, self.max_ts, self.total_time = 0.0, 0.0, 0.0

    def _get_window(self, t_start):
        dur = self.total_time * self.activation_duration_pct
        return t_start, t_start + dur

    def _single_simulation(self, seeds: List[Tuple[Hashable, float]]) -> int:
        """单次模拟"""
        # 自动排序，因此不需要外部约束
        seeds.sort(key=lambda x: x[1])

        activation_window = {}
        max_active_until = -float('inf')

        for n, t in seeds:
            s, e = self._get_window(t)
            activation_window[n] = (s, e)
            max_active_until = max(max_active_until, e)

        activated = set(activation_window.keys())
        attempted_pairs = set()

        for ts in self.sorted_timestamps:
            if ts > max_active_until: break  # 优化

            edges = self.timestamp_to_edges.get(ts, [])
            for u, v, p in edges:
                if v in activated: continue
                if u in activated:
                    s, e = activation_window[u]
                    if s <= ts <= e:
                        if (u, v) in attempted_pairs:
                            continue
                        attempted_pairs.add((u, v))
                        if self.rng.random() < p:
                            ns, ne = self._get_window(ts)
                            activation_window[v] = (ns, ne)
                            activated.add(v)
                            max_active_until = max(max_active_until, ne)
        return len(activated)

    def simulate(self, seed_nodes, num_rounds=5, use_cache=True) -> float:
        if num_rounds <= 0: return 0.0

        seeds = []
        for s in seed_nodes:
            if isinstance(s, tuple):
                seeds.append((s[0], float(s[1])))
            else:
                seeds.append((s, self.min_ts))

        # 缓存逻辑
        cache_key = None
        if use_cache:
            sorted_seeds = tuple(sorted(seeds, key=lambda x: str(x[0]) + str(x[1])))
            cache_key = (sorted_seeds, num_rounds)
            if cache_key in self.influence_cache: return self.influence_cache[cache_key]

        total = 0
        for _ in range(num_rounds):
            total += self._single_simulation(seeds)  # Use _single_simulation

        avg = float(total) / num_rounds
        if use_cache and cache_key: self.influence_cache[cache_key] = avg
        return avg

    def calculate_marginal_gain(self, current_seeds, candidate_node, candidate_t=0.0, num_rounds=5):
        base = self.simulate(current_seeds, num_rounds=num_rounds)
        new_seeds = current_seeds + [(candidate_node, candidate_t)]
        return self.simulate(new_seeds, num_rounds=num_rounds) - base


class t2EICModel:
    """瞬态演化独立级联模型（支持时间段 + 严格IC逻辑）"""

    def __init__(self, temporal_edges: List[Tuple[Hashable, Hashable, float]],
                 activation_prob: float = 1,
                 activation_duration_pct: float = 0.1,  # 修正：添加到参数中
                 random_state: Optional[int] = 42):

        self.activation_prob = activation_prob
        self.activation_duration_pct = activation_duration_pct
        self.edges = defaultdict(list)
        self.all_nodes = set()
        self.rng = random.Random(random_state)
        self.influence_cache = dict()  # Added cache support

        for u, v, t in temporal_edges:
            t = float(t)
            # 存储边
            for src, dst in ((u, v), (v, u)):
                self.edges[(src, dst)].append((t, self.activation_prob))
            self.all_nodes.add(u)
            self.all_nodes.add(v)

        # 预构建时间索引
        self.timestamp_to_edges = defaultdict(list)
        for (u, v), events in self.edges.items():
            for t, p in events:
                self.timestamp_to_edges[t].append((u, v, p))

        self.sorted_timestamps = sorted(self.timestamp_to_edges.keys())

        if self.sorted_timestamps:
            self.min_ts = self.sorted_timestamps[0]
            self.max_ts = self.sorted_timestamps[-1]
            self.total_time = self.max_ts - self.min_ts
        else:
            self.min_ts, self.max_ts, self.total_time = 0.0, 0.0, 0.0

    def _get_window(self, t_start):
        """计算激活窗口"""
        dur = self.total_time * self.activation_duration_pct
        return t_start, t_start + dur

    def _single_simulation(self, seeds: List[Tuple[Hashable, float]]) -> int:
        """单次模拟"""
        # 排序种子（虽然不影响结果，但符合时序直觉）
        seeds.sort(key=lambda x: x[1])

        # 记录每个节点的激活窗口 (start, end)
        activation_window = {}
        # 记录最大活跃时间，用于提前终止优化
        max_active_until = -float('inf')

        # 初始化种子状态
        for n, t in seeds:
            s, e = self._get_window(t)
            activation_window[n] = (s, e)
            max_active_until = max(max_active_until, e)

        # 已激活节点集合
        activated = set(activation_window.keys())

        # 【关键修正】记录已尝试过的传播对 (u -> v)
        # 无论成功与否，u 只能尝试感染 v 一次
        # 按时间轴遍历
        attempted_pairs = set()

        for ts in self.sorted_timestamps:
            # 优化：如果当前时间超过了所有活跃节点的有效期，且没有新节点被激活的可能性（需谨慎，简化版可保留）
            # 但考虑到新激活节点会延长 max_active_until，这里主要用于跳过尾部空白
            if ts > max_active_until:
                break

            current_edges = self.timestamp_to_edges.get(ts, [])

            # 遍历当前时刻的所有边
            for u, v, p in current_edges:
                # 1. 目标 v 已经被感染，跳过
                if v in activated:
                    continue

                # 2. 源 u 是活跃节点
                if u in activated:
                    # 3. 检查 u 是否在有效期窗口内
                    s, e = activation_window[u]
                    if s <= ts <= e:

                        # 【关键修正】检查 u 是否已经尝试过感染 v
                        # 标记为已尝试 (消耗掉这次机会)
                        if (u, v) in attempted_pairs:
                            continue
                        attempted_pairs.add((u, v))
                        # Strict TW-IC allows each ordered pair to try once per simulation.

                        # 4. 尝试激活
                        if self.rng.random() < p:
                            # 激活成功
                            ns, ne = self._get_window(ts)
                            activation_window[v] = (ns, ne)
                            activated.add(v)
                            # 更新系统最大活跃时间，延长模拟
                            max_active_until = max(max_active_until, ne)

        return len(activated)

    def simulate(self, seed_nodes, num_rounds=10, use_cache=True) -> float:
        if num_rounds <= 0: return 0.0

        seeds = []
        # 种子预处理
        for s in seed_nodes:
            if isinstance(s, tuple):
                seeds.append((s[0], float(s[1])))
            else:
                seeds.append((s, self.min_ts))

        # 缓存逻辑
        cache_key = None
        if use_cache:
            sorted_seeds = tuple(sorted(seeds, key=lambda x: str(x[0]) + str(x[1])))
            cache_key = (sorted_seeds, num_rounds)
            if cache_key in self.influence_cache: return self.influence_cache[cache_key]

        total = 0
        for _ in range(num_rounds):
            total += self._single_simulation(seeds)

        avg = float(total) / num_rounds
        if use_cache and cache_key: self.influence_cache[cache_key] = avg
        return avg

    def calculate_marginal_gain(self, current_seeds, candidate_node, candidate_t=0.0, num_rounds=5):
        base = self.simulate(current_seeds, num_rounds=num_rounds)
        new_seeds = current_seeds + [(candidate_node, candidate_t)]
        return self.simulate(new_seeds, num_rounds=num_rounds) - base

import numpy as np

### Utility function and class
class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        self.epoch_count += 1
        
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        return self.num_round >= self.max_round


import numpy as np


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list, ts_list=None):
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)
        self.ts_list = ts_list  # 动态图的时间戳列表（可选）

        # 构建正样本集合（用于过滤已存在的边）
        self.positive_edges = set(zip(src_list, dst_list))
        if ts_list is not None:
            # 若有时间戳，按节点对存储时间戳（用于时间约束）
            self.edge_ts = {(s, d): t for s, d, t in zip(src_list, dst_list, ts_list)}

    def sample(self, size, cur_ts=None):
        """
        采样负样本，支持过滤正样本和时间约束

        参数:
            size: 采样数量
            cur_ts: 当前时间戳（动态图中，负样本需满足时间<cur_ts）
        """
        src_sample, dst_sample = [], []
        while len(src_sample) < size:
            # 随机采样节点对
            s = np.random.choice(self.src_list)
            d = np.random.choice(self.dst_list)

            # 过滤正样本
            if (s, d) in self.positive_edges:
                continue

            # 动态图时间约束：负样本的时间需早于当前时间（若有）
            if cur_ts is not None and (s, d) in self.edge_ts:
                if self.edge_ts[(s, d)] >= cur_ts:
                    continue

            src_sample.append(s)
            dst_sample.append(d)

        return np.array(src_sample), np.array(dst_sample)
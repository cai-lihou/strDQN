"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import random
import sys
import argparse
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.module import TGAN
from src.models.graph import NeighborFinder



### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on link predictions')
parser.add_argument('-d', '--data', type=str, help='data sources to use', default='thiers_2012')
parser.add_argument('--bs', type=int, default=200, help='batch_size')
parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.3, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=128, help='Dimentions of the node embedding')
parser.add_argument('--time_dim', type=int, default=128, help='Dimentions of the time embedding')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method',
                    default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod',
                    help='use dot product attention or mapping based')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty', 'postime'], help='how to use time information',
                    default='postime')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim

MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Utility function and class



# 定义早停监控器类
class EarlyStopMonitor(object):
    # 初始化裁判的“执裁标准”
    def __init__(self, max_round=5, higher_better=True, tolerance=1e-3):
        self.max_round = max_round  # 最大耐心值：允许模型连续多少轮不进步（默认5轮）
        self.num_round = 0  # 当前已经连续几轮没进步了（耐心计数器）

        self.epoch_count = -1  # 记录当前训练到了第几个大轮（Epoch）
        self.best_epoch = 0  # 记录表现最好的那一轮是第几轮

        self.last_best = None  # 记录目前为止的最好成绩
        self.higher_better = higher_better  # 评估标准：成绩是越大越好(如准确率)，还是越小越好(如Loss)
        self.tolerance = tolerance  # 容忍度：成绩至少要提升百分之几，才算真正的进步（默认0.1%）

    # 核心判断函数：每轮训练完，把当前成绩传进来，裁判决定要不要停止
    def early_stop_check(self, curr_val):
        self.epoch_count += 1  # 轮数+1

        # 【核心巧思1】如果成绩是越小越好（比如Loss），就给它乘个负号
        # 这样无论哪种情况，裁判只需要认准“数值变大就是进步”这一个死理即可
        if not self.higher_better:
            curr_val *= -1

        # 如果是第一轮，直接把当前成绩记录为“历史最好成绩”
        if self.last_best is None:
            self.last_best = curr_val

        # 【核心巧思2】计算相对提升比例：(当前成绩 - 历史最好) / |历史最好|
        # 如果这个比例大于容忍度，说明模型取得了有效进步！
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val  # 更新历史最好成绩
            self.num_round = 0  # 归零耐心计数器（裁判重新恢复5次耐心）
            self.best_epoch = self.epoch_count  # 记下这是在哪一轮取得的好成绩

        # 如果没达到容忍度，说明模型在原地踏步甚至退步
        else:
            self.num_round += 1  # 失去1次耐心，计数器+1

        # 返回最终判决：如果失去耐心的次数 >= 最大耐心值，返回 True（触发早停），否则返回 False（继续训练）
        return self.num_round >= self.max_round


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list):
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)
        self.existing_edges = set(zip(src_list, dst_list))

    def sample(self, size):
        """existing_edges是正样本边的集合，比如{(1,2), (3,4)}"""
        src_idx = np.random.randint(0, len(self.src_list), size)
        dst_idx = np.random.randint(0, len(self.dst_list), size)
        src_sample = self.src_list[src_idx]
        dst_sample = self.dst_list[dst_idx]

        # 过滤真实存在的边
        valid_mask = [(s, d) not in self.existing_edges for s, d in zip(src_sample, dst_sample)]
        while not all(valid_mask):
            # 对无效样本重新采样
            invalid_idx = [i for i, v in enumerate(valid_mask) if not v]
            new_src_idx = np.random.randint(0, len(self.src_list), len(invalid_idx))
            new_dst_idx = np.random.randint(0, len(self.dst_list), len(invalid_idx))
            src_sample[invalid_idx] = self.src_list[new_src_idx]
            dst_sample[invalid_idx] = self.dst_list[new_dst_idx]
            valid_mask = [(s, d) not in self.existing_edges for s, d in zip(src_sample, dst_sample)]

        return src_sample, dst_sample

def eval_one_epoch(hint, tgan, sampler, src, dst, ts, label):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    with torch.no_grad():
        tgan = tgan.eval()
        TEST_BATCH_SIZE = 30
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):
            # percent = 100 * k / num_test_batch
            # if k % int(0.2 * num_test_batch) == 0:
            #     logger.info('{0} progress: {1:10.4f}'.format(hint, percent))
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            # label_l_cut = label[s_idx:e_idx]

            size = len(src_l_cut)
            src_l_fake, dst_l_fake = sampler.sample(size)

            pos_prob, neg_prob = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)

            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_acc.append((pred_label == true_label).mean())
            val_ap.append(average_precision_score(true_label, pred_score))
            val_f1.append(f1_score(true_label, pred_label))
            val_auc.append(roc_auc_score(true_label, pred_score))
    mean_acc = np.mean(val_acc) if val_acc else 0.0
    mean_ap = np.mean(val_ap) if val_ap else 0.0
    mean_f1 = np.mean(val_f1) if val_f1 else 0.0
    mean_auc = np.mean(val_auc) if val_auc else 0.0
    return mean_acc, mean_ap, mean_f1, mean_auc


### Load data and train/validation split for TGAT pre-training.
# The final 30% of the chronological edge stream is reserved for StrDQN testing
# and is never used for TGAT checkpoint selection.
g_df = pd.read_csv('./processed/ml_{}.csv'.format(DATA)).sort_values('ts').reset_index(drop=True)
e_feat = np.load('./processed/ml_{}.npy'.format(DATA))
n_feat = np.load('./processed/ml_{}_node.npy'.format(DATA))

src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values

max_src_index = src_l.max()
max_idx = max(src_l.max(), dst_l.max())

pretrain_end_idx = int(len(g_df) * 0.70)
if pretrain_end_idx < 2:
    raise ValueError('Not enough temporal edges for a 70/30 split.')

inner_train_end_idx = max(1, int(pretrain_end_idx * 0.80))
if inner_train_end_idx >= pretrain_end_idx:
    inner_train_end_idx = pretrain_end_idx - 1

train_flag = np.zeros(len(g_df), dtype=bool)
val_flag = np.zeros(len(g_df), dtype=bool)
heldout_flag = np.zeros(len(g_df), dtype=bool)
train_flag[:inner_train_end_idx] = True
val_flag[inner_train_end_idx:pretrain_end_idx] = True
heldout_flag[pretrain_end_idx:] = True

train_src_l = src_l[train_flag]
train_dst_l = dst_l[train_flag]
train_ts_l = ts_l[train_flag]
train_e_idx_l = e_idx_l[train_flag]
train_label_l = label_l[train_flag]

val_src_l = src_l[val_flag]
val_dst_l = dst_l[val_flag]
val_ts_l = ts_l[val_flag]
val_e_idx_l = e_idx_l[val_flag]
val_label_l = label_l[val_flag]

test_src_l = src_l[heldout_flag]
test_dst_l = dst_l[heldout_flag]
test_ts_l = ts_l[heldout_flag]
test_e_idx_l = e_idx_l[heldout_flag]
test_label_l = label_l[heldout_flag]

# Keep the original TGAT interface names, but validation is transductive within
# the first 70% history rather than an inductive final-test evaluation.
nn_val_src_l = val_src_l
nn_val_dst_l = val_dst_l
nn_val_ts_l = val_ts_l
nn_val_e_idx_l = val_e_idx_l
nn_val_label_l = val_label_l
nn_test_src_l = np.array([], dtype=src_l.dtype)
nn_test_dst_l = np.array([], dtype=dst_l.dtype)
nn_test_ts_l = np.array([], dtype=ts_l.dtype)
nn_test_e_idx_l = np.array([], dtype=e_idx_l.dtype)
nn_test_label_l = np.array([], dtype=label_l.dtype)

### Initialize the data structure for causal graph queries.
adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l[:pretrain_end_idx], dst_l[:pretrain_end_idx],
                              e_idx_l[:pretrain_end_idx], ts_l[:pretrain_end_idx]):
    adj_list[src].append((dst, eidx, ts))
    adj_list[dst].append((src, eidx, ts))
train_ngh_finder = NeighborFinder(adj_list, uniform=UNIFORM)
full_ngh_finder = NeighborFinder(adj_list, uniform=UNIFORM)

train_rand_sampler = RandEdgeSampler(train_src_l, train_dst_l)
val_rand_sampler = RandEdgeSampler(src_l[:pretrain_end_idx], dst_l[:pretrain_end_idx])
nn_val_rand_sampler = RandEdgeSampler(nn_val_src_l, nn_val_dst_l)
test_rand_sampler = RandEdgeSampler(src_l[:pretrain_end_idx], dst_l[:pretrain_end_idx])
nn_test_rand_sampler = RandEdgeSampler(nn_val_src_l, nn_val_dst_l)

### Model initialize
device = torch.device('cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu')
tgan = TGAN(train_ngh_finder, n_feat, e_feat,
            num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
            seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)
optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
criterion = torch.nn.BCELoss()
tgan = tgan.to(device)

num_instance = len(train_src_l)
num_batch = math.ceil(num_instance / BATCH_SIZE)

logger.info('num of training instances: {}'.format(num_instance))
logger.info('num of batches per epoch: {}'.format(num_batch))
idx_list = np.arange(num_instance)
np.random.shuffle(idx_list)

early_stopper = EarlyStopMonitor()
for epoch in range(NUM_EPOCH):
    # Training
    # training use only training graph
    tgan.ngh_finder = train_ngh_finder
    acc, ap, f1, auc, m_loss = [], [], [], [], []
    np.random.shuffle(idx_list)
    logger.info('start {} epoch'.format(epoch))
    for k in range(num_batch):
        # percent = 100 * k / num_batch
        # if k % int(0.2 * num_batch) == 0:
        #     logger.info('progress: {0:10.4f}'.format(percent))

        s_idx = k * BATCH_SIZE
        e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
        src_l_cut, dst_l_cut = train_src_l[s_idx:e_idx], train_dst_l[s_idx:e_idx]
        ts_l_cut = train_ts_l[s_idx:e_idx]
        label_l_cut = train_label_l[s_idx:e_idx]
        size = len(src_l_cut)
        src_l_fake, dst_l_fake = train_rand_sampler.sample(size)

        with torch.no_grad():
            pos_label = torch.ones(size, dtype=torch.float, device=device)
            neg_label = torch.zeros(size, dtype=torch.float, device=device)

        optimizer.zero_grad()
        tgan = tgan.train()
        pos_prob, neg_prob = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)

        loss = criterion(pos_prob, pos_label)
        loss += criterion(neg_prob, neg_label)

        loss.backward()
        optimizer.step()
        # get training results
        with torch.no_grad():
            tgan = tgan.eval()
            pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            acc.append((pred_label == true_label).mean())
            ap.append(average_precision_score(true_label, pred_score))
            f1.append(f1_score(true_label, pred_label))
            m_loss.append(loss.item())
            auc.append(roc_auc_score(true_label, pred_score))

    # validation phase use all information
    tgan.ngh_finder = full_ngh_finder
    val_acc, val_ap, val_f1, val_auc = eval_one_epoch('val for old nodes', tgan, val_rand_sampler, val_src_l,
                                                      val_dst_l, val_ts_l, val_label_l)

    nn_val_acc, nn_val_ap, nn_val_f1, nn_val_auc = eval_one_epoch('val for new nodes', tgan, nn_val_rand_sampler,
                                                                  nn_val_src_l,
                                                                  nn_val_dst_l, nn_val_ts_l, nn_val_label_l)

    logger.info('epoch: {}:'.format(epoch))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info('train acc: {}, val acc: {}, new node val acc: {}'.format(np.mean(acc), val_acc, nn_val_acc))
    logger.info('train auc: {}, val auc: {}, new node val auc: {}'.format(np.mean(auc), val_auc, nn_val_auc))
    logger.info('train ap: {}, val ap: {}, new node val ap: {}'.format(np.mean(ap), val_ap, nn_val_ap))
    # logger.info('train f1: {}, val f1: {}, new node val f1: {}'.format(np.mean(f1), val_f1, nn_val_f1))

    if early_stopper.early_stop_check(nn_val_auc):
        logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
        logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
        best_model_path = get_checkpoint_path(early_stopper.best_epoch)
        tgan.load_state_dict(torch.load(best_model_path))
        logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
        tgan.eval()
        break
    else:
        torch.save(tgan.state_dict(), get_checkpoint_path(epoch))

logger.info('Final 30% holdout edges are reserved for StrDQN evaluation and skipped by TGAT pre-training.')
logger.info('Reserved holdout edge count: {}'.format(len(test_src_l)))

logger.info('Saving TGAN model')
torch.save(tgan.state_dict(), MODEL_SAVE_PATH)
logger.info('TGAN models saved')


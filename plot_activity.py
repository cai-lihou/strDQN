import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

# ================= 配置区域 =================
DATA_NAME = 'primaryschool'
DATA_PATH = f'./processed/{DATA_NAME}_main.pkl'
OUTPUT_IMG = f'./result/activity_{DATA_NAME}.png'
BIN_SIZE_SECONDS = 3600  # 统计粒度：1小时 (3600秒)


# ===========================================

def plot_temporal_activity():
    print(f"正在读取数据: {DATA_PATH} ...")

    if not os.path.exists(DATA_PATH):
        print(f"错误: 找不到文件 {DATA_PATH}。请先运行 process.py 生成数据。")
        return

    # 1. 加载数据
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    # 提取时间戳
    # data['temporal_edges'] 结构通常是 [(u, v, t), ...]
    temporal_edges = data['temporal_edges']
    timestamps = [e[2] for e in temporal_edges]

    if len(timestamps) == 0:
        print("数据为空！")
        return

    # 2. 数据处理
    min_t = min(timestamps)
    max_t = max(timestamps)
    duration = max_t - min_t

    print(f"时间跨度: {min_t:.1f} -> {max_t:.1f} (共 {duration / 3600:.1f} 小时)")

    # 将时间戳归一化（从 0 开始），方便画图
    # 如果您希望保留原始时间（比如展示具体是几点钟），可以去掉这行
    # 这里我们保留相对时间，第一天、第二天...
    relative_times = np.array(timestamps) - min_t

    # 3. 分桶统计 (Binning)
    # 创建每小时的桶
    bins = np.arange(0, duration + BIN_SIZE_SECONDS, BIN_SIZE_SECONDS)
    counts, bin_edges = np.histogram(relative_times, bins=bins)

    # 转换小时数作为 X 轴
    hours = bin_edges[:-1] / 3600

    # 4. 绘图 (使用 Seaborn 美化)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # 绘制填充曲线图 (Area Plot)
    plt.fill_between(hours, counts, color="skyblue", alpha=0.4)
    plt.plot(hours, counts, color="Slateblue", alpha=0.8, linewidth=2)

    # 添加装饰
    plt.title(f'Temporal Activity: {DATA_NAME} (Interactions per Hour)', fontsize=16, pad=15)
    plt.xlabel('Time (Hours)', fontsize=14)
    plt.ylabel('Number of Contacts', fontsize=14)
    plt.xlim(0, max(hours))
    plt.ylim(0, max(counts) * 1.1)

    # 标注“白天/黑夜”周期 (可选，根据 peaks 自动标注或手动标注)
    # 这里简单画几条竖线分隔天数 (每24小时)
    for day in range(1, int(max(hours) / 24) + 1):
        plt.axvline(x=day * 24, color='gray', linestyle='--', alpha=0.5)
        plt.text(day * 24 - 12, max(counts) * 0.95, f'Day {day}', ha='center', fontsize=12, color='gray')
        plt.text(day * 24 + 12, max(counts) * 0.95, f'Day {day + 1}', ha='center', fontsize=12, color='gray')

    plt.tight_layout()

    # 5. 保存
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"图表已保存至: {OUTPUT_IMG}")

    # 显示 (如果在本地运行)
    # plt.show()


if __name__ == "__main__":
    plot_temporal_activity()

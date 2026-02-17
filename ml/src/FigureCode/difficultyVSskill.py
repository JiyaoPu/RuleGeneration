#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
本脚本从本地读取9个 CSV 文件，每个文件包含列：
    - "CooperationExtrinsicReward"
    - "CheatExtrinsicReward"
    - "Skill"
对于每个文件，使用 difficulty_translation 将前两列转化为 difficulty，
并对相同 difficulty 的 Skill 求均值，
最后将9个 epoch 对应的 difficulty–skill 曲线绘制在同一张图中。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns


def difficulty_translation(extrinsic_reward, max_reward=0.01):
    """
    将 extrinsic reward 转换为 difficulty 值。

    参数:
        extrinsic_reward (array or list): 包含合作和作弊的奖励，
                                            第一个值对应合作奖励，
                                            第二个值对应作弊奖励，
                                            取值范围均为 [-max_reward, max_reward]。
        max_reward (float): 最大奖励绝对值，默认 3.0。

    返回:
        difficulty (float): 映射后的难度值，范围为 [0, 1]，
                            数值越低代表游戏越简单（奖励越高），
                            数值越高代表游戏越困难（奖励越低）。
    """
    extrinsic_reward = np.array(extrinsic_reward)
    avg_reward = np.mean(extrinsic_reward)
    difficulty = 1 - ((avg_reward + max_reward) / (2 * max_reward))
    return difficulty


def process_csv_for_epoch(csv_file):
    """
    读取一个 CSV 文件，计算每一行的 difficulty（保留两位小数），
    并对相同 difficulty 的 Skill 值求均值，返回排序后的 x,y 数据。

    返回:
        diff_sorted: 按 difficulty 升序排列的 difficulty 数值列表
        skill_sorted: 对应的平均 Skill 列表
    """
    df = pd.read_csv(csv_file)
    required_cols = {"CooperationExtrinsicReward", "CheatExtrinsicReward", "Skill"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"文件 {csv_file} 缺少必要的列。")

    diff_map = defaultdict(list)
    for _, row in df.iterrows():
        coop = row["CooperationExtrinsicReward"]
        cheat = row["CheatExtrinsicReward"]
        skill = (row["Skill"]-200)/100
        d = difficulty_translation([coop, cheat])
        # 四舍五入到两位，便于归类
        d_round = round(d, 2)
        diff_map[d_round].append(skill)

    diff_vals = []
    skill_means = []
    for d, s_list in diff_map.items():
        diff_vals.append(d)
        skill_means.append(np.mean(s_list))

    sorted_pairs = sorted(zip(diff_vals, skill_means), key=lambda x: x[0])
    diff_sorted = [p[0] for p in sorted_pairs]
    skill_sorted = [p[1] for p in sorted_pairs]
    return diff_sorted, skill_sorted


def plot_skill_vs_difficulty_from_csv_files(csv_files, output_path="skill_vs_difficulty_9epochs.png"):
    """
    从多个 CSV 文件中读取数据，每个 CSV 文件对应一个 epoch，
    对每个文件根据 [CooperationExtrinsicReward, CheatExtrinsicReward] 计算 difficulty，
    并求得该 epoch 下相同 difficulty 的 Skill 均值，
    最后将这些 epoch 对应的 difficulty–skill 曲线绘制在同一张图中。

    参数:
        csv_files (list of str): 9个 CSV 文件的路径，文件名中最好包含 epoch 信息（例如 "epoch0.csv", "epoch5.csv", …）
        output_path (str): 保存图像的路径
    """
    # 定义9种颜色（顺序可自定义）
    colors = ['red', 'orange', 'teal', 'green', 'blue', 'indigo', 'violet', 'brown', 'slateblue']

    plt.figure(figsize=(10, 6))

    legend_labels = []

    for idx, csv_file in enumerate(csv_files):
        try:
            x, y = process_csv_for_epoch(csv_file)
        except Exception as e:
            print(f"读取 {csv_file} 时出错：{e}")
            continue

        plt.plot(x, y, marker='o', linestyle='-', color=colors[idx % len(colors)], linewidth=2,
                 label=os.path.splitext(os.path.basename(csv_file))[0])
        legend_labels.append(os.path.splitext(os.path.basename(csv_file))[0])

    plt.xlabel("Difficulty", fontsize=12)
    plt.ylabel("Skill", fontsize=12)
    plt.title("Skill vs Difficulty", fontsize=14)
    plt.legend(title="Epoch", fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    print(f"折线图已保存至 {output_path}")
    plt.show()


def plot_epoch_difficulty_heatmap(csv_files, num_bins=13, output_path="epoch_difficulty_heatmap.png"):
    """
    从多个 CSV 文件（预期为9个，每个对应一个 epoch）中读取数据，
    根据 "CooperationExtrinsicReward" 和 "CheatExtrinsicReward" 计算 difficulty，
    将 difficulty 均匀划分为 num_bins 个区间，并对每个 bin 内的 Skill 求平均，
    构造一个 (n_epochs x num_bins) 的矩阵，最后用热力图展示，
    X 轴为 difficulty（13个 bin），Y 轴为 epoch（9个，均匀映射到 0~1，下方为 0，上方为 1）。
    """

    n_epochs = len(csv_files)  # 预期为9个文件
    # 将 difficulty 均匀划分为 [0,1] 中的 num_bins 个区间
    bins = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # 初始化矩阵：行数为 epoch 数，列数为 bin 数
    heatmap_matrix = np.full((n_epochs, num_bins), np.nan)

    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        # 计算每行的 difficulty
        difficulties = df.apply(lambda row: difficulty_translation(
            [row["CooperationExtrinsicReward"], row["CheatExtrinsicReward"]]), axis=1)
        skills = df["Skill"].values

        # 对每个 bin 内的数据求平均
        for j in range(num_bins):
            in_bin = (difficulties >= bins[j]) & (difficulties < bins[j + 1])
            if np.any(in_bin):
                heatmap_matrix[i, j] = np.mean(skills[in_bin])

    # 如果有空值，用整个矩阵的均值填充，保证热力图无空白
    overall_mean = np.nanmean(heatmap_matrix)
    heatmap_matrix = np.nan_to_num(heatmap_matrix, nan=overall_mean)

    # 将9个 epoch 均匀映射到 0~1 作为 Y 轴标签（下方为0，上方为1）
    y_labels = np.linspace(0, 1, n_epochs)
    # X 轴标签使用 bin_centers（保留两位小数）
    x_labels = np.round(bin_centers, 2)

    plt.figure(figsize=(12, 6))
    # 使用 sns.heatmap 绘图，设置 linewidths=0 消除单元格间隙
    ax = sns.heatmap(heatmap_matrix, xticklabels=x_labels, yticklabels=np.round(y_labels, 2),
                     cmap="viridis", cbar_kws={'label': 'Income'}, linewidths=0)
    ax.set_xlabel("Difficulty", fontsize=12, labelpad=10)
    ax.set_ylabel("Skill", fontsize=12, labelpad=10)
    ax.set_title("Income vs Difficulty", fontsize=16)
    # 翻转 Y 轴，使得最小值 (0) 在下方，最大值 (1) 在上方
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"热力图已保存至 {output_path}")
    plt.show()


def plot_skill_vs_difficulty_with_range(csv_file,color,output_path=None):

    # 读取 CSV 文件
    df = pd.read_csv(csv_file)
    # 计算每行的 difficulty（四舍五入到两位以便归类）
    df['difficulty'] = df.apply(lambda row: round(
        difficulty_translation([row['CooperationExtrinsicReward'], row['CheatExtrinsicReward']]), 2), axis=1)

    # 按 difficulty 分组，计算 Skill 的平均值、最小值和最大值
    grouped = df.groupby('difficulty')['Skill']
    diff_vals = []
    mean_vals = []
    min_vals = []
    max_vals = []
    for d, group in grouped:
        diff_vals.append(d)
        mean_vals.append(group.mean())
        min_vals.append(group.min())
        max_vals.append(group.max())

    # 排序（difficulty 从小到大）
    sorted_idx = np.argsort(diff_vals)
    diff_vals = np.array(diff_vals)[sorted_idx]
    mean_vals = (np.array(mean_vals)[sorted_idx]-200)/100
    min_vals = (np.array(min_vals)[sorted_idx]-200)/100
    max_vals = (np.array(max_vals)[sorted_idx]-200)/100

    # 绘图：平均值折线，并用填充区域显示最小值-最大值范围
    plt.figure(figsize=(10, 6))
    plt.plot(diff_vals, mean_vals, marker='o', color=color, label='Mean Skill')
    plt.fill_between(diff_vals, min_vals, max_vals, color=color, alpha=0.2, label='Skill Range')
    plt.xlabel("Difficulty")
    plt.ylabel("Skill")
    plt.title(f"{os.path.basename(csv_file)}"[:-4])
    plt.legend()
    plt.grid(True)
    if output_path is not None:
        plt.savefig(output_path, dpi=300)
        print(f"图像已保存至 {output_path}")
    plt.show()


if __name__ == "__main__":
    # 假设文件位于当前目录，文件名分别为 "epoch0.csv", "epoch5.csv", …, "epoch40.csv"
    csv_files = [
        "../data/skill_extrinsic_reward_0.01/epoch0.csv",
        "../data/skill_extrinsic_reward_0.01/epoch5.csv",
        "../data/skill_extrinsic_reward_0.01/epoch10.csv",
        "../data/skill_extrinsic_reward_0.01/epoch15.csv",
        "../data/skill_extrinsic_reward_0.01/epoch20.csv",
        "../data/skill_extrinsic_reward_0.01/epoch25.csv",
        "../data/skill_extrinsic_reward_0.01/epoch30.csv",
        "../data/skill_extrinsic_reward_0.01/epoch35.csv",
        "../data/skill_extrinsic_reward_0.01/epoch40.csv"
    ]
    # plot_skill_vs_difficulty_from_csv_files(csv_files)
    # color = ['red', 'orange', 'teal', 'green', 'blue', 'indigo', 'violet', 'brown', 'slateblue']
    # epochid = 8
    # plot_skill_vs_difficulty_with_range(csv_files[epochid],color[epochid])
    plot_epoch_difficulty_heatmap(csv_files,13)
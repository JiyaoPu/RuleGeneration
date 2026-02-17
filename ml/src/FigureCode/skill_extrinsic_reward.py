

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D



def plot_skill_layer_from_csv(csv_file, output_path=None):
    """
    读取 CSV 文件并展示 Skill 层。

    CSV 文件要求包含以下列：
      - "CooperationExtrinsicReward"
      - "CheatExtrinsicReward"
      - "Skill"

    假设 CSV 中的数据是一个网格（flatten 后保存），本函数会自动根据数据行数计算网格维度（n x n）。
    如果数据行数不是完美平方，则采用散点图展示。

    参数:
      csv_file (str): CSV 文件路径。
      output_path (str): 如果指定，则保存绘图到该路径；否则直接显示图像。
    """


    # 读取 CSV 文件
    df = pd.read_csv(csv_file)
    # 提取数据
    X_flat = df["CooperationExtrinsicReward"].values
    Y_flat = df["CheatExtrinsicReward"].values
    Z_flat = df["Skill"].values

    # 尝试根据数据行数重塑成 n x n 的网格
    n = int(np.sqrt(len(df)))
    is_grid = (n * n == len(df))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if is_grid:
        # 重塑数据为 n x n 数组
        X = X_flat.reshape((n, n))
        Y = Y_flat.reshape((n, n))
        Z = Z_flat.reshape((n, n))
        # 绘制曲面图
        surf = ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none', alpha=0.85)
        fig.colorbar(surf, shrink=0.5, aspect=10)
    else:
        # 数据行数不是完美平方，采用散点图
        ax.scatter(X_flat, Y_flat, Z_flat, c=Z_flat, cmap='plasma', marker='o')

    # 设置坐标轴标签和标题
    ax.set_xlabel("Cooperation Extrinsic Reward", fontsize=12, labelpad=10)
    ax.set_ylabel("Cheat Extrinsic Reward", fontsize=12, labelpad=10)
    ax.set_zlabel("Player Income", fontsize=12, labelpad=10)
    ax.set_title("Epoch 40", fontsize=16)
    ax.set_xlim(3, -3)

    plt.tight_layout()
    if output_path is not None:
        # 确保保存目录存在
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig(output_path, dpi=300)
        print(f"图像已保存至 {output_path}")
    plt.show()


def plot_multiple_layers_from_csv_same_figure(csv_files, colors=None, output_path=None):
    """
    读取多个 CSV 文件，每个 CSV 文件应包含以下列：
      - "CooperationExtrinsicReward"
      - "CheatExtrinsicReward"
      - "Skill"
    并在同一 3D 图中用不同颜色绘制各层数据。

    参数:
      csv_files (list of str): CSV 文件路径列表。
      colors (list of str): 每层对应的颜色列表，默认使用 ['red', 'yellow', 'blue', 'green']。
      output_path (str): 如果指定，则保存图像到该路径；否则直接显示图像。
    """


    if colors is None:
        colors = ['blue', 'green', 'yellow', 'red']  # 默认颜色，可根据 csv_files 数量扩充

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    legend_elements = []

    for idx, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        # 提取数据
        X_flat = df["CooperationExtrinsicReward"].values
        Y_flat = df["CheatExtrinsicReward"].values
        Z_flat = df["Skill"].values

        # 尝试重塑为网格
        n = int(np.sqrt(len(df)))
        is_grid = (n * n == len(df))
        if is_grid:
            X = X_flat.reshape((n, n))
            Y = Y_flat.reshape((n, n))
            Z = Z_flat.reshape((n, n))
            ax.plot_surface(X, Y, Z, color=colors[idx % len(colors)], alpha=0.8, edgecolor='none')
        else:
            ax.scatter(X_flat, Y_flat, Z_flat, c=Z_flat, cmap='plasma', marker='o')

        # 使用文件名（去掉路径和扩展名）作为图例标签
        label = os.path.splitext(os.path.basename(csv_file))[0]
        legend_elements.append(Line2D([0], [0], marker='s', color='w',
                                      label=label,
                                      markerfacecolor=colors[idx % len(colors)],
                                      markersize=10))

    ax.set_xlabel("Cooperation Extrinsic Reward", fontsize=12, labelpad=10)
    ax.set_ylabel("Cheat Extrinsic Reward", fontsize=12, labelpad=10)
    ax.set_zlabel("Player Income", fontsize=12, labelpad=10)
    ax.set_title(" ", fontsize=16)

    # # 反转 X 和 Y 轴，使得数值从 3 到 -3（根据你的数据范围）
    ax.set_xlim(3, -3)


    ax.legend(handles=legend_elements, loc='upper left')
    plt.tight_layout()

    if output_path is not None:
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig(output_path, dpi=300)
        print(f"图像已保存至 {output_path}")

    plt.show()


#
plot_skill_layer_from_csv('../data/skill_extrinsic_reward/epoch40.csv')


csv_files = [
    "../data/skill_extrinsic_reward/epoch0.csv",
    "../data/skill_extrinsic_reward/epoch5.csv",
    "../data/skill_extrinsic_reward/epoch10.csv",
    "../data/skill_extrinsic_reward/epoch20.csv",
    "../data/skill_extrinsic_reward/epoch40.csv"
]
# plot_multiple_layers_from_csv_same_figure(csv_files, colors=['violet','yellow', 'green', 'blue', 'red'], output_path="./data/skill_extrinsic_reward/multiple_layers.png")

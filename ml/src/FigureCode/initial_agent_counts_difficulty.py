import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.stats import pearsonr
from matplotlib.lines import Line2D


def visualize_initial_agent_counts_influence(csv_file, output_dir="./visualization/"):
    """
    从 CSV 文件中读取初始代理数量数据（agent_count_1 ~ agent_count_8）及对应的 income，
    生成以下图表：
      1. 对每个 agent_count 维度绘制散点图（X：该维度取值，Y：income），并计算皮尔逊相关系数，
         并拟合趋势曲线，展示每个维度对 income 的影响。
      2. 利用 t-SNE 将 8 维数据降到 2 维，绘制散点图，颜色按 income 着色，
         展示数据整体结构与 income 的关系。
      3. 根据 income 分为低、中、高三组，计算各组各 agent_count 的均值，用雷达图展示不同组的平均分布。

    参数：
      csv_file (str): CSV 文件路径，要求包含 agent_count_1 ~ agent_count_8 及 income 列。
      output_dir (str): 生成的图像保存目录，默认为 "./visualization/"。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 读取数据
    df = pd.read_csv(csv_file)
    feature_cols = ["Random", "Cheater", "Cooperator", "Copycat", "Grudger", "Detective", "AI", "Human"]
    if not set(feature_cols + ["Income"]).issubset(df.columns):
        raise ValueError("CSV 文件中必须包含 " + ", ".join(feature_cols) + " 和 Income 列")
    income = df["Income"].values

    # 1. 每个维度与 income 的散点图和趋势曲线
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()
    poly_degree = 4  # 多项式阶数，可根据数据调整

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        x_vals = df[col].values  # 假设为整数
        # 根据 x 值分组，计算各组对应的 income 数组
        unique_x = np.unique(x_vals)
        groups = [income[x_vals == ux] for ux in unique_x]

        # 绘制箱线图：positions 指定各组的 x 坐标
        bp = ax.boxplot(groups, positions=unique_x, widths=0.6, patch_artist=True,
                        medianprops=dict(color='#54bf98'))
        # 设置箱体颜色为浅绿色
        for patch in bp['boxes']:
            patch.set_facecolor('#54bf98')

        # 计算每个唯一 x 值对应的均值
        mean_incomes = np.array([np.mean(g) for g in groups])

        # 绘制均值点
        ax.scatter(unique_x, mean_incomes, color='yellow', zorder=3, label="Mean")

        # 对聚合后的均值进行回归拟合
        coef = np.polyfit(unique_x, mean_incomes, poly_degree)
        poly_fn = np.poly1d(coef)
        x_line = np.linspace(np.min(unique_x), np.max(unique_x), 200)
        ax.plot(x_line, poly_fn(x_line), color='blue', linewidth=2, label="Trend")

        # 计算皮尔逊相关系数，基于所有原始数据
        r, p = pearsonr(x_vals, income)
        ax.set_xlabel(col, fontsize=14)
        ax.set_ylabel("Income", fontsize=14)
        ax.set_title(f"r = {r:.2f}", fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    plt.suptitle("", fontsize=20)
    scatter_out = os.path.join(output_dir, "agent_counts_boxplot_trend.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(scatter_out, dpi=300)
    print(f"箱线图已保存至 {scatter_out}")
    plt.show()

    # 2. 利用 t-SNE 降维 8 维数据，并绘制二维散点图
    X_data = df[feature_cols].values
    tsne_model = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne_model.fit_transform(X_data)
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=income, cmap="viridis", s=50, alpha=0.8)
    plt.xlabel("t-SNE 1", fontsize=14)
    plt.ylabel("t-SNE 2", fontsize=14)
    plt.title("t-SNE of Initial Agent Counts (colored by Income)", fontsize=16)
    cbar = plt.colorbar(sc)
    cbar.set_label("Income", fontsize=14)
    tsne_out = os.path.join(output_dir, "initial_agent_counts_tsne.png")
    plt.tight_layout()
    plt.savefig(tsne_out, dpi=300)
    print(f"t-SNE 图已保存至 {tsne_out}")
    plt.show()

    # 3. 雷达图展示不同难度组的初始代理数量均值
    # 按 income 分为三组（低、中、高），这里使用 33% 和 66% 分位数
    low_threshold, high_threshold = np.percentile(income, [33, 66])
    groups = {
        "Low Income": df[df["Income"] <= low_threshold],
        "Medium Income": df[(df["Income"] > low_threshold) & (df["Income"] <= high_threshold)],
        "High Income": df[df["Income"] > high_threshold]
    }
    radar_data = {}
    for key, group_df in groups.items():
        # 计算每个组内各 agent_count 的均值
        radar_data[key] = group_df[feature_cols].mean().values

    labels = feature_cols
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for key, values in radar_data.items():
        values = values.tolist()
        values += values[:1]
        ax.plot(angles, values, label=key, linewidth=2)
        ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_title("", fontsize=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
    radar_out = os.path.join(output_dir, "initial_agent_counts_radar.png")
    plt.tight_layout()
    plt.savefig(radar_out, dpi=300)
    print(f"雷达图已保存至 {radar_out}")
    plt.show()

    # 4. 导出不同组的初始代理数量数据到 CSV 文件
    for group_name, group_df in groups.items():
        output_file = os.path.join(output_dir, f"agent_counts_{group_name.replace(' ', '_')}.csv")
        group_df.to_csv(output_file, index=False)
        print(f"已保存 {group_name} 组的数据到 {output_file}")


if __name__ == "__main__":
    # 修改为实际 CSV 文件路径
    csv_file = "../data/initial_agent_counts_difficulty/initial_agent_counts_difficulty.csv"
    visualize_initial_agent_counts_influence(csv_file, output_dir="./visualization/")


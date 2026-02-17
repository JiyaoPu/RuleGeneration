import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.stats import pearsonr
from matplotlib.lines import Line2D

def visualize_tsne(csv_file, output_path=None, perplexity=30, random_state=42):
    """
    从 CSV 文件中读取 trading rule 数据（6 维）和对应的 difficulty，
    使用 t-SNE 将 6 维数据降至 2 维，并用散点图展示，
    点的颜色根据 difficulty 值编码。

    参数：
      csv_file (str): CSV 文件路径，文件中应包含 trading_rule_1 ... trading_rule_6 及 difficulty 列
      output_path (str): 如果指定，则保存图像到该路径，否则直接显示
      perplexity (int): t-SNE 的 perplexity 参数，默认为 30（需小于样本数）
      random_state (int): 随机种子
    """
    # 读取 CSV 数据
    df = pd.read_csv(csv_file)

    # 提取交易规则的6个维度（假设列名为 trading_rule_1 ~ trading_rule_6）
    features = df[[f"trading_rule_{i + 1}" for i in range(6)]].values
    # 提取对应的 difficulty 值
    difficulty = df["income"].values

    # t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    tsne_result = tsne.fit_transform(features)

    # 绘制散点图
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=difficulty, cmap="viridis", s=50)
    plt.xlabel("t-SNE 1", fontsize=14)
    plt.ylabel("t-SNE 2", fontsize=14)
    plt.title("t-SNE of Trading Rule (colored by Difficulty)", fontsize=16)
    cbar = plt.colorbar(sc)
    cbar.set_label("Difficulty", fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"t-SNE 图已保存至 {output_path}")
    plt.show()


def visualize_radar(csv_file, sample_index=0, output_path=None):
    """
    从 CSV 文件中读取交易规则数据（6 维）及对应的 difficulty，
    对指定样本绘制雷达图展示其交易规则参数，并在图中标注 difficulty 值。

    参数：
      csv_file (str): CSV 文件路径，文件中应包含 trading_rule_1 ... trading_rule_6 及 difficulty 列
      sample_index (int): 选择绘制第几个样本，默认 0
      output_path (str): 如果指定，则保存图像到该路径，否则直接显示
    """
    # 读取数据
    df = pd.read_csv(csv_file)

    # 检查是否存在足够样本
    if sample_index >= len(df):
        raise IndexError("sample_index 超出数据行数")

    # 提取交易规则参数和 difficulty
    feature_cols = [f"trading_rule_{i + 1}" for i in range(6)]
    sample = df.iloc[sample_index][feature_cols].values
    diff_val = df.iloc[sample_index]["income"]

    # 为雷达图准备数据：闭合曲线需要重复第一个点
    values = np.concatenate([sample, [sample[0]]])  # 7个点
    labels = feature_cols + [feature_cols[0]]

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, len(sample), endpoint=False)  # 生成6个角度
    angles = np.concatenate([angles, [angles[0]]])  # 追加第一个角度，得到7个角度

    # 绘制雷达图
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='b', linewidth=2, label=f"Difficulty: {diff_val:.2f}")
    ax.fill(angles, values, color='b', alpha=0.25)

    # 设置雷达图标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_cols)
    # 设置 y 轴范围，可以根据数据调整
    ax.set_ylim(np.min(values) - 1, np.max(values) + 1)

    plt.title(f"Trading Rule Radar (Sample {sample_index}, Difficulty={diff_val:.2f})", fontsize=16)
    plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"雷达图已保存至 {output_path}")
    plt.show()


def visualize_trading_rule_influence(csv_file, output_dir="./visualization/"):
    """
    从 CSV 文件中读取 6 维交易规则数据及对应的 income（作为 difficulty 指标），
    生成以下图表：
      1. 对每个交易规则维度绘制散点图（X：该维度取值，Y：income），并标注相关系数，
         用以展示每个维度对 difficulty 的影响。
      2. 利用 t-SNE 将 6 维交易规则数据降至 2 维，绘制散点图，颜色按 income 着色，
         展示交易规则整体结构与 difficulty 的关系。

    参数：
      csv_file (str): 数据 CSV 文件路径，要求包含 trading_rule_1 ~ trading_rule_6 及 income 列。
      output_dir (str): 生成的图像保存目录，默认为 "./visualization/"。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 读取数据
    df = pd.read_csv(csv_file)
    feature_cols = [f"Trade rule {i + 1}" for i in range(6)]
    if not set(feature_cols + ["income"]).issubset(df.columns):
        raise ValueError("CSV 文件中缺少必要的列，请确保包含 trading_rule_1~6 和 income。")

    # 提取特征和目标
    X_data = df[feature_cols].values
    income = df["income"].values

    # 1. 绘制每个维度与 income 的散点图，并计算相关系数
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    poly_degree = 4  # 可以调整多项式的阶数，较高的阶数允许过拟合
    for i, col in enumerate(feature_cols):
        ax = axes[i]
        x_vals = df[col].values
        ax.scatter(x_vals, income, alpha=0.7, color='green', edgecolor='k')
        # 使用高阶多项式拟合所有数据点
        coef = np.polyfit(x_vals, income, poly_degree)
        poly_fn = np.poly1d(coef)
        # 用更多的点绘制曲线，使曲线更平滑
        x_line = np.linspace(np.min(x_vals), np.max(x_vals), 200)
        ax.plot(x_line, poly_fn(x_line), color='blue', linewidth=2, label=f"Trend Line")
        # 计算皮尔逊相关系数
        r, p = pearsonr(x_vals, income)
        ax.set_xlabel(col, fontsize=14)
        ax.set_ylabel("Income", fontsize=14)
        ax.set_title(f"r = {r:.2f}", fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
    plt.suptitle("Each Trade Rule Dimension vs Income", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    scatter_out = os.path.join(output_dir, "trading_rule_vs_income_scatter.png")
    plt.savefig(scatter_out, dpi=300)
    print(f"每维散点图已保存至 {scatter_out}")
    plt.show()

    # 2. 利用 t-SNE 降维并绘制二维散点图
    tsne_model = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne_model.fit_transform(X_data)
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=income, cmap="viridis", s=50, alpha=0.8)
    plt.xlabel("t-SNE 1", fontsize=14)
    plt.ylabel("t-SNE 2", fontsize=14)
    plt.title("t-SNE of Trading Rule Samples (colored by Income)", fontsize=16)
    cbar = plt.colorbar(sc)
    cbar.set_label("Income", fontsize=14)
    plt.tight_layout()
    tsne_out = os.path.join(output_dir, "trading_rule_tsne.png")
    plt.savefig(tsne_out, dpi=300)
    print(f"t-SNE 图已保存至 {tsne_out}")
    plt.show()

    # 3. （可选）雷达图展示不同难度组的交易规则均值
    # 将数据按照 income 分为三组：低、中、高难度（可以根据分位数划分）
    low_threshold, high_threshold = np.percentile(income, [33, 66])
    groups = {
        "Low Income": df[df["income"] <= low_threshold],
        "Medium Income": df[(df["income"] > low_threshold) & (df["income"] <= high_threshold)],
        "High Income": df[df["income"] > high_threshold]
    }

    radar_data = {}
    for key, group_df in groups.items():
        radar_data[key] = group_df[feature_cols].mean().values

    # 构造雷达图数据
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
    radar_out = os.path.join(output_dir, "trading_rule_radar.png")
    plt.tight_layout()
    plt.savefig(radar_out, dpi=300)
    print(f"雷达图已保存至 {radar_out}")
    plt.show()

    # 假设 df 中包含交易规则列（如 "trading_rule_1" ... "trading_rule_6"）和 "income" 列
    low_threshold, high_threshold = np.percentile(df["income"].values, [33, 66])
    groups = {
        "Low_Income": df[df["income"] <= low_threshold],
        "Medium_Income": df[(df["income"] > low_threshold) & (df["income"] <= high_threshold)],
        "High_Income": df[df["income"] > high_threshold]
    }

    # 导出每个组的交易规则到 CSV 文件
    for group_name, group_df in groups.items():
        output_file = f"trade_rules_{group_name}.csv"
        group_df.to_csv(output_file, index=False)
        print(f"已保存 {group_name} 组的交易规则到 {output_file}")


# 示例调用：
# visualize_trading_rule_influence("trading_rule_difficulty.csv")
# 请将文件路径和输出目录按实际情况修改

filelocation = "../data/trading_rule_difficulty/trading_rule_difficulty.csv"
# visualize_tsne(filelocation, output_path="tsne_trading_rule.png", perplexity=10)
# visualize_radar(filelocation, sample_index=0, output_path="radar_sample0.png")
visualize_trading_rule_influence(filelocation, output_dir="../data/trading_rule_difficulty/")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


def visualize_1d_difficulty(csv_file, param_col, income_col, title, output_path):
    """
    从 CSV 文件中读取数据（要求包含指定的参数列和收入列），
    绘制箱线图展示每个参数值下的 Income 分布，并叠加均值点和趋势线，
    同时计算并在标题中显示皮尔逊相关系数。

    参数：
      csv_file (str): CSV 文件路径
      param_col (str): 参数列名（例如 "round_number"）
      income_col (str): 收入列名（例如 "income"）
      title (str): 图表标题
      output_path (str): 输出图像保存路径
    """
    df = pd.read_csv(csv_file)
    if not set([param_col, income_col]).issubset(df.columns):
        raise ValueError(f"CSV 文件中必须包含 {param_col} 和 {income_col} 列")

    # 获取所有唯一的参数值及对应的收入数组
    unique_vals = np.sort(df[param_col].unique())
    groups = [df[df[param_col] == val][income_col].values for val in unique_vals]

    plt.figure(figsize=(10, 6))
    # 绘制箱线图，指定箱体位置为 unique_vals
    bp = plt.boxplot(groups, positions=unique_vals, widths=0.6, patch_artist=True,
                     medianprops=dict(color='black'))
    # 设置箱体颜色（这里设置为浅绿色）
    for patch in bp['boxes']:
        patch.set_facecolor('#54bf98')

    # 计算每个唯一参数值对应的均值
    means = np.array([np.mean(g) for g in groups])
    # 绘制均值点（用黄色标记）
    plt.scatter(unique_vals, means, color='yellow', zorder=3, label="Mean")

    # 对均值点进行线性拟合（这里使用一阶多项式）
    poly_degree = 1
    coef = np.polyfit(unique_vals, means, poly_degree)
    poly_fn = np.poly1d(coef)
    x_line = np.linspace(np.min(unique_vals), np.max(unique_vals), 200)
    plt.plot(x_line, poly_fn(x_line), color='blue', linewidth=2, label="Trend Line")

    # 计算皮尔逊相关系数（基于原始所有数据）
    r, p = pearsonr(df[param_col].values, df[income_col].values)
    plt.xlabel(param_col, fontsize=14)
    plt.ylabel(income_col, fontsize=14)
    plt.title(f"{title}\nr = {r:.2f}", fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"图像已保存至 {output_path}")
    plt.show()


def visualize_mistake_possibility_influence(csv_file, output_dir="./visualization/"):
    """
    从 CSV 文件中读取 mistake_possibility 数据及对应的 income，
    绘制散点图展示 mistake_possibility 与 income 的关系，并计算皮尔逊相关系数，
    同时拟合回归趋势线展示整体趋势。

    参数：
      csv_file (str): CSV 文件路径，要求包含 "mistake_possibility" 和 "income" 列。
      output_dir (str): 图像保存目录，默认为 "./visualization/"。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 读取数据
    df = pd.read_csv(csv_file)
    if not {"mistake_possibility_1", "income"}.issubset(df.columns):
        raise ValueError("CSV 文件中必须包含 mistake_possibility 和 income 列")

    x_vals = df["mistake_possibility_1"].values
    income = df["income"].values

    plt.figure(figsize=(10, 6))
    # 绘制散点图
    plt.scatter(x_vals, income, color='green', alpha=0.7, edgecolor='k', label="Data Points")

    # 使用多项式拟合回归趋势线，阶数可根据数据调整（这里选用4阶）
    poly_degree = 4
    coef = np.polyfit(x_vals, income, poly_degree)
    poly_fn = np.poly1d(coef)
    x_line = np.linspace(np.min(x_vals), np.max(x_vals), 200)
    plt.plot(x_line, poly_fn(x_line), color='blue', linewidth=2, label="Trend Line")

    # 计算皮尔逊相关系数
    r, p = pearsonr(x_vals, income)

    plt.xlabel("Mistake Possibility", fontsize=14)
    plt.ylabel("Income", fontsize=14)
    plt.title(f"\nr = {r:.2f}", fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "mistake_possibility_influence.png")
    plt.savefig(output_path, dpi=300)
    print(f"图像已保存至 {output_path}")
    plt.show()


if __name__ == "__main__":
    # 定义三个 CSV 文件的路径（请根据实际情况修改路径）
    round_csv = "../data/round_number_difficulty/round_number_difficulty.csv"
    reproduction_csv = "../data/reproduction_number_difficulty/reproduction_number_difficulty.csv"
    mistake_csv = "../data/mistake_possibility_difficulty/mistake_possibility_difficulty.csv"

    # 可视化 Round Number 与 Income（箱线图 + 趋势线）
    visualize_1d_difficulty(round_csv,
                            param_col="Round number",
                            income_col="Income",
                            title="",
                            output_path="./visualization/round_number_boxplot.png")

    # 可视化 Reproduction Number 与 Income
    visualize_1d_difficulty(reproduction_csv,
                            param_col="Reproduction number",
                            income_col="Income",
                            title="",
                            output_path="./visualization/reproduction_number_boxplot.png")

    # # 可视化 Mistake Possibility 与 Income
    # visualize_1d_difficulty(mistake_csv,
    #                         param_col="mistake_possibility",
    #                         income_col="income",
    #                         title="Income vs. Mistake Possibility",
    #                         output_path="./visualization/mistake_possibility_boxplot.png")
    visualize_mistake_possibility_influence(mistake_csv, output_dir="./visualization/mistake_possibility_boxplot.png")
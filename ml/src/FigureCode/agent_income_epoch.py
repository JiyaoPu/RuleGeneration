import pandas as pd
import matplotlib.pyplot as plt


def plot_individual_groups(files, group_colors, figsize=(10, 6)):
    """
    对每个难度（High/Medium/Low）的 CSV 文件，
    绘制各组的平滑曲线及该文件内整体（所有组）的均值±2σ的阴影区域。
    """
    for label, filepath in files.items():
        df = pd.read_csv(filepath)
        df = df.sort_values('epoch')

        # 计算全局统计信息：所有组的 income 按 epoch 聚合的均值和标准差
        overall_stats = df.groupby('epoch')['income'].agg(['mean', 'std']).sort_index()
        # 平滑处理（滚动窗口大小为5）
        overall_stats['mean_smooth'] = overall_stats['mean'].rolling(window=5, min_periods=1, center=True).mean()
        overall_stats['std_smooth'] = overall_stats['std'].rolling(window=5, min_periods=1, center=True).mean()

        fig, ax = plt.subplots(figsize=figsize)

        # 绘制全局范围阴影区域（均值±2σ），这里用淡紫色
        ax.fill_between(
            overall_stats.index,
            overall_stats['mean_smooth'] - 2 * overall_stats['std_smooth'],
            overall_stats['mean_smooth'] + 2 * overall_stats['std_smooth'],
            color='lavender',
            alpha=1,
            label='Overall Range'
        )

        # 遍历每个 group，绘制平滑曲线
        groups = sorted(df['group'].unique())
        for i, group in enumerate(groups):
            group_df = df[df['group'] == group].copy().sort_values('epoch')
            group_df['smoothed'] = group_df['income'].rolling(window=5, min_periods=1, center=True).mean()
            color = group_colors[i % len(group_colors)]
            # 使用 group 字符串的最后一个字符作为玩家编号显示，如 'Player 1'
            ax.plot(group_df['epoch'], group_df['smoothed'],
                    label=f'Player {group[-1]}',
                    linewidth=2.0,
                    color=color)

        ax.set_title(f'{label} Difficulty', fontsize=16)
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('Income', fontsize=14)
        ax.legend(fontsize=12)
        plt.tight_layout()
        plt.show()


def plot_aggregated_trends(files, trend_colors, figsize=(10, 6)):
    """
    将 High、Medium、Low 三个难度的总体趋势绘制在一张图上，
    每个难度显示均值的平滑曲线和均值±2σ的阴影区域，
    并取消图例显示。
    """
    plt.figure(figsize=figsize)

    for label, filepath in files.items():
        df = pd.read_csv(filepath)
        df = df.sort_values('epoch')

        # 计算总体均值与标准差（按 epoch 聚合）
        overall_stats = df.groupby('epoch')['income'].agg(['mean', 'std']).sort_index()
        overall_stats['mean_smooth'] = overall_stats['mean'].rolling(window=5, min_periods=1, center=True).mean()
        overall_stats['std_smooth'] = overall_stats['std'].rolling(window=5, min_periods=1, center=True).mean()

        color = trend_colors[label]
        # 绘制阴影区域（均值±2σ），透明度设置为0.2
        plt.fill_between(
            overall_stats.index,
            overall_stats['mean_smooth'] - 2 * overall_stats['std_smooth'],
            overall_stats['mean_smooth'] + 2 * overall_stats['std_smooth'],
            color=color,
            alpha=0.2
        )
        # 绘制平滑后的均值曲线
        plt.plot(
            overall_stats.index,
            overall_stats['mean_smooth'],
            color=color,
            linewidth=2.0
        )

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Income', fontsize=14)
    plt.title('High, Medium, Low Difficulty', fontsize=16)
    plt.grid(True)
    # 不显示图例
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 定义 CSV 文件路径及标签
    files = {
        'Low': '../data/trading_rule_difficulty/training_process_results_high.csv',
        'Medium': '../data/trading_rule_difficulty/training_process_results_medium.csv',
        'High': '../data/trading_rule_difficulty/training_process_results_low.csv'
    }
    # 定义用于绘制各组平滑曲线的颜色（假设最多5个组）
    group_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # 定义用于聚合趋势图的颜色
    trend_colors = {
        'High': '#d62728',  # 红色
        'Medium': '#ff7f0e',  # 橙色
        'Low': '#1f77b4'  # 蓝色
    }

    # 调用函数绘制各个难度下的个体组别趋势图
    plot_individual_groups(files, group_colors, figsize=(10, 6))

    # 调用函数绘制所有难度的总体趋势图（合并在一张图上），并取消图例
    plot_aggregated_trends(files, trend_colors, figsize=(10, 6))

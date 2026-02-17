import numpy as np
import matplotlib.pyplot as plt

# 定义角色标签（6种固定性格 + Human + AI）
categories = ['RandomAction', 'Always Cheat', 'Always Cooperate', 'Copycat', 'Grudger', 'Detective', 'Human', 'AI']

# 生成5组Gini系数数据，每组对应8个角色
np.random.seed(42)
gini_coefficients = np.random.uniform(0, 1, (5, len(categories)))

# 绘制5个独立的雷达图
for round_index in range(5):
    # 数据准备
    values = gini_coefficients[round_index].tolist()
    values += values[:1]  # 闭合图形

    # 设置雷达图的角度
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    # 创建雷达图
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # 绘制雷达图
    ax.fill(angles, values, color='b', alpha=0.25)
    ax.plot(angles, values, color='b', linewidth=2)

    # 设置角色标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # 设置标题
    # ax.set_title(f'Gini Coefficient', size=16, pad=20)

    # 显示图表
    plt.show()

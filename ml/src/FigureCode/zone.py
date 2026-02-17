import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# 指定包含 CSV 文件的目录
dir_path = "../data/0.02/"

# 排除特定的文件
exclude_files = ['BETA_A.csv', 'BETA_B.csv']

# 获取目录中所有的 CSV 文件，排除指定的文件
all_csv_files = [f for f in glob.glob(os.path.join(dir_path, '*.csv')) if os.path.basename(f) not in exclude_files]

# 初始化列表以存储 (episode, file_path) 元组
episode_file_tuples = []
for f in all_csv_files:
    basename = os.path.basename(f)
    name, ext = os.path.splitext(basename)
    try:
        episode = int(name)
        episode_file_tuples.append((episode, f))
    except ValueError:
        pass

# 根据 episode 对元组列表进行排序
episode_file_tuples.sort(key=lambda x: x[0])

# 分离出 episodes 和文件路径
episodes = np.array([ep for ep, f in episode_file_tuples])
new_file_paths = [f for ep, f in episode_file_tuples]

# 将 CSV 文件读取为 DataFrame
new_dataframes = [pd.read_csv(file_path) for file_path in new_file_paths]

# 假设每个文件中的 'Epoch' 和 'Cooperation Rate' 列存在并且一致
epochs = new_dataframes[0]['Epoch'].values

# 获取合作率网格
cooperation_grid = np.array([df['Cooperation Rate'].values for df in new_dataframes])

# 定义四个区域的 Episode 范围
ranges = [(0, 24), (25, 49), (50, 74), (75, 99)]
colors = ['blue', 'green', 'orange', 'red']
labels = ['Episodes 0-24', 'Episodes 25-49', 'Episodes 50-74', 'Episodes 75-99']

# 创建一个 2D 图形
plt.figure(figsize=(12, 8))

# 遍历每个区域，计算统计量并绘制曲线和阴影区域
for i, (start_ep, end_ep) in enumerate(ranges):
    idx = np.where((episodes >= start_ep) & (episodes <= end_ep))[0]
    if len(idx) == 0:
        continue
    current_cooperation = cooperation_grid[idx, :]
    mean_cooperation = np.mean(current_cooperation, axis=0)
    std_cooperation = np.std(current_cooperation, axis=0)

    # 裁剪均值 ± 标准差到 [0, 1] 范围内
    upper_bound = np.clip(mean_cooperation + std_cooperation, 0, 1)
    lower_bound = np.clip(mean_cooperation - std_cooperation, 0, 1)

    # 绘制平均合作率曲线
    plt.plot(epochs, mean_cooperation, color=colors[i], label=labels[i])

    # 绘制裁剪后的标准差范围
    plt.fill_between(epochs, lower_bound, upper_bound, color=colors[i], alpha=0.2)

# 设置坐标轴标签和标题
plt.xlabel('Epoch')
plt.ylabel('Cooperation Rate')
plt.title('Cooperation Rate = 0.02')

plt.legend()
plt.grid(True)

# 显示图形
plt.tight_layout()
plt.show()

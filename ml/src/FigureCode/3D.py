import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# 指定包含 CSV 文件的目录
dir_path = "data/0.50/"

# 排除特定的文件
exclude_files = ['BETA_A.csv', 'BETA_B.csv']

# 获取目录中所有的 CSV 文件，排除指定的文件
all_csv_files = [f for f in glob.glob(os.path.join(dir_path, '*.csv')) if os.path.basename(f) not in exclude_files]

# 初始化列表以存储 (episode, file_path) 元组
episode_file_tuples = []

# 从文件名中提取 episode 编号，并存储元组
for f in all_csv_files:
    basename = os.path.basename(f)
    name, ext = os.path.splitext(basename)
    try:
        episode = int(name)  # 尝试将文件名（不含扩展名）转换为整数
        episode_file_tuples.append((episode, f))
    except ValueError:
        pass  # 跳过无法转换为整数的文件名

# 根据 episode 对元组列表进行排序
episode_file_tuples.sort(key=lambda x: x[0])

# 分离出 episodes 和文件路径
episodes = np.array([ep for ep, f in episode_file_tuples])
new_file_paths = [f for ep, f in episode_file_tuples]

# 将 CSV 文件读取为 DataFrame
new_dataframes = [pd.read_csv(file_path) for file_path in new_file_paths]

# 假设每个文件中的 'Epoch' 和 'Cooperation Rate' 列存在并且一致
epochs = new_dataframes[0]['Epoch'].values  # 假设 Epoch 列在每个文件中一致

# 获取收入网格（以各个 Episode 为行，Epoch 为列）
income_grid = np.array([df['Cooperation Rate'].values for df in new_dataframes])

# 不再对 income_grid 进行转置
# income_grid = income_grid.T  # 注释掉这行

# 生成 epoch 和 episode 的网格，用于绘制 3D 表面图
epoch_grid, episode_grid = np.meshgrid(epochs, episodes)

# 创建一个 3D 表面图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制 3D 表面
ax.plot_surface(epoch_grid, episode_grid, income_grid, cmap='viridis')

# 设置坐标轴标签和标题
ax.set_xlabel('Epoch')
ax.set_ylabel('Episode')
ax.set_zlabel('Cooperation Rate')
ax.set_title('Cooperation Rate = 0.50')

# 调整视角，绕 z 轴旋转
ax.view_init(elev=30, azim=-135)  # 这里的 azim 参数控制绕 z 轴的旋转角度

plt.show()

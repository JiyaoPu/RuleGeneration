
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D

def read_data(folder_name):
    # 文件夹路径，添加'data/'前缀
    folder_path = os.path.join('../data', folder_name)
    # 初始化列表来存储每个episode的数据
    data_list = []
    episode_numbers = []

    # 获取文件夹中所有的CSV文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # 提取文件名中的数字，确保排序正确
    csv_files_with_numbers = []
    for f in csv_files:
        try:
            # 提取文件名中的数字部分，例如 '5.csv' -> 5
            episode_num = int(os.path.splitext(f)[0])
            csv_files_with_numbers.append((episode_num, f))
        except ValueError:
            print(f"文件名 {f} 不是有效的数字，跳过该文件。")
            continue

    # 按照episode编号排序
    csv_files_with_numbers.sort(key=lambda x: x[0])

    # 遍历排序后的文件列表
    for episode_num, file_name in csv_files_with_numbers:
        file_path = os.path.join(folder_path, file_name)
        # 读取CSV文件
        df = pd.read_csv(file_path)
        # 检查是否有需要的列
        if 'Gini Coefficient' in df.columns:
            # 将'Gini Coefficient'列转换为numpy数组
            cooperation_rates = df['Gini Coefficient'].values
            data_list.append(cooperation_rates)
            episode_numbers.append(episode_num)
        else:
            print(f"文件 {file_path} 中缺少 'Gini Coefficient' 列，跳过该文件。")
            continue

    # 将data_list转换为二维numpy数组，行数为episode数量，列数为epoch数量
    data_array = np.array(data_list)
    episode_numbers = np.array(episode_numbers)
    return data_array, episode_numbers

# 定义文件夹名称列表
folder_names = ['0.98', '0.02', '0.50']
# 创建字典来存储每个文件夹的数据
data_dict = {}
episode_numbers_dict = {}

for folder in folder_names:
    folder_path = os.path.join('../data', folder)
    print(f"正在读取文件夹 {folder_path} 的数据...")
    data_array, episode_numbers = read_data(folder)
    data_dict[folder] = data_array
    episode_numbers_dict[folder] = episode_numbers
    print(f"文件夹 {folder_path} 的数据读取完成。")

# 定义颜色、透明度和标签
colors = ['Reds', 'Greens', 'Blues']
alphas = [0.7, 0.7, 0.7]
labels = ['Gini Coefficient = 0.98', 'Gini Coefficient = 0.02', 'Gini Coefficient = 0.50']

# 创建3D图形对象
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 遍历每个文件夹的数据，绘制平滑的曲面
for idx, folder in enumerate(folder_names):
    Z = data_dict[folder]  # 形状为(episode数量, epoch数量)
    episodes = episode_numbers_dict[folder]
    # 检查Z是否为空
    if Z.size == 0:
        print(f"文件夹 data/{folder} 的数据为空，跳过该曲面。")
        continue
    # 创建对应的X和Y
    X, Y = np.meshgrid(np.arange(Z.shape[1]), episodes)

    # # 旋转数据：绕 z 轴逆时针旋转 90 度
    # X_rot = -X
    # Y_rot = -Y
    #
    # X = X_rot
    # Y = Y_rot



    # 对 Z 数据进行高斯滤波
    if folder == '0.50':
        # 对 0.50 文件夹的数据进行两次高斯滤波
        Z_smooth = gaussian_filter(Z, sigma=1)
        Z_smooth = gaussian_filter(Z_smooth, sigma=1)
        Z_smooth = gaussian_filter(Z_smooth, sigma=1)
    else:
        # 对其他文件夹的数据进行一次高斯滤波
        Z_smooth = gaussian_filter(Z, sigma=1)

    # 检查Z的形状是否与X和Y匹配
    if Z_smooth.shape != X.shape:
        print(f"文件夹 data/{folder} 的数据形状与X和Y不匹配，跳过该曲面。")
        continue

    # 绘制平滑后的曲面
    surf = ax.plot_surface(X, Y, Z_smooth, cmap=colors[idx], alpha=alphas[idx])

# 设置坐标轴标签
ax.set_xlabel('Strategy Evolution', fontsize=12)
ax.set_ylabel('Rule Evolution', fontsize=12)
ax.set_zlabel('Envitonment Evolution', fontsize=12)

# 添加图例
from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], color='red', lw=4, label=labels[0]),
                   Line2D([0], [0], color='green', lw=4, label=labels[1]),
                   Line2D([0], [0], color='blue', lw=4, label=labels[2])]
ax.legend(handles=legend_elements, loc='upper left')

# 调整视角
ax.view_init(elev=30, azim=-135)

# 设置标题
ax.set_title('TRD System Visualization', fontsize=16)

# 显示图形
plt.tight_layout()
plt.show()

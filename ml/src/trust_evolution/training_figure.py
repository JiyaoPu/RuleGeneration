# # Import necessary libraries
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Load the local CSV files
# new_file_paths = [
#     "data/0.50/0.csv",
#     "data/0.50/5.csv",
#     "data/0.50/10.csv",
#     "data/0.50/20.csv",
#     "data/0.50/50.csv",
#     "data/0.50/99.csv"
# ]
#
# # Read the CSV files into new dataframes
# new_dataframes = [pd.read_csv(file_path) for file_path in new_file_paths]
#
# # Labels for the new plot
# labels = ['Epoch 0', 'Epoch 5', 'Epoch 10', 'Epoch 20', 'Epoch 50', 'Epoch 99']
#
# # Plot the data using the same approach as before with updated labels and x-axis
# plt.figure(figsize=(10, 6))
#
# for i, df in enumerate(new_dataframes):
#     plt.plot(df['Epoch'], df['Cooperation Rate'], label=labels[i])
#
# plt.xlabel('Episode')
# plt.ylabel('Cooperation Rate')
# plt.title('Cooperation Rate = 0.50')
# plt.legend()
# plt.grid(True)
#
# # Show the plot
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# 加载本地 CSV 文件
new_file_paths = [
    "data/0.50/0.csv",
    "data/0.50/5.csv",
    "data/0.50/10.csv",
    "data/0.50/20.csv",
    "data/0.50/50.csv",
    "data/0.50/99.csv"
]

# 读取 CSV 文件到新的数据框
new_dataframes = [pd.read_csv(file_path) for file_path in new_file_paths]

# 为新图设置标签
labels = ['Episode 0', 'Episode 5', 'Episode 10', 'Episode 20', 'Episode 50', 'Episode 99']

# 绘制数据，使用更新的标签和 x 轴
plt.figure(figsize=(10, 6))

for i, df in enumerate(new_dataframes):
    plt.plot(df['Epoch'], df['Cooperation Rate'], label=labels[i])

plt.xlabel('Epoch')
plt.ylabel('Cooperation Rate')
plt.title('Cooperation Rate = 0.50')

# 将图例放在图形外部右侧
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.grid(True)

# 调整图形边距，避免图例被裁剪
plt.tight_layout(rect=[0, 0, 0.85, 1])

# 显示图形
plt.show()

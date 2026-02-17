import matplotlib.pyplot as plt

# 手动绘制行为树（Copycat为例）
fig, ax = plt.subplots()

# 绘制根节点和子节点
ax.plot([0.5], [1.0], 'bo')  # 根节点
ax.text(0.5, 1.05, "Opponent's Last Action?", ha='center')

ax.plot([0.2, 0.8], [0.5, 0.5], 'bo')  # 子节点
ax.text(0.2, 0.55, "Cooperate if Cooperated", ha='center')
ax.text(0.8, 0.55, "Cheat if Cheated", ha='center')

# 绘制连接线
ax.annotate('', xy=(0.2, 0.5), xytext=(0.5, 1.0),
            arrowprops=dict(facecolor='black', arrowstyle='->'))
ax.annotate('', xy=(0.8, 0.5), xytext=(0.5, 1.0),
            arrowprops=dict(facecolor='black', arrowstyle='->'))

ax.axis('off')
plt.title("Copycat Behavior Tree")
plt.show()

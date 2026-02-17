import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
from agent import Agent

class QLearningAgent(Agent):
    def __init__(self, id=0, actions=['cheat', 'cooperation'], shared_q_table=None, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningAgent, self).__init__(id)
        self.actions = actions  # ['cheat', 'cooperation']
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        # 使用共享的 Q 表，如果没有提供则创建一个新的
        if shared_q_table is not None:
            self.q_table = shared_q_table
        else:
            self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.stereotype = 'Q'  # 标识为 Q-learning 代理
        self.tradeRecords = {}  # 初始化交易记录

    def set_q_table_by_id(self, id):
        filepath = Path('agents/' + str(id) + '.csv')
        self.q_table = pd.read_csv(filepath, index_col=0)

    def get_q_table(self):
        return self.q_table

    def set_greedy(self, e_greedy):
        self.epsilon = e_greedy

    def choose_action(self, observation):
        """
        选择动作，基于 epsilon-greedy 策略。

        参数:
            observation (str): 包含交易记录的字符串，表示状态。

        返回:
            action (str): 选择的动作 ('cheat' 或 'cooperation')。
        """
        # 确保 observation 是字符串
        if not isinstance(observation, str):
            raise TypeError(f"Observation must be a string, got {type(observation)} instead.")

        self.check_state_exist(observation)
        # 动作选择
        if np.random.uniform() < self.epsilon:
            # 选择最优动作
            state_action = self.q_table.loc[observation, :]
            # 为了避免多个动作具有相同的 Q 值，随机选择一个
            max_q = state_action.max()
            actions_with_max_q = state_action[state_action == max_q].index.tolist()
            action = np.random.choice(actions_with_max_q)
        else:
            # 随机选择动作
            action = np.random.choice(self.actions)
        return action

    def learn(self, observation, action, reward, observation_, done):
        """
        更新 Q 表。

        参数:
            observation (str): 当前状态。
            action (str): 当前动作。
            reward (float): 奖励。
            observation_ (str): 下一个状态。
            done (bool): 是否终止。
        """
        # 确保 observation 和 observation_ 是字符串
        if not isinstance(observation, str) or not isinstance(observation_, str):
            raise TypeError("Observations must be strings.")

        if action is None:
            # 如果动作为 None，跳过更新
            return

        self.check_state_exist(observation_)
        q_predict = self.q_table.loc[observation, action]
        if not done:
            q_target = reward + self.gamma * self.q_table.loc[observation_, :].max()
        else:
            q_target = reward  # 终止状态
        # 更新 Q 值
        self.q_table.loc[observation, action] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        """
        检查状态是否存在于 Q 表中，不存在则添加。

        参数:
            state (str): 状态字符串。
        """
        if not isinstance(state, str):
            raise TypeError(f"State must be a string, got {type(state)} instead.")

        if state not in self.q_table.index:
            # 在 Q 表中添加新状态，初始化所有动作的 Q 值为 0
            new_row = pd.Series([0] * len(self.actions), index=self.actions, name=state)
            # 将 Series 转为 DataFrame，并保持索引不变
            self.q_table = pd.concat([self.q_table, new_row.to_frame().T], ignore_index=False)

    def visualize_q_table(self, epoch, save=False):
        """
        可视化 Q 表。

        参数:
            epoch (int): 当前的 Epoch 数。
            save (bool): 是否保存图片。
        """
        # 复制 Q 表，避免修改原始数据
        q_table = self.q_table.copy()

        # 用勾号和叉号替换动作名称
        q_table.rename(columns={'cooperation': '✓', 'cheat': '✗'}, inplace=True)

        # 替换状态名称中的动作名称
        def replace_actions(state):
            if isinstance(state, tuple):
                state_str = ' | '.join(state)
            else:
                state_str = state
            state_str = state_str.replace('cooperation', '✓')
            state_str = state_str.replace('cheat', '✗')
            state_str = state_str.replace('not yet', 'N/A')
            return state_str

        q_table.index = q_table.index.map(replace_actions)

        # 设置索引和列的名称
        q_table.index.name = 'States'
        q_table.columns.name = 'Actions'

        # 绘制热力图
        plt.figure(figsize=(12, 8))
        sns.heatmap(q_table, annot=False, cmap='viridis', fmt=".2f")
        plt.title(f'Q-table at Epoch {epoch}')
        plt.ylabel('States')
        plt.xlabel('Actions')

        if save:
            # 创建目录（如果不存在）
            os.makedirs('visual', exist_ok=True)
            # 保存图片到文件
            plt.savefig(f'visual/q_table_epoch_{epoch}.png')
            plt.close()
        else:
            # 显示图片
            plt.show()

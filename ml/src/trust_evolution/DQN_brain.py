

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
from agent import Agent
import random
from collections import deque
import copy
from collections import namedtuple, deque


Experience = namedtuple('Experience',
                        ('state', 'action', 'reward', 'next_state', 'done'))




class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save an experience"""
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    """
    深度 Q 网络（Deep Q-Network）。
    """
    def __init__(self, input_dim, output_dim):
        """
        初始化 DQN 网络结构。

        参数:
            input_dim (int): 输入层维度。
            output_dim (int): 输出层维度，对应动作的数量。
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        """
        定义前向传播。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 输出 Q 值。
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent(Agent):
    """
    基于深度 Q 网络的代理。
    """
    def __init__(
        self,
        id=0,
        actions=['cheat', 'cooperation'],
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        memory_size=10000,
        target_update=5000,
        state_size=20  # 根据实际状态表示调整


    ):
        """
        初始化 DQN 代理。

        参数:
            id (int): 代理的唯一标识符。
            actions (list): 可选的动作列表。
            learning_rate (float): 学习率。
            gamma (float): 折扣因子 γ。
            epsilon (float): 初始 ε 值，用于 ε-贪心策略。
            epsilon_min (float): 最小 ε 值。
            epsilon_decay (float): ε 的衰减率。
            batch_size (int): 经验回放的批量大小。
            memory_size (int): 经验回放的记忆库大小。
            target_update (int): 目标网络更新的频率（以步为单位）。
            state_size (int): 状态向量的维度。
        """
        super(DQNAgent, self).__init__(id)

        self.stereotype = 'DQN'


        self.actions = actions  # ['cheat', 'cooperation']
        self.ACTION_TO_IDX = {action: idx for idx, action in enumerate(self.actions)}
        self.IDX_TO_ACTION = {idx: action for idx, action in enumerate(self.actions)}

        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = ReplayMemory(memory_size)
        self.target_update = target_update
        self.learn_step_counter = 0

        self.steps_done = 0

        self.char_to_idx = {'o': 0, 'y': 1, 'n': 2}  # 根据实际字符扩展
        self.num_chars = len(self.char_to_idx)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 状态表示的维度
        self.state_size = 60
        self.action_size = 2

        # 初始化策略网络和目标网络
        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器和损失函数
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.SmoothL1Loss()

        # 初始化 Q 表为空
        self.q_table = {}  # {state: {action: Q_value}}

    def one_hot_encode(self, observation):
        max_length = 20
        one_hot = np.zeros((max_length, 3), dtype=np.float32)
        for i, char in enumerate(observation[:max_length]):  # 截断超长的 observation
            if char in self.char_to_idx:
                one_hot[i, self.char_to_idx[char]] = 1.0
        return one_hot.flatten()

    def preprocess_state(self, state):
        """
        将状态字符串转换为数值向量。

        例如，将 'y', 'n', 'o' 转换为 1.0, 0.0, -1.0。

        参数:
            state (str): 状态字符串。

        返回:
            np.ndarray: 数值向量。
        """
        mapping = {'y': 1.0, 'n': 0.0, 'o': -1.0}
        state_vector = [mapping.get(char, -1.0) for char in state]
        return np.array(state_vector, dtype=np.float32)

    def set_q_table_by_id(self, id):
        """
        从指定文件加载 Q 表。

        参数:
            id (int): 代理的唯一标识符。
        """
        filepath = Path(f'agents/{id}.csv')
        if filepath.exists():
            try:
                df = pd.read_csv(filepath, index_col=0)
                self.q_table = df.to_dict(orient='index')
            except Exception as e:
                raise ValueError(f"加载 Q 表失败: {e}")
        else:
            # 如果文件不存在，则初始化 Q 表为空
            self.q_table = {}

    def get_q_table(self):
        """
        获取当前 Q 表的 DataFrame 形式。

        返回:
            pd.DataFrame: 当前 Q 表。
        """
        return pd.DataFrame.from_dict(self.q_table, orient='index')

    def save_q_table(self, id):
        """
        将当前 Q 表保存到指定文件。

        参数:
            id (int): 代理的唯一标识符。
        """
        df = self.get_q_table()
        df.to_csv(f'agents/{id}.csv')



    def choose_action(self, state):
        one_hot_state = self.one_hot_encode(state)
        state_tensor = torch.FloatTensor(one_hot_state).unsqueeze(0)  # 增加batch维度
        if np.random.rand() < self.epsilon:
            action_str = random.choice(self.actions)
            action = self.ACTION_TO_IDX[action_str]
            return action_str
        with torch.no_grad():
            q_values = self.policy_net(state_tensor.to(self.device))
        action = q_values.argmax().item()
        action_str = self.IDX_TO_ACTION[action]
        return action_str



    def remember(self, state, action, reward, next_state, done):
        """
        存储经验到记忆库。

        参数:
            state (str): 当前状态。
            action (str): 当前动作。
            reward (float): 奖励。
            next_state (str): 下一个状态。
            done (bool): 是否终止。
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """
        从记忆库中抽取样本进行训练。
        """
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 预处理状态
        states = torch.FloatTensor([self.one_hot_encode(s) for s in states]).to(self.device)
        next_states = torch.FloatTensor([self.one_hot_encode(s_) for s_ in next_states]).to(self.device)

        # 动作索引
        actions = torch.LongTensor([self.actions.index(a) for a in actions]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 当前 Q 值
        q_values = self.policy_net(states).gather(1, actions)

        # 目标 Q 值
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # 计算损失
        loss = self.loss_fn(q_values, target_q_values)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 衰减 ε
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def set_greedy(self, e_greedy):
        self.epsilon = e_greedy

    def update_target_network(self):
        """
        更新目标网络的参数。
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def learn(self, state, action_str, reward, next_state, done):
        # 编码当前状态和下一个状态
        action = self.ACTION_TO_IDX[action_str]
        encoded_state = self.one_hot_encode(state)
        encoded_next_state = self.one_hot_encode(next_state) if next_state is not None else np.zeros(self.state_size * self.num_chars, dtype=np.float32)

        # 存储经验
        self.memory.push(encoded_state, action, reward, encoded_next_state, done)
        # self.remember(encoded_state, action, reward, encoded_next_state, done)

        # 如果回放缓冲区中经验不足以组成一个批次，则不进行学习
        if len(self.memory) < self.batch_size:
            return

        # 从回放缓冲区中采样一个批次的经验
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        # 转换为张量
        states = torch.FloatTensor(batch.state).to(self.device)
        next_states = torch.FloatTensor(batch.next_state).to(self.device)


        actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)

        dones = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)

        # 计算当前 Q 值
        current_q_values = self.policy_net(states)
        current_q_values = current_q_values.gather(1, actions)
        # 计算目标 Q 值
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # 计算损失
        loss = self.loss_fn(current_q_values, target_q_values)

        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新 epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 更新目标网络
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        """
        保存模型参数。

        参数:
            path (str): 模型保存路径。
        """
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        """
        加载模型参数。

        参数:
            path (str): 模型加载路径。
        """
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def visualize_q_table(self, epoch, save=False):
        """
        可视化 Q 表。

        参数:
            epoch (int): 当前的 Epoch 数。
            save (bool): 是否保存图片。
        """
        # 将 Q 表转换为 DataFrame
        q_table_df = self.get_q_table()

        # 用勾号和叉号替换动作名称
        q_table_df.rename(columns={'cooperation': '✓', 'cheat': '✗'}, inplace=True)

        # 替换状态名称中的动作名称
        def replace_actions(state):
            state_str = state.replace('cooperation', '✓').replace('cheat', '✗').replace('not yet', 'N/A')
            return state_str

        q_table_df.index = q_table_df.index.map(replace_actions)

        # 设置索引和列的名称
        q_table_df.index.name = 'States'
        q_table_df.columns.name = 'Actions'

        # 绘制热力图
        plt.figure(figsize=(12, 8))
        sns.heatmap(q_table_df, annot=False, cmap='viridis', fmt=".2f")
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


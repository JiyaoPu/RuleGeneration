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
    Deep Q-Network
    """
    def __init__(self, input_dim, output_dim):
        """
        Initialize the DQN network architecture.

        Parameters:
            input_dim (int): Dimension of the input layer.
            output_dim (int): Dimension of the output layer, corresponding to the number of actions.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        """
        Define the forward pass.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output Q-values.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent(Agent):
    """
    Agent based on a Deep Q-Network.
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
            Initialize the DQN agent.

            Parameters:
                id (int): Unique identifier of the agent.
                actions (list): List of available actions.
                learning_rate (float): Learning rate.
                gamma (float): Discount factor γ.
                epsilon (float): Initial ε value for the ε-greedy policy.
                epsilon_min (float): Minimum ε value.
                epsilon_decay (float): Decay rate of ε.
                batch_size (int): Batch size for experience replay.
                memory_size (int): Size of the replay memory.
                target_update (int): Frequency of target network updates (in steps).
                state_size (int): Dimension of the state vector.
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

        self.state_size = 60
        self.action_size = 2

        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.SmoothL1Loss()

        self.q_table = {}  # {state: {action: Q_value}}

    def one_hot_encode(self, observation):
        max_length = 20
        one_hot = np.zeros((max_length, 3), dtype=np.float32)
        for i, char in enumerate(observation[:max_length]):  # 截断超长的 observation
            if char in self.char_to_idx:
                one_hot[i, self.char_to_idx[char]] = 1.0
        return one_hot.flatten()

    def preprocess_state(self, state):
        mapping = {'y': 1.0, 'n': 0.0, 'o': -1.0}
        state_vector = [mapping.get(char, -1.0) for char in state]
        return np.array(state_vector, dtype=np.float32)

    def set_q_table_by_id(self, id):
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
        return pd.DataFrame.from_dict(self.q_table, orient='index')

    def save_q_table(self, id):
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
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor([self.one_hot_encode(s) for s in states]).to(self.device)
        next_states = torch.FloatTensor([self.one_hot_encode(s_) for s_ in next_states]).to(self.device)

        actions = torch.LongTensor([self.actions.index(a) for a in actions]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def set_greedy(self, e_greedy):
        self.epsilon = e_greedy

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def learn(self, state, action_str, reward, next_state, done):
        action = self.ACTION_TO_IDX[action_str]
        encoded_state = self.one_hot_encode(state)
        encoded_next_state = self.one_hot_encode(next_state) if next_state is not None else np.zeros(self.state_size * self.num_chars, dtype=np.float32)

        self.memory.push(encoded_state, action, reward, encoded_next_state, done)
        # self.remember(encoded_state, action, reward, encoded_next_state, done)

        if len(self.memory) < self.batch_size:
            return

        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        states = torch.FloatTensor(batch.state).to(self.device)
        next_states = torch.FloatTensor(batch.next_state).to(self.device)


        actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)

        dones = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)

        current_q_values = self.policy_net(states)
        current_q_values = current_q_values.gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def visualize_q_table(self, epoch, save=False):
        q_table_df = self.get_q_table()

        q_table_df.rename(columns={'cooperation': '✓', 'cheat': '✗'}, inplace=True)

        def replace_actions(state):
            state_str = state.replace('cooperation', '✓').replace('cheat', '✗').replace('not yet', 'N/A')
            return state_str

        q_table_df.index = q_table_df.index.map(replace_actions)

        q_table_df.index.name = 'States'
        q_table_df.columns.name = 'Actions'

        plt.figure(figsize=(12, 8))
        sns.heatmap(q_table_df, annot=False, cmap='viridis', fmt=".2f")
        plt.title(f'Q-table at Epoch {epoch}')
        plt.ylabel('States')
        plt.xlabel('Actions')

        if save:
            os.makedirs('visual', exist_ok=True)
            plt.savefig(f'visual/q_table_epoch_{epoch}.png')
            plt.close()
        else:
            plt.show()


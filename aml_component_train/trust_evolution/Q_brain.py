from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from agent import Agent


# =========================================================
# Unified output path helpers
# Default artifact layout:
#   <AZUREML_OUTPUT_DIR>/
#     ├─ agents/
#     └─ data/
#         └─ visual/
# =========================================================

OUTPUT_DIR = Path(os.getenv("AZUREML_OUTPUT_DIR", "outputs")).resolve()
DATA_DIR = OUTPUT_DIR / "data"
AGENTS_DIR = OUTPUT_DIR / "agents"

for d in [OUTPUT_DIR, DATA_DIR, AGENTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def artifact_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (OUTPUT_DIR / p)


def data_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (DATA_DIR / p)


def agent_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (AGENTS_DIR / p)


class QLearningAgent(Agent):
    def __init__(
        self,
        id=0,
        actions=["cheat", "cooperation"],
        shared_q_table=None,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.9,
    ):
        super(QLearningAgent, self).__init__(id)
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        # Use a shared Q-table if provided; otherwise create a new one
        if shared_q_table is not None:
            self.q_table = shared_q_table
        else:
            self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

        self.stereotype = "Q"
        self.tradeRecords = {}

    def set_q_table_by_id(self, id):
        """
        Load a Q-table from the unified agents directory.

        Default path:
            <AZUREML_OUTPUT_DIR>/agents/{id}.csv
        """
        filepath = agent_path(f"{id}.csv")
        self.q_table = pd.read_csv(filepath, index_col=0)

    def get_q_table(self):
        return self.q_table

    def set_greedy(self, e_greedy):
        self.epsilon = e_greedy

    def choose_action(self, observation):
        """
        Choose an action using the epsilon-greedy policy.

        Args:
            observation (str): The current state represented as a string.

        Returns:
            action (str): The chosen action, either 'cheat' or 'cooperation'.
        """
        if not isinstance(observation, str):
            raise TypeError(f"Observation must be a string, got {type(observation)} instead.")

        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            max_q = state_action.max()
            actions_with_max_q = state_action[state_action == max_q].index.tolist()
            action = np.random.choice(actions_with_max_q)
        else:
            action = np.random.choice(self.actions)

        return action

    def learn(self, observation, action, reward, observation_, done):
        """
        Update the Q-table.

        Args:
            observation (str): Current state.
            action (str): Current action.
            reward (float): Received reward.
            observation_ (str): Next state.
            done (bool): Whether the episode has terminated.
        """
        if not isinstance(observation, str) or not isinstance(observation_, str):
            raise TypeError("Observations must be strings.")

        if action is None:
            return

        self.check_state_exist(observation_)
        q_predict = self.q_table.loc[observation, action]

        if not done:
            q_target = reward + self.gamma * self.q_table.loc[observation_, :].max()
        else:
            q_target = reward

        self.q_table.loc[observation, action] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        """
        Ensure the state exists in the Q-table; add it if missing.

        Args:
            state (str): State string.
        """
        if not isinstance(state, str):
            raise TypeError(f"State must be a string, got {type(state)} instead.")

        if state not in self.q_table.index:
            new_row = pd.Series([0] * len(self.actions), index=self.actions, name=state)
            self.q_table = pd.concat([self.q_table, new_row.to_frame().T], ignore_index=False)

    def visualize_q_table(self, epoch, save=False):
        """
        Visualize the Q-table as a heatmap.

        Args:
            epoch (int): Current epoch number.
            save (bool): Whether to save the figure.
        """
        q_table = self.q_table.copy()

        # Replace action labels with symbols
        q_table.rename(columns={"cooperation": "✓", "cheat": "✗"}, inplace=True)

        def replace_actions(state):
            if isinstance(state, tuple):
                state_str = " | ".join(state)
            else:
                state_str = state
            state_str = state_str.replace("cooperation", "✓")
            state_str = state_str.replace("cheat", "✗")
            state_str = state_str.replace("not yet", "N/A")
            return state_str

        q_table.index = q_table.index.map(replace_actions)
        q_table.index.name = "States"
        q_table.columns.name = "Actions"

        plt.figure(figsize=(12, 8))
        sns.heatmap(q_table, annot=False, cmap="viridis", fmt=".2f")
        plt.title(f"Q-table at Epoch {epoch}")
        plt.ylabel("States")
        plt.xlabel("Actions")

        if save:
            out_dir = data_path("visual")
            out_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_dir / f"q_table_epoch_{epoch}.png")
            plt.close()
        else:
            if os.getenv("AZUREML_RUN_ID") is None:
                plt.show()
            else:
                plt.close()
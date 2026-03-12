from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from Q_brain import QLearningAgent
from DQN_brain import DQNAgent


# =========================================================
# Unified output path helpers
# All generated data defaults to:
#   <AZUREML_OUTPUT_DIR>/data/...
# =========================================================

OUTPUT_DIR = Path(os.getenv("AZUREML_OUTPUT_DIR", "outputs")).resolve()
DATA_DIR = OUTPUT_DIR / "data"

for d in [OUTPUT_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def artifact_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (OUTPUT_DIR / p)


def data_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (DATA_DIR / p)


class MiniEnv:
    def __init__(self, trade_rules, round_number, reproduction_number, mistake_possibility, extrinsic_reward):
        # Rules
        self.agents = []
        self.agents_dict = {}
        self.trade_rules = trade_rules
        self.round_number = round_number
        self.reproduction_number = reproduction_number
        self.mistake_possibility = mistake_possibility

        # Fixed settings
        self.trade_number = 5

        # Game records
        self.epoch_list = []
        self.IndividualIncome = []
        self.CooperationRate = []
        self.GiniCoefficient = []

        self.extrinsic_reward = extrinsic_reward
        self.aiType = "Q"

    def plot_results(self):
        """
        Plot Gini coefficient over epochs and save the figure.
        """
        output_path = data_path("Q_money_over_epochs.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(12, 8))
        plt.plot(
            self.epoch_list,
            self.GiniCoefficient,
            marker="o",
            linestyle="-",
            color="b",
            label=self.aiType,
        )

        plt.title(self.aiType, fontsize=16)
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Gini Coefficient", fontsize=14)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(output_path)
        print(f"Figure saved to {output_path}")

        if os.getenv("AZUREML_RUN_ID") is None:
            plt.show()
        plt.close()

    def plot_trust_matrix(self, trust_matrix, labels, filename="trust_matrix_heatmap.png"):
        """
        Plot an 8x8 trust matrix heatmap with values in [0, 1] and save it.

        Args:
            trust_matrix: A nested dictionary such as:
                {
                  'Random': {'Random': 0.49, 'Cheater': 0.001, ...},
                  'Cheater': {'Random': 0.513, 'Cheater': 0.0011, ...},
                  ...
                }
            labels: A list of 8 agent-type labels used for rows and columns.
            filename: Output filename. If relative, it will be saved under DATA_DIR.

        Returns:
            fig: The matplotlib figure object.
        """
        z = []
        for row_label in labels:
            row = []
            for col_label in labels:
                value = trust_matrix.get(row_label, {}).get(col_label, 0.5)
                row.append(value)
            z.append(row)

        z = np.array(z)

        fig, ax = plt.subplots(figsize=(8, 8))
        cax = ax.imshow(z, cmap="viridis", vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_title("8x8 Trust Matrix Heatmap")

        fig.colorbar(cax, ax=ax, label="Trust")

        out_path = data_path(filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")

        if os.getenv("AZUREML_RUN_ID") is None:
            plt.show()
        plt.close()

        return fig

    def save_results_to_csv(self, filename="plotData.csv"):
        """
        Save epoch_list and GiniCoefficient to a CSV file.

        Args:
            filename: Output filename. If relative, it will be saved under DATA_DIR.
        """
        data = {
            "Epoch": self.epoch_list,
            "Gini Coefficient": self.GiniCoefficient,
        }
        df = pd.DataFrame(data)

        out_path = data_path(filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Results saved to {out_path}")

    def skill_translation(self, individual_income):
        """
        Convert the evaluation result into a skill value.

        The AI agent's average income is used as the skill value.

        Args:
            individual_income: A list where index 6 corresponds to the AI agent type.

        Returns:
            skill_value: The AI agent's average income.
        """
        skill_value = individual_income[6]
        return skill_value

    def setup(self, agents, trade_rules, round_number, reproduction_number, mistake_possibility, extrinsic_reward):
        self.agents = agents
        self.agents_dict = {agent.id: agent for agent in self.agents}

        self.trade_rules = trade_rules
        self.round_number = round_number
        self.reproduction_number = reproduction_number
        self.mistake_possibility = mistake_possibility
        self.extrinsic_reward = extrinsic_reward

        # Clear previous state
        self.epoch_list.clear()
        self.IndividualIncome.clear()
        self.CooperationRate.clear()
        self.GiniCoefficient.clear()

    def reset(self, agentList=None, tradeRules=None, extrinsic_reward=None):
        if agentList is None:
            agentList = []
        if tradeRules is None:
            tradeRules = [-3, -3, 0, 2, 5, 5]
        if extrinsic_reward is None:
            extrinsic_reward = [0, 0]

        self.agents.clear()
        self.agents_dict.clear()
        self.setup(
            agentList,
            tradeRules,
            self.round_number,
            self.reproduction_number,
            self.mistake_possibility,
            extrinsic_reward,
        )

    def calculate_rewards(self, action_A, action_B):
        """
        Calculate rewards based on the trade rule matrix.
        """
        if action_A == "cheat" and action_B == "cheat":
            reward_A = self.trade_rules[0]
            reward_B = self.trade_rules[1]
        elif action_A == "cheat" and action_B == "cooperation":
            reward_A = self.trade_rules[2]
            reward_B = self.trade_rules[3]
        elif action_A == "cooperation" and action_B == "cheat":
            reward_A = self.trade_rules[3]
            reward_B = self.trade_rules[2]
        elif action_A == "cooperation" and action_B == "cooperation":
            reward_A = self.trade_rules[4]
            reward_B = self.trade_rules[5]
        else:
            reward_A = 0
            reward_B = 0
        return reward_A, reward_B

    def get_opponent_stereotype(self, opponent_id):
        """
        Get the stereotype of an opponent agent by ID.
        """
        opponent_agent = self.agents_dict.get(opponent_id)
        if opponent_agent is None:
            return None
        return opponent_agent.stereotype

    def imitation_learning(self, ai_agent, top_agent):
        """
        Let the AI agent learn from the best-performing agent's interaction history.

        Args:
            ai_agent: The AI agent instance.
            top_agent: The top-performing agent instance.
        """
        for opponent_id, interactions in top_agent.tradeRecords.items():
            opponent_stereotype = self.get_opponent_stereotype(opponent_id)

            for interaction in interactions:
                top_agent_action, opponent_action = interaction

                state = ai_agent.get_state(opponent_id, opponent_stereotype)
                action = top_agent_action

                reward_top_agent, _ = self.calculate_rewards(top_agent_action, opponent_action)
                reward = reward_top_agent

                next_state = state
                done = False

                ai_agent.learn(state, action, reward, next_state, done)

    def knowledgeTransform(self, RLID, ID_other):
        """
        Perform knowledge transfer through repeated scripted interactions.
        """
        for trainingTimes in range(2000):
            opponent_stereotype = self.get_opponent_stereotype(ID_other)
            RL_stereotype = self.get_opponent_stereotype(RLID)

            for time in range(5):
                agent_A = self.agents_dict[RLID]
                agent_B = self.agents_dict[ID_other]

                observation_A = agent_A.get_state(ID_other, opponent_stereotype)
                observation_B = agent_B.get_state(RLID, RL_stereotype)

                if time == 4:
                    action_A = "cheat"
                else:
                    action_A = "cooperation"

                action_B = "cooperation"

                agent_A.record_action(action_A)
                agent_B.record_action(action_B)

                reward_A, reward_B = self.calculate_rewards(action_A, action_B)

                agent_A.addReward(reward_A)
                agent_B.addReward(reward_B)

                agent_A.Remember(ID_other, action_A, action_B)
                agent_B.Remember(RLID, action_B, action_A)

                if agent_A.stereotype == "Q":
                    next_observation_A = agent_A.get_state(ID_other, opponent_stereotype)
                    done_A = False
                    agent_A.learn(observation_A, action_A, reward_A, next_observation_A, done_A)
                elif agent_A.stereotype == "DQN":
                    next_observation_A = agent_A.get_state(ID_other, opponent_stereotype)
                    done_A = False
                    agent_A.learn(observation_A, action_A, reward_A, next_observation_A, done_A)

                if agent_B.stereotype == "Q":
                    next_observation_B = agent_B.get_state(RLID, RL_stereotype)
                    done_B = False
                    agent_B.learn(observation_B, action_B, reward_B, next_observation_B, done_B)
                elif agent_B.stereotype == "DQN":
                    next_observation_B = agent_B.get_state(RLID, RL_stereotype)
                    done_B = False
                    agent_B.learn(observation_B, action_B, reward_B, next_observation_B, done_B)

    def oneTrade(self, ID_A, ID_B):
        """
        Execute one trade between two agents.
        """
        agent_A = self.agents_dict[ID_A]
        agent_B = self.agents_dict[ID_B]

        A_stereotype = self.get_opponent_stereotype(ID_A)
        B_stereotype = self.get_opponent_stereotype(ID_B)

        observation_A = agent_A.get_state(ID_B, B_stereotype)
        observation_B = agent_B.get_state(ID_A, A_stereotype)

        if agent_A.stereotype == "Q":
            action_A = agent_A.choose_action(observation_A)
        elif agent_A.stereotype == "DQN":
            action_A = agent_A.choose_action(observation_A)
        else:
            action_A = agent_A.StereotypeAction(ID_B)

        if agent_B.stereotype == "Q":
            action_B = agent_B.choose_action(observation_B)
        elif agent_B.stereotype == "DQN":
            action_B = agent_B.choose_action(observation_B)
        else:
            action_B = agent_B.StereotypeAction(ID_A)

        if random.random() < self.mistake_possibility:
            action_A = "cooperation" if action_A == "cheat" else "cheat"
        if random.random() < self.mistake_possibility:
            action_B = "cooperation" if action_B == "cheat" else "cheat"

        agent_A.record_action(action_A)
        agent_B.record_action(action_B)

        reward_A, reward_B = self.calculate_rewards(action_A, action_B)

        # Add extrinsic rewards
        if action_A == "cooperation":
            reward_A += self.extrinsic_reward[0]
        else:
            reward_A += self.extrinsic_reward[1]

        if action_B == "cooperation":
            reward_B += self.extrinsic_reward[0]
        else:
            reward_B += self.extrinsic_reward[1]

        agent_A.addReward(reward_A)
        agent_B.addReward(reward_B)

        agent_A.Remember(ID_B, action_A, action_B)
        agent_B.Remember(ID_A, action_B, action_A)

        if agent_A.stereotype == "Q":
            next_observation_A = agent_A.get_state(ID_B, B_stereotype)
            done_A = False
            agent_A.learn(observation_A, action_A, reward_A, next_observation_A, done_A)
        elif agent_A.stereotype == "DQN":
            next_observation_A = agent_A.get_state(ID_B, B_stereotype)
            done_A = False
            agent_A.learn(observation_A, action_A, reward_A, next_observation_A, done_A)

        if agent_B.stereotype == "Q":
            next_observation_B = agent_B.get_state(ID_A, A_stereotype)
            done_B = False
            agent_B.learn(observation_B, action_B, reward_B, next_observation_B, done_B)
        elif agent_B.stereotype == "DQN":
            next_observation_B = agent_B.get_state(ID_A, A_stereotype)
            done_B = False
            agent_B.learn(observation_B, action_B, reward_B, next_observation_B, done_B)

        return action_A, action_B

    def oneCompetition(self, episode_id, agent_type_names, agent_types_order, record_data, save_model, printGame):
        """
        Run one competition consisting of multiple rounds, including trading,
        elimination, and reproduction.
        """
        self.aiType = agent_type_names["AI"]
        total_agents = len(self.agents)

        for round_num_id in range(self.round_number):
            if printGame:
                print(f"--- Episode {episode_id}, Round {round_num_id + 1} begin ---")

            # Reset all agent states before each round
            for agent in self.agents:
                agent.money = 0
                agent.tradeRecords = {}
                agent.tradeNum = 0
                agent.cooperation_count = 0
                agent.cheat_count = 0

            # Each agent trades with every other agent
            for agent_A in self.agents:
                for agent_B in self.agents:
                    if agent_A.id != agent_B.id:
                        for _ in range(self.trade_number):
                            self.oneTrade(agent_A.id, agent_B.id)
                        agent_A.tradeNum += self.trade_number
                        agent_B.tradeNum += self.trade_number

            sorted_agents_desc = sorted(self.agents, key=lambda x: x.money, reverse=True)
            for rank, agent in enumerate(sorted_agents_desc, start=1):
                if agent.stereotype == "Q" and printGame:
                    print(f"Agent ID {agent.id} ({agent.stereotype}) rank: {rank} / {len(self.agents)}")
            if printGame:
                print("============================\n")

            # Trust metric
            agent_types_order = [
                "Random",
                "Cheater",
                "Cooperator",
                "Copycat",
                "Grudger",
                "Detective",
                agent_type_names["AI"],
                "Human",
            ]

            group_stats = {s: {t: {"coop": 0, "total": 0} for t in agent_types_order} for s in agent_types_order}

            for agent in self.agents:
                s_type = agent.stereotype
                for opponent_id, interactions in agent.tradeRecords.items():
                    opponent = self.agents_dict.get(opponent_id)
                    if opponent is None:
                        continue
                    t_type = opponent.stereotype
                    for interaction in interactions:
                        if str(interaction[1]).lower() == "cooperation":
                            group_stats[s_type][t_type]["coop"] += 1
                        group_stats[s_type][t_type]["total"] += 1

            trust_matrix = {s: {t: 0 for t in agent_types_order} for s in agent_types_order}
            for s in agent_types_order:
                for t in agent_types_order:
                    coop = group_stats[s][t]["coop"]
                    total = group_stats[s][t]["total"]
                    trust_value = (coop + 1) / (total + 2) if total > 0 else 0.5
                    trust_matrix[s][t] = trust_value

            # self.plot_trust_matrix(trust_matrix, agent_types_order, "trust_matrix_heatmap.png")

            # Aggregate per-agent-type statistics
            agent_stats = {}
            total_agents_current = len(self.agents)
            for agent in self.agents:
                agent_type = agent.stereotype
                agent_name = agent_type_names.get(agent_type, f"Type {agent_type}")
                if agent_type not in agent_stats:
                    agent_stats[agent_type] = {
                        "name": agent_name,
                        "count": 0,
                        "total_money": 0,
                        "cooperation_count": 0,
                        "cheat_count": 0,
                    }
                agent_stats[agent_type]["count"] += 1
                agent_stats[agent_type]["total_money"] += agent.money
                agent_stats[agent_type]["cooperation_count"] += agent.cooperation_count
                agent_stats[agent_type]["cheat_count"] += agent.cheat_count

            for stats in agent_stats.values():
                stats["proportion"] = stats["count"] / total_agents_current * 100

            if printGame:
                print("----------------------")
            for agent_type in agent_types_order:
                stats = agent_stats.get(agent_type)
                if stats:
                    if printGame:
                        print(
                            f"{stats['name']}: count={stats['count']}, proportion={stats['proportion']:.2f}%, "
                            f"total_money={stats['total_money']}, cooperation_count={stats['cooperation_count']}, cheat_count={stats['cheat_count']}"
                        )
                else:
                    if printGame:
                        print(f"{agent_type_names.get(agent_type, f'Type {agent_type}')}: no data")
            if printGame:
                print(f"--- Epoch {episode_id + 1}, Round {round_num_id + 1} end ---")
                print("----------------------")

            # Record AI-related metrics for this round
            ai_agents = [agent for agent in self.agents if agent.stereotype == agent_type_names["AI"]]
            if ai_agents:
                total_ai_money = sum(agent.money for agent in ai_agents)
                average_individual_income_ai = total_ai_money / len(ai_agents)
                total_cooperation_count = sum(agent.cooperation_count for agent in ai_agents)
                total_trade_count = sum(agent.tradeNum for agent in ai_agents)
                cooperation_rate = total_cooperation_count / total_trade_count if total_trade_count > 0 else 0

                incomes = np.array([agent.money for agent in ai_agents])
                if len(incomes) > 1:
                    incomes_sorted = np.sort(incomes)
                    n = len(incomes_sorted)
                    cumulative_incomes = np.cumsum(incomes_sorted)
                    gini_numerator = np.sum((2 * np.arange(1, n + 1) - n - 1) * incomes_sorted)
                    gini_denominator = n * cumulative_incomes[-1]
                    gini_coefficient = gini_numerator / gini_denominator if gini_denominator != 0 else 0
                else:
                    gini_coefficient = 0

                if printGame:
                    print(f"Record Epoch {episode_id + 1}: AI average income = {average_individual_income_ai}")
                    print(f"Record Epoch {episode_id + 1}: cooperation rate = {cooperation_rate:.2f}")
                    print(f"Record Epoch {episode_id + 1}: Gini coefficient = {gini_coefficient:.2f}")
            else:
                if printGame:
                    print("No AI agent data recorded.\n")

            # Elimination and reproduction
            self.agents.sort(key=lambda x: x.money, reverse=True)
            top_agent = self.agents[0]

            ai_agents = [agent for agent in self.agents if agent.stereotype == agent_type_names["AI"]]
            for ai_agent in ai_agents:
                self.imitation_learning(ai_agent, top_agent)

            ai_agents = [agent for agent in self.agents if agent.stereotype == agent_type_names["AI"]]
            num_ai_agents = len(ai_agents)

            agents_to_eliminate = []
            ai_agents_eliminated = 0

            for agent in reversed(self.agents):
                if len(agents_to_eliminate) >= self.reproduction_number:
                    break
                if agent.stereotype == agent_type_names["AI"]:
                    if num_ai_agents - ai_agents_eliminated > 1:
                        agents_to_eliminate.append(agent)
                        ai_agents_eliminated += 1
                    else:
                        continue
                else:
                    agents_to_eliminate.append(agent)

            actual_reproduction_number = len(agents_to_eliminate)
            top_agents = [agent for agent in self.agents if agent not in agents_to_eliminate]
            copied_agents = top_agents[:actual_reproduction_number]

            id_counter = max(agent.id for agent in self.agents) + 1
            new_agents = []
            for agent in copied_agents:
                if hasattr(agent, "clone") and callable(getattr(agent, "clone")):
                    new_agent = agent.clone(new_id=id_counter)
                else:
                    new_agent = copy.deepcopy(agent)
                    new_agent.id = id_counter
                new_agents.append(new_agent)
                id_counter += 1

            self.agents = top_agents + new_agents
            self.agents_dict = {agent.id: agent for agent in self.agents}

            if len(self.agents) != total_agents:
                raise ValueError("The total number of agents has changed unexpectedly.")

        # Compute per-type metrics and overall metrics
        agent_types_order = ["Random", "Cheater", "Cooperator", "Copycat", "Grudger", "Detective", "AI", "Human"]
        gini_list = []
        coop_rate_list = []
        income_list = []

        for t in agent_types_order:
            type_agents = [agent for agent in self.agents if agent.stereotype == agent_type_names[t]]
            if not type_agents:
                gini_list.append(0)
                coop_rate_list.append(0)
                income_list.append(0)
            else:
                incomes = np.array([agent.money for agent in type_agents])
                avg_income = np.mean(incomes)
                income_list.append(avg_income)

                total_coop = sum(agent.cooperation_count for agent in type_agents)
                total_trades = sum(agent.tradeNum for agent in type_agents)
                coop_rate = total_coop / total_trades if total_trades > 0 else 0
                coop_rate_list.append(coop_rate)

                if len(incomes) > 1:
                    sorted_incomes = np.sort(incomes)
                    n = len(sorted_incomes)
                    cumulative = np.cumsum(sorted_incomes)
                    gini_numer = np.sum((2 * np.arange(1, n + 1) - n - 1) * sorted_incomes)
                    gini = gini_numer / (n * cumulative[-1]) if cumulative[-1] != 0 else 0
                else:
                    gini = 0
                gini_list.append(gini)

        if self.agents:
            all_incomes = np.array([agent.money for agent in self.agents])
            overall_income = np.mean(all_incomes)
            overall_total_coop = sum(agent.cooperation_count for agent in self.agents)
            overall_total_trades = sum(agent.tradeNum for agent in self.agents)
            overall_coop_rate = overall_total_coop / overall_total_trades if overall_total_trades > 0 else 0

            if len(all_incomes) > 1:
                sorted_all = np.sort(all_incomes)
                n_all = len(sorted_all)
                cumulative_all = np.cumsum(sorted_all)
                overall_gini_numer = np.sum((2 * np.arange(1, n_all + 1) - n_all - 1) * sorted_all)
                overall_gini = overall_gini_numer / (n_all * cumulative_all[-1]) if cumulative_all[-1] != 0 else 0
            else:
                overall_gini = 0
        else:
            overall_income = 0
            overall_coop_rate = 0
            overall_gini = 0

        gini_list.append(overall_gini)
        coop_rate_list.append(overall_coop_rate)
        income_list.append(overall_income)

        skill = self.skill_translation(income_list)

        return {
            "gini_coefficient": gini_list,
            "cooperation_rate": coop_rate_list,
            "individual_income": income_list,
        }
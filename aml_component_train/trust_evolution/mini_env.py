from __future__ import annotations

from Q_brain import QLearningAgent
from DQN_brain import DQNAgent
import matplotlib.pyplot as plt
import pandas as pd  # 确保在文件顶部导入 pandas
import numpy as np
import random
import plotly.graph_objects as go
import copy
from pathlib import Path
import os, json, sys
import pandas as pd


def repo_root() -> Path:
    """
    从当前文件位置向上找，直到找到包含 'ml' 目录或 '.git' 的目录。
    """
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "ml").exists() or (parent / ".git").exists():
            return parent
    return Path.cwd()

def resolve_path(path_like: str | Path, base: Path | None = None) -> Path:
    """
    把相对路径解析为绝对路径：默认相对 repo_root()
    """
    base = base or repo_root()
    path = Path(path_like)
    return path if path.is_absolute() else (base / path).resolve()

class MiniEnv:
    def __init__(self, trade_rules, round_number, reproduction_number, mistake_possibility, extrinsic_reward):  #[-3, -3, 0, 2, 5, 5] [0,0,3,-1,2,2 ]

        # rules
        self.agents = []
        self.agents_dict = {}
        self.trade_rules = trade_rules
        self.round_number = round_number
        self.reproduction_number = reproduction_number
        self.mistake_possibility = mistake_possibility

        # fixed rules
        self.trade_number = 5

        # Game record
        self.epoch_list = []
        self.IndividualIncome = []
        self.CooperationRate = []
        self.GiniCoefficient = []


        self.extrinsic_reward = extrinsic_reward
        self.aiType = 'Q'


    def plot_results(self):
        """
        绘制 Epoch 与 DQN 总资金 / 2 的关系图。
        """

        # 使用 resolve_path 解析路径
        output_path = resolve_path("ml/data/Q_money_over_epochs.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(12, 8))
        plt.plot(self.epoch_list, self.GiniCoefficient,
                 marker='o', linestyle='-', color='b',
                 label=self.aiType)

        plt.title(self.aiType, fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Gini Coefficient', fontsize=14)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(output_path)
        print(f"图像已保存至 {output_path}")

        # Azure 上不显示
        if os.getenv("AZUREML_RUN_ID") is None:
            plt.show()
    def plot_trust_matrix(self, trust_matrix, labels, filename="trust_matrix_heatmap.png"):
        """
        绘制一个范围为0到1的8×8 Trust Matrix热力图，并使用 plt.savefig 保存为图片。

        参数：
            trust_matrix: 字典形式，格式如下：
                {
                  'Random': {'Random': 0.49, 'Cheater': 0.001, ...},
                  'Cheater': {'Random': 0.513, 'Cheater': 0.0011, ...},
                  ...
                }
            labels: 长度为8的列表，每个元素为代理类别标签（按行/列对应）。
            filename: 保存图片的文件名，默认为 "trust_matrix_heatmap.png"。

        返回：
            fig: 绘制好的 matplotlib 图形对象。
        """
        # 将字典转换为二维数组（列表的列表），顺序按照 labels
        z = []
        for row_label in labels:
            row = []
            for col_label in labels:
                # 如果某一对不存在，则默认为0.5（中立状态）
                value = trust_matrix.get(row_label, {}).get(col_label, 0.5)
                row.append(value)
            z.append(row)

        # 转换为 numpy 数组
        z = np.array(z)

        # 创建图形和坐标轴
        fig, ax = plt.subplots(figsize=(8, 8))

        # 绘制热力图
        cax = ax.imshow(z, cmap='viridis', vmin=0, vmax=1)

        # 设置坐标轴刻度和标签
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        # 调整 x 轴标签的显示方式
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # 添加标题
        ax.set_title("8×8 Trust Matrix Heatmap")

        # 添加颜色条
        fig.colorbar(cax, ax=ax, label="Trust")

        # 保存图像
        out_path = resolve_path(filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        # 显示图像
        plt.show()

        return fig

    # 调用函数绘制并保存热力图
    def save_results_to_csv(self, filename='plotData.csv'):
        """
        将 epoch_list 和 dqn_total_money_div2_list 保存为 CSV 文件。

        参数:
            filename (str): 保存的文件名，默认为 'dqn_money_over_epochs.csv'
        """
        data = {
            'Epoch': self.epoch_list,
            'Gini Coefficient': self.GiniCoefficient
        }
        df = pd.DataFrame(data)
        out_path = resolve_path(filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"结果已保存到 {out_path}")

    def skill_translation(self, individual_income):
        """
        根据 oneCompetition 返回的 evaluation 结果计算 AI 玩家获得的平均资金，
        并将该值作为 Skill 值。

        参数:
            evaluation (dict): oneCompetition 的返回结果，其中包含键 'individual_income'，
                               它对应一个长度为9的列表，索引6对应 AI 玩家。

        返回:
            skill_value (float): AI 玩家平均获得的资金，作为 Skill 值。
        """
        # 提取 AI 玩家获得的平均资金（列表中第7项，对应索引6）
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
        if agentList is None: agentList = []
        if tradeRules is None: tradeRules = [-3, -3, 0, 2, 5, 5]
        if extrinsic_reward is None: extrinsic_reward = [0, 0]

        self.agents.clear()
        self.agents_dict.clear()
        self.setup(agentList, tradeRules, self.round_number, self.reproduction_number, self.mistake_possibility,
                   extrinsic_reward)

    def calculate_rewards(self, action_A, action_B):
        # 根据 tradeRules 定义奖励矩阵
        if action_A == 'cheat' and action_B == 'cheat':
            reward_A = self.trade_rules[0]
            reward_B = self.trade_rules[1]
        elif action_A == 'cheat' and action_B == 'cooperation':
            reward_A = self.trade_rules[2]
            reward_B = self.trade_rules[3]
        elif action_A == 'cooperation' and action_B == 'cheat':
            reward_A = self.trade_rules[3]
            reward_B = self.trade_rules[2]
        elif action_A == 'cooperation' and action_B == 'cooperation':
            reward_A = self.trade_rules[4]
            reward_B = self.trade_rules[5]
        else:
            reward_A = 0
            reward_B = 0
        return reward_A, reward_B

    def get_opponent_stereotype(self, opponent_id):
        opponent_agent = self.agents_dict.get(opponent_id)
        if opponent_agent is None:
            return None  # 如果找不到对手代理，返回 None
        return opponent_agent.stereotype
    def imitation_learning(self, ai_agent, top_agent):
        """
        让 AI 代理从表现最好的代理的交易记录中学习。

        参数：
            ai_agent: AI 代理实例。
            top_agent: 表现最好的代理实例。
        """
        for opponent_id, interactions in top_agent.tradeRecords.items():
            opponent_stereotype = self.get_opponent_stereotype(opponent_id)

            for interaction in interactions:
                # 解包动作
                top_agent_action, opponent_action = interaction

                # 构造状态，对于 AI 代理，与对手的交互历史可能不同
                # 需要确保状态表示一致
                state = ai_agent.get_state(opponent_id, opponent_stereotype)

                # 动作就是 top_agent 的行动
                action = top_agent_action

                # 使用 calculate_rewards() 计算奖励
                reward_top_agent, _ = self.calculate_rewards(top_agent_action, opponent_action)
                reward = reward_top_agent  # 我们关注的是 top_agent 的奖励

                # 假设 next_state 与 state 相同，或者根据实际情况更新
                next_state = state
                done = False  # 根据需要设置

                # 让 AI 代理学习
                ai_agent.learn(state, action, reward, next_state, done)


    def knowledgeTransform(self, RLID, ID_other):
        for trainingTimes in range(2000):
            opponent_stereotype = self.get_opponent_stereotype(ID_other)
            RL_stereotype = self.get_opponent_stereotype(RLID)

            for time in range(5):
                agent_A = self.agents_dict[RLID]
                agent_B = self.agents_dict[ID_other]

                # 获取状态
                observation_A = agent_A.get_state(ID_other,opponent_stereotype)
                observation_B = agent_B.get_state(RLID, RL_stereotype)

                # 选择行动
                if time == 4:
                    action_A = 'cheat'
                else:
                    action_A = 'cooperation'

                action_B = 'cooperation'

                # 记录自己的行动
                agent_A.record_action(action_A)
                agent_B.record_action(action_B)

                # 根据行动计算奖励
                reward_A, reward_B = self.calculate_rewards(action_A, action_B)

                # 更新代理的资金
                agent_A.addReward(reward_A)
                agent_B.addReward(reward_B)

                # 记录自己和对手的行动
                agent_A.Remember(ID_other, action_A, action_B)
                agent_B.Remember(RLID, action_B, action_A)

                # 学习（Q-learning 和 DQN 代理）
                if agent_A.stereotype == 'Q':
                    next_observation_A = agent_A.get_state(ID_other,opponent_stereotype)
                    done_A = False  # 根据需要设置
                    agent_A.learn(observation_A, action_A, reward_A, next_observation_A, done_A)
                elif agent_A.stereotype == 'DQN':
                    next_observation_A = agent_A.get_state(ID_other,opponent_stereotype)
                    done_A = False
                    agent_A.learn(observation_A, action_A, reward_A, next_observation_A, done_A)

                if agent_B.stereotype == 'Q':
                    next_observation_B = agent_B.get_state(RLID, RL_stereotype)
                    done_B = False  # 根据需要设置
                    agent_B.learn(observation_B, action_B, reward_B, next_observation_B, done_B)
                elif agent_B.stereotype == 'DQN':
                    next_observation_B = agent_B.get_state(RLID, RL_stereotype)
                    done_B = False
                    agent_B.learn(observation_B, action_B, reward_B, next_observation_B, done_B)


    def oneTrade(self, ID_A, ID_B):
        agent_A = self.agents_dict[ID_A]
        agent_B = self.agents_dict[ID_B]

        A_stereotype = self.get_opponent_stereotype(ID_A)
        B_stereotype = self.get_opponent_stereotype(ID_B)

        # 获取状态
        observation_A = agent_A.get_state(ID_B, B_stereotype)
        observation_B = agent_B.get_state(ID_A, A_stereotype)

        # 选择行动
        if agent_A.stereotype == 'Q':
            action_A = agent_A.choose_action(observation_A)
        elif agent_A.stereotype == 'DQN':
            action_A = agent_A.choose_action(observation_A)  # 假设 DQNAgent 有类似的方法
        else:
            action_A = agent_A.StereotypeAction(ID_B)

        if agent_B.stereotype == 'Q':
            action_B = agent_B.choose_action(observation_B)
        elif agent_B.stereotype == 'DQN':
            action_B = agent_B.choose_action(observation_B)  # 假设 DQNAgent 有类似的方法
        else:
            action_B = agent_B.StereotypeAction(ID_A)

        # mistake possibility
        if random.random() < self.mistake_possibility:
            action_A = 'cooperation' if action_A == 'cheat' else 'cheat'
        if random.random() < self.mistake_possibility:
            action_B = 'cooperation' if action_B == 'cheat' else 'cheat'

        # 记录自己的行动
        agent_A.record_action(action_A)
        agent_B.record_action(action_B)

        # 根据行动计算奖励
        reward_A, reward_B = self.calculate_rewards(action_A, action_B)



        #-----------------extrinsic_reward
        if action_A == 'cooperation':
            reward_A += self.extrinsic_reward[0]
        else:
            reward_A += self.extrinsic_reward[1]
        if action_B == 'cooperation':
            reward_B += self.extrinsic_reward[0]
        else:
            reward_B += self.extrinsic_reward[1]

        # 更新代理的资金
        agent_A.addReward(reward_A)
        agent_B.addReward(reward_B)
        # 记录自己和对手的行动
        agent_A.Remember(ID_B, action_A, action_B)
        agent_B.Remember(ID_A, action_B, action_A)



        # 学习（Q-learning 和 DQN 代理）
        if agent_A.stereotype == 'Q':
            next_observation_A = agent_A.get_state(ID_B, B_stereotype)
            done_A = False  # 根据需要设置
            agent_A.learn(observation_A, action_A, reward_A, next_observation_A, done_A)
        elif agent_A.stereotype == 'DQN':
            next_observation_A = agent_A.get_state(ID_B, B_stereotype)
            done_A = False
            agent_A.learn(observation_A, action_A, reward_A, next_observation_A, done_A)

        if agent_B.stereotype == 'Q':
            next_observation_B = agent_B.get_state(ID_A, A_stereotype)
            done_B = False  # 根据需要设置
            agent_B.learn(observation_B, action_B, reward_B, next_observation_B, done_B)
        elif agent_B.stereotype == 'DQN':
            next_observation_B = agent_B.get_state(ID_A, A_stereotype)
            done_B = False
            agent_B.learn(observation_B, action_B, reward_B, next_observation_B, done_B)

        return action_A, action_B

    def oneCompetition(self, episode_id, agent_type_names, agent_types_order, record_data, save_model, printGame):
        # ------------------------------
        # 进行一轮比赛（多个回合），包括交易、淘汰、复制等操作
        # ------------------------------
        self.aiType = agent_type_names['AI']
        total_agents = len(self.agents)  # 代理的总数

        for round_num_id in range(self.round_number):
            if printGame:
                print(f"--- Episode {episode_id}, Round {round_num_id + 1} begin ---")

            # 重置所有代理的状态
            for agent in self.agents:
                agent.money = 0
                agent.tradeRecords = {}
                agent.tradeNum = 0
                agent.cooperation_count = 0
                agent.cheat_count = 0

            # 每个代理与其他代理进行交易
            for agent_A in self.agents:
                for agent_B in self.agents:
                    if agent_A.id != agent_B.id:
                        for _ in range(self.trade_number):
                            self.oneTrade(agent_A.id, agent_B.id)
                        agent_A.tradeNum += self.trade_number
                        agent_B.tradeNum += self.trade_number

            sorted_agents_desc = sorted(self.agents, key=lambda x: x.money, reverse=True)
            for rank, agent in enumerate(sorted_agents_desc, start=1):
                if agent.stereotype == 'Q' and printGame:
                    print(f"代理ID {agent.id} ({agent.stereotype}) 排名: {rank} / {len(self.agents)}")
            if printGame:
                print("============================\n")

            # =========Trust Metric=================
            # 定义代理类型顺序
            agent_types_order = ['Random', 'Cheater', 'Cooperator', 'Copycat', 'Grudger', 'Detective', agent_type_names['AI'], 'Human']

            # 初始化一个字典用于统计各群体间的合作次数和交互总次数
            group_stats = {s: {t: {'coop': 0, 'total': 0} for t in agent_types_order} for s in agent_types_order}

            # 遍历所有代理，统计每个代理与其他代理间的交互情况
            for agent in self.agents:
                s_type = agent.stereotype  # 本代理所属类型
                # 遍历本代理的所有交互记录，键为对手 id
                for opponent_id, interactions in agent.tradeRecords.items():
                    # 根据对手 id 从 self.agents_dict 中获取对手对象，提取其类型
                    opponent = self.agents_dict.get(opponent_id)
                    if opponent is None:
                        continue
                    t_type = opponent.stereotype
                    # 遍历所有交互记录（列表中的每个元组）
                    for interaction in interactions:
                        # 这里我们认为 interaction[1] 表示对手的行为，'cooperation' 记为合作，否则为背叛
                        if str(interaction[1]).lower() == 'cooperation':
                            group_stats[s_type][t_type]['coop'] += 1
                        group_stats[s_type][t_type]['total'] += 1

            # 计算每个群体之间的信任值，采用贝叶斯信任模型：T = (合作次数+1) / (总交互次数+2)
            trust_matrix = {s: {t: 0 for t in agent_types_order} for s in agent_types_order}
            for s in agent_types_order:
                for t in agent_types_order:
                    coop = group_stats[s][t]['coop']
                    total = group_stats[s][t]['total']
                    # 如果该组间没有任何交互，设定信任值为中立0.5
                    trust_value = (coop + 1) / (total + 2) if total > 0 else 0.5
                    trust_matrix[s][t] = trust_value

            # self.plot_trust_matrix(trust_matrix, agent_types_order, "trust_matrix_heatmap.png")

            # ==========================
            # 统计每个代理的类型、资金、合作次数、欺骗次数
            agent_stats = {}
            total_agents_current = len(self.agents)
            for agent in self.agents:
                agent_type = agent.stereotype
                agent_name = agent_type_names.get(agent_type, f"Type {agent_type}")
                if agent_type not in agent_stats:
                    agent_stats[agent_type] = {
                        'name': agent_name,
                        'count': 0,
                        'total_money': 0,
                        'cooperation_count': 0,
                        'cheat_count': 0
                    }
                agent_stats[agent_type]['count'] += 1
                agent_stats[agent_type]['total_money'] += agent.money
                agent_stats[agent_type]['cooperation_count'] += agent.cooperation_count
                agent_stats[agent_type]['cheat_count'] += agent.cheat_count

            for stats in agent_stats.values():
                stats['proportion'] = stats['count'] / total_agents_current * 100
            if printGame:
                print("----------------------")
            for agent_type in agent_types_order:
                stats = agent_stats.get(agent_type)
                if stats:
                    if printGame:
                        print(f"{stats['name']}: 数量={stats['count']}, 比例={stats['proportion']:.2f}%, "
                              f"总资金={stats['total_money']}, 合作次数={stats['cooperation_count']}, 欺骗次数={stats['cheat_count']}")
                else:
                    if printGame:
                        print(f"{agent_type_names.get(agent_type, f'Type {agent_type}')}: 无数据")
            if printGame:
                print(f"--- Epoch {episode_id + 1}, Round {round_num_id + 1} 结束 ---")
                print("----------------------")

            # 记录每一轮的评价结果（此处仅对 AI 代理进行记录）
            ai_agents = [agent for agent in self.agents if agent.stereotype == agent_type_names['AI']]
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
                    print(f"记录 Epoch {episode_id + 1}: AI 平均资金 = {average_individual_income_ai}\n")
                    print(f"记录 Epoch {episode_id + 1}: 合作率 = {cooperation_rate:.2f}\n")
                    print(f"记录 Epoch {episode_id + 1}: Gini 系数 = {gini_coefficient:.2f}\n")
            else:
                if printGame:
                    print("没有 AI 代理数据记录。\n")

            # 代理淘汰和复制过程（原有代码，不再修改）
            self.agents.sort(key=lambda x: x.money, reverse=True)
            top_agent = self.agents[0]
            ai_agents = [agent for agent in self.agents if agent.stereotype == agent_type_names['AI']]
            for ai_agent in ai_agents:
                self.imitation_learning(ai_agent, top_agent)
            ai_agents = [agent for agent in self.agents if agent.stereotype == agent_type_names['AI']]
            num_ai_agents = len(ai_agents)
            agents_to_eliminate = []
            ai_agents_eliminated = 0
            for agent in reversed(self.agents):
                if len(agents_to_eliminate) >= self.reproduction_number:
                    break
                if agent.stereotype == agent_type_names['AI']:
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
                if hasattr(agent, 'clone') and callable(getattr(agent, 'clone')):
                    new_agent = agent.clone(new_id=id_counter)
                else:
                    new_agent = copy.deepcopy(agent)
                    new_agent.id = id_counter
                new_agents.append(new_agent)
                id_counter += 1
            self.agents = top_agents + new_agents
            self.agents_dict = {agent.id: agent for agent in self.agents}
            if len(self.agents) != total_agents:
                raise ValueError("代理总数发生变化！")

        # 计算每一类代理的指标及总体指标
        # 定义代理类型顺序（8种）和返回列表
        agent_types_order = ['Random', 'Cheater', 'Cooperator', 'Copycat', 'Grudger', 'Detective', 'AI', 'Human']
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

        # 计算总体指标（所有代理）
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

        # 将总体指标附加到各列表末尾
        gini_list.append(overall_gini)
        coop_rate_list.append(overall_coop_rate)
        income_list.append(overall_income)

        skill = self.skill_translation(income_list)
        # 返回一个字典，其中三个键对应的值均为长度为9的列表
        return {
            'gini_coefficient': gini_list,
            'cooperation_rate': coop_rate_list,
            'individual_income': income_list
        }



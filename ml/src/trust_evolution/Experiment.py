import argparse
import os
import math
import random
import torchvision.transforms as transforms
from torchvision.utils import save_image
from agent import Agent
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import winsound
import seaborn as sns
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch
from pathlib import Path
from mini_env import MiniEnv
from Q_brain import QLearningAgent
from DQN_brain import DQNAgent
from warnings import simplefilter
import json, sys
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 确保已安装
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc
from sklearn.manifold import TSNE



simplefilter(action="ignore",category=FutureWarning)
simplefilter(action="ignore",category=UserWarning)

os.makedirs("agents", exist_ok=True)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

# Rule
parser.add_argument("--random_count", type=int, default=4, help="0, number of Random Action agents")
parser.add_argument("--cheater_count", type=int, default=4, help="1, number of Always Cheat agents")
parser.add_argument("--cooperator_count", type=int, default=4, help="2, number of Always Cooperate agents")
parser.add_argument("--copycat_count", type=int, default=4, help="3, number of Copycat agents")
parser.add_argument("--grudger_count", type=int, default=4, help="4, number of Grudger agents")
parser.add_argument("--detective_count", type=int, default=4, help="5, number of Detective agents")
parser.add_argument("--ai_count", type=int, default=1, help="6, number of AI agents")
parser.add_argument("--human_count", type=int, default=2, help="7, number of human agents")

parser.add_argument("--trade_rules", type=float, nargs=6, default=[0, 0, 3, -1, 2, 2], help="8 to 13, Trade rules as a list")
parser.add_argument("--round_number", type=int, default=3, help="14, The round number of a competition")
parser.add_argument("--reproduction_number", type=int, default=0, help="15, The reproduction number of each round")
parser.add_argument("--mistake_possibility", type=float, default=0.00, help="16, the possibility to take opposite action")

parser.add_argument("--fixed_rule", type=str2bool, default = "True", help="Use some fixed rule for agent training and testing")

# Rule for Game Flow
parser.add_argument("--extrinsic_reward", type=float, nargs=2, default=[0, 0], help="the reward to cooperation and cheat behaviour itself")


# Strategy
parser.add_argument("--humanPlayer", type=str2bool, default=False, help="True means there is human player")
parser.add_argument("--ai_type", type=str, default='Q', help="type of AI agent (e.g., 'Q', 'DQN')")

# Evaluation
parser.add_argument("--cooperationRate", type=float, default= 1, help="cooperation rate")
parser.add_argument("--individualIncome", type=float, default= 2, help="individual income")
parser.add_argument("--giniCoefficient", type=float, default= 0.50, help="Gini Coefficient")
# Evaluation for Game Flow
parser.add_argument("--difficulty", type=float, default= 0.01, help="difficulty")


# Designer and Evaluator
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.01, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--RuleDimension", type=int, default= 3, help="trade rule dimension")
parser.add_argument("--DE_train_episode", type=int, default=1, help="number of episode during Q table training")
parser.add_argument("--DE_test_episode", type=int, default=1, help="number of episode during test")
parser.add_argument("--layersNum", type=int, default=1, help="layer number of the generator output")
parser.add_argument("--evaluationSize", type=int, default=1, help="size of the evaluation metrics")

# Agent training
parser.add_argument("--agent_train_epoch", type=int, default=40, help="number of epoch of training agent")
parser.add_argument("--gamma", type=float, default=0.99, help="discount factor for rewards")
parser.add_argument("--epsilon", type=float, default=1.0, help="initial epsilon for epsilon-greedy")
parser.add_argument("--epsilon_decay", type=float, default=0.999, help="epsilon decay rate")
parser.add_argument("--epsilon_min", type=float, default=0.1, help="minimum epsilon")
parser.add_argument("--memory_size", type=int, default=10000, help="memory size for experience replay")
parser.add_argument("--target_update", type=int, default=10, help="how often to update the target network")
parser.add_argument("--state_size", type=int, default=20, help="size of the state vector")

# Other
# parser.add_argument("--publish", type=bool, default=True, help="True means publish data on web")

actionlist = ["cheat", "cooperation"]

publish = False
printQtable = False
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False

# 在模块顶层定义全局变量，用于判断是否已经初始化 Excel 文件
excel_initialized = False
first_save = False


def publish_excel_update(update_data,
                         excel_path="C:/Users/hilab/OneDrive/Desktop/Rule_Generation/WebDash/data/dataupdates.xlsx"):
    global excel_initialized

    # 第一次调用时，删除已有的 Excel 文件（或清空）
    if not excel_initialized:
        if os.path.exists(excel_path):
            os.remove(excel_path)
        excel_initialized = True

    # 如果文件存在，则读取已有数据并追加，否则直接创建新 DataFrame
    if os.path.exists(excel_path):
        df_existing = pd.read_excel(excel_path)
        df_new = pd.DataFrame([update_data])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = pd.DataFrame([update_data])

    df_combined.to_excel(excel_path, index=False)
    print("UPDATE_EXCEL:" + json.dumps({"excel_updated": True}))
    sys.stdout.flush()


def plot_q_table(q_table, output_path="q_table_heatmap.png"):
    """
    绘制 Q-table 的热图，并保存为图像文件。
    数据根据 observation 前缀（格式如 "0_..."）分为 8 组，
    分别对应 8 种性格（前缀 0 到 7），在一张图中自上而下绘制 8 个热力图。

    参数：
      q_table: pandas DataFrame，包含至少三列：'cheat', 'cooperation' 和 'observation'。
      output_path: 图像保存的路径，默认保存为 "q_table_heatmap.png"。
    """
    # 如果没有 'observation' 列，则使用索引作为 observation
    if "observation" not in q_table.columns:
        q_table = q_table.copy()
        q_table["observation"] = q_table.index.astype(str)

    # 按照 observation 排序
    df_sorted = q_table.sort_values(by="observation").copy()

    # 提取前缀，假设 observation 格式为 "0_XXXXXXXX", "1_XXXXXX" 等
    def get_prefix(obs):
        try:
            return obs.split('_')[0]
        except Exception:
            return None

    df_sorted['prefix'] = df_sorted["observation"].apply(get_prefix)

    # 定义前缀对应的性格映射
    personality_map = {
        "0": "Random",
        "1": "Cheater",
        "2": "Cooperator",
        "3": "Copycat",
        "4": "Grudger",
        "5": "Detective",
        "6": "AI",
        "7": "Human"
    }

    # 建立 8 个子图（nrows=8）
    fig, axes = plt.subplots(nrows=8, figsize=(12, 4 * 8))

    for i in range(8):
        prefix_str = str(i)
        group = df_sorted[df_sorted['prefix'] == prefix_str]
        ax = axes[i]
        if group.empty:
            # 若无数据，显示性格名称及提示 "No Data"，关闭坐标轴
            ax.set_title(f"{personality_map.get(prefix_str, prefix_str)}: No Data")
            ax.axis('off')
        else:
            # 对该组数据归一化 cheat 和 cooperation 列
            combined_min = min(group["cheat"].min(), group["cooperation"].min())
            combined_max = max(group["cheat"].max(), group["cooperation"].max())
            group = group.copy()
            group["cheat_norm"] = (group["cheat"] - combined_min) / (combined_max - combined_min)
            group["cooperation_norm"] = (group["cooperation"] - combined_min) / (combined_max - combined_min)
            # 绘制热图
            data = group[["cheat_norm", "cooperation_norm"]].T
            sns.heatmap(data, cmap="coolwarm", annot=False,
                        xticklabels=group["observation"], yticklabels=["cheat", "cooperation"],
                        ax=ax)
            ax.set_title(f"{personality_map.get(prefix_str, prefix_str)} (n={len(group)})")
            ax.tick_params(axis='x', rotation=90, labelsize=8)
    plt.tight_layout()

    # 如果目标文件已存在则删除
    if os.path.exists(output_path):
        os.remove(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_test_results_to_excel(test_result, epoch,
                               excel_path="C:/Users/hilab/OneDrive/Desktop/Rule_Generation/WebDash/data/test_results.xlsx"):
    """
    将测试结果保存到 Excel 文件中。
    格式：
        epoch, gini_coefficient, cooperation_rate, individual_income
    其中 epoch 为数字，其它3个字段为 9 维列表（以字符串形式保存）。

    如果是第一次调用（即文件存在时），则删除文件后重写；后续调用则追加记录。

    参数：
        test_result: 字典，包含 'gini_coefficient', 'cooperation_rate', 'individual_income'
                     每个键对应一个长度为9的列表。
        epoch: 数字，表示当前测试的 epoch 数量
        excel_path: Excel 文件保存路径
    """
    global first_save

    # 将 9 维列表转换为字符串保存
    row = {
        "epoch": epoch,
        "gini_coefficient": str(test_result['gini_coefficient']),
        "cooperation_rate": str(test_result['cooperation_rate']),
        "individual_income": str(test_result['individual_income'])
    }
    df_new = pd.DataFrame([row])

    # 第一次调用时，如果文件已存在则删除文件
    if not first_save:
        if os.path.exists(excel_path):
            try:
                os.remove(excel_path)
                print(f"已删除旧文件：{excel_path}")
            except Exception as e:
                print("删除旧Excel文件时出错：", e)
        # 保存新文件
        df_new.to_excel(excel_path, index=False)
        first_save = True
    else:
        # 后续调用时追加记录
        try:
            if os.path.exists(excel_path):
                df_existing = pd.read_excel(excel_path)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_excel(excel_path, index=False)
            else:
                df_new.to_excel(excel_path, index=False)
        except Exception as e:
            print("保存Excel数据时出错：", e)
    print(f"测试结果已保存到 {excel_path}")
# def save_test_results_to_excel(test_result, epoch,
#                                excel_path="C:/Users/hilab/OneDrive/Desktop/Rule_Generation/WebDash/data/test_results.xlsx"):
#     """
#     将测试结果保存到 Excel 文件中。
#     格式：
#         epoch, gini_coefficient, cooperation_rate, individual_income
#     其中 epoch 为数字，其它3个字段为 9 维列表（以字符串形式保存）。
#
#     如果文件存在，则追加记录；否则新建文件。
#
#     参数：
#         test_result: 字典，包含 'gini_coefficient', 'cooperation_rate', 'individual_income'
#                      每个键对应一个长度为9的列表。
#         epoch: 数字，表示当前测试的 epoch 数量
#         excel_path: Excel 文件保存路径
#     """
#     # 将 9 维列表转换为字符串保存
#
#     row = {
#         "epoch": epoch,
#         "gini_coefficient": str(test_result['gini_coefficient']),
#         "cooperation_rate": str(test_result['cooperation_rate']),
#         "individual_income": str(test_result['individual_income'])
#     }
#     df_new = pd.DataFrame([row])
#
#     if os.path.exists(excel_path):
#         try:
#             df_existing = pd.read_excel(excel_path)
#             df_combined = pd.concat([df_existing, df_new], ignore_index=True)
#             df_combined.to_excel(excel_path, index=False)
#         except Exception as e:
#             print("保存Excel数据时出错：", e)
#     else:
#         df_new.to_excel(excel_path, index=False)
#     print(f"测试结果已保存到 {excel_path}")




class RuleDesigner(nn.Module):
    def __init__(self):
        super(RuleDesigner, self).__init__()

        def block(in_feat, out_feat, normalize=False):  # batch size 为1时无需 BatchNorm
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.evaluationSize, 8, normalize=False),
            *block(8, 16),
            nn.Linear(16, int(np.prod((opt.layersNum, opt.RuleDimension)))),
            nn.Softmax(dim=1)   # 指定在第1维上进行归一化
        )

    def forward(self, z):
        output = self.model(z)
        output = output.view(output.size(0), int(np.prod((opt.layersNum, opt.RuleDimension))))
        return output



class Evaluator(nn.Module):
    def __init__(self):
        super(Evaluator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod((opt.layersNum, opt.RuleDimension))), 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

class Environment(nn.Module):
    def __init__(self):
        super(Environment, self).__init__()

        trade_rules = [0, 0, 3, -1, 2, 2]
        round_number = 1
        reproduction_number = 5
        mistake_possibility = 0.05
        extrinsic_reward = [0, 0]

        self.shared_q_table = pd.DataFrame(columns=actionlist, dtype=np.float64)

        self.env = MiniEnv(trade_rules, round_number, reproduction_number, mistake_possibility, extrinsic_reward)  # [-3, -3, 0, 2, 5, 5]  [0,0,3,-1,2,2 ]
        self.agents = []

    def reset_shared_q_table(self):
        self.shared_q_table = pd.DataFrame(columns=self.shared_q_table.columns, dtype=np.float64)

        for agent in self.agents:
            if isinstance(agent, QLearningAgent):
                agent.q_table = self.shared_q_table

    def create_agents(self, agent_counts):
        # 如果 agent_counts 是 numpy 数组，则转换为字典
        if isinstance(agent_counts, np.ndarray):
            agent_counts = {
                'Random': int(agent_counts[0]),
                'Cheater': int(agent_counts[1]),
                'Cooperator': int(agent_counts[2]),
                'Copycat': int(agent_counts[3]),
                'Grudger': int(agent_counts[4]),
                'Detective': int(agent_counts[5]),
                'AI': int(agent_counts[6]),
                'Human': int(agent_counts[7])
            }

        agents = []
        id_counter = 0

        for stereotype, count in agent_counts.items():
            for _ in range(count):
                if stereotype == 'AI':
                    # 根据 opt.ai_type 来区分创建 Q-learning 或 DQN 代理
                    if opt.ai_type == 'Q':
                        agent = QLearningAgent(
                            id=id_counter,
                            actions=actionlist,
                            shared_q_table=self.shared_q_table,  # 传入共享的 Q 表
                            learning_rate=opt.lr,  # 使用全局学习率
                            reward_decay=opt.gamma,  # 使用全局折扣因子
                            e_greedy=opt.epsilon  # 使用全局探索率
                        )
                    elif opt.ai_type == 'DQN':
                        agent = DQNAgent(
                            id=id_counter,
                            actions=actionlist,
                            learning_rate=opt.lr,
                            gamma=opt.gamma,
                            epsilon=opt.epsilon,
                            epsilon_min=opt.epsilon_min,
                            epsilon_decay=opt.epsilon_decay,
                            batch_size=opt.batch_size,
                            memory_size=opt.memory_size,
                            target_update=opt.target_update,
                            state_size=opt.state_size
                        )
                    else:
                        # 如果未指定具体的AI类型，则退回创建通用Agent
                        agent = Agent(id=id_counter, stereotype=stereotype)
                else:
                    # 对于其他代理类型，直接创建 Agent
                    agent = Agent(id=id_counter, stereotype=stereotype)
                agents.append(agent)
                id_counter += 1
        return agents

    def trainAgent(self, initial_agent_counts, trade_rules, round_number, reproduction_number, mistake_possibility,
                        extrinsic_reward, DesignerEpochID, save_model):

        # initialize
        difficulty = difficulty_translation(extrinsic_reward)

        if not self.agents:
            self.reset_shared_q_table()
            self.agents = self.create_agents(initial_agent_counts)

        record_data = True

        skill_list = []

        # 外层循环：进行 num_episodes 次游戏（Epoch）
        for epoch_id in range(opt.agent_train_epoch):
            if printGame:
                print(f"========== Epoch {epoch_id + 1} 开始 ==========")

            self.env.setup(self.agents, trade_rules, round_number, reproduction_number, mistake_possibility, extrinsic_reward)
            evaluation = self.env.oneCompetition(epoch_id, agent_type_names, agent_types_order, record_data, save_model, printGame)
            skill_value = evaluation['individual_income'][6]
            skill_list.append(skill_value)

            if printGame:
                print(f"========== Epoch {epoch_id + 1} 结束 ==========")

        # self.env.plot_results()  # 训练结束后绘制图表
        # self.env.save_results_to_csv()  # 保存结果为 CSV 文件

        save_dir = os.path.join('data', agent_type_names['AI'])
        os.makedirs(save_dir, exist_ok=True)
        fileName = os.path.join(save_dir, f"{DesignerEpochID}.csv")
        self.env.save_results_to_csv(fileName)

        # 所有 Epoch 结束后的统计
        print("所有游戏（Epoch）结束，统计最终代理的种类及数量：")
        agent_type_counts = {}
        for agent in self.agents:
            agent_type = agent.stereotype
            agent_type_counts[agent_type] = agent_type_counts.get(agent_type, 0) + 1

        for agent_type in agent_types_order:
            count = agent_type_counts.get(agent_type, 0)
            agent_name = agent_type_names.get(agent_type, f"Type {agent_type}")
            print(f"{agent_name}: 数量={count}")

        return difficulty, skill_list


    def testAgent(self, initial_agent_counts, trade_rules, round_number, reproduction_number, mistake_possibility,
                  test_epoch_number=1):
        """
        测试AI代理在给定交易规则下的表现，并统计每一类及总体的：
          - gini_coefficient,
          - cooperation_rate,
          - individual_income
        返回一个包含上述3个指标的字典，每个指标为长度为9的列表，
        前8项对应各类型（顺序见 agent_types_order），最后一项为总体结果。
        本函数通过多次调用 oneCompetition 后，将每次的结果累加再平均。
        """
        # 初始化累加器（长度为9）
        accum_gini = np.zeros(9)
        accum_coop = np.zeros(9)
        accum_income = np.zeros(9)

        # 循环运行多个 epoch，并累计各指标
        for epoch_id in range(test_epoch_number):
            # 这里假设 oneCompetition 返回一个字典，统计当前 epoch 的指标
            results = self.env.oneCompetition(1, agent_type_names, agent_types_order, False, False, printGame)
            # results['gini_coefficient'], results['cooperation_rate'], results['individual_income'] 均为长度为9的列表
            accum_gini += np.array(results['gini_coefficient'])
            accum_coop += np.array(results['cooperation_rate'])
            accum_income += np.array(results['individual_income'])

        # 求各指标的平均值
        final_gini = (accum_gini / test_epoch_number).tolist()
        final_coop = (accum_coop / test_epoch_number).tolist()
        final_income = (accum_income / test_epoch_number).tolist()

        return {
            'gini_coefficient': final_gini,
            'cooperation_rate': final_coop,
            'individual_income': final_income
        }


    def forward(self, initial_agent_counts, trade_rules, round_number, reproduction_number, mistake_possibility,
                extrinsic_reward, DesignerEpochID,
                save_model):

        # initialize
        self.agents = []
        self.env.setup(self.agents, trade_rules, round_number, reproduction_number, mistake_possibility, extrinsic_reward)

        # train
        difficulty, skill_list = self.trainAgent(initial_agent_counts, trade_rules, round_number, reproduction_number, mistake_possibility,
                        extrinsic_reward, DesignerEpochID, save_model)

        # # test
        # Evaluation = self.testAgent(initial_agent_counts, trade_rules, round_number, reproduction_number, mistake_possibility,
        #                                   test_epoch_number= 10)
        if printQtable:
            for agent in self.agents:
                if isinstance(agent, QLearningAgent):
                    plot_q_table(agent.q_table, output_path="C:/Users/hilab/OneDrive/Desktop/Rule_Generation/WebDash/data/q_table_heatmap.png")
                    break
        # return Evaluation
        return difficulty, skill_list

def save_extrinsic_reward_results_to_csv(filename='extrinsic_reward.csv', epoch_list=[], extrinsic_reward_list=[]):
    """
    将 epoch_list 和 dqn_total_money_div2_list 保存为 CSV 文件。

    参数:
        filename (str): 保存的文件名，默认为 'dqn_money_over_epochs.csv'
    """
    data = {
        'Epoch': epoch_list,
        'Extrinsic Reward': extrinsic_reward_list
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"结果已保存到 {filename}")




def plot_skill_vs_difficulty(all_difficulty, all_skill, num_epochs, output_path="skill_vs_difficulty.png"):
    """
    绘制 Skill 与 Difficulty 的折线图，每隔5个 epoch 画一条曲线。

    参数:
        all_difficulty (list): 每个 DesignerEpoch 对应的 difficulty 值列表，
                               长度为 DesignerEpoch 的个数。
        all_skill (list): 每个 DesignerEpoch 下的 skill 列表，
                          每个元素为长度为 num_epochs 的列表，
                          表示各 epoch 的 Skill 值。
        num_epochs (int): 每个 DesignerEpoch 内的 epoch 数量。
        output_path (str): 保存图像的路径，默认为 "skill_vs_difficulty.png"。
    """
    plt.figure(figsize=(10, 6))

    # 每隔5个 epoch 绘制一条曲线
    for epoch_idx in range(0, num_epochs, 5):
        x = all_difficulty
        y = [skill_list[epoch_idx] for skill_list in all_skill]
        plt.plot(x, y, marker='o', label=f'Epoch {epoch_idx + 1}')

    plt.xlabel('Difficulty')
    plt.ylabel('Skill (AI Avg Income)')
    plt.title('Skill vs Difficulty')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()


def get_fixed_rules_vector(opt):
    return np.array([
        opt.random_count,        # rules[0]: Number of Random Action agents
        opt.cheater_count,       # rules[1]: Number of Always Cheat agents
        opt.cooperator_count,    # rules[2]: Number of Always Cooperate agents
        opt.copycat_count,       # rules[3]: Number of Copycat agents
        opt.grudger_count,       # rules[4]: Number of Grudger agents
        opt.detective_count,     # rules[5]: Number of Detective agents
        opt.ai_count,            # rules[6]: Number of AI agents
        opt.human_count,         # rules[7]: Number of Human agents

        opt.trade_rules[0],      # rules[8]: Payoff for both agents cheating
        opt.trade_rules[1],      # rules[9]: Payoff for the other agent when one cheats and the other cooperates
        opt.trade_rules[2],      # rules[10]: Payoff for agent A when A cheats and B cooperates
        opt.trade_rules[3],      # rules[11]: Payoff for agent B when B cheats and A cooperates
        opt.trade_rules[4],      # rules[12]: Payoff for both agents cooperating
        opt.trade_rules[5],      # rules[13]: Payoff for both agents cooperating (second dimension)

        opt.round_number,        # rules[14]: Number of rounds in each competition
        opt.reproduction_number, # rules[15]: Reproduction number of top players each round
        opt.mistake_possibility  # rules[16]: Mistake probability
    ])

def rule_translation(rule_vector):
    # 提取前 8 个值，代表不同类型代理的数量
    total_agents = 25
    initial_counts_raw = rule_vector[:8]

    # 将这 8 个值归一化，使它们的和为 1
    normalized_counts = initial_counts_raw / np.sum(initial_counts_raw)

    # 将归一化后的比例乘以总数量，并四舍五入为整数
    initial_counts = np.round(normalized_counts * total_agents).astype(int)

    # 确保总和为 total_agents，如果不为 total_agents，进行调整
    current_total = np.sum(initial_counts)
    if current_total != total_agents:
        diff = total_agents - current_total
        # 根据差值进行调整，随机选择一个索引来增加或减少
        while diff != 0:
            index = np.random.choice(8)
            if diff > 0:
                initial_counts[index] += 1
                diff -= 1
            elif diff < 0 and initial_counts[index] > 0:
                initial_counts[index] -= 1
                diff += 1

    # 提取支付矩阵部分
    payoff_raw = rule_vector[8:14]
    min_payoff = -5
    max_payoff = 5
    # 线性映射到期望范围
    payoff_matrix = payoff_raw * (max_payoff - min_payoff) + min_payoff

    # 提取轮数部分并映射为整数
    min_round = 5
    max_round = 20
    round_number = int(rule_vector[14] * (max_round - min_round) + min_round)

    # 提取复制比例部分并映射为整数
    min_rate = 1
    max_rate = 10
    reproduction_rate = int(rule_vector[15] * (max_rate - min_rate) + min_rate)

    # 提取犯错概率并映射为期望范围
    max_probability = 0.5
    mistake_probability = rule_vector[16] * max_probability

    return initial_counts, payoff_matrix, round_number, reproduction_rate, mistake_probability


def extrinsic_reward_translation(rule_vector):

    # 提取支付矩阵部分
    extrinsic_reward_raw = rule_vector[:2]
    min_extrinsic_reward = -opt.difficulty
    max_extrinsic_reward = opt.difficulty
    # 线性映射到期望范围
    extrinsic_reward = extrinsic_reward_raw * (max_extrinsic_reward - min_extrinsic_reward) + min_extrinsic_reward

    return extrinsic_reward


def difficulty_translation(extrinsic_reward, max_reward=opt.difficulty):
    """
    将 extrinsic reward 转换为 difficulty 值。

    参数:
        extrinsic_reward (array or list): 包含合作和作弊的奖励，
                                            第一个值对应合作奖励，
                                            第二个值对应作弊奖励，
                                            取值范围均为 [-max_reward, max_reward]。
        max_reward (float): 最大奖励绝对值，默认 3.0。

    返回:
        difficulty (float): 映射后的难度值，范围为 [0, 1]，
                            数值越低代表游戏越简单（奖励越高），
                            数值越高代表游戏越困难（奖励越低）。
    """
    extrinsic_reward = np.array(extrinsic_reward)
    # 计算平均奖励
    avg_reward = np.mean(extrinsic_reward)
    # 将平均奖励从 [-max_reward, max_reward] 映射到 [1, 0]，即:
    # 当 avg_reward = 3 -> difficulty = 1 - ((3+3)/(6)) = 0
    # 当 avg_reward = -3 -> difficulty = 1 - ((-3+3)/(6)) = 1
    difficulty = 1 - ((avg_reward + max_reward) / (2 * max_reward))
    return difficulty







def plot_skill_surfaces_at_epochs(desired_epochs,
                                  initial_agent_counts,
                                  trade_rules,
                                  round_number,
                                  reproduction_number,
                                  mistake_possibility,
                                  DE_epoch_id,
                                  save_model,
                                  extrinsic_reward_range=(-opt.difficulty,opt.difficulty),
                                  num_points=13,
                                  output_path="./data/skill_extrinsic_reward/skill_surfaces.png",
                                  csv_output_dir="./data/skill_extrinsic_reward/"):
    """
    对 extrinsic_reward 的第一维和第二维分别从 extrinsic_reward_range[0] 到 extrinsic_reward_range[1] 分成 num_points 份，
    共生成 num_points*num_points 个 extrinsic_reward 组合（二维向量）。
    对于每个组合，调用 environment 得到 (difficulty, skill_list)，
    然后分别提取 desired_epochs 中每个 epoch 对应的 Skill 值，构造多个 z_skill 曲面，
    最后在同一 3D 图中绘制这些曲面：
        X轴：extrinsic_reward 第一维 (合作奖励) —— 从 3 到 -3
        Y轴：extrinsic_reward 第二维 (欺骗奖励)
        Z轴：Skill 值 (AI 玩家平均资金)
    同时，将每个 desired epoch 对应的 X, Y, Skill 数据保存为 CSV 文件，如 epoch0.csv, epoch10.csv。
    """
    # 生成 extrinsic_reward 的离散取值
    reward_values = np.linspace(extrinsic_reward_range[0], extrinsic_reward_range[1], num_points)
    X, Y = np.meshgrid(reward_values, reward_values)

    # 初始化一个字典，每个 desired epoch 对应一个 Z 数组（与 X, Y 形状相同）
    Z_dict = {ep: np.zeros_like(X) for ep in desired_epochs}

    # 遍历网格，计算每个 extrinsic_reward 组合对应的 Skill 值
    for i in range(num_points):
        for j in range(num_points):
            # 构造 extrinsic_reward：[合作奖励, 欺骗奖励]
            extrinsic_reward = [X[i, j], Y[i, j]]
            # 调用 environment 进行一次比赛，返回 difficulty 和 skill_list
            # 注意：这里假设 environment 返回 (difficulty, skill_list)
            difficulty, skill_list = environment(initial_agent_counts, trade_rules, round_number, reproduction_number,
                                                 mistake_possibility,
                                                 extrinsic_reward, DE_epoch_id, save_model)
            # 对每个 desired epoch，取出对应的 Skill 值
            for ep in desired_epochs:
                Z_dict[ep][i, j] = skill_list[ep]

    # 对每个 Z 数据进行高斯平滑，使曲面更平滑
    for ep in desired_epochs:
        Z_dict[ep] = gaussian_filter(Z_dict[ep], sigma=1)

    # 保存 CSV 文件，每个 desired epoch 一个文件
    for ep in desired_epochs:
        # 构造 DataFrame，平铺 X, Y, 以及该 epoch 对应的 Skill 数据
        df = pd.DataFrame({
            "CooperationExtrinsicReward": X.flatten(),
            "CheatExtrinsicReward": Y.flatten(),
            "Skill": Z_dict[ep].flatten()
        })
        csv_filename = f"{csv_output_dir}epoch{ep}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"已保存 CSV 文件：{csv_filename}")

    # 定义颜色映射（假设 desired_epochs 有 4 个元素）
    # 如果 desired_epochs 个数不为 4，可根据实际情况修改颜色列表
    color_list =  ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'brown', 'pink']
    colors = {ep: color_list[idx] for idx, ep in enumerate(desired_epochs)}

    # 绘制 3D 曲面图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制每个 desired epoch 对应的 Skill 曲面
    for ep in desired_epochs:
        ax.plot_surface(X, Y, Z_dict[ep], color=colors[ep], alpha=0.8, edgecolor='none')

    # 设置坐标轴标签与标题
    ax.set_xlabel("Cooperation Extrinsic Reward", fontsize=12, labelpad=10)
    ax.set_ylabel("Cheat Extrinsic Reward", fontsize=12, labelpad=10)
    ax.set_zlabel("Skill (Avg Income)", fontsize=12, labelpad=10)
    ax.set_title("Skill Surfaces for Epochs " + ", ".join(str(ep) for ep in desired_epochs), fontsize=16)

    # 反转 X 轴，使得 Cooperation 奖励从 3 到 -3
    ax.set_xlim(3, -3)

    # 添加图例，使用代理艺术家
    legend_elements = [Line2D([0], [0], marker='s', color='w', label=f"Epoch {ep}",
                              markerfacecolor=colors[ep], markersize=10)
                       for ep in desired_epochs]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()


def trading_rule_difficulty(initial_agent_counts,
                            trade_rules,
                            round_number,
                            reproduction_number,
                            mistake_possibility,
                            DE_epoch_id,
                            save_model,
                            extrinsic_reward,
                            num_samples=10, output_csv="trading_rule_difficulty.csv",
                            tsne_output="trading_rule_tsne.png"):
    """
    1. 利用拉丁超立方采样，在6维空间（每个维度取值范围[-3,3]）中采样 num_samples 个 trading rule 样本。
    2. 对每个交易规则样本，将其传入 Environment 模块进行测试，得到 (dummy_difficulty, skill_list)。
       我们以 skill_list 的最终值（例如最后一个 epoch 的 Individual income）作为该交易规则的“难度”参考指标。
    3. 将每个交易规则及其对应的测得指标保存到 CSV 文件中。
    4. 对所有采样的 6 维交易规则数据应用 t-SNE 降维，将结果在二维散点图中展示，颜色表示测得的收入值（难度指标）。

    参数:
      initial_agent_counts: 固定的初始代理数量
      round_number: 游戏轮数
      reproduction_number: 每轮复制数量
      mistake_possibility: 犯错概率
      DE_epoch_id: 设计训练轮次编号（用于环境调用）
      save_model: 是否保存模型（传递给 environment）
      num_samples (int): 拉丁超立方采样的样本数，默认180
      output_csv (str): 结果保存的 CSV 文件路径
      tsne_output (str): t-SNE 可视化结果保存的图像路径
    """
    # 1. 拉丁超立方采样：6维，每个维度均匀采样 num_samples 个点
    sampler = qmc.LatinHypercube(d=6)
    sample = sampler.random(n=num_samples)  # 样本在 [0,1] 区间内
    # 将 [0,1] 线性映射到 [-3,3]： x_mapped = x * 6 - 3
    trading_rules = sample * 6 - 3  # shape = (num_samples, 6)

    extrinsic_reward = [0, 0]
    # 存储结果：每行包含6个 trading rule 参数和对应的 measured_income（作为 difficulty 的参考指标）
    results = []

    # 这里假设对于每个 trading rule，我们调用 environment 得到 (dummy_difficulty, skill_list)
    # 我们选取 skill_list 中最后一个值作为该样本对应的 Individual income
    for rule in trading_rules:
        # rule 为一个6维数组
        # 调用 environment，这里将 trading_rule 传入相应参数位置（请根据实际接口修改）
        difficulty_dummy, skill_list = environment(initial_agent_counts, rule, round_number, reproduction_number,
                                                   mistake_possibility, extrinsic_reward, DE_epoch_id, save_model)
        # 选择最后一个 epoch 的 skill 作为指标
        measured_income = skill_list[-1]
        results.append(np.concatenate([rule, [measured_income]]))

    results = np.array(results)  # shape: (num_samples, 7)

    # 2. 保存数据到 CSV 文件，列名为 trading_rule_1, ..., trading_rule_6, income
    col_names = [f"trading_rule_{i + 1}" for i in range(6)] + ["income"]
    df_results = pd.DataFrame(results, columns=col_names)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_results.to_csv(output_csv, index=False)
    print(f"结果已保存到 {output_csv}")

    # 3. 使用 t-SNE 降维，将6维 trading rule 降到2维
    tsne_model = TSNE(n_components=2, perplexity=5, random_state=42)
    tsne_result = tsne_model.fit_transform(trading_rules)  # shape: (num_samples, 2)

    # 4. 绘制 t-SNE 散点图，颜色根据 measured_income（即 results 中最后一列）着色
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=results[:, -1], cmap="viridis", s=50)
    plt.xlabel("t-SNE 1", fontsize=14)
    plt.ylabel("t-SNE 2", fontsize=14)
    plt.title("t-SNE of 6D Trading Rule Samples (colored by Income)", fontsize=16)
    plt.colorbar(sc, label="Income")
    plt.tight_layout()
    os.makedirs(os.path.dirname(tsne_output), exist_ok=True)
    plt.savefig(tsne_output, dpi=300)
    print(f"t-SNE图已保存至 {tsne_output}")
    plt.show()


def mistake_difficulty(initial_agent_counts,
                            trade_rules,
                            round_number,
                            reproduction_number,
                            mistake_possibility,
                            DE_epoch_id,
                            save_model,
                            extrinsic_reward,
                            num_samples=10, output_csv="mistake_possibility_difficulty.csv",
                            plot_output="mistake_possibility_tsne.png"):
    """
    1. 利用拉丁超立方采样，在6维空间（每个维度取值范围[-3,3]）中采样 num_samples 个 trading rule 样本。
    2. 对每个交易规则样本，将其传入 Environment 模块进行测试，得到 (dummy_difficulty, skill_list)。
       我们以 skill_list 的最终值（例如最后一个 epoch 的 Individual income）作为该交易规则的“难度”参考指标。
    3. 将每个交易规则及其对应的测得指标保存到 CSV 文件中。
    4. 对所有采样的 6 维交易规则数据应用 t-SNE 降维，将结果在二维散点图中展示，颜色表示测得的收入值（难度指标）。

    参数:
      initial_agent_counts: 固定的初始代理数量
      round_number: 游戏轮数
      reproduction_number: 每轮复制数量
      mistake_possibility: 犯错概率
      DE_epoch_id: 设计训练轮次编号（用于环境调用）
      save_model: 是否保存模型（传递给 environment）
      num_samples (int): 拉丁超立方采样的样本数，默认180
      output_csv (str): 结果保存的 CSV 文件路径
      tsne_output (str): t-SNE 可视化结果保存的图像路径
    """
    # 1. 拉丁超立方采样：6维，每个维度均匀采样 num_samples 个点
    sampler = qmc.LatinHypercube(d=1)
    sample = sampler.random(n=num_samples)  # 样本在 [0,1] 区间内
    mistake_possibility = sample * 0.5

    extrinsic_reward = [0, 0]
    # 存储结果：每行包含6个 trading rule 参数和对应的 measured_income（作为 difficulty 的参考指标）
    results = []

    # 这里假设对于每个 trading rule，我们调用 environment 得到 (dummy_difficulty, skill_list)
    # 我们选取 skill_list 中最后一个值作为该样本对应的 Individual income
    for rule in mistake_possibility:
        # rule 为一个6维数组
        # 调用 environment，这里将 trading_rule 传入相应参数位置（请根据实际接口修改）
        difficulty_dummy, skill_list = environment(initial_agent_counts, trade_rules, round_number, reproduction_number,
                                                   rule, extrinsic_reward, DE_epoch_id, save_model)
        # 选择最后一个 epoch 的 skill 作为指标
        measured_income = skill_list[-1]
        results.append(np.concatenate([rule, [measured_income]]))

    results = np.array(results)  # shape: (num_samples, 7)

    # 2. 保存数据到 CSV 文件，列名为 trading_rule_1, ..., trading_rule_6, income
    col_names = [f"mistake_possibility_{i + 1}" for i in range(1)] + ["income"]
    df_results = pd.DataFrame(results, columns=col_names)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_results.to_csv(output_csv, index=False)
    print(f"结果已保存到 {output_csv}")

    # # 3. 使用 t-SNE 降维，将6维 trading rule 降到2维
    # tsne_model = TSNE(n_components=1, perplexity=5, random_state=42)
    # tsne_result = tsne_model.fit_transform(mistake_possibility)  # shape: (num_samples, 2)
    #
    # # 4. 绘制 t-SNE 散点图，颜色根据 measured_income（即 results 中最后一列）着色
    # plt.figure(figsize=(10, 8))
    # sc = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=results[:, -1], cmap="viridis", s=50)
    # plt.xlabel("t-SNE 1", fontsize=14)
    # plt.ylabel("t-SNE 2", fontsize=14)
    # plt.title("t-SNE of 6D Trading Rule Samples (colored by Income)", fontsize=16)
    # plt.colorbar(sc, label="Income")
    # plt.tight_layout()
    # os.makedirs(os.path.dirname(tsne_output), exist_ok=True)
    # plt.savefig(tsne_output, dpi=300)
    # print(f"t-SNE图已保存至 {tsne_output}")
    # plt.show()
    # 4. 绘图：X轴 mistake possibility，Y轴 Income，颜色根据 Income 着色
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(results[:, 0], results[:, 1], c=results[:, 1], cmap="viridis", s=50)
    plt.xlabel("Mistake Possibility", fontsize=14)
    plt.ylabel("Income", fontsize=14)
    plt.title("Income vs. Mistake Possibility", fontsize=16)
    cbar = plt.colorbar(sc)
    cbar.set_label("Income", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_output), exist_ok=True)
    plt.savefig(plot_output, dpi=300)
    print(f"图像已保存至 {plot_output}")
    plt.show()


def initial_agent_counts_difficulty(initial_agent_counts,
                                    trade_rules,  # 固定交易规则
                                    round_number,
                                    reproduction_number,
                                    mistake_possibility,
                                    extrinsic_reward,
                                    DE_epoch_id,
                                    save_model,
                                    num_samples=180,
                                    output_csv="initial_agent_counts_difficulty.csv",
                                    tsne_output="initial_agent_counts_tsne.png",
                                    counts_dim=8,
                                    counts_range_min=0,
                                    counts_range_max=50):
    """
    1. 利用拉丁超立方采样，在 counts_dim 维初始代理数量空间（每个维度取值范围 [counts_range_min, counts_range_max]）中采样 num_samples 个样本。
       为了模拟离散代理数量，采样后将数值取整。
    2. 对于每个初始代理配置，将其传入 Environment 模块（其它参数固定）测试，得到 (dummy_difficulty, skill_list)；
       取 skill_list 中最后一个值作为该样本对应的 Individual income，作为“difficulty”的参考指标。
    3. 将每个样本的初始代理配置及其对应的 measured_income 保存到 CSV 文件中。
    4. 使用 t-SNE 将初始代理配置（高维）降到 2 维，并绘制散点图，点颜色根据 measured_income 着色。

    参数：
      trading_rule: 固定的交易规则（不变），用于环境测试
      round_number: 游戏轮数
      reproduction_number: 每轮复制数量
      mistake_possibility: 犯错概率
      extrinsic_reward: 固定的外部奖励参数
      DE_epoch_id: 设计训练轮次编号（传递给 environment）
      save_model: 是否保存模型（传递给 environment）
      num_samples (int): 拉丁超立方采样的样本数，默认 180
      output_csv (str): 结果保存的 CSV 文件路径
      tsne_output (str): t-SNE 可视化结果保存的图像路径
      counts_dim (int): 初始代理数量的维数（例如代理类型数），默认8
      counts_range_min (float): 每个维度的最小值，默认0
      counts_range_max (float): 每个维度的最大值，默认50
    """
    # 1. 拉丁超立方采样：在 counts_dim 维空间内采样 num_samples 个点
    sampler = qmc.LatinHypercube(d=counts_dim)
    sample = sampler.random(n=num_samples)  # 采样点在 [0,1]^counts_dim
    # 将采样点线性映射到 [counts_range_min, counts_range_max]
    initial_counts_samples = qmc.scale(sample, counts_range_min, counts_range_max)
    # 由于代理数量为整数，取整
    initial_counts_samples = np.rint(initial_counts_samples).astype(int)

    # 存储结果：每行包含 counts_dim 个初始代理数量及对应的 measured_income
    results = []

    # 2. 对每个初始代理配置调用 environment 得到结果（这里 trading_rule、round_number、reproduction_number、mistake_possibility、
    # extrinsic_reward、DE_epoch_id、save_model 均固定，只改变 initial_agent_counts）
    for counts in initial_counts_samples:
        # 注意：这里的 environment 接口需要能够接受 counts 作为初始代理配置
        # 例如 environment(initial_agent_counts, trading_rule, round_number, reproduction_number, mistake_possibility, extrinsic_reward, DE_epoch_id, save_model)
        # 请根据实际接口修改，下面假设 counts 就是 initial_agent_counts
        dummy_diff, skill_list = environment(counts, trade_rules, round_number, reproduction_number,
                                             mistake_possibility, extrinsic_reward, DE_epoch_id, save_model)
        # 取最后一个 epoch 的 skill（Individual income）作为指标
        measured_income = skill_list[-1]
        results.append(np.concatenate([counts, [measured_income]]))

    results = np.array(results)  # shape: (num_samples, counts_dim+1)

    # 3. 保存结果到 CSV 文件
    col_names = [f"agent_count_{i + 1}" for i in range(counts_dim)] + ["income"]
    df_results = pd.DataFrame(results, columns=col_names)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_results.to_csv(output_csv, index=False)
    print(f"结果已保存到 {output_csv}")

    # 4. 使用 t-SNE 将初始代理配置降到 2 维，并用散点图展示，颜色根据 income 着色
    tsne_model = TSNE(n_components=2, perplexity=5, random_state=42)
    tsne_result = tsne_model.fit_transform(initial_counts_samples)

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=results[:, -1], cmap="viridis", s=50)
    plt.xlabel("t-SNE 1", fontsize=14)
    plt.ylabel("t-SNE 2", fontsize=14)
    plt.title("t-SNE of Initial Agent Counts (colored by Income)", fontsize=16)
    cbar = plt.colorbar(sc)
    cbar.set_label("Income", fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(tsne_output), exist_ok=True)
    plt.savefig(tsne_output, dpi=300)
    print(f"t-SNE图已保存至 {tsne_output}")
    plt.show()





def plot_training_process_for_trade_rule_groups(initial_agent_counts,
                                                round_number,
                                                reproduction_number,
                                                mistake_possibility,
                                                extrinsic_reward,
                                                DE_epoch_id,
                                                save_model,
                                                num_epochs,
                                                output_csv="training_process_results_high.csv",
                                                output_plot="training_process.png"):
    """
    测试不同交易规则组对 agent 训练过程中 Individual income 的影响。

    参数:
      initial_agent_counts: 固定的初始代理数量
      round_number: 游戏轮数
      reproduction_number: 每轮复制数量
      mistake_possibility: 犯错概率
      extrinsic_reward: 固定的外部奖励参数
      DE_epoch_id: 设计训练轮次编号（用于环境调用）
      save_model: 是否保存模型（传递给 environment）
      trade_rule_groups (dict): 一个字典，键为组名（例如 "Low", "Medium", "High"），值为对应的6维 trading rule 向量
      num_epochs (int): 训练的 epoch 数量（例如 40）
      output_csv (str): 保存训练过程中各 epoch 结果的 CSV 文件路径
      output_plot (str): 保存绘制图像的文件路径
    """
    # 用于存储所有组的训练过程数据，每行包括: group, epoch, income

    trade_rule_groups = {
        "Low": np.array([-1.177282578, -1.482345946, -1.087466925, -2.907925356, -1.252640696, -0.788812408]),
        "Medium": np.array([-0.41097735, 2.361120932, -0.386333482, -1.977072218, 1.475039731, 0.702690951]),
        "High": np.array([-0.230105264, -2.389890292, -1.183604627, 2.589601439, 2.999660671, 2.272005864])
    }

    # trade_rule_groups = {
    #     "Low 1": np.array([]),
    #     "Low 2": np.array([]),
    #     "Low 3": np.array([]),
    #     "Low 4": np.array([]),
    #     "Low 5": np.array([])
    # }
    # trade_rule_groups = {
    #     "High 1": np.array([-1.970610399,	0.367538147,	-1.955744721,	2.722505338,	2.885549556,	1.467340952]),
    #     "High 2": np.array([-1.970610399,	0.367538147,	-1.955744721,	2.722505338,	2.885549556,	1.467340952]),
    #     "High 3": np.array([0.482584307,	-1.640214185,	2.919970043,	2.744662269,	-2.883049343,	0.47393471]),
    #     "High 4": np.array([-1.759425565,	0.774120033,	-0.305843851,	2.002150144,	2.913378955,	2.410853437]),
    #     "High 5": np.array([2.890421738,	1.332293284,	2.868411304,	2.06078246,	-2.307293618,	1.149497733])
    # }
    # trade_rule_groups = {
    #     "Medium 1": np.array([    -2.570783182,0.883897888,- 1.761860351,- 0.026318802,2.86499443, - 1.949292571]),
    #     "Medium 2": np.array([    -2.122077561,- 0.378099919,- 0.786817044,0.764101232,1.014065489,0.357228063]),
    #     "Medium 3": np.array([    -0.484877983, - 2.883851145, - 0.738393807,1.345635969,0.642108088,- 0.359373669]),
    #     "Medium 4": np.array([    -0.41097735,2.361120932,- 0.386333482,- 1.977072218,1.475039731,0.702690951]),
    #     "Medium 5": np.array([    -1.105055502,1.240737673,- 1.471400858,- 2.178101685,0.557541218,2.492708917])
    # }
    # trade_rule_groups = {
    #     "Low 1": np.array([-1.177282578, - 1.482345946, - 1.087466925, - 2.907925356, - 1.252640696, - 0.788812408]),
    #     "Low 2": np.array([-2.926014838, - 0.063973044, - 2.550610722, - 1.484016735, - 0.727814734, - 0.695829967]),
    #     "Low 3": np.array([-2.393065458, - 2.44798662, 1.399448713, - 2.960723016, - 0.68905033, - 2.758099553]),
    #     "Low 4": np.array([2.397393453, - 2.412930847, - 2.286878572, - 0.804228109, - 1.305893385, - 1.581059454]),
    #     "Low 5": np.array([-0.533547143, - 2.751772241, - 2.221207739, 0.582099827, - 1.386868919, - 1.036599245])
    # }

    records = []

    # 遍历每个交易规则组
    for group_name, trading_rule in trade_rule_groups.items():
        print(f"正在测试组 {group_name} 的交易规则：{trading_rule}")
        # 假设 environment 接口为：
        # evaluation = environment(initial_agent_counts, trading_rule, round_number, reproduction_number, mistake_possibility, extrinsic_reward, DE_epoch_id, save_model)
        # evaluation 中假设包含 key "individual_income"，它是一个列表，长度为 num_epochs
        dummy_diff, skill_list = environment(initial_agent_counts, trading_rule, round_number, reproduction_number,
                                 mistake_possibility, extrinsic_reward, DE_epoch_id, save_model)
        # 这里假设 evaluation["individual_income"] 返回各 epoch 的收入（或只返回最后一个值，如果你希望记录所有 epoch，就要求 environment 返回一个列表）
        income_over_epochs = skill_list
        if len(income_over_epochs) != num_epochs:
            print(f"警告：组 {group_name} 返回的 epoch 数 ({len(income_over_epochs)}) 不等于预期的 {num_epochs}")

        # 记录每个 epoch 的结果
        for epoch in range(len(income_over_epochs)):
            records.append({
                "group": group_name,
                "epoch": epoch,
                "income": income_over_epochs[epoch]
            })

    # 保存所有数据到 CSV
    df_records = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_records.to_csv(output_csv, index=False)
    print(f"训练过程数据已保存至 {output_csv}")

    # 绘制图像：X轴 epoch，Y轴 income，每个组用不同颜色显示曲线
    plt.figure(figsize=(10, 6))
    groups = df_records["group"].unique()
    # 为不同组指定颜色，可以自定义
    color_map = {"Low": "red", "Medium": "blue", "High": "green"}
    for group in groups:
        group_data = df_records[df_records["group"] == group]
        # 按 epoch 排序
        group_data = group_data.sort_values("epoch")
        plt.plot(group_data["epoch"], group_data["income"],
                 marker='o', linewidth=2, color=color_map.get(group, None), label=group)

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Individual Income", fontsize=14)
    plt.title("Training Process: Income vs. Epoch for Different Trading Rule Groups", fontsize=16)
    plt.legend(title="Trading Rule Group")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.savefig(output_plot, dpi=300)
    print(f"训练过程图已保存至 {output_plot}")
    plt.show()



def round_number_difficulty(initial_agent_counts, trade_rules, reproduction_number, mistake_possibility,
                            extrinsic_reward, DE_epoch_id, save_model,
                            num_samples=180,
                            output_csv="./data/round_number_difficulty/round_number_difficulty.csv",
                            plot_output="./data/round_number_difficulty/round_number_tsne.png"):
    """
    1. 使用拉丁超立方采样，在1维空间中采样 num_samples 个点，
       将采样结果线性映射到 [1,5]，得到 180 个 round_number 样本；
    2. 对每个 round_number 样本，将其传入 Environment 模块进行测试，
       得到返回的 (dummy_difficulty, skill_list)，取 skill_list[-1] 作为该样本对应的 Individual income；
    3. 将每个 round_number 与对应的 measured_income 保存到 CSV 文件中；
    4. 绘制图像：X 轴为 round_number，Y 轴为 measured_income，并拟合一条趋势线。

    参数：
      initial_agent_counts: 固定的初始代理数量
      trade_rules: 固定的交易规则
      reproduction_number: 每轮复制数量
      mistake_possibility: 犯错概率
      extrinsic_reward: 固定的外部奖励参数
      DE_epoch_id: 设计训练轮次编号（传递给 environment）
      save_model: 是否保存模型（传递给 environment）
      num_samples (int): 拉丁超立方采样的样本数，默认180
      output_csv (str): 结果保存的 CSV 文件路径
      plot_output (str): 绘图保存的图像文件路径
    """
    # 1. 拉丁超立方采样，1维采样
    sampler = qmc.LatinHypercube(d=1)
    sample = sampler.random(n=num_samples)  # 样本在 [0,1] 区间内
    # 将 [0,1] 线性映射到 [1,5]： round_number = sample*4 + 1
    round_numbers_cont = sample * 4 + 1  # 连续值
    # 取整：假设环境要求整数
    round_numbers = np.rint(round_numbers_cont).astype(int).flatten()
    extrinsic_reward = [0,0]
    results = []
    # 2. 对每个 round_number 调用 environment 得到 skill_list，取最后一个值作为 Individual income
    for rn in round_numbers:
        # 注意：调用环境的接口请根据实际修改
        dummy_diff, skill_list = environment(initial_agent_counts, trade_rules, rn, reproduction_number,
                                             mistake_possibility, extrinsic_reward, DE_epoch_id, save_model)
        measured_income = skill_list[-1]
        results.append([rn, measured_income])
        print(f"Round number {rn}: Income = {measured_income}")

    results = np.array(results)  # shape: (num_samples, 2)

    # 3. 保存数据到 CSV 文件
    df_results = pd.DataFrame(results, columns=["round_number", "income"])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_results.to_csv(output_csv, index=False)
    print(f"结果已保存到 {output_csv}")

    # 4. 绘制图像：X 轴为 round_number，Y 轴为 income，叠加趋势线
    plt.figure(figsize=(10, 6))
    plt.scatter(results[:, 0], results[:, 1], color='blue', alpha=0.7, label="Data Points")
    # 线性回归拟合趋势线
    coef = np.polyfit(results[:, 0], results[:, 1], 1)
    poly_fn = np.poly1d(coef)
    x_line = np.linspace(np.min(results[:, 0]), np.max(results[:, 0]), 100)
    plt.plot(x_line, poly_fn(x_line), color='red', linewidth=2, label="Trend Line")
    plt.xlabel("Round Number", fontsize=14)
    plt.ylabel("Income", fontsize=14)
    plt.title("Income vs. Round Number", fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_output), exist_ok=True)
    plt.savefig(plot_output, dpi=300)
    print(f"图像已保存至 {plot_output}")
    plt.show()



def reproduction_number_difficulty(initial_agent_counts, trade_rules, round_number, mistake_possibility,
                                   extrinsic_reward, DE_epoch_id, save_model,
                                   num_samples=10,
                                   output_csv="./data/reproduction_number_difficulty/reproduction_number_difficulty.csv",
                                   plot_output="./data/reproduction_number_difficulty/reproduction_number_tsne.png"):
    """
    1. 使用拉丁超立方采样，在1维空间中采样 num_samples 个 reproduction_number 样本，
       将采样结果线性映射到 [1, 10]，得到连续值后取整。
    2. 对每个 reproduction_number 样本，将其传入 Environment 模块进行模拟，
       得到 (dummy_difficulty, skill_list)，选取 skill_list 的最后一个值作为该样本对应的 Individual income，
       作为“难度”参考指标。
    3. 将每个 reproduction_number 与对应的 measured_income 保存到 CSV 文件中。
    4. 绘制图像：X 轴为 reproduction_number，Y 轴为 measured_income，并叠加趋势线展示变化趋势。

    参数：
      initial_agent_counts: 固定的初始代理数量
      trade_rules: 固定的交易规则
      round_number: 游戏轮数（固定）
      mistake_possibility: 犯错概率
      extrinsic_reward: 固定的外部奖励参数
      DE_epoch_id: 设计训练轮次编号（传递给 environment）
      save_model: 是否保存模型（传递给 environment）
      num_samples (int): 拉丁超立方采样的样本数，默认10
      output_csv (str): 结果保存的 CSV 文件路径
      plot_output (str): 绘图保存的图像文件路径
    """
    # 定义 reproduction_number 的取值范围，这里假设为 [1, 10]
    min_rep = 0
    max_rep = 10

    # 1. 使用拉丁超立方采样在1维空间中采样 num_samples 个点
    sampler = qmc.LatinHypercube(d=1)
    sample = sampler.random(n=num_samples)  # 样本在 [0,1] 区间内
    # 将 [0,1] 映射到 [min_rep, max_rep]
    reproduction_numbers_cont = sample * (max_rep - min_rep) + min_rep
    # 取整，并转换为一维数组
    reproduction_numbers = np.rint(reproduction_numbers_cont).astype(int).flatten()

    results = []
    # 2. 对每个 reproduction_number 样本调用 environment
    for rep in reproduction_numbers:
        # 调用 environment 模块，接口假定为：
        # environment(initial_agent_counts, trade_rules, round_number, reproduction_number, mistake_possibility, extrinsic_reward, DE_epoch_id, save_model)
        # 返回 (dummy_difficulty, skill_list)，这里选取 skill_list[-1] 作为最终 Individual income
        dummy_diff, skill_list = environment(initial_agent_counts, trade_rules, round_number, rep,
                                             mistake_possibility, extrinsic_reward, DE_epoch_id, save_model)
        measured_income = skill_list[-1]
        results.append([rep, measured_income])
        print(f"Reproduction number {rep}: Income = {measured_income}")

    results = np.array(results)  # shape: (num_samples, 2)

    # 3. 保存数据到 CSV 文件
    df_results = pd.DataFrame(results, columns=["reproduction_number", "income"])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_results.to_csv(output_csv, index=False)
    print(f"结果已保存到 {output_csv}")

    # 4. 绘制图像：X轴 reproduction_number, Y轴 income，并叠加趋势线
    plt.figure(figsize=(10, 6))
    plt.scatter(results[:, 0], results[:, 1], color='blue', alpha=0.7, label="Data Points")
    # 拟合线性回归趋势线
    coef = np.polyfit(results[:, 0], results[:, 1], 1)
    poly_fn = np.poly1d(coef)
    x_line = np.linspace(np.min(results[:, 0]), np.max(results[:, 0]), 100)
    plt.plot(x_line, poly_fn(x_line), color='red', linewidth=2, label="Trend Line")
    plt.xlabel("Reproduction Number", fontsize=14)
    plt.ylabel("Income", fontsize=14)
    plt.title("Income vs. Reproduction Number", fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_output), exist_ok=True)
    plt.savefig(plot_output, dpi=300)
    print(f"图像已保存至 {plot_output}")
    plt.show()









#########################################################


# Loss function
MSELoss = torch.nn.MSELoss()

# Initialize generator and discriminator
ruleDesigner = RuleDesigner()
evaluator = Evaluator()
environment = Environment()


if cuda:
    ruleDesigner.cuda()
    evaluator.cuda()
    environment.cuda()
    MSELoss.cuda()
    device = torch.device('cuda')

print(cuda)


# Optimizers
optimizer_Designer = torch.optim.Adam(ruleDesigner.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))   # betas=(opt.b1, opt.b2) ???
optimizer_Evaluator = torch.optim.Adam(evaluator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



save_model = False


reward_cooperation_list = []
reward_cheat_list = []

epoch_list = []

# 定义代理类型编号与名称的映射
agent_type_names = {
    'Random': 'Random',
    'Cheater': 'Cheater',
    'Cooperator': 'Cooperator',
    'Copycat': 'Copycat',
    'Grudger': 'Grudger',
    'Detective': 'Detective',
    'AI': opt.ai_type,  # 假设 opt.ai_type 为 'AI' 或实际的类型名称
    'Human': 'Human'
}

# 定义代理类型顺序
agent_types_order = ['Random', 'Cheater', 'Cooperator', 'Copycat', 'Grudger', 'Detective', 'AI', 'Human']



# -----------------
#  Train Designer
# -----------------
printSER = False
printGame = True


all_difficulty = []  # 每个元素对应一个 DesignerEpoch 的 difficulty（标量）
all_skill = []       # 每个元素为一个列表，长度为 opt.agent_train_epoch（例如6个skill值）


for DE_epoch_id in range(opt.DE_train_episode):

    if DE_epoch_id == opt.DE_train_episode - 1:
        save_model = True

    optimizer_Designer.zero_grad()
    # Sample noise as generator input
    noise_std = 0.005
    evaluation_requirement = torch.normal(
        mean=opt.difficulty,
        std=noise_std,
        size=(opt.batch_size, opt.evaluationSize),
        device=device
    )

    # Generate a batch of agents
    rule_vector = ruleDesigner(evaluation_requirement)      # inistates and reward
    loss_g = MSELoss(evaluator(rule_vector), evaluation_requirement)
    loss_g.backward()
    optimizer_Designer.step()

    optimizer_Evaluator.zero_grad()

    # start of one batch
    for i in range(opt.batch_size):
        #  Rule translation
        rules = rule_vector[i].detach().cpu().numpy()

        if opt.fixed_rule:
            initial_agent_counts = {
                'Random': opt.random_count,
                'Cheater': opt.cheater_count,
                'Cooperator': opt.cooperator_count,
                'Copycat': opt.copycat_count,
                'Grudger': opt.grudger_count,
                'Detective': opt.detective_count,
                opt.ai_type: opt.ai_count,
                'Human': opt.human_count
            }
            initial_agent_counts = np.array(list(initial_agent_counts.values()))
            trade_rules = opt.trade_rules
            round_number = opt.round_number
            reproduction_number = opt.reproduction_number
            mistake_possibility = opt.mistake_possibility
            extrinsic_reward = opt.extrinsic_reward

        else:
            # initial_agent_counts, trade_rules, round_number, reproduction_number, mistake_possibility = rule_translation(
            #     rules)
            initial_agent_counts = {
                'Random': opt.random_count,
                'Cheater': opt.cheater_count,
                'Cooperator': opt.cooperator_count,
                'Copycat': opt.copycat_count,
                'Grudger': opt.grudger_count,
                'Detective': opt.detective_count,
                opt.ai_type: opt.ai_count,
                'Human': opt.human_count
            }
            initial_agent_counts = np.array(list(initial_agent_counts.values()))
            trade_rules = opt.trade_rules
            round_number = opt.round_number
            reproduction_number = opt.reproduction_number
            mistake_possibility = opt.mistake_possibility
            extrinsic_reward = opt.extrinsic_reward

            extrinsic_reward = extrinsic_reward_translation(rules)



        difficulty = difficulty_translation(extrinsic_reward)

        if publish:
            # 构造更新数据字典
            update_data = {
                "epoch": DE_epoch_id,
                "initial_agent_counts": initial_agent_counts.tolist() if isinstance(initial_agent_counts, np.ndarray) else initial_agent_counts,
                "trade_rules": trade_rules.tolist() if isinstance(trade_rules, np.ndarray) else trade_rules,
                "round_number": round_number,
                "reproduction_number": reproduction_number,
                "mistake_possibility": mistake_possibility
            }
            # 发布更新数据
            publish_excel_update(update_data)

        # # extrinsic_reward = (rules - 0.5) * 10
        # extrinsic_reward = [0,0]

        # One Game
        # evaluation_e_temp = environment(initial_agent_counts, trade_rules, round_number, reproduction_number, mistake_possibility,
        #             extrinsic_reward, DE_epoch_id,
        #             save_model)
        difficulty, skill_list = environment(initial_agent_counts, trade_rules, round_number, reproduction_number, mistake_possibility,
                    extrinsic_reward, DE_epoch_id,
                    save_model)

        all_difficulty.append(difficulty)
        all_skill.append(skill_list)

        # save_test_results_to_excel(evaluation_e_temp, DE_epoch_id)
        #
        # evaluation_e_temp= np.expand_dims(evaluation_e_temp, axis=0)

        # # 确保 gini_coefficient_e 是数组（长度至少为1）
        # if i == 0:
        #     gini_coefficient_e = np.array([evaluation_e_temp[0]['gini_coefficient'][-1]])
        # else:
        #     gini_coefficient_e = np.append(gini_coefficient_e, evaluation_e_temp[0]['gini_coefficient'][-1])

        # difficulty
        if i == 0:
            difficulty_e = np.array([difficulty])
        else:
            difficulty_e = np.append(difficulty_e, difficulty)

        # end of one batch

        environment_evaluation_result = Variable(Tensor(difficulty_e), requires_grad=False)
        environment_evaluation_result.to(device)



    loss_d = MSELoss(evaluator(rule_vector.detach()), environment_evaluation_result)
    loss_d.backward()
    optimizer_Evaluator.step()

    average_extrinsic_reward = (rule_vector.mean(dim=0) - 0.5) * 10

    reward_cooperation_list.append(average_extrinsic_reward[0].cpu().item())
    reward_cheat_list.append(average_extrinsic_reward[1].cpu().item())
    epoch_list.append(DE_epoch_id + 1)

    # plot_skill_surfaces_at_epochs([0,5,10,15,20,25,30,35,40],
    #                             initial_agent_counts,
    #                             trade_rules,
    #                             round_number,
    #                             reproduction_number,
    #                             mistake_possibility,
    #                             DE_epoch_id,
    #                             save_model,
    #                             extrinsic_reward_range=(-opt.difficulty, opt.difficulty),
    #                             num_points=13)

    # mistake_difficulty(initial_agent_counts,
    #                         trade_rules,
    #                         round_number,
    #                         reproduction_number,
    #                         mistake_possibility,
    #                         DE_epoch_id,
    #                         save_model,
    #                         extrinsic_reward,
    #                         num_samples=10,
    #                         output_csv="./data/mistake_possibility_difficulty/mistake_possibility_difficulty.csv",
    #                         plot_output="./data/mistake_possibility_difficulty/mistake_possibility_tsne.png")

    # initial_agent_counts_difficulty(initial_agent_counts, trade_rules, round_number, reproduction_number, mistake_possibility,
    #                                 extrinsic_reward, DE_epoch_id, save_model,
    #                                 num_samples=10,
    #                                 output_csv="./data/initial_agent_counts_difficulty/initial_agent_counts_difficulty.csv",
    #                                 tsne_output="./data/initial_agent_counts_difficulty/initial_agent_counts_tsne.png",
    #                                 counts_dim=8,
    #                                 counts_range_min=0,
    #                                 counts_range_max=5)

    # plot_training_process_for_trade_rule_groups(initial_agent_counts, round_number, reproduction_number, mistake_possibility, extrinsic_reward, DE_epoch_id, save_model,
    #                                             num_epochs=40,
    #                                             output_csv="./data/trading_rule_difficulty/training_process_results.csv",
    #                                             output_plot="./data/trading_rule_difficulty/training_process.png")

    # round_number_difficulty(initial_agent_counts, trade_rules, reproduction_number, mistake_possibility,
    #                         extrinsic_reward, DE_epoch_id, save_model,
    #                         num_samples=10,
    #                         output_csv="./data/round_number_difficulty/round_number_difficulty.csv",
    #                         plot_output="./data/round_number_difficulty/round_number_tsne.png")

    # reproduction_number_difficulty(initial_agent_counts, trade_rules, round_number, mistake_possibility, extrinsic_reward,
    #                                DE_epoch_id, save_model,
    #                                num_samples=180,
    #                                output_csv="./data/reproduction_number_difficulty/reproduction_number_difficulty.csv",
    #                                plot_output="./data/reproduction_number_difficulty/reproduction_number_tsne.png")

    if save_model:
        PATH = "./designer/designer.pth"
        torch.save(ruleDesigner.state_dict(), PATH)
    if printSER:
        print('----- SER training, Epoch ID: ', DE_epoch_id, ' -----')
        print('  loss_g: ', loss_g,'  loss_d: ', loss_d)
        print('  Gini Coefficient: ', environment_evaluation_result)
        print('  Expectation: ', evaluation_requirement.squeeze())

plot_skill_vs_difficulty(all_difficulty, all_skill, opt.agent_train_epoch)
save_extrinsic_reward_results_to_csv('extrinsic_reward_A.csv', epoch_list, reward_cooperation_list)
save_extrinsic_reward_results_to_csv('extrinsic_reward_B.csv', epoch_list, reward_cheat_list)
print("=========done========")

# 在代码末尾添加
duration = 1000  # 持续时间，毫秒
freq = 440  # 频率，赫兹
winsound.Beep(freq, duration)



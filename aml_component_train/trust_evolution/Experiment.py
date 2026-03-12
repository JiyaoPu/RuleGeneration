from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from agent import Agent
from DQN_brain import DQNAgent
from mini_env import MiniEnv
from mpl_toolkits.mplot3d import Axes3D
from Q_brain import QLearningAgent
from scipy.ndimage import gaussian_filter
from scipy.stats import qmc
from sklearn.manifold import TSNE
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
import torchvision.transforms as transforms
from matplotlib.lines import Line2D


simplefilter(action="ignore", category=FutureWarning)
simplefilter(action="ignore", category=UserWarning)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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

parser.add_argument("--fixed_rule", type=str2bool, default="False", help="Use some fixed rule for agent training and testing")

# Rule for Game Flow
parser.add_argument("--extrinsic_reward", type=float, nargs=2, default=[0, 0], help="the reward to cooperation and cheat behaviour itself")

# Strategy
parser.add_argument("--humanPlayer", type=str2bool, default=False, help="True means there is human player")
parser.add_argument("--ai_type", type=str, default="Q", help="type of AI agent (e.g., 'Q', 'DQN')")

# Evaluation
parser.add_argument("--cooperationRate", type=float, default=1, help="cooperation rate")
parser.add_argument("--individualIncome", type=float, default=2, help="individual income")
parser.add_argument("--giniCoefficient", type=float, default=0.50, help="Gini Coefficient")
# Evaluation for Game Flow
parser.add_argument("--difficulty", type=float, default=0.01, help="difficulty")

# Designer and Evaluator
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.01, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--RuleDimension", type=int, default=3, help="trade rule dimension")
parser.add_argument("--DE_train_episode", type=int, default=5, help="number of episode during Q table training")
parser.add_argument("--DE_test_episode", type=int, default=1, help="number of episode during test")
parser.add_argument("--layersNum", type=int, default=1, help="layer number of the generator output")
parser.add_argument("--evaluationSize", type=int, default=1, help="size of the evaluation metrics")

# Agent training
parser.add_argument("--agent_train_epoch", type=int, default=10, help="number of epoch of training agent")
parser.add_argument("--gamma", type=float, default=0.99, help="discount factor for rewards")
parser.add_argument("--epsilon", type=float, default=1.0, help="initial epsilon for epsilon-greedy")
parser.add_argument("--epsilon_decay", type=float, default=0.999, help="epsilon decay rate")
parser.add_argument("--epsilon_min", type=float, default=0.1, help="minimum epsilon")
parser.add_argument("--memory_size", type=int, default=10000, help="memory size for experience replay")
parser.add_argument("--target_update", type=int, default=10, help="how often to update the target network")
parser.add_argument("--state_size", type=int, default=20, help="size of the state vector")

# Other
parser.add_argument(
    "--output_dir",
    type=str,
    default=os.getenv("AZUREML_OUTPUT_DIR", "outputs"),
    help="All artifacts will be written under this directory (Azure ML component output).",
)

actionlist = ["cheat", "cooperation"]

publish = False
printQtable = True
opt = parser.parse_args()

# === Unified output dirs (AML-friendly) ===
OUTPUT_DIR = Path(opt.output_dir).resolve()
DATA_DIR = OUTPUT_DIR / "data"
AGENTS_DIR = OUTPUT_DIR / "agents"
MODELS_DIR = OUTPUT_DIR / "models"

for d in [OUTPUT_DIR, DATA_DIR, AGENTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def artifact_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (OUTPUT_DIR / p)


def data_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (DATA_DIR / p)


# Redirect legacy env-based paths to OUTPUT_DIR/data by default
os.environ.setdefault("RULEGEN_EXCEL_PATH", str(data_path("dataupdates.xlsx")))
os.environ.setdefault("RULEGEN_TEST_RESULTS_PATH", str(data_path("test_results.xlsx")))
os.environ.setdefault("RULEGEN_QTABLE_PATH", str(data_path("q_table_heatmap.png")))
os.environ.setdefault("RULEGEN_SKILL_SURFACE_PATH", str(data_path("skill_extrinsic_reward/skill_surfaces.png")))
os.environ.setdefault("RULEGEN_SKILL_SURFACE_CSV_DIR", str(data_path("skill_extrinsic_reward")))
os.environ.setdefault("RULEGEN_ROUND_DIFF_CSV", str(data_path("round_number_difficulty/round_number_difficulty.csv")))
os.environ.setdefault("RULEGEN_ROUND_DIFF_PLOT", str(data_path("round_number_difficulty/round_number_tsne.png")))
os.environ.setdefault("RULEGEN_REPRO_DIFF_CSV", str(data_path("reproduction_number_difficulty/reproduction_number_difficulty.csv")))
os.environ.setdefault("RULEGEN_REPRO_DIFF_PLOT", str(data_path("reproduction_number_difficulty/reproduction_number_tsne.png")))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = device.type == "cuda"
print(f"[Device] {device} | cuda_available={torch.cuda.is_available()}")

# 全局变量
excel_initialized = False
first_save = False


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def to_serializable_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return x


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def save_metrics_json(run_id: str, status: str, epochs_payload: list[dict], metrics_path: Path):
    final_obj = {}
    if epochs_payload:
        last_epoch = epochs_payload[-1]
        coop = last_epoch.get("cooperation_rate", [])
        income = last_epoch.get("individual_income", [])
        gini = last_epoch.get("gini_coefficient", [])

        final_obj = {
            "epoch": last_epoch.get("epoch"),
            "cooperation_rate_overall": safe_float(coop[-1], 0.0) if coop else 0.0,
            "avg_income_overall": safe_float(income[-1], 0.0) if income else 0.0,
            "gini_overall": safe_float(gini[-1], 0.0) if gini else 0.0,
        }

    payload = {
        "run_id": run_id,
        "status": status,
        "last_updated": utc_now_iso(),
        "epochs": epochs_payload,
        "final": final_obj,
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[Artifact] saved metrics to {metrics_path}")


def publish_excel_update(update_data, excel_path: str | Path | None = None):
    global excel_initialized

    excel_path = Path(excel_path) if excel_path else Path(os.getenv("RULEGEN_EXCEL_PATH", str(data_path("dataupdates.xlsx"))))
    if not excel_path.is_absolute():
        excel_path = data_path(excel_path)

    excel_path.parent.mkdir(parents=True, exist_ok=True)

    if not excel_initialized:
        if excel_path.exists():
            excel_path.unlink()
        excel_initialized = True

    if excel_path.exists():
        df_existing = pd.read_excel(excel_path)
        df_new = pd.DataFrame([update_data])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = pd.DataFrame([update_data])

    df_combined.to_excel(excel_path, index=False)
    print("UPDATE_EXCEL:" + json.dumps({"excel_updated": True, "path": str(excel_path)}))
    sys.stdout.flush()


def plot_q_table(q_table, output_path="q_table_heatmap.png"):
    if "observation" not in q_table.columns:
        q_table = q_table.copy()
        q_table["observation"] = q_table.index.astype(str)

    df_sorted = q_table.sort_values(by="observation").copy()

    def get_prefix(obs):
        try:
            return obs.split("_")[0]
        except Exception:
            return None

    df_sorted["prefix"] = df_sorted["observation"].apply(get_prefix)

    personality_map = {
        "0": "Random",
        "1": "Cheater",
        "2": "Cooperator",
        "3": "Copycat",
        "4": "Grudger",
        "5": "Detective",
        "6": "AI",
        "7": "Human",
    }

    fig, axes = plt.subplots(nrows=8, figsize=(12, 4 * 8))

    for i in range(8):
        prefix_str = str(i)
        group = df_sorted[df_sorted["prefix"] == prefix_str]
        ax = axes[i]
        if group.empty:
            ax.set_title(f"{personality_map.get(prefix_str, prefix_str)}: No Data")
            ax.axis("off")
        else:
            combined_min = min(group["cheat"].min(), group["cooperation"].min())
            combined_max = max(group["cheat"].max(), group["cooperation"].max())
            group = group.copy()

            if combined_max == combined_min:
                group["cheat_norm"] = 0.0
                group["cooperation_norm"] = 0.0
            else:
                group["cheat_norm"] = (group["cheat"] - combined_min) / (combined_max - combined_min)
                group["cooperation_norm"] = (group["cooperation"] - combined_min) / (combined_max - combined_min)

            data = group[["cheat_norm", "cooperation_norm"]].T
            sns.heatmap(
                data,
                cmap="coolwarm",
                annot=False,
                xticklabels=group["observation"],
                yticklabels=["cheat", "cooperation"],
                ax=ax,
            )
            ax.set_title(f"{personality_map.get(prefix_str, prefix_str)} (n={len(group)})")
            ax.tick_params(axis="x", rotation=90, labelsize=8)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_test_results_to_excel(test_result, epoch, excel_path=None):
    global first_save

    excel_path = Path(excel_path) if excel_path else Path(os.getenv("RULEGEN_TEST_RESULTS_PATH", str(data_path("test_results.xlsx"))))
    if not excel_path.is_absolute():
        excel_path = data_path(excel_path)

    excel_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "epoch": epoch,
        "gini_coefficient": str(test_result["gini_coefficient"]),
        "cooperation_rate": str(test_result["cooperation_rate"]),
        "individual_income": str(test_result["individual_income"]),
    }
    df_new = pd.DataFrame([row])

    if not first_save:
        if excel_path.exists():
            try:
                excel_path.unlink()
                print(f"Old file deleted: {excel_path}")
            except Exception as e:
                print("Error occurred while deleting the old Excel file:", e)
        df_new.to_excel(excel_path, index=False)
        first_save = True
    else:
        try:
            if excel_path.exists():
                df_existing = pd.read_excel(excel_path)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_excel(excel_path, index=False)
            else:
                df_new.to_excel(excel_path, index=False)
        except Exception as e:
            print("Error occurred while saving Excel data:", e)
    print(f"Test results have been saved to {excel_path}")


class RuleDesigner(nn.Module):
    def __init__(self):
        super(RuleDesigner, self).__init__()

        def block(in_feat, out_feat, normalize=False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.evaluationSize, 8, normalize=False),
            *block(8, 16),
            nn.Linear(16, int(np.prod((opt.layersNum, opt.RuleDimension)))),
            nn.Softmax(dim=1),
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
        self.env = MiniEnv(trade_rules, round_number, reproduction_number, mistake_possibility, extrinsic_reward)
        self.agents = []

    def reset_shared_q_table(self):
        self.shared_q_table = pd.DataFrame(columns=self.shared_q_table.columns, dtype=np.float64)

        for agent in self.agents:
            if isinstance(agent, QLearningAgent):
                agent.q_table = self.shared_q_table

    def create_agents(self, agent_counts):
        if isinstance(agent_counts, np.ndarray):
            agent_counts = {
                "Random": int(agent_counts[0]),
                "Cheater": int(agent_counts[1]),
                "Cooperator": int(agent_counts[2]),
                "Copycat": int(agent_counts[3]),
                "Grudger": int(agent_counts[4]),
                "Detective": int(agent_counts[5]),
                "AI": int(agent_counts[6]),
                "Human": int(agent_counts[7]),
            }

        agents = []
        id_counter = 0

        for stereotype, count in agent_counts.items():
            for _ in range(count):
                if stereotype == "AI":
                    if opt.ai_type == "Q":
                        agent = QLearningAgent(
                            id=id_counter,
                            actions=actionlist,
                            shared_q_table=self.shared_q_table,
                            learning_rate=opt.lr,
                            reward_decay=opt.gamma,
                            e_greedy=opt.epsilon,
                        )
                    elif opt.ai_type == "DQN":
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
                            state_size=opt.state_size,
                        )
                    else:
                        agent = Agent(id=id_counter, stereotype=stereotype)
                else:
                    agent = Agent(id=id_counter, stereotype=stereotype)
                agents.append(agent)
                id_counter += 1
        return agents

    def trainAgent(
        self,
        initial_agent_counts,
        trade_rules,
        round_number,
        reproduction_number,
        mistake_possibility,
        extrinsic_reward,
        DesignerEpochID,
        save_model,
    ):
        difficulty = difficulty_translation(extrinsic_reward)

        if not self.agents:
            self.reset_shared_q_table()
            self.agents = self.create_agents(initial_agent_counts)

        record_data = True
        skill_list = []
        last_evaluation = None

        for epoch_id in range(opt.agent_train_epoch):
            if printGame:
                print(f"========== Epoch {epoch_id + 1} begin ==========")

            self.env.setup(self.agents, trade_rules, round_number, reproduction_number, mistake_possibility, extrinsic_reward)
            evaluation = self.env.oneCompetition(epoch_id, agent_type_names, agent_types_order, record_data, save_model, printGame)
            last_evaluation = evaluation
            skill_value = evaluation["individual_income"][6]
            skill_list.append(skill_value)

            if printGame:
                print(f"========== Epoch {epoch_id + 1} end ==========")

        save_dir = data_path(agent_type_names["AI"])
        save_dir.mkdir(parents=True, exist_ok=True)
        file_name = save_dir / f"{DesignerEpochID}.csv"
        self.env.save_results_to_csv(str(file_name))

        print("All games (epochs) have finished. Count the final types of agents and their quantities:")
        agent_type_counts = {}
        for agent in self.agents:
            agent_type = agent.stereotype
            agent_type_counts[agent_type] = agent_type_counts.get(agent_type, 0) + 1

        for agent_type in agent_types_order:
            count = agent_type_counts.get(agent_type, 0)
            agent_name = agent_type_names.get(agent_type, f"Type {agent_type}")
            print(f"{agent_name}: count={count}")

        return difficulty, skill_list, last_evaluation

    def testAgent(
        self,
        initial_agent_counts,
        trade_rules,
        round_number,
        reproduction_number,
        mistake_possibility,
        test_epoch_number=1,
    ):
        accum_gini = np.zeros(9)
        accum_coop = np.zeros(9)
        accum_income = np.zeros(9)

        for epoch_id in range(test_epoch_number):
            results = self.env.oneCompetition(1, agent_type_names, agent_types_order, False, False, printGame)
            accum_gini += np.array(results["gini_coefficient"])
            accum_coop += np.array(results["cooperation_rate"])
            accum_income += np.array(results["individual_income"])

        final_gini = (accum_gini / test_epoch_number).tolist()
        final_coop = (accum_coop / test_epoch_number).tolist()
        final_income = (accum_income / test_epoch_number).tolist()

        return {
            "gini_coefficient": final_gini,
            "cooperation_rate": final_coop,
            "individual_income": final_income,
        }

    def forward(
        self,
        initial_agent_counts,
        trade_rules,
        round_number,
        reproduction_number,
        mistake_possibility,
        extrinsic_reward,
        DesignerEpochID,
        save_model,
    ):
        self.agents = []
        self.env.setup(self.agents, trade_rules, round_number, reproduction_number, mistake_possibility, extrinsic_reward)

        difficulty, skill_list, last_evaluation = self.trainAgent(
            initial_agent_counts,
            trade_rules,
            round_number,
            reproduction_number,
            mistake_possibility,
            extrinsic_reward,
            DesignerEpochID,
            save_model,
        )

        if printQtable:
            for agent in self.agents:
                if isinstance(agent, QLearningAgent):
                    output_path = Path(os.getenv("RULEGEN_QTABLE_PATH", str(data_path("q_table_heatmap.png"))))
                    if not output_path.is_absolute():
                        output_path = data_path(output_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    plot_q_table(agent.q_table, output_path=str(output_path))
                    break

        return difficulty, skill_list, last_evaluation


def save_extrinsic_reward_results_to_csv(filename="extrinsic_reward.csv", epoch_list=None, extrinsic_reward_list=None):
    epoch_list = epoch_list or []
    extrinsic_reward_list = extrinsic_reward_list or []
    data = {
        "Epoch": epoch_list,
        "Extrinsic Reward": extrinsic_reward_list,
    }
    df = pd.DataFrame(data)
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"extrinsic_reward result saved at {filename}")


def plot_skill_vs_difficulty(all_difficulty, all_skill, num_epochs, output_path="skill_vs_difficulty.png"):
    plt.figure(figsize=(10, 6))

    for epoch_idx in range(0, num_epochs, 5):
        x = all_difficulty
        y = [skill_list[epoch_idx] for skill_list in all_skill]
        plt.plot(x, y, marker="o", label=f"Epoch {epoch_idx + 1}")

    plt.xlabel("Difficulty")
    plt.ylabel("Skill (AI Avg Income)")
    plt.title("Skill vs Difficulty")
    plt.legend()
    plt.grid(True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)

    if os.getenv("AZUREML_RUN_ID") is None:
        plt.show()
    plt.close()


def get_fixed_rules_vector(opt):
    return np.array(
        [
            opt.random_count,
            opt.cheater_count,
            opt.cooperator_count,
            opt.copycat_count,
            opt.grudger_count,
            opt.detective_count,
            opt.ai_count,
            opt.human_count,
            opt.trade_rules[0],
            opt.trade_rules[1],
            opt.trade_rules[2],
            opt.trade_rules[3],
            opt.trade_rules[4],
            opt.trade_rules[5],
            opt.round_number,
            opt.reproduction_number,
            opt.mistake_possibility,
        ]
    )


def rule_translation(rule_vector):
    total_agents = 25
    initial_counts_raw = rule_vector[:8]
    normalized_counts = initial_counts_raw / np.sum(initial_counts_raw)
    initial_counts = np.round(normalized_counts * total_agents).astype(int)

    current_total = np.sum(initial_counts)
    if current_total != total_agents:
        diff = total_agents - current_total
        while diff != 0:
            index = np.random.choice(8)
            if diff > 0:
                initial_counts[index] += 1
                diff -= 1
            elif diff < 0 and initial_counts[index] > 0:
                initial_counts[index] -= 1
                diff += 1

    payoff_raw = rule_vector[8:14]
    min_payoff = -5
    max_payoff = 5
    payoff_matrix = payoff_raw * (max_payoff - min_payoff) + min_payoff

    min_round = 5
    max_round = 20
    round_number = int(rule_vector[14] * (max_round - min_round) + min_round)

    min_rate = 1
    max_rate = 10
    reproduction_rate = int(rule_vector[15] * (max_rate - min_rate) + min_rate)

    max_probability = 0.5
    mistake_probability = rule_vector[16] * max_probability

    return initial_counts, payoff_matrix, round_number, reproduction_rate, mistake_probability


def extrinsic_reward_translation(rule_vector):
    extrinsic_reward_raw = rule_vector[:2]
    min_extrinsic_reward = -opt.difficulty
    max_extrinsic_reward = opt.difficulty
    extrinsic_reward = extrinsic_reward_raw * (max_extrinsic_reward - min_extrinsic_reward) + min_extrinsic_reward
    return extrinsic_reward


def difficulty_translation(extrinsic_reward, max_reward=opt.difficulty):
    extrinsic_reward = np.array(extrinsic_reward)
    avg_reward = np.mean(extrinsic_reward)
    difficulty = 1 - ((avg_reward + max_reward) / (2 * max_reward))
    return difficulty


def plot_skill_surfaces_at_epochs(
    desired_epochs,
    initial_agent_counts,
    trade_rules,
    round_number,
    reproduction_number,
    mistake_possibility,
    DE_epoch_id,
    save_model,
    extrinsic_reward_range=(-opt.difficulty, opt.difficulty),
    num_points=13,
    output_path=None,
    csv_output_dir=None,
):
    output_path = Path(output_path) if output_path else Path(os.getenv("RULEGEN_SKILL_SURFACE_PATH", str(data_path("skill_extrinsic_reward/skill_surfaces.png"))))
    csv_output_dir = Path(csv_output_dir) if csv_output_dir else Path(os.getenv("RULEGEN_SKILL_SURFACE_CSV_DIR", str(data_path("skill_extrinsic_reward"))))

    if not output_path.is_absolute():
        output_path = data_path(output_path)
    if not csv_output_dir.is_absolute():
        csv_output_dir = data_path(csv_output_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_output_dir.mkdir(parents=True, exist_ok=True)

    reward_values = np.linspace(extrinsic_reward_range[0], extrinsic_reward_range[1], num_points)
    X, Y = np.meshgrid(reward_values, reward_values)
    Z_dict = {ep: np.zeros_like(X) for ep in desired_epochs}

    for i in range(num_points):
        for j in range(num_points):
            extrinsic_reward = [X[i, j], Y[i, j]]
            difficulty, skill_list, _ = environment(
                initial_agent_counts,
                trade_rules,
                round_number,
                reproduction_number,
                mistake_possibility,
                extrinsic_reward,
                DE_epoch_id,
                save_model,
            )
            for ep in desired_epochs:
                Z_dict[ep][i, j] = skill_list[ep]

    for ep in desired_epochs:
        Z_dict[ep] = gaussian_filter(Z_dict[ep], sigma=1)

    for ep in desired_epochs:
        df = pd.DataFrame(
            {
                "CooperationExtrinsicReward": X.flatten(),
                "CheatExtrinsicReward": Y.flatten(),
                "Skill": Z_dict[ep].flatten(),
            }
        )
        csv_filename = csv_output_dir / f"epoch{ep}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"CSV file saved: {csv_filename}")

    color_list = ["red", "orange", "yellow", "green", "blue", "indigo", "violet", "brown", "pink"]
    colors = {ep: color_list[idx] for idx, ep in enumerate(desired_epochs)}

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    for ep in desired_epochs:
        ax.plot_surface(X, Y, Z_dict[ep], color=colors[ep], alpha=0.8, edgecolor="none")

    ax.set_xlabel("Cooperation Extrinsic Reward", fontsize=12, labelpad=10)
    ax.set_ylabel("Cheat Extrinsic Reward", fontsize=12, labelpad=10)
    ax.set_zlabel("Skill (Avg Income)", fontsize=12, labelpad=10)
    ax.set_title("Skill Surfaces for Epochs " + ", ".join(str(ep) for ep in desired_epochs), fontsize=16)
    ax.set_xlim(3, -3)

    legend_elements = [
        Line2D([0], [0], marker="s", color="w", label=f"Epoch {ep}", markerfacecolor=colors[ep], markersize=10)
        for ep in desired_epochs
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    if os.getenv("AZUREML_RUN_ID") is None:
        plt.show()
    plt.close()


def trading_rule_difficulty(
    initial_agent_counts,
    trade_rules,
    round_number,
    reproduction_number,
    mistake_possibility,
    DE_epoch_id,
    save_model,
    extrinsic_reward,
    num_samples=10,
    output_csv="trading_rule_difficulty/trading_rule_difficulty.csv",
    tsne_output="trading_rule_difficulty/trading_rule_tsne.png",
):
    output_csv_path = data_path(output_csv)
    tsne_output_path = data_path(tsne_output)

    sampler = qmc.LatinHypercube(d=6)
    sample = sampler.random(n=num_samples)
    trading_rules = sample * 6 - 3

    extrinsic_reward = [0, 0]
    results = []

    for rule in trading_rules:
        difficulty_dummy, skill_list, _ = environment(
            initial_agent_counts,
            rule,
            round_number,
            reproduction_number,
            mistake_possibility,
            extrinsic_reward,
            DE_epoch_id,
            save_model,
        )
        measured_income = skill_list[-1]
        results.append(np.concatenate([rule, [measured_income]]))

    results = np.array(results)

    col_names = [f"trading_rule_{i + 1}" for i in range(6)] + ["income"]
    df_results = pd.DataFrame(results, columns=col_names)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_csv_path, index=False)
    print(f"trading_rule result saved at {output_csv_path}")

    tsne_model = TSNE(n_components=2, perplexity=5, random_state=42)
    tsne_result = tsne_model.fit_transform(trading_rules)

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=results[:, -1], cmap="viridis", s=50)
    plt.xlabel("t-SNE 1", fontsize=14)
    plt.ylabel("t-SNE 2", fontsize=14)
    plt.title("t-SNE of 6D Trading Rule Samples (colored by Income)", fontsize=16)
    plt.colorbar(sc, label="Income")
    plt.tight_layout()

    tsne_output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(tsne_output_path, dpi=300)
    print(f"t-SNE saved at {tsne_output_path}")

    if os.getenv("AZUREML_RUN_ID") is None:
        plt.show()
    plt.close()


def mistake_difficulty(
    initial_agent_counts,
    trade_rules,
    round_number,
    reproduction_number,
    mistake_possibility,
    DE_epoch_id,
    save_model,
    extrinsic_reward,
    num_samples=10,
    output_csv="mistake_possibility_difficulty/mistake_possibility_difficulty.csv",
    plot_output="mistake_possibility_difficulty/mistake_possibility_tsne.png",
):
    output_csv_path = data_path(output_csv)
    plot_output_path = data_path(plot_output)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    plot_output_path.parent.mkdir(parents=True, exist_ok=True)

    sampler = qmc.LatinHypercube(d=1)
    sample = sampler.random(n=num_samples)
    mistake_possibility_samples = sample * 0.5

    extrinsic_reward = [0, 0]
    results = []

    for mp in mistake_possibility_samples:
        difficulty_dummy, skill_list, _ = environment(
            initial_agent_counts,
            trade_rules,
            round_number,
            reproduction_number,
            mp,
            extrinsic_reward,
            DE_epoch_id,
            save_model,
        )
        measured_income = skill_list[-1]
        results.append(np.concatenate([mp, [measured_income]]))

    results = np.array(results)

    col_names = ["mistake_possibility", "income"]
    df_results = pd.DataFrame(results, columns=col_names)

    df_results.to_csv(output_csv_path, index=False)
    print(f"Results saved at {output_csv_path}")

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(results[:, 0], results[:, 1], c=results[:, 1], cmap="viridis", s=50)
    plt.xlabel("Mistake Possibility", fontsize=14)
    plt.ylabel("Income", fontsize=14)
    plt.title("Income vs. Mistake Possibility", fontsize=16)
    cbar = plt.colorbar(sc)
    cbar.set_label("Income", fontsize=14)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(plot_output_path, dpi=300)
    print(f"Plot saved at {plot_output_path}")

    if os.getenv("AZUREML_RUN_ID") is None:
        plt.show()
    plt.close()


def initial_agent_counts_difficulty(
    initial_agent_counts,
    trade_rules,
    round_number,
    reproduction_number,
    mistake_possibility,
    extrinsic_reward,
    DE_epoch_id,
    save_model,
    num_samples=180,
    output_csv="initial_agent_counts_difficulty/initial_agent_counts_difficulty.csv",
    tsne_output="initial_agent_counts_difficulty/initial_agent_counts_tsne.png",
    counts_dim=8,
    counts_range_min=0,
    counts_range_max=50,
):
    output_csv_path = data_path(output_csv)
    tsne_output_path = data_path(tsne_output)

    sampler = qmc.LatinHypercube(d=counts_dim)
    sample = sampler.random(n=num_samples)
    initial_counts_samples = qmc.scale(sample, counts_range_min, counts_range_max)
    initial_counts_samples = np.rint(initial_counts_samples).astype(int)

    results = []

    for counts in initial_counts_samples:
        dummy_diff, skill_list, _ = environment(
            counts,
            trade_rules,
            round_number,
            reproduction_number,
            mistake_possibility,
            extrinsic_reward,
            DE_epoch_id,
            save_model,
        )
        measured_income = skill_list[-1]
        results.append(np.concatenate([counts, [measured_income]]))

    results = np.array(results)

    col_names = [f"agent_count_{i + 1}" for i in range(counts_dim)] + ["income"]
    df_results = pd.DataFrame(results, columns=col_names)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_csv_path, index=False)
    print(f"Initial_agent_counts result saved at {output_csv_path}")

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

    tsne_output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(tsne_output_path, dpi=300)
    print(f"t-SNE saved at {tsne_output_path}")

    if os.getenv("AZUREML_RUN_ID") is None:
        plt.show()
    plt.close()


def plot_training_process_for_trade_rule_groups(
    initial_agent_counts,
    round_number,
    reproduction_number,
    mistake_possibility,
    extrinsic_reward,
    DE_epoch_id,
    save_model,
    num_epochs,
    output_csv="training_process/training_process_results_high.csv",
    output_plot="training_process/training_process.png",
):
    output_csv_path = data_path(output_csv)
    output_plot_path = data_path(output_plot)

    trade_rule_groups = {
        "Low": np.array([-1.177282578, -1.482345946, -1.087466925, -2.907925356, -1.252640696, -0.788812408]),
        "Medium": np.array([-0.41097735, 2.361120932, -0.386333482, -1.977072218, 1.475039731, 0.702690951]),
        "High": np.array([-0.230105264, -2.389890292, -1.183604627, 2.589601439, 2.999660671, 2.272005864]),
    }

    records = []

    for group_name, trading_rule in trade_rule_groups.items():
        print(f"Testing {group_name} rules: {trading_rule}")

        dummy_diff, skill_list, _ = environment(
            initial_agent_counts,
            trading_rule,
            round_number,
            reproduction_number,
            mistake_possibility,
            extrinsic_reward,
            DE_epoch_id,
            save_model,
        )

        income_over_epochs = skill_list
        if len(income_over_epochs) != num_epochs:
            print(f"Warning: Group {group_name} returned epoch id ({len(income_over_epochs)}) different from expect epoch {num_epochs}")

        for epoch in range(len(income_over_epochs)):
            records.append(
                {
                    "group": group_name,
                    "epoch": epoch,
                    "income": income_over_epochs[epoch],
                }
            )

    df_records = pd.DataFrame(records)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_records.to_csv(output_csv_path, index=False)
    print(f"Training data saved at {output_csv_path}")

    plt.figure(figsize=(10, 6))
    groups = df_records["group"].unique()
    color_map = {"Low": "red", "Medium": "blue", "High": "green"}

    for group in groups:
        group_data = df_records[df_records["group"] == group].sort_values("epoch")
        plt.plot(
            group_data["epoch"],
            group_data["income"],
            marker="o",
            linewidth=2,
            color=color_map.get(group, None),
            label=group,
        )

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Individual Income", fontsize=14)
    plt.title("Training Process: Income vs. Epoch for Different Trading Rule Groups", fontsize=16)
    plt.legend(title="Trading Rule Group")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path, dpi=300)
    print(f"Training progress saved at {output_plot_path}")

    if os.getenv("AZUREML_RUN_ID") is None:
        plt.show()
    plt.close()


def round_number_difficulty(
    initial_agent_counts,
    trade_rules,
    reproduction_number,
    mistake_possibility,
    extrinsic_reward,
    DE_epoch_id,
    save_model,
    num_samples=180,
    output_csv=None,
    plot_output=None,
):
    output_csv_path = Path(output_csv) if output_csv else Path(os.getenv("RULEGEN_ROUND_DIFF_CSV", str(data_path("round_number_difficulty/round_number_difficulty.csv"))))
    plot_output_path = Path(plot_output) if plot_output else Path(os.getenv("RULEGEN_ROUND_DIFF_PLOT", str(data_path("round_number_difficulty/round_number_tsne.png"))))

    if not output_csv_path.is_absolute():
        output_csv_path = data_path(output_csv_path)
    if not plot_output_path.is_absolute():
        plot_output_path = data_path(plot_output_path)

    sampler = qmc.LatinHypercube(d=1)
    sample = sampler.random(n=num_samples)
    round_numbers_cont = sample * 4 + 1
    round_numbers = np.rint(round_numbers_cont).astype(int).flatten()
    extrinsic_reward = [0, 0]

    results = []
    for rn in round_numbers:
        dummy_diff, skill_list, _ = environment(
            initial_agent_counts,
            trade_rules,
            rn,
            reproduction_number,
            mistake_possibility,
            extrinsic_reward,
            DE_epoch_id,
            save_model,
        )
        measured_income = skill_list[-1]
        results.append([rn, measured_income])
        print(f"Round number {rn}: Income = {measured_income}")

    results = np.array(results)

    df_results = pd.DataFrame(results, columns=["round_number", "income"])
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_csv_path, index=False)
    print(f"Difficulty result saved at {output_csv_path}")

    plt.figure(figsize=(10, 6))
    plt.scatter(results[:, 0], results[:, 1], color="blue", alpha=0.7, label="Data Points")
    coef = np.polyfit(results[:, 0], results[:, 1], 1)
    poly_fn = np.poly1d(coef)
    x_line = np.linspace(np.min(results[:, 0]), np.max(results[:, 0]), 100)
    plt.plot(x_line, poly_fn(x_line), color="red", linewidth=2, label="Trend Line")
    plt.xlabel("Round Number", fontsize=14)
    plt.ylabel("Income", fontsize=14)
    plt.title("Income vs. Round Number", fontsize=16)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plot_output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_output_path, dpi=300)
    print(f"Plot saved at {plot_output_path}")

    if os.getenv("AZUREML_RUN_ID") is None:
        plt.show()
    plt.close()


def reproduction_number_difficulty(
    initial_agent_counts,
    trade_rules,
    round_number,
    mistake_possibility,
    extrinsic_reward,
    DE_epoch_id,
    save_model,
    num_samples=10,
    output_csv=None,
    plot_output=None,
):
    output_csv_path = Path(output_csv) if output_csv else Path(os.getenv("RULEGEN_REPRO_DIFF_CSV", str(data_path("reproduction_number_difficulty/reproduction_number_difficulty.csv"))))
    plot_output_path = Path(plot_output) if plot_output else Path(os.getenv("RULEGEN_REPRO_DIFF_PLOT", str(data_path("reproduction_number_difficulty/reproduction_number_tsne.png"))))

    if not output_csv_path.is_absolute():
        output_csv_path = data_path(output_csv_path)
    if not plot_output_path.is_absolute():
        plot_output_path = data_path(plot_output_path)

    min_rep = 0
    max_rep = 10

    sampler = qmc.LatinHypercube(d=1)
    sample = sampler.random(n=num_samples)
    reproduction_numbers_cont = sample * (max_rep - min_rep) + min_rep
    reproduction_numbers = np.rint(reproduction_numbers_cont).astype(int).flatten()

    results = []
    for rep in reproduction_numbers:
        dummy_diff, skill_list, _ = environment(
            initial_agent_counts,
            trade_rules,
            round_number,
            rep,
            mistake_possibility,
            extrinsic_reward,
            DE_epoch_id,
            save_model,
        )
        measured_income = skill_list[-1]
        results.append([rep, measured_income])
        print(f"Reproduction number {rep}: Income = {measured_income}")

    results = np.array(results)

    df_results = pd.DataFrame(results, columns=["reproduction_number", "income"])
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_csv_path, index=False)
    print(f"Reproduction result saved at {output_csv_path}")

    plt.figure(figsize=(10, 6))
    plt.scatter(results[:, 0], results[:, 1], color="blue", alpha=0.7, label="Data Points")
    coef = np.polyfit(results[:, 0], results[:, 1], 1)
    poly_fn = np.poly1d(coef)
    x_line = np.linspace(np.min(results[:, 0]), np.max(results[:, 0]), 100)
    plt.plot(x_line, poly_fn(x_line), color="red", linewidth=2, label="Trend Line")
    plt.xlabel("Reproduction Number", fontsize=14)
    plt.ylabel("Income", fontsize=14)
    plt.title("Income vs. Reproduction Number", fontsize=16)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plot_output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_output_path, dpi=300)
    print(f"Reproduction image saved at {plot_output_path}")

    if os.getenv("AZUREML_RUN_ID") is None:
        plt.show()
    plt.close()


# =========================================================
# Init
# =========================================================

MSELoss = torch.nn.MSELoss()

ruleDesigner = RuleDesigner()
evaluator = Evaluator()
environment = Environment()

if cuda:
    ruleDesigner.cuda()
    evaluator.cuda()
    environment.cuda()
    MSELoss.cuda()

print(cuda)

optimizer_Designer = torch.optim.Adam(ruleDesigner.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_Evaluator = torch.optim.Adam(evaluator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

save_model = False

reward_cooperation_list = []
reward_cheat_list = []
epoch_list = []

agent_type_names = {
    "Random": "Random",
    "Cheater": "Cheater",
    "Cooperator": "Cooperator",
    "Copycat": "Copycat",
    "Grudger": "Grudger",
    "Detective": "Detective",
    "AI": opt.ai_type,
    "Human": "Human",
}

agent_types_order = ["Random", "Cheater", "Cooperator", "Copycat", "Grudger", "Detective", "AI", "Human"]

printSER = False
printGame = True

all_difficulty = []
all_skill = []
metrics_epochs = []

run_id = os.getenv("AZUREML_RUN_ID", f"manual_{datetime.now().strftime('%Y%m%d%H%M%S')}")
metrics_json_path = data_path("metrics.json")

# metrics.json
save_metrics_json(run_id=run_id, status="running", epochs_payload=metrics_epochs, metrics_path=metrics_json_path)

for DE_epoch_id in range(opt.DE_train_episode):
    if DE_epoch_id == opt.DE_train_episode - 1:
        save_model = True

    optimizer_Designer.zero_grad()

    noise_std = 0.005
    evaluation_requirement = torch.normal(
        mean=opt.difficulty,
        std=noise_std,
        size=(opt.batch_size, opt.evaluationSize),
        device=device,
    )

    rule_vector = ruleDesigner(evaluation_requirement)
    loss_g = MSELoss(evaluator(rule_vector), evaluation_requirement)
    loss_g.backward()
    optimizer_Designer.step()

    optimizer_Evaluator.zero_grad()

    last_epoch_payload = None

    for i in range(opt.batch_size):
        rules = rule_vector[i].detach().cpu().numpy()

        if opt.fixed_rule:
            initial_agent_counts = {
                "Random": opt.random_count,
                "Cheater": opt.cheater_count,
                "Cooperator": opt.cooperator_count,
                "Copycat": opt.copycat_count,
                "Grudger": opt.grudger_count,
                "Detective": opt.detective_count,
                opt.ai_type: opt.ai_count,
                "Human": opt.human_count,
            }
            initial_agent_counts = np.array(list(initial_agent_counts.values()))
            trade_rules = opt.trade_rules
            round_number = opt.round_number
            reproduction_number = opt.reproduction_number
            mistake_possibility = opt.mistake_possibility
            extrinsic_reward = opt.extrinsic_reward
        else:
            initial_agent_counts = {
                "Random": opt.random_count,
                "Cheater": opt.cheater_count,
                "Cooperator": opt.cooperator_count,
                "Copycat": opt.copycat_count,
                "Grudger": opt.grudger_count,
                "Detective": opt.detective_count,
                opt.ai_type: opt.ai_count,
                "Human": opt.human_count,
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
            update_data = {
                "epoch": DE_epoch_id,
                "initial_agent_counts": initial_agent_counts.tolist() if isinstance(initial_agent_counts, np.ndarray) else initial_agent_counts,
                "trade_rules": trade_rules.tolist() if isinstance(trade_rules, np.ndarray) else trade_rules,
                "round_number": round_number,
                "reproduction_number": reproduction_number,
                "mistake_possibility": mistake_possibility,
            }
            publish_excel_update(update_data)

        difficulty, skill_list, final_evaluation = environment(
            initial_agent_counts,
            trade_rules,
            round_number,
            reproduction_number,
            mistake_possibility,
            extrinsic_reward,
            DE_epoch_id,
            save_model,
        )

        all_difficulty.append(difficulty)
        all_skill.append(skill_list)

        if i == 0:
            difficulty_e = np.array([difficulty])
        else:
            difficulty_e = np.append(difficulty_e, difficulty)

        if final_evaluation is None:
            final_evaluation = {
                "cooperation_rate": [],
                "individual_income": [],
                "gini_coefficient": [],
            }

        last_epoch_payload = {
            "epoch": DE_epoch_id + 1,
            "initial_agent_counts": to_serializable_list(initial_agent_counts),
            "trade_rules": to_serializable_list(trade_rules),
            "round_number": int(round_number),
            "reproduction_number": int(reproduction_number),
            "mistake_possibility": float(mistake_possibility),
            "cooperation_rate": to_serializable_list(final_evaluation.get("cooperation_rate", [])),
            "individual_income": to_serializable_list(final_evaluation.get("individual_income", [])),
            "gini_coefficient": to_serializable_list(final_evaluation.get("gini_coefficient", [])),
        }

        environment_evaluation_result = Variable(Tensor(difficulty_e), requires_grad=False)
        environment_evaluation_result.to(device)

    loss_d = MSELoss(evaluator(rule_vector.detach()), environment_evaluation_result)
    loss_d.backward()
    optimizer_Evaluator.step()

    average_extrinsic_reward = (rule_vector.mean(dim=0) - 0.5) * 10

    reward_cooperation_list.append(average_extrinsic_reward[0].cpu().item())
    reward_cheat_list.append(average_extrinsic_reward[1].cpu().item())
    epoch_list.append(DE_epoch_id + 1)

    if last_epoch_payload is not None:
        metrics_epochs.append(last_epoch_payload)
        save_metrics_json(
            run_id=run_id,
            status="running" if DE_epoch_id < opt.DE_train_episode - 1 else "succeeded",
            epochs_payload=metrics_epochs,
            metrics_path=metrics_json_path,
        )

    if save_model:
        model_path = MODELS_DIR / "designer.pth"
        torch.save(ruleDesigner.state_dict(), model_path)
        print(f"[Artifact] saved designer to {model_path}")

    if printSER:
        print("----- SER training, Epoch ID: ", DE_epoch_id, " -----")
        print("  loss_g: ", loss_g, "  loss_d: ", loss_d)
        print("  Gini Coefficient: ", environment_evaluation_result)
        print("  Expectation: ", evaluation_requirement.squeeze())

plot_skill_vs_difficulty(
    all_difficulty,
    all_skill,
    opt.agent_train_epoch,
    output_path=str(data_path("skill_vs_difficulty.png")),
)

save_extrinsic_reward_results_to_csv(str(data_path("extrinsic_reward_A.csv")), epoch_list, reward_cooperation_list)
save_extrinsic_reward_results_to_csv(str(data_path("extrinsic_reward_B.csv")), epoch_list, reward_cheat_list)

# succeeded
save_metrics_json(
    run_id=run_id,
    status="succeeded",
    epochs_payload=metrics_epochs,
    metrics_path=metrics_json_path,
)

print("=========done========")
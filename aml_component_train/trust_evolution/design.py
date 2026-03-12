from __future__ import annotations

import random
from pathlib import Path
import os

import numpy as np
import torch
import UnityEngine as ue
from torch.autograd import Variable

from Q_brain import QLearningAgent
from mini_env import MiniEnv
from Experiment import RuleDesigner
from Experiment import Environment


# =========================================================
# Unified output path helpers
# All generated artifacts default to:
#   <AZUREML_OUTPUT_DIR>/...
# =========================================================

OUTPUT_DIR = Path(os.getenv("AZUREML_OUTPUT_DIR", "outputs")).resolve()
DATA_DIR = OUTPUT_DIR / "data"
MODELS_DIR = OUTPUT_DIR / "models"

for d in [OUTPUT_DIR, DATA_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def artifact_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (OUTPUT_DIR / p)


def data_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (DATA_DIR / p)


def model_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (MODELS_DIR / p)


def OneGame(AgentNum, TradeRules):
    """
    Run one game and return the trade list and action list.
    """
    env = MiniEnv(AgentNum, TradeRules)
    env.setup(AgentNum, TradeRules)
    TradeList = env.tradeList
    ActionList = []

    AgentsDict = {}
    for i in range(AgentNum):
        agent_id = manager.roles[i].GetComponent("Role").id
        agent_temp = QLearningAgent(actions=actionlist)
        agent_temp.set_q_table_by_id(agent_id)
        AgentsDict.update({agent_id: agent_temp})

    for tradeID in range(len(env.tradeList)):
        ID_A, ID_B = env.tradeList[tradeID]
        action_A, action_B = OneTrade(env, AgentsDict, ID_A, ID_B)
        ActionList.append([action_A, action_B])

    return TradeList, ActionList


def OneTrade(env, AgentsDict, ID_A, ID_B):
    """
    Execute one trade step between two agents.
    """
    state_A = env.agents[ID_A].money
    state_B = env.agents[ID_B].money

    action_A = AgentsDict[ID_A].choose_action(str(state_A))
    action_B = AgentsDict[ID_B].choose_action(str(state_B))

    # Agent takes action and gets next state and reward
    state_A_, state_B_, reward_A, reward_B, done_A, done_B = env.trade(ID_A, ID_B, action_A, action_B)

    return action_A, action_B


def AITrade():
    """
    Let AI agents perform trades in the Unity environment.
    """
    TradeList = []
    ActionList = []
    TradeRules = [0, 0, 3, -1, 2, 2]

    AgentNum = manager.GetPlayerNum()
    TradeList, ActionList = OneGame(AgentNum, TradeRules)

    for i in range(len(TradeList)):
        [id_A, id_B] = TradeList[i]
        [action_A, action_B] = ActionList[i]
        PythonManager.addTrade(int(id_A), int(id_B), str(action_A), str(action_B))

    manager.SetTradePoints()
    manager.TradeArrangement()


def AIDesigner():
    """
    Generate environment and gameplay rules from the trained designer model,
    then push them into the Unity environment.
    """
    cooperationRate = 10

    Tensor = torch.FloatTensor
    cooperationRate = Variable(
        Tensor(np.random.normal(cooperationRate, 0, (1, 1))),
        requires_grad=False,
    )

    generator = RuleDesigner()

    # Load the trained designer model from the unified model directory
    path = model_path("designer.pth")
    generator.load_state_dict(torch.load(str(path), map_location="cpu"))
    generator.eval()

    sr = generator(cooperationRate)
    rules = sr[0]
    rules = rules.detach().cpu().numpy()
    rules = list(rules)

    ue.Debug.Log(str(len(rules)))

    mapSize = [int(rules[0] * 18 + 6), int(rules[1] * 18 + 6)]
    forestSize = mapSize + [2, 2]
    agentNum = int(rules[2] * 8 + 3)

    canChopWood = bool(rules[3] > 0.5)

    initiallocation = []
    initialCoin = []
    Personality = []
    speed = []

    for a in range(agentNum):
        initiallocation.append(randomLocation(mapSize))
        initialCoin.append(rules[4] * 20)
        Personality.append(int(rules[5] * 7))
        speed.append(rules[6] * 10)

    canDestroy = bool(rules[7] > 0.5)
    DestroyDeadline = rules[8] * 10 - 5
    tradeRules = [
        rules[9] * 20 - 10,
        rules[10] * 20 - 10,
        rules[11] * 20 - 10,
        rules[12] * 20 - 10,
        rules[13] * 20 - 10,
        rules[14] * 20 - 10,
    ]

    # ======================== Unity ========================
    EnviromentManager.forestSize[0] = int(forestSize[0])
    EnviromentManager.forestSize[1] = int(forestSize[1])
    EnviromentManager.mapSize[0] = int(mapSize[0])
    EnviromentManager.mapSize[1] = int(mapSize[1])
    manager.canChopWood = bool(canChopWood)

    EnviromentManager.agentNum = int(agentNum)
    for i in range(agentNum):
        EnviromentManager.addAgentInformation(
            float(initiallocation[i][0]),
            float(initiallocation[i][1]),
            int(Personality[i]),
            float(initialCoin[i]),
            float(speed[i]),
        )

    for t in range(len(tradeRules)):
        manager.trade_rules[t] = float(tradeRules[t])

    manager.canDestroy = bool(canDestroy)
    manager.DestroyDeadline = float(DestroyDeadline)


def randomLocation(mapSize):
    """
    Generate a random location inside the map bounds.
    """
    return [
        random.uniform(-abs(mapSize[0]) / 2 - 1, abs(mapSize[0]) / 2 - 1),
        random.uniform(-abs(mapSize[1]) / 2 - 1, abs(mapSize[1]) / 2 - 1),
    ]


# =========================================================
# Unity bindings
# =========================================================

actionlist = ["cheat", "cooperation"]
manager = ue.GameObject.Find("GameManager").GetComponent("GameManager")
PythonManager = ue.GameObject.Find("PythonManager").GetComponent("PythonManager")
EnviromentManager = ue.GameObject.Find("EnvironmentBuilder").GetComponent("EnvironmentBuilder")

AIDesigner()
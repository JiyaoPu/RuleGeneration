import UnityEngine as ue
from Q_brain import QLearningTable
import torch
from mini_env import Game

def OneGame(AgentNum,TradeRules):
    env = Game(AgentNum,TradeRules)
    env.setup(AgentNum,TradeRules)
    TradeList = env.tradeList
    ActionList = []

    AgentsDict = {}
    for i in range(AgentNum):  # create agents
        id = manager.roles[i].GetComponent("Role").id
        agent_temp = QLearningTable(actions=actionlist)
        agent_temp.set_q_table_by_id(id)
        AgentsDict.update({id: agent_temp})

    for tradeID in range(len(env.tradeList)):
        ID_A, ID_B = env.tradeList[tradeID]
        action_A, action_B = OneTrade(env, AgentsDict, ID_A, ID_B)
        ActionList.append([action_A, action_B])

    return TradeList, ActionList

def OneTrade(env, AgentsDict, ID_A, ID_B):

    state_A = env.agents[ID_A].money
    state_B = env.agents[ID_B].money

    action_A = AgentsDict[ID_A].choose_action(str(state_A))
    action_B = AgentsDict[ID_B].choose_action(str(state_B))

    # agent take action and get next state and reward
    state_A_,state_B_,reward_A,reward_B,done_A,done_B = env.trade(ID_A,ID_B,action_A, action_B)

    return action_A,action_B


def AITrade():
    TradeList = []          # [id_A, id_B]
    ActionList = []      # [action_A, action_B]
    TradeRules = [0,0,3,-1,2,2 ]

    AgentNum = manager.GetPlayerNum()
    # ue.Debug.Log('-----')
    TradeList, ActionList = OneGame(AgentNum,TradeRules)
    for i in range(len(TradeList)):
        [id_A,id_B] = TradeList[i]
        [action_A,action_B] = ActionList[i]
        PythonManager.addTrade(int(id_A),int(id_B),str(action_A),str(action_B))

    manager.SetTradePoints()
    manager.TradeArrangement()

def AIDesigner():
    mapSize = [6, 6]
    canChopWood = False
    forestSize = [8, 8]
    agentNum = 3
    initiallocation = [[1, 0.3],[-0.5, -1],[-2, 2, 1]]

    Personality =  [4, 1, 2]
    initialCoin = [5, 2, 3]
    speed = [6, 3, 4]
    tradeRules = [-1, -1, 3, -2, 1, 1]
    canDestroy = False
    DestroyDeadline = 2

    EnviromentManager.forestSize[0] = int(forestSize[0])
    EnviromentManager.forestSize[1] = int(forestSize[1])
    EnviromentManager.mapSize[0] = int(mapSize[0])
    EnviromentManager.mapSize[1] = int(mapSize[1])
    manager.canChopWood = bool(canChopWood)

    EnviromentManager.agentNum = int(agentNum)
    for i in range(agentNum):
        EnviromentManager.addAgentInformation(float(initiallocation[i][0]), float(initiallocation[i][1]),int(Personality[i]),int(initialCoin[i]), float(speed[i]))

    for i in range(len(tradeRules)):
        manager.trade_rules[i] = int(tradeRules[i])

    manager.canDestroy = bool(canDestroy)
    manager.DestroyDeadline = int(DestroyDeadline)


actionlist = ["cheat", "cooperation"]
manager = ue.GameObject.Find("GameManager").GetComponent("GameManager")
PythonManager = ue.GameObject.Find("PythonManager").GetComponent("PythonManager")
EnviromentManager = ue.GameObject.Find("EnvironmentBuilder").GetComponent("EnvironmentBuilder")
AITrade()
# AIDesigner()

import UnityEngine as ue
from Q_brain import QLearningTable
import torch
from mini_env import Game
from main import Generator
from torch.autograd import Variable
import numpy as np
import random
from main import Environment



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

def AIDesigner():
    # rules = []  # [mapsizex, mapsizey, agentNum, canChopWood, initialCoin, Personality, speed, canDestroy, DestroyDeadline, tradeRules0, tradeRules1, tradeRules2, tradeRules3, tradeRules4, tradeRules5]  0~1  length = 15

    cooperationRate = 10

    Tensor = torch.FloatTensor
    cooperationRate = Variable(Tensor(np.random.normal(cooperationRate, 0, (1, 1))),
                               requires_grad=False)  # here we use success rate as the measurement.

    generator = Generator()
    PATH = "D:/OneDrive - Durham University/1.Durham University/7.Automated game design/code/Eden/Maze/variableMaze_Vladimir/designer/designer.pth"
    generator.load_state_dict(torch.load(PATH))
    generator.eval()
    sr = generator(cooperationRate)
    rules = sr[0]
    rules = rules.detach().cpu().numpy()
    rules = list(rules)
    # tradeRules = [-1, -1, 3, -2, 1, 1]
    ue.Debug.Log(str(len(rules)))

    mapSize = [int(rules[0] * 18 +6), int(rules[1] * 18 +6)]    # (2,20)
    forestSize = mapSize + [2, 2]

    agentNum = int(rules[2] * 8 + 3)


    if(rules[3]>0.5):
        canChopWood = True
    else:
        canChopWood = False
    # mapSize = [8, 14]
    # agentNum = 3
    # initiallocation = [[3, 0.3],[-2.5, -1],[-2, 3]]
    # Personality = [1, 2, 3]
    # initialCoin = [5, 5, 5]
    # speed = [3, 3, 3]
    initiallocation = []
    initialCoin = []
    Personality = []
    speed = []
    for a in range(agentNum):
        initiallocation.append(randomLocation(mapSize))
        initialCoin.append(rules[4]*20)     # (0,20)
        Personality.append(int(rules[5]*7))     # (0,7)
        speed.append(rules[6]*10)       # (0,10)
    # tradeRules = [-1, -1, 3, -2, 1, 1]
    if (rules[7] > 0.5):
        canDestroy = True
    else:
        canDestroy = False
    # canDestroy = False
    DestroyDeadline = rules[8]*10 - 5
    tradeRules = [rules[9]*20-10,rules[10]*20-10,rules[11]*20-10,rules[12]*20-10,rules[13]*20-10,rules[14]*20-10]  # (-10,10)
    # tradeRules = rules[-6:]*20-10         # (-10,10)

    ######################## Unity ########################
    EnviromentManager.forestSize[0] = int(forestSize[0])
    EnviromentManager.forestSize[1] = int(forestSize[1])
    EnviromentManager.mapSize[0] = int(mapSize[0])
    EnviromentManager.mapSize[1] = int(mapSize[1])
    manager.canChopWood = bool(canChopWood)

    EnviromentManager.agentNum = int(agentNum)
    for i in range(agentNum):
        EnviromentManager.addAgentInformation(float(initiallocation[i][0]), float(initiallocation[i][1]),int(Personality[i]),float(initialCoin[i]), float(speed[i]))

    for t in range(len(tradeRules)):
        manager.trade_rules[t] = float(tradeRules[t])

    manager.canDestroy = bool(canDestroy)
    manager.DestroyDeadline = float(DestroyDeadline)

def randomLocation(mapSize):
    return [random.uniform(-abs(mapSize[0])/2-1, abs(mapSize[0])/2-1), random.uniform(-abs(mapSize[1])/2-1, abs(mapSize[1])/2-1)]



############################################################################################


actionlist = ["cheat", "cooperation"]
manager = ue.GameObject.Find("GameManager").GetComponent("GameManager")
PythonManager = ue.GameObject.Find("PythonManager").GetComponent("PythonManager")
EnviromentManager = ue.GameObject.Find("EnvironmentBuilder").GetComponent("EnvironmentBuilder")

AIDesigner()


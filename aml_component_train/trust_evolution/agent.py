import numpy as np
import copy


stereotype_mapping = {
    'Random': 0,
    'Cheater': 1,
    'Cooperator': 2,
    'Copycat': 3,
    'Grudger': 4,
    'Detective': 5,
    'Human': 6,
    'Q': 7,
    'DQN': 8
}


class Agent(object):
    def __init__(self, id, stereotype=0):
        self.id = id
        self.money = 5
        # self.speed = 2
        self.tradeNum = 0
        self.stereotype = stereotype
        self.tradeRecords = {}  # {对手ID: [(own_action, opponent_action), ...]}
        self.location = [0, 0]
        # 添加行动计数器
        self.cooperation_count = 0
        self.cheat_count = 0


    def addReward(self, reward):
        self.money += reward

    def StereotypeAction(self, othersID):
        if self.stereotype == 'Random':
            action = self.RandomAction()
        elif self.stereotype == 'Cheater':
            action = "cheat"
        elif self.stereotype == 'Cooperator':
            action = "cooperation"
        elif self.stereotype == 'Copycat':
            action = self.Copycat(othersID)
        elif self.stereotype == 'Grudger':
            action = self.Grudger(othersID)
        elif self.stereotype == 'Detective':
            action = self.Detective(othersID)
        elif self.stereotype == 'Human':
            action = self.Human(othersID)
        else:
            action = self.RandomAction()
        return action

    def Remember(self, opponent_id, own_action, opponent_action):
        if opponent_id in self.tradeRecords:
            self.tradeRecords[opponent_id].append((own_action, opponent_action))
        else:
            self.tradeRecords[opponent_id] = [(own_action, opponent_action)]

    def RandomAction(self):
        return np.random.choice(["cooperation", "cheat"])

    def Copycat(self, othersID):
        action = "cooperation"
        if othersID in self.tradeRecords:
            last_interaction = self.tradeRecords[othersID][-1]
            opponent_last_action = last_interaction[1]
            action = opponent_last_action
        return action

    def Grudger(self, othersID):
        action = "cooperation"
        if othersID in self.tradeRecords:
            for interaction in self.tradeRecords[othersID]:
                if interaction[1] == "cheat":
                    action = "cheat"
                    break
        return action

    def Detective(self, othersID):
        action = "cooperation"
        if othersID in self.tradeRecords:
            history = self.tradeRecords[othersID]
            if len(history) == 1:
                action = "cheat"
            elif len(history) == 2 or len(history) == 3:
                action = "cooperation"
            elif len(history) > 3:
                initial_interactions = history[:4]
                cheated = any(interaction[1] == "cheat" for interaction in initial_interactions)
                if cheated:
                    action = self.Copycat(othersID)
                else:
                    action = "cheat"
        return action

    def Human(self, othersID):
        """
        Human 代理的行为决策：
          - 如果对方人格为 Cooperator、Random、Cheater，则始终返回 "cheat"；
          - 如果对方人格为 Human、Copycat、Grudger，则始终返回 "cooperation"；
          - 如果对方人格为 Detective，则在与该对手交互次数不足4次时返回 "cheat"，否则返回 "cooperation"；
          - 如果对方人格为 AI，则返回随机动作。
        """
        # 尝试从 self.opponent_types 中获取对手人格（请确保在 Human 代理中设置此属性）
        var_opponent = None
        if hasattr(self, 'opponent_types'):
            var_opponent = self.opponent_types.get(othersID, None)
        # 如果没有记录对手人格，则返回随机
        if var_opponent is None:
            return self.RandomAction()

        if var_opponent in ['Cooperator', 'Random', 'Cheater']:
            return "cheat"
        elif var_opponent in ['Human', 'Copycat', 'Grudger']:
            return "cooperation"
        elif var_opponent == 'Detective':
            # 根据与该对手的交互次数决定：少于4次则 cheat，否则 cooperation
            var_count = len(self.tradeRecords.get(othersID, []))
            if var_count < 4:
                return "cheat"
            else:
                return "cooperation"
        elif var_opponent == 'AI':
            return self.RandomAction()
        else:
            return self.RandomAction()

    def get_state(self, opponent_id, opponent_stereotype):
        """
        获取与指定对手的最近5次完整交互记录作为观察值，并包含对手的 stereotype。

        观察值格式：
        - 首先是对手的 stereotype 编码，以字符串形式表示。
        - 然后是一个下划线 '_'，用于分隔。
        - 接下来是最近5次交互记录，每个交互记录用两个字符表示：
            - 'y' 代表合作 (cooperation)
            - 'n' 代表欺骗 (cheat)
            - 'oo' 代表未完成 (not yet)
        - 最终返回一个字符串，例如 '3_ooooooyynn'。

        参数：
            opponent_id (int): 对手的 ID。
            opponent_stereotype (str): 对手的 stereotype 字符串。

        返回：
            state (str): 状态字符串。
        """
        # 获取与指定对手的所有交互记录
        records = self.tradeRecords.get(opponent_id, [])

        # 初始化交易记录字符串
        interactions = ""

        # 遍历最近5次交互记录
        for interaction in records[-5:]:
            own_action, opponent_action = interaction
            # 将动作映射为字符
            own_char = 'y' if own_action == 'cooperation' else 'n'
            opp_char = 'y' if opponent_action == 'cooperation' else 'n'
            # 拼接成两字符表示
            interactions += own_char + opp_char

        # 如果交互次数不足5次，前面填充 'oo'
        missing_trades = 5 - len(records[-5:])
        interactions = 'oo' * missing_trades + interactions

        # 获取对手 stereotype 的编码
        opponent_code = stereotype_mapping.get(opponent_stereotype, -1)

        # 将对手的 stereotype 编码添加到状态字符串的开头
        state = f"{opponent_code}_{interactions}"

        return state

    # 记录自己的行动
    def record_action(self, action):
        if action == 'cooperation':
            self.cooperation_count += 1
        elif action == 'cheat':
            self.cheat_count += 1

    def get_opponent_id(self):
        return next(reversed(self.tradeRecords))

    def get_last_action(self):
        return self.tradeRecords.get(next(reversed(self.tradeRecords)))[-1][0]
import numpy as np
import pandas as pd
import os


def map_value(d, a, b, r):
    """
    根据 difficulty d (0~1)，参数取值范围 [a, b] 以及相关系数 r，
    将 d 映射为参数值，采用多个阈值区分 r 的不同档位：
      - 当 r < -0.5：采用较高斜率，使得随着 d 增加，参数迅速减小
      - 当 -0.5 <= r < -0.1：采用中等斜率
      - 当 -0.1 <= r <= 0.1：参数固定为 (a+b)/2
      - 当 0.1 < r <= 0.3：采用中等正斜率
      - 当 r > 0.3：采用较高正斜率，使得随着 d 增加，参数迅速增大
    """
    if r < -0.5:
        # 强负相关：参数从 b 到 a，变化更剧烈，比如放大比例1.2
        return b - (b - a)*d
    elif r < -0.1:
        # 中等负相关
        return b - 0.8*(b - a)*d
    elif abs(r) <= 0.1:
        return (a + b) / 2
    elif r <= 0.3:
        # 弱正相关
        return a + 0.8*(b - a)*d
    else:
        # 强正相关：放大比例1.2
        return a + (b - a)*d



def generate_rule_difficulty_table(output_csv="rule_difficulty_table.csv"):
    """
    构造一个表格，展示不同 difficulty 档次下（Very Easy, Easy, Medium, Hard, Very Hard, Extreme）
    17个规则参数的预期值。各参数按维度使用各自的相关系数映射：

    - Initial Population (8维)：范围 [0, 5]，r: [0.21, 0.01, 0.48, 0.18, 0.10, 0.15, 0.53, 0.32]
    - Trade Rules (6维)：范围 [-3, 3]，r: [0.32, 0.24, 0.44, 0.5, 0.19, 0.17]
    - Round Number：范围 [1, 5]，r: -0.02
    - Reproduction Number：范围 [0, 10]，r: -0.52
    - Mistake Possibility：范围 [0, 0.5]，r: -0.10

    难度档次：
      Very Easy: d = 0.0
      Easy:      d = 0.2
      Medium:    d = 0.4
      Hard:      d = 0.6
      Very Hard: d = 0.8
      Extreme:   d = 1.0

    表格的行索引为难度档次，列为 17 个参数。
    """
    # 参数名称定义
    initial_pop_names = ["Initial_Random", "Initial_Cheater", "Initial_Cooperator", "Initial_Copycat",
                         "Initial_Grudger", "Initial_Detective", "Initial_AI", "Initial_Human"]
    trade_rule_names = [f"TradeRule_{i + 1}" for i in range(6)]
    other_names = ["RoundNumber", "ReproductionNumber", "MistakePossibility"]
    param_names = initial_pop_names + trade_rule_names + other_names

    # 各参数对应的相关系数
    r_initial = [0.21, 0.01, 0.48, 0.18, 0.10, 0.15, 0.53, 0.32]
    r_trade = [0.32, 0.24, 0.44, 0.5, 0.19, 0.17]
    r_other = [-0.02, -0.52, -0.10]
    r_values = r_initial + r_trade + r_other

    # 参数取值范围
    range_dict = {}
    for name in initial_pop_names:
        range_dict[name] = (0, 5)
    for name in trade_rule_names:
        range_dict[name] = (-3, 3)
    range_dict["RoundNumber"] = (1, 5)
    range_dict["ReproductionNumber"] = (0, 10)
    range_dict["MistakePossibility"] = (0, 0.5)

    # 定义难度档次和对应 d 值
    difficulty_levels = ["Very Easy", "Easy", "Medium", "Hard", "Very Hard", "Extreme"]
    d_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # 构造结果数据
    table_data = {}
    for param, r in zip(param_names, r_values):
        a, b = range_dict[param]
        # 对于每个难度档次计算预期值
        values = [map_value(d, a, b, r) for d in d_values]
        table_data[param] = values

    # 构造 DataFrame，行索引为难度档次
    df_table = pd.DataFrame(table_data, index=difficulty_levels)
    df_table.index.name = "DifficultyLevel"

    # 保存 CSV 文件
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_table.to_csv(output_csv, index=True)
    print(f"规则-难度映射表已保存至 {output_csv}")

    return df_table


if __name__ == "__main__":
    table = generate_rule_difficulty_table(output_csv="./rule_difficulty_table.csv")
    print(table)

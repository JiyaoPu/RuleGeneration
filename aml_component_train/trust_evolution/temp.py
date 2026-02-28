import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Q-table 数据
data = """observation,cheat,cooperate
oooooooooo,2.276668,54.177297
ooooooooyy,53.229622,0.163712
ooooooyynn,54.672189,2.232561
ooooyynnny,52.824680,1.946036
ooyynnnyny,50.488198,0.085950
yynnnynynn,51.652347,0.237557
nnnynynnyn,0.030000,2.315394
nynynnynyy,2.652996,0.040610
nynnynyyny,2.908316,0.161084
nnynyynyyn,0.162245,0.020000
ynyynyynny,0.687783,0.188298
ooooooyyyn,2.263764,0.020000
ooooyyynny,1.826097,0.020000
ooyyynnyny,1.329284,-0.028816
yyynnynynn,6.378296,-0.028523
ynnynynnyn,0.059772,0.091425
nynynnynny,0.011226,0.000000
nynnynnynn,0.642188,0.070695
nnynnynnnn,0.740793,0.182701
ynnynnnnyn,0.179985,-0.013293
yynnnynyyn,0.000000,1.852627
nnnynyynyy,2.023917,0.020403
nynyynyyny,1.831516,-0.009691
nyynyynynn,8.705000,-0.010308
ynyynynnnn,27.829798,19.382969
yynynnnnnn,53.423924,41.786627
nnnynynnnn,52.694001,0.032779
nynynnnnnn,53.729780,0.842065
nynnnnnnyn,4.468567,0.040070
nnnnnnynny,7.220795,4.016015
nnnnynnyyn,0.090367,0.001188
nynynnnnyn,0.064741,1.719058
nynnnnynyy,0.000000,1.935507
nnnnynyyyy,0.159502,2.883299
nnynyyyyyy,5.366673,5.543322
nynnnnnnnn,54.758861,45.659926
nnnnnnnnnn,55.546857,55.170291
nnnnnnnnny,7.976384,3.450065
ooyynnnynn,4.734371,-0.028816
yynnnynnyn,0.089151,0.000000
nnnynnynny,0.100366,0.010924
nynnynnyyn,0.030000,0.020000
nnynnyynny,0.000000,0.000000
ynnyynnynn,0.148452,-0.007349
nyynnynnyn,0.090373,-0.008514
nynynnnnny,1.093739,-0.010000
nynnnnnyyn,0.180811,0.011578
nnnnnyynny,0.357418,0.029025
nnnyynnyyn,0.180025,0.001779
ooooyynnnn,1.953249,0.000000
ooyynnnnny,1.308167,-0.019702
yynnnnnynn,6.751407,-0.036572
nnnnnynnyn,0.000000,0.146967
nnnynnynyy,0.373826,0.118877
nynnynyyyy,0.095228,0.215830
ynyyyyyyyy,0.154724,0.260079
ynyynynnyn,0.389574,0.142786
yynynnynny,0.119163,0.011560
nnnnynnynn,7.839832,4.805948
nynnnnnynn,7.076249,-0.028306
nnynyynynn,9.803299,1.040886
nynnnnnnny,3.457050,-0.008291
nnnnnnnyyn,0.092633,0.000000
ooooyynnyy,2.383101,0.039998
ooyynnyyny,2.185389,-0.019702
yynnyynyyn,0.000000,0.040398
nnyynyynyy,0.000000,0.081149
yynyynyyyy,0.122715,0.154627
nyynyyyyyy,0.331751,0.100868
ynyyyyyyny,0.240724,0.137794
yyyyyynynn,0.392653,0.356392
ynnynynnnn,23.551000,0.007019
yynnnynyny,2.408923,-0.010000
nnnynynyyn,0.000000,0.020000
nynynyynyy,0.000000,0.060207
nynyynyyyy,0.000000,0.081766
ooyynnnyyy,0.059700,2.069160
yynnnyyyyy,2.434357,0.000000
nnnyyyyyny,2.995672,-0.019603
nyyyyynyyn,0.089173,0.021760
yyyynyynny,0.010703,-0.019297
yynyynnyyn,0.150867,0.054078
nyynnyynny,0.269633,0.044582
ooooooyyny,0.000000,1.979011
ooooyynyyy,2.283240,0.020000
ooyynyyyny,2.780835,-0.010000
yynyyynynn,12.083449,-0.010000
nyyynynnnn,36.286260,-0.018123
yynnnnnyyn,0.000000,0.040097
nnnnnyynyy,0.059717,0.000000
nnnyynyyny,0.073130,-0.010000
nyynyynyyn,0.212561,0.038844
yynyynnynn,0.351736,0.184759
nnnnnnnnyn,13.718040,12.095083
nnnnnnynyn,0.060367,0.000891
nnnynynynn,9.854809,-0.010000
nynynynnnn,29.023823,0.000000
ooooooooyn,0.850371,-0.010000
ooooooynnn,0.030000,2.160204
ooooynnnyy,0.000000,2.273166
"""

# 读取数据
df = pd.read_csv(io.StringIO(data))

# 归一化Q值以便可视化
# 合并cheat和cooperate列
combined_min = min(df["cheat"].min(), df["cooperate"].min())
combined_max = max(df["cheat"].max(), df["cooperate"].max())

# 正则化
df["cheat_norm"] = (df["cheat"] - combined_min) / (combined_max - combined_min)
df["cooperate_norm"] = (df["cooperate"] - combined_min) / (combined_max - combined_min)


# 可视化
plt.figure(figsize=(12, 4))
df_sorted = df.sort_values(by="observation")

ax = sns.heatmap(df_sorted[["cheat_norm", "cooperate_norm"]].T, cmap="coolwarm", annot=False, xticklabels=df_sorted["observation"], yticklabels=["cheat", "cooperate"])

plt.xticks(rotation=90, fontsize=8)  # 调整字体大小
plt.subplots_adjust(bottom=0.3)  # 调整底部间距
plt.title("Q-table Heatmap (Normalized)")
plt.show()

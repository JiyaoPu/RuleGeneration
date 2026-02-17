import pandas as pd
import matplotlib.pyplot as plt


# Load BETA_A and BETA_B CSV files

beta_a_df = pd.read_csv("../data/cooperation rate/0.98/BETA/BETA_A_0.csv")
beta_b_df = pd.read_csv("../data/cooperation rate/0.98/BETA/BETA_B_0.csv")

# Plot the data
plt.figure(figsize=(10, 6))

plt.plot(beta_a_df['Epoch'], beta_a_df['Beta'], label='Cooperation Reward', color='blue')
plt.plot(beta_b_df['Epoch'], beta_b_df['Beta'], label='Cheat Punishment', color='red')

plt.xlabel('Episode')
plt.ylabel('Extrinsic Reward')
plt.title('Gini Coefficient = 0.98')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()


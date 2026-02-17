import pandas as pd
import matplotlib.pyplot as plt
import os

# Define folder and filenames
folder = "../data/cooperation rate/0.98/BETA/"
beta_a_files = [f"BETA_A_{i}.csv" for i in range(4)]
beta_b_files = [f"BETA_B_{i}.csv" for i in range(4)]

# Initialize lists to collect data for overall range
all_beta_a_values = []
all_beta_b_values = []

# Load BETA_A files and collect data
for file in beta_a_files:
    file_path = os.path.join(folder, file)
    beta_a_df = pd.read_csv(file_path)
    all_beta_a_values.append(beta_a_df['Beta'])

# Load BETA_B files and collect data
for file in beta_b_files:
    file_path = os.path.join(folder, file)
    beta_b_df = pd.read_csv(file_path)
    all_beta_b_values.append(beta_b_df['Beta'])

# Concatenate all beta values into DataFrames for easier calculations
all_beta_a_df = pd.concat(all_beta_a_values, axis=1)
all_beta_b_df = pd.concat(all_beta_b_values, axis=1)

# Calculate the mean and standard deviation for each epoch across all BETA_A and BETA_B files
beta_a_mean = all_beta_a_df.mean(axis=1)
beta_a_std = all_beta_a_df.std(axis=1)
beta_b_mean = all_beta_b_df.mean(axis=1)
beta_b_std = all_beta_b_df.std(axis=1)

# Assume all files have the same epochs, get the epochs from any DataFrame
epochs = beta_a_df['Epoch']

# Plot the lines for each individual BETA file
plt.figure(figsize=(12, 8))

# Plot BETA_A lines in blue shades
colors_a = ['#1f77b4', '#3399ff', '#66b3ff', '#99ccff']
for idx, file in enumerate(beta_a_files):
    file_path = os.path.join(folder, file)
    beta_a_df = pd.read_csv(file_path)
    plt.plot(epochs, beta_a_df['Beta'], label=f'Cooperation Reward {idx}', linestyle='-', linewidth=1.5, color=colors_a[idx])

# Plot BETA_B lines in red shades
colors_b = ['#d62728', '#ff6666', '#ff9999', '#ffcccc']
for idx, file in enumerate(beta_b_files):
    file_path = os.path.join(folder, file)
    beta_b_df = pd.read_csv(file_path)
    plt.plot(epochs, beta_b_df['Beta'], label=f'Cheat Punishment {idx}', linestyle='--', linewidth=1.5, color=colors_b[idx])

# Add fill_between for the possible range of BETA_A and BETA_B
plt.fill_between(epochs, beta_a_mean - 2 * beta_a_std, beta_a_mean + 2 * beta_a_std, color='blue', alpha=0.1, label='Possible Range of Cooperation Reward (±2σ)')
plt.fill_between(epochs, beta_b_mean - 2 * beta_b_std, beta_b_mean + 2 * beta_b_std, color='red', alpha=0.1, label='Possible Range of Cheat Punishment (±2σ)')

# Customize the plot
plt.xlabel('Episode')
plt.ylabel('Extrinsic Reward')
plt.title('Cooperation Rate = 0.98')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

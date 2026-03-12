from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Unified output path helpers
# =========================================================

OUTPUT_DIR = Path(os.getenv("AZUREML_OUTPUT_DIR", "outputs")).resolve()
DATA_DIR = OUTPUT_DIR / "data"
PLOTS_DIR = OUTPUT_DIR / "plots"

for d in [OUTPUT_DIR, DATA_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def data_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (DATA_DIR / p)


def plot_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (PLOTS_DIR / p)


# =========================================================
# Load CSV files
# =========================================================

file_names = [
    "0.50/0.csv",
    "0.50/5.csv",
    "0.50/10.csv",
    "0.50/20.csv",
    "0.50/50.csv",
    "0.50/99.csv",
]

file_paths = [data_path(f) for f in file_names]

dataframes = [pd.read_csv(fp) for fp in file_paths]

labels = [
    "Episode 0",
    "Episode 5",
    "Episode 10",
    "Episode 20",
    "Episode 50",
    "Episode 99",
]


# =========================================================
# Plot
# =========================================================

plt.figure(figsize=(10, 6))

for i, df in enumerate(dataframes):
    plt.plot(df["Epoch"], df["Cooperation Rate"], label=labels[i])

plt.xlabel("Epoch")
plt.ylabel("Cooperation Rate")
plt.title("Cooperation Rate = 0.50")

plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.grid(True)

plt.tight_layout(rect=[0, 0, 0.85, 1])


# =========================================================
# Save figure
# =========================================================

output_file = plot_path("cooperation_rate_050.png")
plt.savefig(output_file)

print(f"Plot saved to: {output_file}")

# Show locally but not in Azure ML
if os.getenv("AZUREML_RUN_ID") is None:
    plt.show()
else:
    plt.close()
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ======================
# Paths
# ======================

input_file = Path(
    "../../interpretability/causal_results/causal_results/qwen25_delta_ce_v5/full_delta_ce.csv"
)

output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)


# ======================
# Load data
# ======================

df = pd.read_csv(input_file)


# Keep only positive causal changes
df["positive_delta_ce"] = df["delta_ce"].clip(lower=0)


# Aggregate by layer
layer_score = (
    df.groupby("layer")["positive_delta_ce"]
    .sum()
    .reindex(range(28), fill_value=0)
)


layers = layer_score.index.values
values = layer_score.values


# ======================
# Plot
# ======================

fig, ax = plt.subplots(
    figsize=(8,4)
)


bars = ax.bar(
    layers,
    values
)


ax.set_xlabel(
    "Transformer Layer"
)

ax.set_ylabel(
    "Positive $\Delta$CE contribution"
)


ax.set_title(
    "Layer-wise distribution of causal contribution"
)


ax.set_xticks(
    range(0,28,2)
)


ax.grid(
    axis="y",
    alpha=0.3
)


fig.tight_layout()


fig.savefig(
    output_dir/"figure8_layerwise_delta_ce.pdf",
    bbox_inches="tight"
)


fig.savefig(
    output_dir/"figure8_layerwise_delta_ce.png",
    dpi=300,
    bbox_inches="tight"
)


print("Figure 8 generated successfully.")
print(layer_score)
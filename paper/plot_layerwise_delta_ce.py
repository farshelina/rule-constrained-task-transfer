from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from visual_palette import POSITIVE, INK

input_file = Path("../../interpretability/causal_results/causal_results/qwen25_delta_ce_v5/full_delta_ce.csv")
output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

df = pd.read_csv(input_file)
df["positive_delta_ce"] = df["delta_ce"].clip(lower=0)
layer_score = df.groupby("layer")["positive_delta_ce"].sum().reindex(range(28), fill_value=0)

layers = layer_score.index.values
values = layer_score.values
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(layers, values, color=POSITIVE, edgecolor=INK, linewidth=0.55)
ax.set_xlabel("Transformer Layer")
ax.set_ylabel(r"Positive $\Delta CE$ contribution")
ax.set_title("Layer-wise distribution of positive causal change")
ax.set_xticks(range(0, 28, 2))
ax.grid(axis="y", alpha=0.22)
fig.tight_layout()
fig.savefig(output_dir / "figure8_layerwise_delta_ce.pdf", bbox_inches="tight")
fig.savefig(output_dir / "figure8_layerwise_delta_ce.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Figure 8 generated successfully.")
print(layer_score)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from visual_palette import SIGNED_CMAP, INK

input_file = r"D:\github\doudizhu\interpretability\causal_results\causal_results\qwen25_delta_ce_v5\full_delta_ce.csv"
output_dir = "figures"
output_pdf = os.path.join(output_dir, "figure_attention_head_heatmap.pdf")
output_png = os.path.join(output_dir, "figure_attention_head_heatmap.png")

df = pd.read_csv(input_file)
for c in ["layer", "head", "delta_ce"]:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

n_layers = int(df["layer"].max()) + 1
n_heads = int(df["head"].max()) + 1
heatmap = np.zeros((n_layers, n_heads))
for _, row in df.iterrows():
    heatmap[int(row["layer"]), int(row["head"])] = float(row["delta_ce"])

top_heads = df.sort_values("delta_ce", ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8.5, 7))

# Fixed for the current full scan (observed range is approximately -2.26 to 7.06).
# Zero is the semantic boundary; endpoint hues match all other signed-result figures.
norm = TwoSlopeNorm(vmin=-2.5, vcenter=0.0, vmax=7.5)
im = ax.imshow(heatmap, origin="lower", cmap=SIGNED_CMAP, norm=norm, aspect="equal")

ax.set_xlabel("Attention Head Index", fontsize=12)
ax.set_ylabel("Transformer Layer", fontsize=12)
ax.set_xticks(np.arange(n_heads))
ax.set_yticks(np.arange(n_layers))
ax.set_xticklabels(np.arange(n_heads), fontsize=8)
ax.set_yticklabels(np.arange(n_layers), fontsize=8)

ax.set_xticks(np.arange(-0.5, n_heads, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n_layers, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=0.45, alpha=0.65)
ax.tick_params(which="minor", bottom=False, left=False)

for _, row in top_heads.iterrows():
    layer = int(row["layer"])
    head = int(row["head"])
    ax.add_patch(Rectangle(
        (head - 0.5, layer - 0.5), 1, 1,
        linewidth=1.4, edgecolor=INK, facecolor="none"
    ))

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r"$\Delta CE$ (LoRA-v2 $-$ Base)", fontsize=12)
ax.set_title(r"Attention-Head Causal Change ($\Delta CE$)", fontsize=13)

plt.tight_layout()
os.makedirs(output_dir, exist_ok=True)
plt.savefig(output_pdf, bbox_inches="tight")
plt.savefig(output_png, dpi=300, bbox_inches="tight")
plt.close()
print("Saved:")
print(output_pdf)
print(output_png)

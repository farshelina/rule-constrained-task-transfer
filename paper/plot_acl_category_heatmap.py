from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ======================
# Output directory
# ======================
output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

# ======================
# Data: Base-to-LoRA-v2 category-level changes
# Rows: Counter, Hierarchy, Pattern
# Cols: DeepSeek, Qwen2.5, Llama3.1
# ======================
data = np.array([
    [13, -14,  2],   # Counter Logic
    [12,  36,  3],   # Hierarchy Logic
    [ 5,  34, 36],   # Pattern Recognition
], dtype=float)

row_labels = [
    "Counter Logic",
    "Hierarchy Logic",
    "Pattern Recognition"
]

col_labels = [
    "DeepSeek",
    "Qwen2.5",
    "Llama3.1"
]

# ======================
# Plot
# ======================
fig, ax = plt.subplots(figsize=(7.2, 4.8))

im = ax.imshow(data, cmap="coolwarm", aspect="auto")

# ticks
ax.set_xticks(np.arange(len(col_labels)))
ax.set_yticks(np.arange(len(row_labels)))
ax.set_xticklabels(col_labels)
ax.set_yticklabels(row_labels)

# title and labels
ax.set_title("Category-level ACL Bench transfer (Base-to-LoRA-v2 change)")
ax.set_xlabel("Model family")
ax.set_ylabel("ACL Bench category")

# annotate each cell
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        value = data[i, j]
        text_str = f"{value:+.0f}"
        ax.text(
            j, i, text_str,
            ha="center", va="center",
            fontsize=11
        )

# colorbar
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Accuracy change (percentage points)")

fig.tight_layout()

fig.savefig(output_dir / "figure7_acl_category_heatmap.pdf", bbox_inches="tight")
fig.savefig(output_dir / "figure7_acl_category_heatmap.png", dpi=300, bbox_inches="tight")

print("Figure 7 generated successfully.")
print(output_dir / "figure7_acl_category_heatmap.pdf")
print(output_dir / "figure7_acl_category_heatmap.png")
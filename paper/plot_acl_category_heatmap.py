from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from visual_palette import SIGNED_CMAP, INK

output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

data = np.array([
    [13, -14,  2],
    [12,  36,  3],
    [ 5,  34, 36],
], dtype=float)

row_labels = ["Counter Logic", "Hierarchy Logic", "Pattern Recognition"]
col_labels = ["DeepSeek", "Qwen2.5", "Llama3.1"]

fig, ax = plt.subplots(figsize=(7.2, 4.8))

# Fixed directional semantics: negative -> white -> positive.
# The numerical scale is symmetric because this figure shows accuracy deltas.
norm = TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40)
im = ax.imshow(data, cmap=SIGNED_CMAP, norm=norm, aspect="auto")

ax.set_xticks(np.arange(len(col_labels)))
ax.set_yticks(np.arange(len(row_labels)))
ax.set_xticklabels(col_labels)
ax.set_yticklabels(row_labels)
ax.set_xlabel("Model family")
ax.set_ylabel("ACL Bench category")
ax.set_title("Category-level ACL Bench transfer (Base-to-LoRA-v2 change)")

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        value = data[i, j]
        ax.text(j, i, f"{value:+.0f}", ha="center", va="center",
                fontsize=11, color=INK, fontweight="bold")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Accuracy change (percentage points)")
fig.tight_layout()

# Match the filename used by results.tex.
fig.savefig(output_dir / "figure7_acl_category_transfer.pdf", bbox_inches="tight")
fig.savefig(output_dir / "figure7_acl_category_transfer.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Generated:")
print(output_dir / "figure7_acl_category_transfer.pdf")
print(output_dir / "figure7_acl_category_transfer.png")

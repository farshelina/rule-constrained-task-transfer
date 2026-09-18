from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from visual_palette import SIGNED_CMAP, INK

output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

# Rows: Counter Logic, Hierarchy Logic, Pattern Recognition
# Columns: DeepSeek, Qwen2.5, Llama3.1
base = np.array([
    [24, 34, 34],
    [35, 36, 58],
    [26, 36, 43],
], dtype=float)

dd_lora = np.array([
    [28, 39, 30],
    [30, 60, 51],
    [37, 63, 46],
], dtype=float)

rc_lora = np.array([
    [37, 46, 30],
    [40, 65, 58],
    [37, 65, 52],
], dtype=float)

base_to_dd = dd_lora - base
dd_to_rc = rc_lora - dd_lora

row_labels = ["Counter Logic", "Hierarchy Logic", "Pattern Recognition"]
col_labels = ["DeepSeek", "Qwen2.5", "Llama3.1"]

# Use one common symmetric scale so the two stages can be compared directly.
all_delta = np.concatenate([base_to_dd.ravel(), dd_to_rc.ravel()])
limit = max(10.0, float(np.ceil(np.max(np.abs(all_delta)) / 5.0) * 5.0))
norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

# Use a dedicated third column for the colorbar. This keeps it completely
# outside the right heatmap instead of letting it overlap the Llama3.1 column.
fig = plt.figure(figsize=(11.2, 4.7))
gs = fig.add_gridspec(
    nrows=1,
    ncols=3,
    width_ratios=[1.0, 1.0, 0.045],
    wspace=0.24,
)
ax_left = fig.add_subplot(gs[0, 0])
ax_right = fig.add_subplot(gs[0, 1], sharey=ax_left)
cax = fig.add_subplot(gs[0, 2])
axes = [ax_left, ax_right]

for ax, data, title in zip(
    axes,
    [base_to_dd, dd_to_rc],
    ["Base $\\rightarrow$ DD-LoRA", "DD-LoRA $\\rightarrow$ RC-LoRA"],
):
    im = ax.imshow(data, cmap=SIGNED_CMAP, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_title(title)
    ax.set_xlabel("Model checkpoint")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            ax.text(
                j,
                i,
                f"{value:+.0f}",
                ha="center",
                va="center",
                fontsize=10,
                color=INK,
                fontweight="bold",
            )

ax_left.set_yticks(np.arange(len(row_labels)))
ax_left.set_yticklabels(row_labels)
ax_left.set_ylabel("ACL Bench category")

# Keep the shared y-axis labels only on the left panel.
ax_right.tick_params(axis="y", labelleft=False)

cbar = fig.colorbar(im, cax=cax)
cbar.set_label("Accuracy change (percentage points)", labelpad=10)

fig.suptitle(
    "Category-level ACL Bench transfer across training stages",
    y=0.98,
)
# Reserve explicit margins for the title and axis labels. The colorbar position
# is governed by GridSpec, so no right-side overlap occurs.
fig.subplots_adjust(left=0.11, right=0.94, bottom=0.17, top=0.82)

fig.savefig(output_dir / "figure7_acl_category_transfer.pdf", bbox_inches="tight")
fig.savefig(
    output_dir / "figure7_acl_category_transfer.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

print("Generated:")
print(output_dir / "figure7_acl_category_transfer.pdf")
print(output_dir / "figure7_acl_category_transfer.png")

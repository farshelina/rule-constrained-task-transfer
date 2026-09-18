from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from visual_palette import POSITIVE, NEUTRAL, INK

output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

models = ["DeepSeek", "Qwen2.5", "Llama3.1"]
base = np.array([28.3, 35.3, 45.0])
dd_lora = np.array([31.7, 54.0, 42.3])
rc_lora = np.array([38.0, 58.7, 46.7])
delta_rc_dd = rc_lora - dd_lora

fig, ax = plt.subplots(figsize=(8.2, 4.9))
x = np.arange(len(models))
width = 0.24

bars_base = ax.bar(
    x - width, base, width,
    label="Base", color="white", edgecolor=NEUTRAL, linewidth=1.0, hatch="///"
)
bars_dd = ax.bar(
    x, dd_lora, width,
    label="DD-LoRA", color=NEUTRAL, edgecolor=INK, linewidth=0.8
)
bars_rc = ax.bar(
    x + width, rc_lora, width,
    label="RC-LoRA", color=POSITIVE, edgecolor=INK, linewidth=0.8
)

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("ACL Bench Accuracy (%)")
ax.set_title("Overall ACL Bench accuracy across training regimes")
ax.set_ylim(0, 70)
ax.grid(axis="y", alpha=0.22)
ax.legend(ncol=3, loc="upper left")

for bars in (bars_base, bars_dd, bars_rc):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.9,
            f"{h:.1f}", ha="center", va="bottom", fontsize=8, color=INK
        )

# The annotation reports the manuscript's primary incremental comparison:
# RC-LoRA relative to the corresponding DD-LoRA checkpoint.
for xi, d, y in zip(x, delta_rc_dd, rc_lora):
    ax.text(
        xi + width, y + 4.0, f"RC-DD {d:+.1f}", ha="center",
        fontsize=8, color=POSITIVE, fontweight="bold"
    )

fig.tight_layout()
fig.savefig(output_dir / "figure6_acl_transfer.pdf", bbox_inches="tight")
fig.savefig(output_dir / "figure6_acl_transfer.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Figure 6 generated.")

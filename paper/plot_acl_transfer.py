from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from visual_palette import POSITIVE, NEGATIVE, NEUTRAL, INK

output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

models = ["DeepSeek", "Qwen2.5", "Llama3.1"]
base = np.array([28.0, 35.3, 45.0])
lora_v2 = np.array([38.0, 54.0, 58.7])
delta = lora_v2 - base

fig, ax = plt.subplots(figsize=(7.6, 4.8))
x = np.arange(len(models))
width = 0.34

bars_base = ax.bar(
    x - width / 2, base, width,
    label="Base", color=NEUTRAL, edgecolor=INK, linewidth=0.8
)

lora_colors = [POSITIVE if d >= 0 else NEGATIVE for d in delta]
bars_lora = ax.bar(
    x + width / 2, lora_v2, width,
    label="LoRA-v2", color=lora_colors, edgecolor=INK, linewidth=0.8
)

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("ACL Bench Accuracy (%)")
ax.set_title("Overall cross-task transfer after fine-tuning")
ax.set_ylim(0, 70)
ax.grid(axis="y", alpha=0.22)
ax.legend()

for bars in (bars_base, bars_lora):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                f"{h:.1f}", ha="center", fontsize=8, color=INK)

for xi, d, y in zip(x, delta, lora_v2):
    ax.text(xi + width/2, y + 4.0, f"{d:+.1f}", ha="center",
            fontsize=8, color=POSITIVE if d >= 0 else NEGATIVE,
            fontweight="bold")

fig.tight_layout()
fig.savefig(output_dir / "figure6_acl_transfer.pdf", bbox_inches="tight")
fig.savefig(output_dir / "figure6_acl_transfer.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Figure 6 generated.")

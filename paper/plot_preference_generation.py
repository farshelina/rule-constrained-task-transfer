from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from visual_palette import POSITIVE, NEGATIVE, NEUTRAL, INK

output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

models = ["Base", "LoRA-v2"]
margin = np.array([1.390711, 4.371395])
legal_rate = np.array([0.333, 0.0])
forbidden_rate = np.array([0.667, 1.0])

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

# Panel A: Base is neutral; the higher LoRA-v2 BCDM is a positive-direction result.
ax = axes[0]
x = np.arange(len(models))
bars = ax.bar(x, margin, color=[NEUTRAL, POSITIVE], edgecolor=INK, linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Best Constraint Delta Margin")
ax.set_title("(a) Preference reallocation")
for bar, value in zip(bars, margin):
    ax.text(bar.get_x()+bar.get_width()/2, value+0.1, f"{value:.2f}",
            ha="center", fontsize=9, color=INK)
ax.set_ylim(0, max(margin)+1)

# Panel B: legal generation is positive-direction; forbidden generation is negative-direction.
ax = axes[1]
width = 0.35
x = np.arange(len(models))
bars1 = ax.bar(x-width/2, legal_rate, width, label="Legal action",
               color=POSITIVE, edgecolor=INK, linewidth=0.8)
bars2 = ax.bar(x+width/2, forbidden_rate, width, label="Forbidden action",
               color=NEGATIVE, edgecolor=INK, linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Generation rate")
ax.set_title("(b) Direct generation diagnostic")
ax.legend()
ax.set_ylim(0, 1.1)

fig.tight_layout()
fig.savefig(output_dir / "figure5_preference_generation.pdf", bbox_inches="tight")
fig.savefig(output_dir / "figure5_preference_generation.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Figure 5 generated.")

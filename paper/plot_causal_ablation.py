from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from visual_palette import POSITIVE, NEUTRAL, INK

output_dir = Path("figures")
output_dir.mkdir(parents=True, exist_ok=True)

k_labels = ["Top-5", "Top-10", "Top-27"]
k_numeric = [5, 10, 27]
topk_drop = np.array([7.488516, 6.986626, 5.280776])
global_random_mean = np.array([0.281428, 0.186322, 2.638228])
global_random_std  = np.array([0.771978, 0.972558, 0.316980])
matched_random_mean = np.array([1.173079, 1.129682, 1.320021])
matched_random_std  = np.array([1.092576, 1.953802, 1.537377])

# Figure 4a
fig, ax = plt.subplots(figsize=(6.8, 4.6))
x = np.arange(len(k_labels))
bars = ax.bar(x, topk_drop, color=POSITIVE, edgecolor=INK, linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(k_labels)
ax.set_ylabel("Behavioral degradation (drop)")
ax.set_xlabel(r"Ablated $\Delta CE$-selected heads")
ax.set_title(r"Top-$k$ causal ablation of selected attention heads")
for bar, value in zip(bars, topk_drop):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
            f"{value:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
ax.set_ylim(0, max(topk_drop) + 1.0)
ax.grid(axis="y", alpha=0.22)
fig.tight_layout()
fig.savefig(output_dir / "figure4a_topk_ablation.pdf", bbox_inches="tight")
fig.savefig(output_dir / "figure4a_topk_ablation.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure 4b: one supportive hue plus one neutral hue. Random controls are
# distinguished by hatching instead of introducing extra categorical colors.
fig, ax = plt.subplots(figsize=(8.2, 4.8))
x = np.arange(len(k_numeric))
width = 0.24

bars1 = ax.bar(x-width, topk_drop, width=width,
               label=r"$\Delta CE$ Top-$k$", color=POSITIVE,
               edgecolor=INK, linewidth=0.8)
bars2 = ax.bar(x, global_random_mean, width=width,
               yerr=global_random_std, capsize=4,
               label="Global Random", color="white",
               edgecolor=NEUTRAL, linewidth=1.0, hatch="///")
bars3 = ax.bar(x+width, matched_random_mean, width=width,
               yerr=matched_random_std, capsize=4,
               label="Matched Random", color=NEUTRAL,
               edgecolor=INK, linewidth=0.8, hatch="..")

ax.set_xticks(x)
ax.set_xticklabels([str(k) for k in k_numeric])
ax.set_xlabel("Number of ablated heads ($k$)")
ax.set_ylabel("Behavioral degradation (drop)")
ax.set_title(r"Top-$k$ ablation compared with random controls")
ax.legend()
ax.grid(axis="y", alpha=0.22)

def annotate_bars(container, values):
    for bar, value in zip(container, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.08,
                f"{value:.2f}", ha="center", va="bottom", fontsize=8, color=INK)

annotate_bars(bars1, topk_drop)
annotate_bars(bars2, global_random_mean)
annotate_bars(bars3, matched_random_mean)
upper_bound = max(np.max(topk_drop),
                  np.max(global_random_mean + global_random_std),
                  np.max(matched_random_mean + matched_random_std))
ax.set_ylim(0, upper_bound + 1.0)
fig.tight_layout()
fig.savefig(output_dir / "figure4b_random_controls.pdf", bbox_inches="tight")
fig.savefig(output_dir / "figure4b_random_controls.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Done.")

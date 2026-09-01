from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)


# ==========================
# Data
# ==========================

models = ["Base", "LoRA-v2"]

# Best constraint delta margin
margin = np.array([
    1.368485,
    5.954524
])


# Generation diagnostic
legal_rate = np.array([
    0.333,
    0.0
])

forbidden_rate = np.array([
    0.667,
    1.0
])


# ==========================
# Figure
# ==========================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(9,4)
)


# ---- Panel A ----

ax = axes[0]

x = np.arange(len(models))

bars = ax.bar(
    x,
    margin
)

ax.set_xticks(x)
ax.set_xticklabels(models)

ax.set_ylabel(
    "Best Constraint Delta Margin"
)

ax.set_title(
    "(a) Preference reallocation"
)


for bar, value in zip(bars, margin):
    ax.text(
        bar.get_x()+bar.get_width()/2,
        value+0.1,
        f"{value:.2f}",
        ha="center",
        fontsize=9
    )


ax.set_ylim(
    0,
    max(margin)+1
)


# ---- Panel B ----

ax = axes[1]

width = 0.35

x = np.arange(len(models))


bars1 = ax.bar(
    x-width/2,
    legal_rate,
    width,
    label="Legal action"
)


bars2 = ax.bar(
    x+width/2,
    forbidden_rate,
    width,
    label="Forbidden action"
)


ax.set_xticks(x)
ax.set_xticklabels(models)

ax.set_ylabel(
    "Generation rate"
)

ax.set_title(
    "(b) Direct generation diagnostic"
)


ax.legend()


ax.set_ylim(
    0,
    1.1
)


fig.tight_layout()


fig.savefig(
    output_dir/"figure5_preference_generation.pdf",
    bbox_inches="tight"
)

fig.savefig(
    output_dir/"figure5_preference_generation.png",
    dpi=300,
    bbox_inches="tight"
)


print("Figure 5 generated.")
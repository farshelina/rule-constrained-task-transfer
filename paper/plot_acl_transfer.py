from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)


# ======================
# Data
# ======================

models = [
    "DeepSeek",
    "Qwen2.5",
    "Llama3.1"
]


base = np.array([
    28.3,
    35.3,
    45.0
])

lora_v1 = np.array([
    31.7,
    54.0,
    42.3
])

lora_v2 = np.array([
    38.0,
    58.7,
    46.7
])


# ======================
# Plot
# ======================

fig, ax = plt.subplots(
    figsize=(8,4.8)
)


x = np.arange(len(models))

width = 0.25


bars1 = ax.bar(
    x-width,
    base,
    width,
    label="Base"
)

bars2 = ax.bar(
    x,
    lora_v1,
    width,
    label="LoRA-v1"
)

bars3 = ax.bar(
    x+width,
    lora_v2,
    width,
    label="LoRA-v2"
)


ax.set_xticks(x)
ax.set_xticklabels(models)

ax.set_ylabel(
    "ACL Bench Accuracy (%)"
)

ax.set_title(
    "Cross-task transfer performance after fine-tuning"
)


ax.legend()

ax.set_ylim(
    0,
    70
)


ax.grid(
    axis="y",
    alpha=0.3
)


def annotate(container):

    for bar in container:

        height = bar.get_height()

        ax.text(
            bar.get_x()+bar.get_width()/2,
            height+1,
            f"{height:.1f}",
            ha="center",
            fontsize=8
        )


annotate(bars1)
annotate(bars2)
annotate(bars3)


fig.tight_layout()


fig.savefig(
    output_dir/"figure6_acl_transfer.pdf",
    bbox_inches="tight"
)


fig.savefig(
    output_dir/"figure6_acl_transfer.png",
    dpi=300,
    bbox_inches="tight"
)


print("Figure 6 generated.")
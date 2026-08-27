# Website Update Notes — 2026-08-27

This version updates the interactive research website with the completed Top-k causal-ablation experiment.

## Main updates

- Updated paper title:
  - **Rule-Constrained Fine-Tuning and Task Transfer in Large Language Models: Behavioral Evaluation and Causal Analysis of Attention Heads**
- Added completed Top-k joint ablation results at:
  - `k = 5`
  - `k = 10`
  - `k = 27`
- Added two random controls:
  - **Global random**: randomly sample k heads from all 784 attention heads.
  - **Matched random**: randomly sample k heads while matching the layer-count distribution of the Top-k set.
- Added a grouped Top-k causal-validation chart to:
  - Method Journey → Metric & Causal Method
  - Causal Heads
- Added an interactive k selector with the exact recorded behavioral drops.
- Moved Top-k validation out of Future Work because the experiment is now completed.
- Updated Future Work to focus on:
  - ACL causal evaluation
  - clean-v1 / clean-v2 reruns
  - cross-model mechanism comparison
  - exploratory benchmark retesting

## Recorded Top-k results

| k | Top-k drop | Global random drop | Matched random drop |
|---:|---:|---:|---:|
| 5 | 7.489 | 0.281 | 1.173 |
| 10 | 6.987 | 0.186 | 1.130 |
| 27 | 5.281 | 2.638 | 1.320 |

At all tested k values, the Top-k behavioral drop is larger than both random controls. The effect is not monotonic in k, so the website interprets the result as evidence of selective functional importance rather than simple additive contributions across heads.

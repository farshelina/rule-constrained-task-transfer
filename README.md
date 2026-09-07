
# Interactive Research Website

## Title

Rule-Constrained Fine-Tuning and Task Transfer in Large Language Models:
Behavioral Evaluation and Causal Analysis of Attention Heads

## Live demo

**Interactive demo:** https://rule-constrained-task-transfer.streamlit.app/

The Streamlit app is deployed on Streamlit Community Cloud. The source code,
evaluation scripts, figures, and reproducibility materials are maintained in
this GitHub repository.

## Main pages

- Overview
- Method Journey
  - Research Route
  - Benchmark Exploration
  - Dataset Evolution
  - Metric & Causal Method
- Dataset
- ACL Bench
- Training
- Constraint Behavior
- Causal Heads
- Future Work

## New content in this version

- Public BBH pilot results
- Initial ACL pilot results
- Multi-hop ACL pilot results
- Benchmark branch structure: ACL as the first self-developed benchmark,
  with Multi-hop, Struct and Anomaly as exploratory extensions
- Real sample browser for six self-developed benchmark versions
- Thirteen retained dataset snapshots
- Record-level and field-level dataset iteration charts
- Four-stage behavior-metric evolution
- Triad screening summary
- Debug head scan
- Interactive Delta_CE calculation for every one of the 784 heads
- Completed Top-k joint ablation at k = 5, 10, 27
- Global random and layer-matched random controls for Top-k validation
- Original vs. ablated scores for Base and LoRA-v2
- Stable Windows launcher using 127.0.0.1:8501

## Run locally

Double-click:

```text
run_app.bat
```

The launcher waits until the Streamlit port is ready and then opens:

```text
http://127.0.0.1:8501
```

Keep the `Research Website Server` window open while using the site.

## Update later

Most result updates can be made by replacing CSV files in `data/`.
The completed Top-k results are stored in `data/topk_ablation.csv`.
The main page structure does not need to be rewritten.

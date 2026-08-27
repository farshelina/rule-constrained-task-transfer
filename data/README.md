
# Data inventory

## Current main results
- dataset_summary.csv
- constraint_pairs.csv
- acl_bench_questions.csv
- acl_results_v2.csv
- acl_predictions_v2.csv
- acl_results_v1_historical.csv
- training_summary.csv
- training_curves.csv
- behavior_summary.csv
- behavior_by_scenario.csv
- behavior_records.csv
- delta_ce.csv
- layer_summary.csv
- topk_ablation.csv

## Method Journey
- benchmark_catalog.csv
- benchmark_samples.csv
- pilot_benchmark_results.csv
- pilot_acl_category_profile.csv
- dataset_evolution.csv
- dataset_field_changes.csv
- metric_evolution.csv
- triad_summary.csv
- debug_head_scan.csv


## Top-k causal validation
`topk_ablation.csv` contains the completed joint-ablation comparison:
- `k`: number of jointly ablated heads
- `topk_drop`: drop after ablating the Delta_CE-ranked Top-k heads
- `global_random_drop`: drop after randomly sampling k heads from all 784 heads
- `matched_random_drop`: drop after sampling k heads while matching the Top-k layer-count distribution

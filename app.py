
from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "assets" / "figures"

FULL_TITLE = (
    "Rule-Constrained Fine-Tuning and Task Transfer in Large Language Models: "
    "Behavioral Evaluation and Causal Analysis of Attention Heads"
)

st.set_page_config(
    page_title=FULL_TITLE,
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1280px; padding-top: 2rem; padding-bottom: 4rem;}
      .hero {
        padding: 2.35rem 2.55rem;
        border-radius: 25px;
        background: linear-gradient(135deg, #edf1ff 0%, #f8faff 62%, #ffffff 100%);
        border: 1px solid #dbe3ff;
        margin-bottom: 1.5rem;
      }
      .hero h1 {font-size: 2.7rem; line-height: 1.12; margin: 0 0 0.8rem; color: #17213a;}
      .hero p {font-size: 1.08rem; color: #536079; max-width: 980px; margin: 0;}
      .tag {
        display:inline-block; padding:.28rem .72rem; border-radius:999px;
        background:#e8edff; color:#4058bd; font-weight:700; font-size:.82rem;
        margin-bottom:.75rem;
      }
      .card {
        border:1px solid #e0e6f1; border-radius:18px; padding:1.05rem 1.15rem;
        background:white; min-height:135px;
      }
      .card h4 {margin:0 0 .45rem; color:#17213a;}
      .card p {margin:0; color:#5c6780;}
      .flow {
        border:1px solid #dce3f2; border-radius:18px; padding:1rem 1.1rem;
        background:#ffffff; text-align:center; min-height:120px;
      }
      .flow strong {display:block; color:#1f2b46; margin-bottom:.35rem;}
      .flow span {color:#6b7690; font-size:.92rem;}
      .arrow {
        text-align:center; font-size:1.8rem; color:#7c89ad; padding-top:2.1rem;
      }
      .metric-card {
        border:1px solid #e0e6f1; border-radius:18px; padding:1rem 1.1rem;
        background:#fafbff; min-height:215px;
      }
      .metric-card h4 {margin:0 0 .45rem; color:#17213a;}
      .metric-card p {margin:.35rem 0; color:#5c6780; font-size:.92rem;}
      .pending {
        border:2px dashed #cdd6e8; border-radius:18px; padding:1.55rem;
        background:#fafbfe; min-height:180px;
      }
      .pending h4 {margin:0 0 .55rem; color:#46536f;}
      .pending p {margin:0; color:#76819a;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


@st.cache_data
def json_file(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def grouped_score_chart(frame: pd.DataFrame, title: str):
    figure = px.bar(
        frame,
        x="model",
        y="score",
        color="version",
        barmode="group",
        text_auto=".1f",
        labels={"model": "Model", "score": "Accuracy (%)", "version": "Checkpoint"},
        title=title,
    )
    figure.update_layout(
        yaxis_range=[0, 100],
        legend_title_text="Checkpoint",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return figure


st.markdown(
    """
    <div class="hero">
      <div class="tag">Interactive Research Record</div>
      <h1>
        Rule-Constrained Fine-Tuning and Task Transfer in Large Language Models:<br>
        Behavioral Evaluation and Causal Analysis of Attention Heads
      </h1>
      <p>
        The website connects benchmark exploration, training-data iteration,
        task-transfer evaluation, constraint-sensitive behavior and causal
        attention-head analysis in one interactive research narrative.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.image(str(FIGURES / "pipeline.svg"), width="stretch")

(
    overview_tab,
    journey_tab,
    data_tab,
    acl_tab,
    training_tab,
    behavior_tab,
    heads_tab,
    future_tab,
) = st.tabs(
    [
        "Overview",
        "Method Journey",
        "Dataset",
        "ACL Bench",
        "Training",
        "Constraint Behavior",
        "Causal Heads",
        "Future Work",
    ]
)

# ================================================================
# Overview
# ================================================================
with overview_tab:
    st.markdown('<div class="tag">Research Story</div>', unsafe_allow_html=True)
    st.header("Data → Transfer → Behavior → Mechanism")

    col1, col2, col3, col4 = st.columns(4)
    cards = [
        (
            col1,
            "Data Design",
            "Iteratively refine mixed instruction and Dou Dizhu decision data, "
            "then introduce action-level constraints.",
        ),
        (
            col2,
            "Task Transfer",
            "Evaluate whether fine-tuning affects abstract reasoning beyond the "
            "original card-game environment.",
        ),
        (
            col3,
            "Constraint Behavior",
            "Measure whether a prohibited action causes preference to move "
            "toward the strongest legal alternative.",
        ),
        (
            col4,
            "Causal Mechanism",
            "Ablate all 784 attention heads and compare causal contributions "
            "before and after LoRA fine-tuning.",
        ),
    ]
    for column, heading, body in cards:
        with column:
            st.markdown(
                f"""
                <div class="card">
                  <h4>{heading}</h4>
                  <p>{body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("Current interactive evidence")
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Final ACL questions", "300")
    e2.metric("Constraint probes", "60")
    e3.metric("Causal head scans", "784 × 2")
    e4.metric("Top-k settings", "5 / 10 / 27")
    e5.metric("Dataset snapshots", "13")

    st.markdown(
        """
        The main result pages focus on the current experiments. The
        **Method Journey** page records the pilot benchmarks, data iterations,
        metric revisions, full single-head scan and the completed Top-k causal-validation step.
        """
    )

# ================================================================
# Method Journey
# ================================================================
with journey_tab:
    st.markdown('<div class="tag">Research Process</div>', unsafe_allow_html=True)
    st.header("From Pilot Evaluation to Causal Analysis")

    route_tab, bench_tab, dataset_journey_tab, metric_tab = st.tabs(
        [
            "Research Route",
            "Benchmark Exploration",
            "Dataset Evolution",
            "Metric & Causal Method",
        ]
    )

    with route_tab:
        st.subheader("Two parallel development tracks")

        cols = st.columns([1.2, 0.35, 1.2, 0.35, 1.2])
        with cols[0]:
            st.markdown(
                """
                <div class="flow">
                  <strong>Public Benchmarks</strong>
                  <span>BBH subsets and an early public mathematical-reasoning evaluation</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)
        with cols[2]:
            st.markdown(
                """
                <div class="flow">
                  <strong>Initial ACL Bench</strong>
                  <span>The first self-developed benchmark and the central prototype</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with cols[3]:
            st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)
        with cols[4]:
            st.markdown(
                """
                <div class="flow">
                  <strong>Final ACL Bench</strong>
                  <span>Refined task set retained for the primary transfer experiments</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### Parallel benchmark extensions from the initial ACL design")
        b1, b2, b3 = st.columns(3)
        for column, heading, body in [
            (
                b1,
                "Multi-hop ACL",
                "Extends hierarchy reasoning into explicit 1-hop, 2-hop and "
                "3-hop transitive chains.",
            ),
            (
                b2,
                "Struct Bench",
                "Extends structural transfer into legality, abstraction and "
                "analogy across abstract and card-symbol domains.",
            ),
            (
                b3,
                "Anomaly Bench",
                "Extends pattern recognition into broken-straight, broken-pair "
                "and intruder-detection tasks.",
            ),
        ]:
            with column:
                st.markdown(
                    f"""
                    <div class="card">
                      <h4>{heading}</h4>
                      <p>{body}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("#### Data and analysis track")
        dcols = st.columns([1, 0.25, 1, 0.25, 1, 0.25, 1])
        labels = [
            ("Early mixed data", "Mixing, formatting and initial debugging"),
            ("Dataset refinement", "Cleaning and repeated output revisions"),
            ("Final v1 / v2", "Standard and constraint-injected corpora"),
            ("Behavior & causality", "Metric evolution, full head scan and Top-k validation"),
        ]
        for index, (heading, body) in enumerate(labels):
            with dcols[index * 2]:
                st.markdown(
                    f"""
                    <div class="flow">
                      <strong>{heading}</strong>
                      <span>{body}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            if index < len(labels) - 1:
                with dcols[index * 2 + 1]:
                    st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)

    with bench_tab:
        st.subheader("Pilot benchmark landscape")
        catalog = csv("benchmark_catalog.csv")
        st.dataframe(catalog, width="stretch", hide_index=True)

        pilot = csv("pilot_benchmark_results.csv")

        st.markdown("#### Public BBH pilot results")
        public_results = pilot[pilot["group"] == "Public BBH"]
        public_fig = px.bar(
            public_results,
            x="benchmark",
            y="score",
            color="checkpoint",
            barmode="group",
            text_auto=".1f",
            labels={
                "benchmark": "Task",
                "score": "Accuracy (%)",
                "checkpoint": "Checkpoint",
            },
        )
        public_fig.update_layout(
            yaxis_range=[0, 100],
            margin=dict(l=10, r=10, t=35, b=10),
        )
        st.plotly_chart(public_fig, width="stretch")

        left, right = st.columns(2)
        with left:
            st.markdown("#### Initial ACL pilot")
            early_acl = pilot[pilot["group"] == "Initial ACL"]
            fig = px.bar(
                early_acl,
                x="checkpoint",
                y="score",
                text_auto=".1f",
                labels={"checkpoint": "Checkpoint", "score": "Accuracy (%)"},
            )
            fig.update_layout(
                yaxis_range=[0, 100],
                margin=dict(l=10, r=10, t=35, b=10),
            )
            st.plotly_chart(fig, width="stretch")

        with right:
            st.markdown("#### Multi-hop ACL pilot")
            multihop_results = pilot[pilot["group"] == "Multi-hop ACL"]
            fig = px.bar(
                multihop_results,
                x="benchmark",
                y="score",
                color="checkpoint",
                barmode="group",
                text_auto=".1f",
                labels={
                    "benchmark": "Reasoning depth",
                    "score": "Accuracy (%)",
                    "checkpoint": "Checkpoint",
                },
            )
            fig.update_layout(
                yaxis_range=[0, 100],
                margin=dict(l=10, r=10, t=35, b=10),
            )
            st.plotly_chart(fig, width="stretch")

        st.caption(
            "Pilot results are displayed as records of early evaluation work. "
            "They use pilot-stage checkpoints and are not merged with the final "
            "v1/v2 comparisons."
        )

        st.subheader("Exploratory benchmark browser")
        samples = csv("benchmark_samples.csv")
        benchmark_name = st.selectbox(
            "Benchmark",
            samples["benchmark"].drop_duplicates().tolist(),
            key="journey_benchmark",
        )
        sample_subset = samples[samples["benchmark"] == benchmark_name].reset_index(
            drop=True
        )

        category_values = [
            value
            for value in sample_subset["category"].dropna().drop_duplicates().tolist()
            if str(value)
        ]
        chosen_category = st.selectbox(
            "Category",
            ["All"] + category_values,
            key="journey_category",
        )
        if chosen_category != "All":
            sample_subset = sample_subset[
                sample_subset["category"] == chosen_category
            ].reset_index(drop=True)

        sample_number = st.number_input(
            "Sample number",
            min_value=1,
            max_value=max(1, len(sample_subset)),
            value=1,
            step=1,
            key="journey_sample_number",
        )

        if not sample_subset.empty:
            sample = sample_subset.iloc[int(sample_number) - 1]
            st.markdown(
                f"**Category:** `{sample['category']}`"
                + (
                    f" &nbsp;&nbsp; **Mode:** `{sample['mode']}`"
                    if pd.notna(sample["mode"]) and str(sample["mode"])
                    else ""
                )
            )
            st.write(sample["question"])
            try:
                st.write(json.loads(sample["options"]))
            except Exception:
                st.write(sample["options"])
            with st.expander("Show reference answer"):
                st.success(f"Correct option: {sample['answer']}")

    with dataset_journey_tab:
        st.subheader("Thirteen retained dataset snapshots")
        evolution = csv("dataset_evolution.csv")
        fields = csv("dataset_field_changes.csv")

        c1, c2, c3 = st.columns(3)
        c1.metric("Snapshots", len(evolution))
        c2.metric(
            "Initial examples",
            f"{int(evolution.iloc[0]['total_examples']):,}",
        )
        c3.metric(
            "Final examples",
            f"{int(evolution.iloc[-1]['total_examples']):,}",
        )

        total_fig = px.line(
            evolution,
            x="version",
            y="total_examples",
            markers=True,
            hover_data=["phase", "general_examples", "doudizhu_examples"],
            labels={"version": "Dataset snapshot", "total_examples": "Examples"},
            title="Training-corpus size across retained snapshots",
        )
        total_fig.update_layout(
            xaxis_tickangle=-35,
            margin=dict(l=10, r=10, t=60, b=100),
        )
        st.plotly_chart(total_fig, width="stretch")

        changed = evolution.dropna(
            subset=["changed_records_from_previous"]
        ).copy()
        changed_fig = px.bar(
            changed,
            x="version",
            y="changed_records_from_previous",
            color="phase",
            text_auto=True,
            labels={
                "version": "Dataset snapshot",
                "changed_records_from_previous": "Changed records",
                "phase": "Iteration phase",
            },
            title="Record-level revisions from each preceding snapshot",
        )
        changed_fig.update_layout(
            xaxis_tickangle=-35,
            margin=dict(l=10, r=10, t=60, b=100),
        )
        st.plotly_chart(changed_fig, width="stretch")

        field_pivot = fields.dropna(
            subset=["changed_fields_from_previous"]
        )
        field_fig = px.bar(
            field_pivot,
            x="version",
            y="changed_fields_from_previous",
            color="field",
            barmode="stack",
            labels={
                "version": "Dataset snapshot",
                "changed_fields_from_previous": "Changed fields",
                "field": "Field",
            },
            title="Which fields changed during each iteration",
        )
        field_fig.update_layout(
            xaxis_tickangle=-35,
            margin=dict(l=10, r=10, t=60, b=100),
        )
        st.plotly_chart(field_fig, width="stretch")

        st.dataframe(
            evolution[
                [
                    "version",
                    "phase",
                    "total_examples",
                    "general_examples",
                    "doudizhu_examples",
                    "changed_records_from_previous",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

    with metric_tab:
        st.subheader("Behavior-metric evolution")
        metric_data = csv("metric_evolution.csv")

        stage_columns = st.columns(4)
        for stage in sorted(metric_data["stage"].unique()):
            stage_data = metric_data[metric_data["stage"] == stage]
            base_row = stage_data[stage_data["checkpoint"] == "Base"].iloc[0]
            lora_row = stage_data[stage_data["checkpoint"] == "LoRA-v2"].iloc[0]
            with stage_columns[int(stage) - 1]:
                st.markdown(
                    f"""
                    <div class="metric-card">
                      <h4>Stage {int(stage)} · {base_row['short_name']}</h4>
                      <p><strong>Base:</strong> {base_row['mean']:.3f} ± {base_row['std']:.3f}</p>
                      <p><strong>LoRA-v2:</strong> {lora_row['mean']:.3f} ± {lora_row['std']:.3f}</p>
                      <p>{base_row['definition']}</p>
                      <p><strong>Next:</strong> {base_row['next_step']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        selected_stage = st.selectbox(
            "Inspect metric stage",
            sorted(metric_data["stage"].unique()),
            format_func=lambda value: (
                f"Stage {int(value)} · "
                f"{metric_data[metric_data['stage'] == value]['short_name'].iloc[0]}"
            ),
        )
        selected_metric = metric_data[metric_data["stage"] == selected_stage]
        metric_fig = px.bar(
            selected_metric,
            x="checkpoint",
            y="mean",
            error_y="std",
            text_auto=".3f",
            labels={"checkpoint": "Checkpoint", "mean": "Metric mean"},
            title=selected_metric["metric"].iloc[0],
        )
        metric_fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(metric_fig, width="stretch")

        st.markdown("#### From attention correlation to causal intervention")
        triad = csv("triad_summary.csv")
        triad_fig = px.bar(
            triad,
            x="stage",
            y="heads",
            color="run",
            barmode="group",
            text_auto=True,
            labels={"stage": "Screening stage", "heads": "Heads retained", "run": "Run"},
            title="Triad attention-head screening",
        )
        triad_fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(triad_fig, width="stretch")

        st.markdown(
            """
            The correlation-oriented screening was followed by behavioral
            intervention: each attention head was zero-ablated and the selected
            behavior metric was recomputed.
            """
        )

        debug = csv("debug_head_scan.csv").copy()
        debug["head_label"] = debug.apply(
            lambda row: f"L{int(row['layer'])}-H{int(row['head'])}",
            axis=1,
        )
        debug = debug.sort_values("causal_effect")
        debug_fig = px.bar(
            debug,
            x="causal_effect",
            y="head_label",
            orientation="h",
            text_auto=".3f",
            labels={"causal_effect": "Causal effect", "head_label": "Head"},
            title="Debug scan: first 10 LoRA-v2 heads on 20 probes",
        )
        debug_fig.update_yaxes(type="category")
        debug_fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(debug_fig, width="stretch")

        s1, s2, s3 = st.columns(3)
        s1.metric("Debug scan", "10 heads × 20 probes")
        s2.metric("Full Base scan", "784 heads × 60 probes")
        s3.metric("Full LoRA scan", "784 heads × 60 probes")


        st.markdown("#### Top-k causal validation")
        topk_journey = csv("topk_ablation.csv")
        topk_long = topk_journey.melt(
            id_vars="k",
            value_vars=["topk_drop", "global_random_drop", "matched_random_drop"],
            var_name="condition",
            value_name="behavior_drop",
        )
        topk_long["condition"] = topk_long["condition"].map(
            {
                "topk_drop": "Top-k heads",
                "global_random_drop": "Global random",
                "matched_random_drop": "Matched random",
            }
        )
        topk_journey_fig = px.bar(
            topk_long,
            x="k",
            y="behavior_drop",
            color="condition",
            barmode="group",
            text_auto=".3f",
            labels={
                "k": "Number of ablated heads (k)",
                "behavior_drop": "Behavior-score drop",
                "condition": "Ablation set",
            },
            title="Top-k vs. global and layer-matched random controls",
        )
        topk_journey_fig.update_layout(
            margin=dict(l=10, r=10, t=60, b=10),
            xaxis=dict(tickmode="array", tickvals=topk_journey["k"].tolist()),
        )
        st.plotly_chart(topk_journey_fig, width="stretch")
        st.caption(
            "At every tested k, ablating the Delta_CE-ranked Top-k heads produces a "
            "larger drop than both random controls. Matched random preserves the "
            "Top-k layer-count distribution, providing a stricter control for layer location."
        )

# ================================================================
# Dataset
# ================================================================
with data_tab:
    st.markdown('<div class="tag">Final Training Data</div>', unsafe_allow_html=True)
    st.header("Mixed Fine-Tuning Corpora")

    summary = csv("dataset_summary.csv")
    v2 = summary[summary["dataset"] == "LoRA-v2 mixed training set"]
    counts = dict(zip(v2["component"], v2["count"]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total examples", f"{counts['Total examples']:,}")
    c2.metric("General instructions", f"{counts['General instruction examples']:,}")
    c3.metric("Dou Dizhu examples", f"{counts['Dou Dizhu examples']:,}")
    c4.metric("Constraint-injected", f"{counts['Constraint-injected examples']:,}")

    composition = pd.DataFrame(
        {
            "component": ["General instructions", "Dou Dizhu"],
            "count": [
                counts["General instruction examples"],
                counts["Dou Dizhu examples"],
            ],
        }
    )
    pie = px.pie(
        composition,
        names="component",
        values="count",
        hole=0.48,
        title="Composition of the final mixed corpus",
    )
    pie.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(pie, width="stretch")

    v1_col, v2_col = st.columns(2)
    with v1_col:
        st.markdown(
            """
            <div class="card">
              <h4>LoRA-v1 · mixed_train_v10_repaired</h4>
              <p>
                General instruction data mixed with standard Dou Dizhu
                decision examples.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with v2_col:
        st.markdown(
            """
            <div class="card">
              <h4>LoRA-v2 · mixed_train_v11_constraints</h4>
              <p>
                The same mixed-data structure, with action-level constraints
                introduced into a subset of Dou Dizhu decision examples.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Paired real training examples")
    pairs = csv("constraint_pairs.csv")
    pair_number = st.number_input(
        "Example number",
        min_value=1,
        max_value=len(pairs),
        value=1,
        step=1,
    )
    pair = pairs.iloc[int(pair_number) - 1]

    left, right = st.columns(2)
    with left:
        st.markdown("#### LoRA-v1 target")
        st.write(pair["v1_instruction"])
        st.code(pair["input"], language=None)
        st.success(pair["v1_output"])
    with right:
        st.markdown("#### LoRA-v2 constrained target")
        st.write(pair["v2_instruction"])
        st.code(pair["input"], language=None)
        st.success(pair["v2_output"])

# ================================================================
# ACL Bench
# ================================================================
with acl_tab:
    st.markdown('<div class="tag">Primary Evaluation</div>', unsafe_allow_html=True)
    st.header("Final Repaired ACL Bench")

    categories = csv("acl_categories.csv")
    category_columns = st.columns(3)
    for index, row in categories.iterrows():
        with category_columns[index]:
            st.markdown(
                f"""
                <div class="card">
                  <h4>{row['display_name']} · {int(row['count'])}</h4>
                  <p>{row['description']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(
        "The benchmark contains 300 four-choice instances: 100 per category."
    )

    results_v2 = csv("acl_results_v2.csv")
    overall_v2 = results_v2[results_v2["task"] == "Overall"]
    st.plotly_chart(
        grouped_score_chart(
            overall_v2,
            "Base vs. LoRA-v2 on the final repaired ACL Bench",
        ),
        width="stretch",
    )

    inspect_model = st.selectbox(
        "Category-level model",
        overall_v2["model"].drop_duplicates().tolist(),
    )
    category_results = results_v2[
        (results_v2["model"] == inspect_model)
        & (results_v2["task"] != "Overall")
    ].copy()
    category_results["task"] = category_results["task"].map(
        {
            "counter_logic": "Counter Logic",
            "hierarchy_logic": "Hierarchy Logic",
            "pattern_recognition": "Pattern Recognition",
        }
    )
    category_figure = px.bar(
        category_results,
        x="task",
        y="score",
        color="version",
        barmode="group",
        text_auto=".1f",
        labels={"task": "Category", "score": "Accuracy (%)", "version": "Checkpoint"},
        title=f"{inspect_model}: category-level accuracy",
    )
    category_figure.update_layout(
        yaxis_range=[0, 100],
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(category_figure, width="stretch")

    st.subheader("Question browser")
    questions = csv("acl_bench_questions.csv")
    selected_category = st.selectbox(
        "Question category",
        ["All"] + categories["category"].tolist(),
    )
    if selected_category == "All":
        filtered_questions = questions.reset_index(drop=True)
    else:
        filtered_questions = questions[
            questions["category"] == selected_category
        ].reset_index(drop=True)

    question_number = st.number_input(
        "Question number",
        min_value=1,
        max_value=len(filtered_questions),
        value=1,
        step=1,
        key="question_number",
    )
    question = filtered_questions.iloc[int(question_number) - 1]
    st.markdown(
        f"**Category:** `{question['category']}` &nbsp;&nbsp; "
        f"**ID:** `{question['id']}`"
    )
    st.write(question["question"])
    st.write(
        {
            "A": question["option_A"],
            "B": question["option_B"],
            "C": question["option_C"],
            "D": question["option_D"],
        }
    )
    with st.expander("Show reference answer"):
        st.success(f"Correct option: {question['answer']}")

    st.subheader("Model prediction browser")
    predictions = csv("acl_predictions_v2.csv")
    p1, p2, p3 = st.columns(3)
    with p1:
        prediction_model = st.selectbox(
            "Prediction model",
            predictions["model"].drop_duplicates().tolist(),
        )
    with p2:
        available_versions = (
            predictions[predictions["model"] == prediction_model]["version"]
            .drop_duplicates()
            .tolist()
        )
        prediction_version = st.selectbox("Prediction checkpoint", available_versions)
    with p3:
        result_filter = st.selectbox(
            "Prediction filter",
            ["All", "Correct only", "Incorrect only"],
        )

    filtered_predictions = predictions[
        (predictions["model"] == prediction_model)
        & (predictions["version"] == prediction_version)
    ].copy()
    if result_filter == "Correct only":
        filtered_predictions = filtered_predictions[
            filtered_predictions["is_correct"] == True
        ]
    elif result_filter == "Incorrect only":
        filtered_predictions = filtered_predictions[
            filtered_predictions["is_correct"] == False
        ]
    filtered_predictions = filtered_predictions.reset_index(drop=True)

    prediction_number = st.number_input(
        "Prediction number",
        min_value=1,
        max_value=max(1, len(filtered_predictions)),
        value=1,
        step=1,
        key="prediction_number",
    )
    if not filtered_predictions.empty:
        prediction = filtered_predictions.iloc[int(prediction_number) - 1]
        st.write(prediction["question"])
        st.write(
            {
                "A": prediction["option_A"],
                "B": prediction["option_B"],
                "C": prediction["option_C"],
                "D": prediction["option_D"],
            }
        )
        status = "Correct" if bool(prediction["is_correct"]) else "Incorrect"
        st.write(
            f"**Prediction:** {prediction['prediction']} · "
            f"**Reference:** {prediction['ground_truth']} · **{status}**"
        )
    else:
        st.warning("No predictions match the current filter.")

    with st.expander("LoRA-v1 evaluation results"):
        results_v1 = csv("acl_results_v1_historical.csv")
        overall_v1 = results_v1[results_v1["task"] == "Overall"]
        st.plotly_chart(
            grouped_score_chart(overall_v1, "Base vs. LoRA-v1 results"),
            width="stretch",
        )
        st.dataframe(
            overall_v1[["model", "version", "score"]].sort_values(
                ["model", "version"]
            ),
            width="stretch",
            hide_index=True,
        )

# ================================================================
# Training
# ================================================================
with training_tab:
    st.markdown('<div class="tag">Training Records</div>', unsafe_allow_html=True)
    st.header("Training Configuration and Curves")

    qwen_config = json_file("qwen_training_config.json")
    configuration_rows = []
    for key, value in qwen_config.items():
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        else:
            value = str(value)
        configuration_rows.append({"parameter": str(key), "value": value})

    st.dataframe(
        pd.DataFrame(configuration_rows).astype(str),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Preserved training histories")
    training_summary = csv("training_summary.csv")
    display_summary = training_summary.copy()
    for column in ["last_logged_loss", "train_loss", "runtime_hours"]:
        display_summary[column] = display_summary[column].round(4)
    st.dataframe(
        display_summary[
            [
                "run",
                "global_steps",
                "epochs",
                "last_logged_loss",
                "train_loss",
                "runtime_hours",
                "source",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    curves = csv("training_curves.csv")
    selected_run = st.selectbox(
        "Training curve",
        curves["run"].drop_duplicates().tolist(),
    )
    selected_curve = curves[curves["run"] == selected_run].sort_values("step").copy()
    selected_curve["smoothed_loss"] = (
        selected_curve["loss"].rolling(window=15, min_periods=1).mean()
    )

    loss_chart = px.line(
        selected_curve,
        x="step",
        y="smoothed_loss",
        labels={"step": "Training step", "smoothed_loss": "Smoothed loss"},
        title=f"{selected_run}: training loss",
    )
    loss_chart.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(loss_chart, width="stretch")

    lr_chart = px.line(
        selected_curve,
        x="step",
        y="learning_rate",
        labels={"step": "Training step", "learning_rate": "Learning rate"},
        title=f"{selected_run}: learning-rate schedule",
    )
    lr_chart.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(lr_chart, width="stretch")

# ================================================================
# Constraint behavior
# ================================================================
with behavior_tab:
    st.markdown('<div class="tag">Behavior</div>', unsafe_allow_html=True)
    st.header("Constraint-Response Analysis")

    with st.expander("How the behavior metric was developed"):
        metric_data = csv("metric_evolution.csv")
        st.dataframe(
            metric_data[
                [
                    "stage",
                    "short_name",
                    "checkpoint",
                    "mean",
                    "std",
                    "n",
                    "definition",
                ]
            ].round({"mean": 4, "std": 4}),
            width="stretch",
            hide_index=True,
        )

    behavior_summary = csv("behavior_summary.csv")
    behavior_chart = go.Figure()
    behavior_chart.add_trace(
        go.Bar(
            x=behavior_summary["version"],
            y=behavior_summary["mean"],
            error_y=dict(
                type="data",
                array=behavior_summary["std"],
                visible=True,
            ),
            text=[f"{value:.3f}" for value in behavior_summary["mean"]],
            textposition="outside",
        )
    )
    behavior_chart.update_layout(
        title="Best Constraint Delta Margin (mean ± SD, n=60)",
        xaxis_title="Checkpoint",
        yaxis_title="Metric score",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(behavior_chart, width="stretch")

    by_scenario = csv("behavior_by_scenario.csv")
    scenario_chart = px.bar(
        by_scenario,
        x="scenario_name",
        y="mean",
        color="version",
        barmode="group",
        error_y="std",
        labels={
            "scenario_name": "Probe scenario",
            "mean": "Mean metric score",
            "version": "Checkpoint",
        },
        title="Constraint response by probe scenario",
    )
    scenario_chart.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(scenario_chart, width="stretch")

    records = csv("behavior_records.csv")
    st.subheader("Real probe record browser")
    selected_scenario = st.selectbox(
        "Probe scenario",
        records["scenario_name"].drop_duplicates().tolist(),
    )
    scenario_records = records[
        records["scenario_name"] == selected_scenario
    ].copy()
    probe_ids = scenario_records["probe_id"].drop_duplicates().tolist()
    selected_probe = st.selectbox("Probe ID", probe_ids)
    probe_records = scenario_records[
        scenario_records["probe_id"] == selected_probe
    ].copy()

    reference = probe_records.iloc[0]
    st.write(
        {
            "forbidden_action": reference["forbidden_action"],
            "reference_allowed_action": reference["allowed_action"],
            "legal_alternatives": reference["legal_alternatives"],
            "response_prefix": reference["response_prefix"],
        }
    )
    st.dataframe(
        probe_records[
            [
                "version",
                "positive_best_allowed_action",
                "negative_best_allowed_action",
                "best_constraint_delta_margin",
                "metric_score",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

# ================================================================
# Causal heads
# ================================================================
with heads_tab:
    st.markdown('<div class="tag">Mechanism</div>', unsafe_allow_html=True)
    st.header("Qwen2.5 Attention-Head Causal Scan")

    delta = csv("delta_ce.csv")
    st.caption(
        "The scan covers all 28 × 28 = 784 attention heads. "
        "Delta_CE = CE_LoRA-v2 − CE_Base."
    )

    st.subheader("How Delta_CE is obtained")
    selected_head_label = st.selectbox(
        "Inspect a layer-head pair",
        [
            f"L{int(row.layer)}-H{int(row.head)}"
            for row in delta.itertuples()
        ],
    )
    layer_value = int(selected_head_label.split("-")[0][1:])
    head_value = int(selected_head_label.split("-")[1][1:])
    selected_head = delta[
        (delta["layer"] == layer_value) & (delta["head"] == head_value)
    ].iloc[0]

    base_original = selected_head["orig_base"]
    base_ablated = selected_head["ablated_base"]
    lora_original = selected_head["orig_lora"]
    lora_ablated = selected_head["ablated_lora"]
    ce_base = selected_head["ce_base"]
    ce_lora = selected_head["ce_lora"]
    delta_ce_value = selected_head["delta_ce"]

    row1 = st.columns(3)
    row1[0].metric("Base original", f"{base_original:.4f}")
    row1[1].metric("Base after ablation", f"{base_ablated:.4f}")
    row1[2].metric("CE Base", f"{ce_base:.4f}")

    row2 = st.columns(3)
    row2[0].metric("LoRA-v2 original", f"{lora_original:.4f}")
    row2[1].metric("LoRA-v2 after ablation", f"{lora_ablated:.4f}")
    row2[2].metric("CE LoRA-v2", f"{ce_lora:.4f}")

    st.markdown(
        f"""
        **CE Base** = {base_original:.4f} − {base_ablated:.4f}
        = **{ce_base:.4f}**

        **CE LoRA-v2** = {lora_original:.4f} − {lora_ablated:.4f}
        = **{ce_lora:.4f}**

        **Delta_CE** = {ce_lora:.4f} − {ce_base:.4f}
        = **{delta_ce_value:.4f}**
        """
    )

    calculation_frame = pd.DataFrame(
        [
            {
                "Checkpoint": "Base",
                "Original": base_original,
                "After ablation": base_ablated,
                "Causal effect": ce_base,
            },
            {
                "Checkpoint": "LoRA-v2",
                "Original": lora_original,
                "After ablation": lora_ablated,
                "Causal effect": ce_lora,
            },
        ]
    )
    calc_fig = px.bar(
        calculation_frame.melt(
            id_vars="Checkpoint",
            value_vars=["Original", "After ablation"],
            var_name="Condition",
            value_name="Behavior score",
        ),
        x="Checkpoint",
        y="Behavior score",
        color="Condition",
        barmode="group",
        text_auto=".3f",
        title=f"{selected_head_label}: original vs. ablated behavior score",
    )
    calc_fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(calc_fig, width="stretch")

    st.subheader("Delta_CE ranking")
    top_n = st.slider(
        "Number of top heads",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
    )
    top_heads = delta.nlargest(top_n, "delta_ce").sort_values("delta_ce").copy()
    top_heads["head_label"] = top_heads.apply(
        lambda row: f"L{int(row['layer'])}-H{int(row['head'])}",
        axis=1,
    )

    ranking_chart = px.bar(
        top_heads,
        x="delta_ce",
        y="head_label",
        orientation="h",
        text_auto=".3f",
        labels={"delta_ce": "Delta_CE", "head_label": "Layer-Head"},
        title=f"Top {top_n} heads by Delta_CE",
        category_orders={"head_label": top_heads["head_label"].tolist()},
        hover_data={
            "layer": True,
            "head": True,
            "ce_base": ":.4f",
            "ce_lora": ":.4f",
            "delta_ce": ":.4f",
            "head_label": False,
        },
    )
    ranking_chart.update_yaxes(
        type="category",
        categoryorder="array",
        categoryarray=top_heads["head_label"].tolist(),
        title_text="Layer-Head",
    )
    ranking_chart.update_xaxes(title_text="Delta_CE")
    ranking_chart.update_layout(
        height=max(440, top_n * 25),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(ranking_chart, width="stretch")

    metric_name = st.selectbox(
        "Heatmap metric",
        ["Delta_CE", "Base causal effect", "LoRA-v2 causal effect"],
    )
    metric_column = {
        "Delta_CE": "delta_ce",
        "Base causal effect": "ce_base",
        "LoRA-v2 causal effect": "ce_lora",
    }[metric_name]

    heatmap_data = (
        delta.pivot(index="layer", columns="head", values=metric_column)
        .sort_index()
    )
    heatmap = px.imshow(
        heatmap_data,
        aspect="auto",
        labels={"x": "Head", "y": "Layer", "color": metric_name},
        title=f"28 × 28 heatmap: {metric_name}",
    )
    heatmap.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(heatmap, width="stretch")

    st.subheader("Layer-range summary")
    st.dataframe(csv("layer_summary.csv"), width="stretch", hide_index=True)

    st.subheader("Top-head table")
    top_table = delta.nlargest(20, "delta_ce")[
        [
            "layer",
            "head",
            "orig_base",
            "ablated_base",
            "ce_base",
            "orig_lora",
            "ablated_lora",
            "ce_lora",
            "delta_ce",
        ]
    ].copy()
    top_table.insert(
        0,
        "head_id",
        top_table.apply(
            lambda row: f"L{int(row['layer'])}-H{int(row['head'])}",
            axis=1,
        ),
    )
    for column in [
        "orig_base",
        "ablated_base",
        "ce_base",
        "orig_lora",
        "ablated_lora",
        "ce_lora",
        "delta_ce",
    ]:
        top_table[column] = top_table[column].round(4)
    st.dataframe(top_table, width="stretch", hide_index=True)

    st.subheader("Top-k causal validation")
    st.write(
        "Single-head Delta_CE ranks heads independently. The completed Top-k experiment "
        "tests whether the highest-ranked heads remain disproportionately important when "
        "removed jointly, and compares them with two random controls."
    )

    topk = csv("topk_ablation.csv")
    topk_long = topk.melt(
        id_vars="k",
        value_vars=["topk_drop", "global_random_drop", "matched_random_drop"],
        var_name="condition",
        value_name="behavior_drop",
    )
    topk_long["condition"] = topk_long["condition"].map(
        {
            "topk_drop": "Top-k heads",
            "global_random_drop": "Global random",
            "matched_random_drop": "Matched random",
        }
    )

    topk_fig = px.bar(
        topk_long,
        x="k",
        y="behavior_drop",
        color="condition",
        barmode="group",
        text_auto=".3f",
        labels={
            "k": "Number of ablated heads (k)",
            "behavior_drop": "Drop in constraint-sensitive behavior",
            "condition": "Ablation set",
        },
        title="Cumulative causal validation: ranked heads vs. random controls",
    )
    topk_fig.update_layout(
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(tickmode="array", tickvals=topk["k"].tolist()),
    )
    st.plotly_chart(topk_fig, width="stretch")

    k_choice = st.selectbox(
        "Inspect one Top-k setting",
        topk["k"].tolist(),
        format_func=lambda value: f"k = {int(value)}",
        key="topk_inspect_k",
    )
    topk_row = topk[topk["k"] == k_choice].iloc[0]
    t1, t2, t3 = st.columns(3)
    t1.metric("Top-k drop", f"{topk_row['topk_drop']:.3f}")
    t2.metric("Global random drop", f"{topk_row['global_random_drop']:.3f}")
    t3.metric("Matched random drop", f"{topk_row['matched_random_drop']:.3f}")

    control1, control2 = st.columns(2)
    with control1:
        st.markdown(
            """
            <div class="card">
              <h4>Global random control</h4>
              <p>Randomly sample k heads from the complete 784-head model. This tests whether removing any k heads would cause a similar drop.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with control2:
        st.markdown(
            """
            <div class="card">
              <h4>Matched random control</h4>
              <p>Randomly sample k heads while matching the Top-k layer-count distribution. This controls for the possibility that the effect is explained only by layer location.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.success(
        "Across k = 5, 10 and 27, Top-k ablation produces a larger behavioral drop "
        "than both Global random and Matched random controls. This strengthens the "
        "evidence that the Delta_CE-ranked heads form a functionally important subset "
        "for the measured constraint-sensitive behavior."
    )
    st.caption(
        "The Top-k effect is not monotonic in k, so the result is interpreted as evidence "
        "of selective functional importance rather than simple additive head contributions."
    )

# ================================================================
# Future work
# ================================================================
with future_tab:
    st.markdown('<div class="tag">Next Experiments</div>', unsafe_allow_html=True)
    st.header("Next Steps After Top-k Causal Validation")

    st.success(
        "Top-k joint ablation and both random-control comparisons are now recorded as "
        "completed experiments. The next goal is to connect this causal subset more "
        "directly to task transfer and to strengthen controlled training comparisons."
    )

    f1, f2 = st.columns(2)
    with f1:
        st.markdown(
            """
            <div class="pending">
              <h4>ACL causal evaluation</h4>
              <p>Re-evaluate ACL Bench after ablating the identified Top-k heads to test whether the same causal subset also supports cross-task transfer.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            """
            <div class="pending">
              <h4>Clean v1 / v2 reruns</h4>
              <p>Retrain matched standard and constraint-injected datasets under a unified pipeline to isolate the incremental effect of constraint examples.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    f3, f4 = st.columns(2)
    with f3:
        st.markdown(
            """
            <div class="pending">
              <h4>Cross-model mechanism comparison</h4>
              <p>Extend the causal mechanism analysis to additional model families and compare whether similar functional patterns emerge after fine-tuning.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f4:
        st.markdown(
            """
            <div class="pending">
              <h4>Exploratory benchmark retest</h4>
              <p>Re-evaluate Multi-hop, Struct and Anomaly benchmark branches with the final training and evaluation pipeline.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

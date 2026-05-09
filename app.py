from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DATA_PATH = Path(__file__).with_name("illness_dataset.csv")

PRIMARY = "#0f766e"
ACCENT = "#f97316"
INK = "#0f172a"
MUTED = "#64748b"
SURFACE = "#f8fafc"
COLOR_SEQUENCE = [
    "#0f766e",
    "#f97316",
    "#4f46e5",
    "#be123c",
    "#0891b2",
    "#7c3aed",
    "#16a34a",
    "#ca8a04",
]

LABEL_OVERRIDES = {
    "hot _flashes": "Hot flashes",
    "flact_affect": "Flat affect",
    "strong_carvings": "Strong cravings",
    "repetitve_behaviors": "Repetitive behaviors",
    "required_perfectness": "Required perfection",
    "anger_outbrusts": "Anger outbursts",
    "persisitent_worry_about_losing_attachement_figures": (
        "Persistent worry about losing attachment figures"
    ),
    "excessive_distress_when_anticipating_sepration": (
        "Excessive distress when anticipating separation"
    ),
    "nightmares_involving_sepration_themes": "Nightmares involving separation themes",
    "panic_symptoms_during_sepration": "Panic symptoms during separation",
    "clingy_behavior_and_need_for_constant_reassuarance": (
        "Clingy behavior and need for constant reassurance"
    ),
    "neglecting_responsibilites": "Neglecting responsibilities",
    "giving_up_important_activites": "Giving up important activities",
    "low_self-esteem": "Low self-esteem",
    "recurrent_suicidal_behavior,self_harm": (
        "Recurrent suicidal behavior / self-harm"
    ),
}


st.set_page_config(
    page_title="Mental Health Symptoms Dashboard | Hieu Nguyen",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --primary: #0f766e;
                --accent: #f97316;
                --ink: #0f172a;
                --muted: #64748b;
                --surface: #f8fafc;
                --line: #e2e8f0;
            }

            .block-container {
                padding-top: 1.75rem;
                padding-bottom: 2rem;
                max-width: 1440px;
            }

            h1, h2, h3 {
                letter-spacing: 0;
                color: var(--ink);
            }

            .app-hero {
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 24px 26px;
                background:
                    linear-gradient(135deg, rgba(15, 118, 110, .08), rgba(249, 115, 22, .08)),
                    #ffffff;
                margin-bottom: 1.25rem;
            }

            .app-eyebrow {
                color: var(--primary);
                font-weight: 700;
                font-size: .78rem;
                letter-spacing: .08em;
                text-transform: uppercase;
                margin-bottom: .35rem;
            }

            .app-title {
                font-size: clamp(2rem, 4vw, 3.35rem);
                line-height: 1.03;
                font-weight: 800;
                margin: 0 0 .55rem 0;
                letter-spacing: 0;
            }

            .app-subtitle {
                color: var(--muted);
                font-size: 1.04rem;
                max-width: 900px;
                margin: 0;
            }

            .creator-line {
                display: inline-flex;
                align-items: center;
                gap: .5rem;
                margin-top: 1rem;
                color: var(--ink);
                font-weight: 700;
            }

            .creator-dot {
                width: 8px;
                height: 8px;
                border-radius: 999px;
                background: var(--accent);
                display: inline-block;
            }

            .note-box {
                border-left: 4px solid var(--accent);
                background: #fff7ed;
                color: #7c2d12;
                border-radius: 6px;
                padding: .75rem .9rem;
                font-size: .92rem;
                margin: .6rem 0 1rem 0;
            }

            .section-label {
                color: var(--muted);
                font-size: .78rem;
                font-weight: 700;
                letter-spacing: .08em;
                text-transform: uppercase;
                margin: .4rem 0 .2rem;
            }

            .footer {
                margin-top: 2.25rem;
                padding-top: 1rem;
                border-top: 1px solid var(--line);
                color: var(--muted);
                font-size: .9rem;
            }

            div[data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 14px 16px;
                box-shadow: 0 1px 2px rgba(15, 23, 42, .04);
            }

            div[data-testid="stMetricLabel"] p {
                color: var(--muted);
                font-weight: 700;
            }

            div[data-testid="stMetricValue"] {
                color: var(--ink);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def display_label(name: str) -> str:
    if name in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[name]
    label = name.replace(",", " / ").replace("_", " ").replace("-", " ")
    return " ".join(label.split()).title()


def compact_number(value: float | int) -> str:
    if abs(float(value)) >= 1_000:
        return f"{value:,.0f}"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def format_pct(value: float) -> str:
    return f"{value:.1f}%"


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    df.columns = [column.strip() for column in df.columns]

    if "Disease" not in df.columns:
        raise ValueError("The dataset must include a 'Disease' column.")

    df["Disease"] = df["Disease"].astype(str).str.strip()
    symptom_cols = [column for column in df.columns if column != "Disease"]
    symptom_frame = (
        df[symptom_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .clip(lower=0, upper=1)
        .astype(int)
    )
    df = pd.concat([df[["Disease"]], symptom_frame], axis=1)

    df["Active symptom count"] = df[symptom_cols].sum(axis=1)
    return df, symptom_cols


def filter_data(
    df: pd.DataFrame,
    conditions: Iterable[str],
    symptom_cols: list[str],
    symptom_focus: list[str],
    count_range: tuple[int, int],
    require_all_focus: bool,
) -> pd.DataFrame:
    filtered = df[df["Disease"].isin(list(conditions))].copy()
    filtered = filtered[
        filtered["Active symptom count"].between(count_range[0], count_range[1])
    ]

    valid_focus = [symptom for symptom in symptom_focus if symptom in symptom_cols]
    if valid_focus:
        focus_matrix = filtered[valid_focus].eq(1)
        if require_all_focus:
            filtered = filtered[focus_matrix.all(axis=1)]
        else:
            filtered = filtered[focus_matrix.any(axis=1)]

    return filtered


def make_empty_chart(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 16, "color": MUTED},
    )
    fig.update_layout(
        height=360,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 30, "b": 30},
    )
    return fig


def style_chart(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=COLOR_SEQUENCE,
        font={"family": "Inter, Segoe UI, Arial, sans-serif", "color": INK},
        margin={"l": 20, "r": 20, "t": 52, "b": 35},
        legend_title_text="",
        title_font={"size": 19, "color": INK},
        hoverlabel={"bgcolor": "white", "font_size": 13},
    )
    fig.update_xaxes(
        gridcolor="#e2e8f0",
        linecolor="#cbd5e1",
        zerolinecolor="#e2e8f0",
        title_font={"color": MUTED},
        tickfont={"color": MUTED},
    )
    fig.update_yaxes(
        gridcolor="#e2e8f0",
        linecolor="#cbd5e1",
        zerolinecolor="#e2e8f0",
        title_font={"color": MUTED},
        tickfont={"color": MUTED},
    )
    return fig


def symptom_prevalence_frame(
    df: pd.DataFrame, symptom_cols: list[str], top_n: int = 15
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Symptom", "Records", "Prevalence"])

    counts = df[symptom_cols].sum().sort_values(ascending=False).head(top_n)
    frame = counts.rename("Records").reset_index().rename(columns={"index": "Symptom"})
    frame["Prevalence"] = frame["Records"] / len(df) * 100
    frame["Symptom label"] = frame["Symptom"].map(display_label)
    return frame


def condition_profile(
    df: pd.DataFrame, symptom_cols: list[str], condition: str, top_n: int = 15
) -> pd.DataFrame:
    condition_df = df[df["Disease"] == condition]
    if condition_df.empty:
        return pd.DataFrame()

    condition_rate = condition_df[symptom_cols].mean() * 100
    baseline_rate = df[symptom_cols].mean() * 100
    profile = pd.DataFrame(
        {
            "Symptom": symptom_cols,
            "Condition prevalence": condition_rate.values,
            "Population prevalence": baseline_rate.values,
        }
    )
    profile["Lift"] = profile["Condition prevalence"] - profile["Population prevalence"]
    profile["Symptom label"] = profile["Symptom"].map(display_label)
    return profile.sort_values("Condition prevalence", ascending=False).head(top_n)


def distinctiveness_profile(
    df: pd.DataFrame, symptom_cols: list[str], condition: str, top_n: int = 15
) -> pd.DataFrame:
    condition_df = df[df["Disease"] == condition]
    other_df = df[df["Disease"] != condition]
    if condition_df.empty or other_df.empty:
        return pd.DataFrame()

    condition_rate = condition_df[symptom_cols].mean() * 100
    other_rate = other_df[symptom_cols].mean() * 100
    frame = pd.DataFrame(
        {
            "Symptom": symptom_cols,
            "Condition prevalence": condition_rate.values,
            "Other conditions prevalence": other_rate.values,
        }
    )
    frame["Lift vs others"] = (
        frame["Condition prevalence"] - frame["Other conditions prevalence"]
    )
    frame["Symptom label"] = frame["Symptom"].map(display_label)
    frame = frame[frame["Condition prevalence"] > 0]
    return frame.sort_values("Lift vs others", ascending=False).head(top_n)


def condition_distribution_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return make_empty_chart("No records match the current filters.")

    counts = (
        df["Disease"]
        .value_counts()
        .rename_axis("Condition")
        .reset_index(name="Records")
        .sort_values("Records", ascending=True)
    )
    fig = px.bar(
        counts,
        x="Records",
        y="Condition",
        orientation="h",
        title="Condition distribution",
        color_discrete_sequence=[PRIMARY],
        text="Records",
    )
    fig.update_traces(
        marker_line_width=0,
        texttemplate="%{text:,}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Records: %{x:,}<extra></extra>",
    )
    fig.update_layout(yaxis_title="", xaxis_title="Records")
    return style_chart(fig, height=max(420, 26 * len(counts) + 90))


def top_symptoms_chart(df: pd.DataFrame, symptom_cols: list[str], top_n: int) -> go.Figure:
    symptoms = symptom_prevalence_frame(df, symptom_cols, top_n)
    if symptoms.empty:
        return make_empty_chart("No symptoms to display for the current filters.")

    symptoms = symptoms.sort_values("Records", ascending=True)
    fig = px.bar(
        symptoms,
        x="Prevalence",
        y="Symptom label",
        orientation="h",
        title=f"Top {len(symptoms)} symptoms by prevalence",
        color="Prevalence",
        color_continuous_scale=["#ccfbf1", PRIMARY],
        text=symptoms["Prevalence"].map(format_pct),
    )
    fig.update_traces(
        marker_line_width=0,
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Prevalence: %{x:.1f}%<extra></extra>",
    )
    fig.update_layout(
        yaxis_title="",
        xaxis_title="Records with symptom",
        coloraxis_showscale=False,
    )
    return style_chart(fig, height=max(420, 28 * len(symptoms) + 95))


def symptom_load_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return make_empty_chart("No symptom-load data after filtering.")

    fig = px.histogram(
        df,
        x="Active symptom count",
        nbins=24,
        title="Symptom load across records",
        color_discrete_sequence=[ACCENT],
    )
    fig.update_traces(
        marker_line_width=0,
        hovertemplate="Active symptoms: %{x}<br>Records: %{y:,}<extra></extra>",
    )
    fig.update_layout(yaxis_title="Records", xaxis_title="Active symptoms per record")
    return style_chart(fig, height=380)


def class_balance_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return make_empty_chart("No class-balance data after filtering.")

    summary = (
        df.groupby("Disease")["Active symptom count"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    fig = px.scatter(
        summary,
        x="count",
        y="mean",
        size="median",
        color="Disease",
        title="Condition volume vs average symptom load",
        hover_name="Disease",
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_traces(
        marker={"opacity": 0.86, "line": {"width": 1, "color": "white"}},
        hovertemplate=(
            "<b>%{hovertext}</b><br>Records: %{x:,}<br>"
            "Avg active symptoms: %{y:.1f}<extra></extra>"
        ),
    )
    fig.update_layout(
        xaxis_title="Records",
        yaxis_title="Average active symptoms",
        showlegend=False,
    )
    return style_chart(fig, height=380)


def condition_profile_charts(
    df: pd.DataFrame, symptom_cols: list[str], condition: str
) -> tuple[go.Figure, go.Figure, go.Figure, pd.DataFrame]:
    top_profile = condition_profile(df, symptom_cols, condition, top_n=15)
    distinct = distinctiveness_profile(df, symptom_cols, condition, top_n=15)

    if top_profile.empty:
        empty = make_empty_chart("Select a condition with matching records.")
        return empty, empty, empty, top_profile

    profile_chart = px.bar(
        top_profile.sort_values("Condition prevalence", ascending=True),
        x="Condition prevalence",
        y="Symptom label",
        orientation="h",
        title=f"Most common symptoms: {condition}",
        color_discrete_sequence=[PRIMARY],
        text=top_profile.sort_values("Condition prevalence", ascending=True)[
            "Condition prevalence"
        ].map(format_pct),
    )
    profile_chart.update_traces(
        marker_line_width=0,
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Condition prevalence: %{x:.1f}%<extra></extra>",
    )
    profile_chart.update_layout(yaxis_title="", xaxis_title="Prevalence in condition")

    distinct_chart = px.bar(
        distinct.sort_values("Lift vs others", ascending=True),
        x="Lift vs others",
        y="Symptom label",
        orientation="h",
        title="Symptoms that separate this condition",
        color="Lift vs others",
        color_continuous_scale=["#e0f2fe", "#38bdf8", PRIMARY],
        text=distinct.sort_values("Lift vs others", ascending=True)[
            "Lift vs others"
        ].map(lambda value: f"+{value:.1f}"),
    )
    distinct_chart.update_traces(
        marker_line_width=0,
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Lift vs other conditions: %{x:.1f} pts<extra></extra>",
    )
    distinct_chart.update_layout(
        yaxis_title="",
        xaxis_title="Percentage-point lift",
        coloraxis_showscale=False,
    )

    top_symptoms = top_profile["Symptom"].head(10).tolist()
    condition_df = df[df["Disease"] == condition]
    matrix = condition_df[top_symptoms].T.dot(condition_df[top_symptoms]) / max(
        len(condition_df), 1
    )
    matrix = matrix * 100
    heatmap = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=[display_label(column) for column in matrix.columns],
            y=[display_label(index) for index in matrix.index],
            colorscale=[[0, "#f8fafc"], [0.5, "#99f6e4"], [1, PRIMARY]],
            hovertemplate="<b>%{y}</b> with <b>%{x}</b><br>Co-occurrence: %{z:.1f}%<extra></extra>",
            colorbar={"title": "%"},
        )
    )
    heatmap.update_layout(title="Co-occurrence among top symptoms")

    table_frame = top_profile.copy()
    for column in ["Condition prevalence", "Population prevalence", "Lift"]:
        table_frame[column] = table_frame[column].map(lambda value: f"{value:.1f}%")
    table_frame = table_frame[
        ["Symptom label", "Condition prevalence", "Population prevalence", "Lift"]
    ].rename(
        columns={
            "Symptom label": "Symptom",
            "Condition prevalence": "Condition",
            "Population prevalence": "Dataset",
        }
    )

    return (
        style_chart(profile_chart, height=500),
        style_chart(distinct_chart, height=500),
        style_chart(heatmap, height=520),
        table_frame,
    )


def symptom_explorer(
    df: pd.DataFrame, symptom_cols: list[str], selected_symptom: str, bundle: list[str]
) -> tuple[go.Figure, go.Figure, pd.DataFrame]:
    condition_rates = (
        df.groupby("Disease")[selected_symptom]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .reset_index(name="Prevalence")
    )
    condition_rates = condition_rates[condition_rates["Prevalence"] > 0]

    if condition_rates.empty:
        condition_chart = make_empty_chart("This symptom does not appear in the data.")
    else:
        condition_chart = px.bar(
            condition_rates.sort_values("Prevalence", ascending=True),
            x="Prevalence",
            y="Disease",
            orientation="h",
            title=f"Conditions associated with {display_label(selected_symptom)}",
            color="Prevalence",
            color_continuous_scale=["#ffedd5", ACCENT],
            text=condition_rates.sort_values("Prevalence", ascending=True)[
                "Prevalence"
            ].map(format_pct),
        )
        condition_chart.update_traces(
            marker_line_width=0,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Prevalence: %{x:.1f}%<extra></extra>",
        )
        condition_chart.update_layout(
            yaxis_title="",
            xaxis_title="Condition prevalence",
            coloraxis_showscale=False,
        )
        condition_chart = style_chart(
            condition_chart, height=max(430, 26 * len(condition_rates) + 100)
        )

    selected_rows = df[df[selected_symptom] == 1]
    related = pd.DataFrame()
    if not selected_rows.empty:
        related_rate = selected_rows[symptom_cols].mean() * 100
        baseline_rate = df[symptom_cols].mean() * 100
        related = pd.DataFrame(
            {
                "Symptom": symptom_cols,
                "With selected symptom": related_rate.values,
                "Dataset baseline": baseline_rate.values,
            }
        )
        related = related[related["Symptom"] != selected_symptom]
        related["Lift"] = related["With selected symptom"] - related["Dataset baseline"]
        related["Symptom label"] = related["Symptom"].map(display_label)
        related = related.sort_values("Lift", ascending=False).head(15)

    if related.empty:
        related_chart = make_empty_chart("No related symptoms to display.")
    else:
        related_chart = px.bar(
            related.sort_values("Lift", ascending=True),
            x="Lift",
            y="Symptom label",
            orientation="h",
            title="Symptoms most overrepresented with this selection",
            color="Lift",
            color_continuous_scale=["#dbeafe", "#60a5fa", PRIMARY],
            text=related.sort_values("Lift", ascending=True)["Lift"].map(
                lambda value: f"+{value:.1f}"
            ),
        )
        related_chart.update_traces(
            marker_line_width=0,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Lift: %{x:.1f} pts<extra></extra>",
        )
        related_chart.update_layout(
            yaxis_title="",
            xaxis_title="Percentage-point lift",
            coloraxis_showscale=False,
        )
        related_chart = style_chart(related_chart, height=500)

    valid_bundle = [symptom for symptom in bundle if symptom in symptom_cols]
    if valid_bundle:
        mask = df[valid_bundle].eq(1).all(axis=1)
        bundle_rates = (
            df[mask]["Disease"]
            .value_counts()
            .rename_axis("Condition")
            .reset_index(name="Matching records")
        )
        bundle_rates["Share of bundle"] = (
            bundle_rates["Matching records"] / max(bundle_rates["Matching records"].sum(), 1)
        ) * 100
    else:
        bundle_rates = pd.DataFrame(columns=["Condition", "Matching records", "Share of bundle"])

    return condition_chart, related_chart, bundle_rates


def prediction_scores(
    df: pd.DataFrame, symptom_cols: list[str], selected_symptoms: list[str]
) -> pd.DataFrame:
    valid_symptoms = [symptom for symptom in selected_symptoms if symptom in symptom_cols]
    if not valid_symptoms:
        return pd.DataFrame()

    profile = df.groupby("Disease")[symptom_cols].mean()
    intersection = profile[valid_symptoms].sum(axis=1)
    selected_coverage = intersection / len(valid_symptoms)
    condition_signature_size = profile.sum(axis=1).replace(0, np.nan)
    soft_jaccard = intersection / (
        condition_signature_size + len(valid_symptoms) - intersection
    )
    score = (0.58 * selected_coverage + 0.42 * soft_jaccard).fillna(0)

    results = pd.DataFrame(
        {
            "Condition": score.index,
            "Match score": score.values * 100,
            "Selected symptom coverage": selected_coverage.values * 100,
            "Signature fit": soft_jaccard.fillna(0).values * 100,
        }
    )
    return results.sort_values("Match score", ascending=False)


def prediction_chart(results: pd.DataFrame) -> go.Figure:
    if results.empty:
        return make_empty_chart("Choose symptoms to generate a similarity ranking.")

    top_results = results.head(8).sort_values("Match score", ascending=True)
    fig = px.bar(
        top_results,
        x="Match score",
        y="Condition",
        orientation="h",
        title="Top condition similarity scores",
        color="Match score",
        color_continuous_scale=["#ccfbf1", PRIMARY],
        text=top_results["Match score"].map(format_pct),
    )
    fig.update_traces(
        marker_line_width=0,
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b><br>Match score: %{x:.1f}%<br>"
            "This is an analytical similarity score, not a diagnosis.<extra></extra>"
        ),
    )
    fig.update_layout(
        yaxis_title="",
        xaxis_title="Similarity score",
        coloraxis_showscale=False,
    )
    return style_chart(fig, height=430)


def explain_prediction(
    df: pd.DataFrame,
    symptom_cols: list[str],
    condition: str,
    selected_symptoms: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile = df[df["Disease"] == condition][symptom_cols].mean() * 100
    selected = [symptom for symptom in selected_symptoms if symptom in symptom_cols]
    matched = (
        profile[selected]
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "Symptom", 0: "Condition prevalence"})
    )
    matched["Symptom"] = matched["Symptom"].map(display_label)
    matched["Condition prevalence"] = matched["Condition prevalence"].map(format_pct)

    missing_key = profile.drop(labels=selected, errors="ignore").sort_values(ascending=False)
    missing_key = (
        missing_key.head(8)
        .reset_index()
        .rename(columns={"index": "Symptom", 0: "Condition prevalence"})
    )
    missing_key["Symptom"] = missing_key["Symptom"].map(display_label)
    missing_key["Condition prevalence"] = missing_key["Condition prevalence"].map(format_pct)
    return matched, missing_key


def data_quality_frame(df: pd.DataFrame, symptom_cols: list[str]) -> pd.DataFrame:
    missing = df[["Disease", *symptom_cols]].isna().sum().sum()
    duplicate_rows = df.duplicated(subset=["Disease", *symptom_cols]).sum()
    disease_counts = df["Disease"].value_counts()
    imbalance_ratio = disease_counts.max() / disease_counts.min()
    zero_variance = int((df[symptom_cols].nunique() <= 1).sum())
    one_prevalence = int((df[symptom_cols].mean() == 1).sum())

    return pd.DataFrame(
        [
            {"Check": "Missing values", "Result": compact_number(missing), "Status": "Clean" if missing == 0 else "Review"},
            {"Check": "Duplicate rows", "Result": compact_number(duplicate_rows), "Status": "Expected in symptom templates" if duplicate_rows else "Clean"},
            {"Check": "Class imbalance ratio", "Result": f"{imbalance_ratio:.2f}x", "Status": "Balanced" if imbalance_ratio < 1.5 else "Review"},
            {"Check": "Zero-variance symptom columns", "Result": compact_number(zero_variance), "Status": "Clean" if zero_variance == 0 else "Review"},
            {"Check": "Always-on symptom columns", "Result": compact_number(one_prevalence), "Status": "Clean" if one_prevalence == 0 else "Review"},
        ]
    )


def render_header(total_rows: int, condition_count: int, symptom_count: int) -> None:
    st.markdown(
        f"""
        <div class="app-hero">
            <div class="app-eyebrow">Portfolio analytics dashboard</div>
            <div class="app-title">Mental Health Symptoms & Illness Prediction</div>
            <p class="app-subtitle">
                Interactive analysis of {total_rows:,} symptom records across
                {condition_count:,} mental-health condition labels and {symptom_count:,}
                binary symptom features.
            </p>
            <div class="creator-line">
                <span class="creator-dot"></span>
                Created by Hieu Nguyen
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="note-box">
            This dashboard is for analytics and portfolio demonstration only. It is not
            designed to diagnose, treat, or replace care from a qualified professional.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(
    df: pd.DataFrame, symptom_cols: list[str]
) -> tuple[list[str], list[str], tuple[int, int], bool, int]:
    with st.sidebar:
        st.title("Dashboard Controls")
        st.caption("Created by Hieu Nguyen")

        conditions = sorted(df["Disease"].dropna().unique())
        selected_conditions = st.multiselect(
            "Conditions",
            options=conditions,
            default=conditions,
        )

        focus_symptoms = st.multiselect(
            "Focus symptoms",
            options=symptom_cols,
            format_func=display_label,
            help="Filter the dataset to records containing at least one selected symptom.",
        )
        require_all_focus = st.toggle(
            "Require every focus symptom",
            value=False,
            help="When enabled, records must include every selected focus symptom.",
        )

        min_count = int(df["Active symptom count"].min())
        max_count = int(df["Active symptom count"].max())
        count_range = st.slider(
            "Active symptom count",
            min_value=min_count,
            max_value=max_count,
            value=(min_count, max_count),
        )

        top_n = st.slider("Top symptoms shown", 8, 25, 15)
        st.divider()
        st.caption("Tip: use the tabs to move from executive overview to detailed symptom exploration.")

    return (
        selected_conditions,
        focus_symptoms,
        count_range,
        require_all_focus,
        top_n,
    )


def main() -> None:
    inject_css()
    df, symptom_cols = load_data(DATA_PATH)

    selected_conditions, focus_symptoms, count_range, require_all_focus, top_n = (
        render_sidebar(df, symptom_cols)
    )
    filtered_df = filter_data(
        df=df,
        conditions=selected_conditions,
        symptom_cols=symptom_cols,
        symptom_focus=focus_symptoms,
        count_range=count_range,
        require_all_focus=require_all_focus,
    )

    render_header(
        total_rows=len(df),
        condition_count=df["Disease"].nunique(),
        symptom_count=len(symptom_cols),
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Records in view", f"{len(filtered_df):,}", f"{len(df):,} total")
    metric_cols[1].metric("Condition labels", f"{filtered_df['Disease'].nunique():,}")
    metric_cols[2].metric("Symptom features", f"{len(symptom_cols):,}")
    metric_cols[3].metric(
        "Avg active symptoms",
        compact_number(float(filtered_df["Active symptom count"].mean()) if not filtered_df.empty else 0),
    )

    overview_tab, profile_tab, symptom_tab, prediction_tab, quality_tab = st.tabs(
        [
            "Overview",
            "Condition Profiles",
            "Symptom Explorer",
            "Prediction Sandbox",
            "Data Quality",
        ]
    )

    with overview_tab:
        st.markdown('<div class="section-label">Executive view</div>', unsafe_allow_html=True)
        left, right = st.columns([1.05, 1])
        with left:
            st.plotly_chart(condition_distribution_chart(filtered_df), use_container_width=True)
        with right:
            st.plotly_chart(
                top_symptoms_chart(filtered_df, symptom_cols, top_n),
                use_container_width=True,
            )

        lower_left, lower_right = st.columns([1, 1])
        with lower_left:
            st.plotly_chart(symptom_load_chart(filtered_df), use_container_width=True)
        with lower_right:
            st.plotly_chart(class_balance_chart(filtered_df), use_container_width=True)

    with profile_tab:
        st.markdown('<div class="section-label">Condition deep dive</div>', unsafe_allow_html=True)
        available_conditions = sorted(filtered_df["Disease"].unique()) if not filtered_df.empty else []
        if not available_conditions:
            st.info("No conditions match the current filter settings.")
        else:
            default_index = available_conditions.index("Major Depressive Disorder") if "Major Depressive Disorder" in available_conditions else 0
            condition = st.selectbox(
                "Select a condition",
                options=available_conditions,
                index=default_index,
            )
            condition_records = int((filtered_df["Disease"] == condition).sum())
            st.caption(f"{condition_records:,} records in the current filtered view.")

            profile_chart, distinct_chart, heatmap, table_frame = condition_profile_charts(
                filtered_df, symptom_cols, condition
            )
            first, second = st.columns([1, 1])
            with first:
                st.plotly_chart(profile_chart, use_container_width=True)
            with second:
                st.plotly_chart(distinct_chart, use_container_width=True)
            st.plotly_chart(heatmap, use_container_width=True)
            st.dataframe(table_frame, use_container_width=True, hide_index=True)

    with symptom_tab:
        st.markdown('<div class="section-label">Symptom relationships</div>', unsafe_allow_html=True)
        default_symptom = "sleep_disturbances" if "sleep_disturbances" in symptom_cols else symptom_cols[0]
        selected_symptom = st.selectbox(
            "Select a symptom",
            options=symptom_cols,
            index=symptom_cols.index(default_symptom),
            format_func=display_label,
        )
        bundle = st.multiselect(
            "Optional symptom bundle",
            options=symptom_cols,
            default=[selected_symptom],
            format_func=display_label,
            help="Shows the condition mix for records that contain every selected symptom.",
        )
        if len(bundle) > 6:
            st.warning("For readability, only the first 6 bundle symptoms are used.")
            bundle = bundle[:6]

        condition_chart, related_chart, bundle_rates = symptom_explorer(
            filtered_df, symptom_cols, selected_symptom, bundle
        )
        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(condition_chart, use_container_width=True)
        with right:
            st.plotly_chart(related_chart, use_container_width=True)

        st.markdown('<div class="section-label">Bundle match table</div>', unsafe_allow_html=True)
        if bundle_rates.empty:
            st.info("No records match the selected symptom bundle.")
        else:
            formatted_bundle = bundle_rates.copy()
            formatted_bundle["Share of bundle"] = formatted_bundle["Share of bundle"].map(format_pct)
            st.dataframe(formatted_bundle, use_container_width=True, hide_index=True)

    with prediction_tab:
        st.markdown('<div class="section-label">Analytical scoring sandbox</div>', unsafe_allow_html=True)
        st.caption(
            "Select symptoms to rank condition profiles by similarity. This is an explainable data product feature, not clinical guidance."
        )
        default_prediction = [
            symptom
            for symptom in [
                "sleep_disturbances",
                "difficulty_concentrating",
                "fatigue",
                "low_mood",
            ]
            if symptom in symptom_cols
        ]
        selected_prediction_symptoms = st.multiselect(
            "Symptoms present",
            options=symptom_cols,
            default=default_prediction,
            format_func=display_label,
        )
        if len(selected_prediction_symptoms) > 12:
            st.warning("For a clearer demonstration, the score uses the first 12 selected symptoms.")
            selected_prediction_symptoms = selected_prediction_symptoms[:12]

        results = prediction_scores(filtered_df, symptom_cols, selected_prediction_symptoms)
        left, right = st.columns([1.15, 0.85])
        with left:
            st.plotly_chart(prediction_chart(results), use_container_width=True)
        with right:
            if results.empty:
                st.info("Choose at least one symptom to view ranked matches.")
            else:
                top_condition = results.iloc[0]["Condition"]
                st.metric("Highest similarity", top_condition, f"{results.iloc[0]['Match score']:.1f}%")
                st.dataframe(
                    results.head(8).assign(
                        **{
                            "Match score": results.head(8)["Match score"].map(format_pct),
                            "Selected symptom coverage": results.head(8)[
                                "Selected symptom coverage"
                            ].map(format_pct),
                            "Signature fit": results.head(8)["Signature fit"].map(format_pct),
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        if not results.empty:
            top_condition = str(results.iloc[0]["Condition"])
            matched, missing_key = explain_prediction(
                filtered_df, symptom_cols, top_condition, selected_prediction_symptoms
            )
            exp_left, exp_right = st.columns([1, 1])
            with exp_left:
                st.markdown('<div class="section-label">Selected symptoms inside top profile</div>', unsafe_allow_html=True)
                st.dataframe(matched, use_container_width=True, hide_index=True)
            with exp_right:
                st.markdown('<div class="section-label">Other key symptoms for top profile</div>', unsafe_allow_html=True)
                st.dataframe(missing_key, use_container_width=True, hide_index=True)

    with quality_tab:
        st.markdown('<div class="section-label">Dataset readiness</div>', unsafe_allow_html=True)
        quality = data_quality_frame(df, symptom_cols)
        st.dataframe(quality, use_container_width=True, hide_index=True)

        disease_counts = (
            df["Disease"].value_counts().rename_axis("Condition").reset_index(name="Records")
        )
        symptom_counts = symptom_prevalence_frame(df, symptom_cols, top_n=25)
        left, right = st.columns([1, 1])
        with left:
            st.dataframe(disease_counts, use_container_width=True, hide_index=True)
        with right:
            st.dataframe(
                symptom_counts[
                    ["Symptom label", "Records", "Prevalence"]
                ].rename(columns={"Symptom label": "Symptom"}),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown('<div class="section-label">Sample records</div>', unsafe_allow_html=True)
        sample_cols = ["Disease", "Active symptom count", *symptom_cols[:18]]
        sample = filtered_df[sample_cols].head(100).rename(columns=display_label)
        st.dataframe(sample, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="footer">
            Created by Hieu Nguyen. Built with Python, Streamlit, pandas, and Plotly.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

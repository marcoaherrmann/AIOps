"""
DelayPredict — Plotly Dash Data Analytics Dashboard
-----------------------------------------------------
Data-focused visualization: dataset statistics, model performance over
training rounds, incremental learning curve, and PSI drift per feature.
Includes training controls (retrain, incremental training, rollback).

Runs on port 8050.
"""

import os
import json
import requests as http
import pandas as pd
from pathlib import Path
from dash import Dash, html, dcc, Output, Input, State, no_update
import plotly.graph_objects as go

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://api:8000")

# ── Paths ─────────────────────────────────────────────────────────────────────
RETRAIN_HISTORY_PATH     = Path("/app/data/processed/retrain_history.json")
INCREMENTAL_HISTORY_PATH = Path("/app/data/processed/incremental_history.json")
RAW_DATA_PATH            = Path("/app/data/raw/airlines_delay.csv")

# ── Theme colors (matches the HTML dashboard) ─────────────────────────────────
BG     = "#0f172a"
CARD   = "#1e293b"
BORDER = "#334155"
TEXT   = "#e2e8f0"
MUTED  = "#64748b"
BLUE   = "#38bdf8"
GREEN  = "#34d399"
YELLOW = "#fbbf24"
RED    = "#f87171"
ORANGE = "#fb923c"

def base_layout(title="", **kwargs):
    return dict(
        title=dict(text=title, font=dict(color=MUTED, size=13), x=0),
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=TEXT, family="Arial, sans-serif", size=12),
        margin=dict(l=50, r=20, t=40, b=40),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
        legend=dict(font=dict(color=TEXT, size=11), bgcolor="rgba(0,0,0,0)"),
        **kwargs,
    )

# ── Pre-compute static dataset aggregations (loaded once at startup) ───────────
df = pd.read_csv(RAW_DATA_PATH)
df["DepartureHour"] = df["Time"] // 60
df["Route"]         = df["AirportFrom"] + "-" + df["AirportTo"]

DAY_NAMES = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
df_dow = (
    df.groupby("DayOfWeek")["Delay"].mean().reset_index()
      .assign(day=lambda x: x["DayOfWeek"].map(DAY_NAMES))
)
df_hour    = df.groupby("DepartureHour")["Delay"].mean().reset_index()
top_airlines = df["Airline"].value_counts().head(12).index
df_airline = (
    df[df["Airline"].isin(top_airlines)]
      .groupby("Airline")["Delay"].mean().reset_index()
      .sort_values("Delay")
)

DELAY_RATE     = df["Delay"].mean() * 100
TOTAL_ROWS     = len(df)
TOTAL_AIRLINES = df["Airline"].nunique()
TOTAL_ROUTES   = df["Route"].nunique()

# ── App ───────────────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    title="DelayPredict — Analytics",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# ── Helper components ─────────────────────────────────────────────────────────
def card(children, style=None):
    base = {"backgroundColor": CARD, "borderRadius": "12px",
            "border": f"1px solid {BORDER}", "padding": "20px"}
    if style:
        base.update(style)
    return html.Div(children, style=base)

def kpi(title, value, sub=""):
    return card([
        html.Div(title, style={"color": MUTED, "fontSize": "12px",
                               "textTransform": "uppercase", "letterSpacing": "1px",
                               "marginBottom": "8px"}),
        html.Div(value, style={"color": BLUE, "fontSize": "28px", "fontWeight": "bold"}),
        html.Div(sub,   style={"color": MUTED, "fontSize": "12px", "marginTop": "4px"}),
    ])

def btn(label, btn_id, color=BLUE):
    return html.Button(label, id=btn_id, n_clicks=0, style={
        "background": "#1d4ed8", "color": "#fff", "border": "none",
        "padding": "10px 20px", "borderRadius": "8px", "cursor": "pointer",
        "fontSize": "14px", "fontWeight": "bold", "marginRight": "12px",
    })

GRID2 = {"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginBottom": "24px"}
GRID4 = {"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "16px", "marginBottom": "24px"}

def empty_fig(title=""):
    """Dark empty figure shown before the first callback fires."""
    fig = go.Figure(layout=base_layout(title))
    fig.add_annotation(text="Loading...", x=0.5, y=0.5,
                       xref="paper", yref="paper", showarrow=False,
                       font=dict(color=MUTED, size=13))
    return fig

# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={"backgroundColor": BG, "minHeight": "100vh",
           "padding": "24px", "fontFamily": "Arial, sans-serif"},
    children=[
        html.H1("✈️ DelayPredict — Data Analytics",
                style={"color": BLUE, "marginBottom": "4px"}),
        html.P("Dataset statistics · Model performance · Drift analysis",
               style={"color": MUTED, "fontSize": "14px", "marginBottom": "32px"}),

        dcc.Interval(id="interval", interval=10_000, n_intervals=0),

        # Static dataset KPIs
        html.Div([
            kpi("Total Flights",  f"{TOTAL_ROWS:,}",    "in dataset"),
            kpi("Delay Rate",     f"{DELAY_RATE:.1f}%", "flights delayed > 15 min"),
            kpi("Airlines",       str(TOTAL_AIRLINES),  "unique carriers"),
            kpi("Routes",         f"{TOTAL_ROUTES:,}",  "unique A→B pairs"),
        ], style=GRID4),

        # Live model KPIs
        html.Div(id="live-kpis", style=GRID4),

        # ── Training Controls ─────────────────────────────────────────────────
        card([
            html.H3("Training Controls",
                    style={"color": MUTED, "fontSize": "13px", "textTransform": "uppercase",
                           "letterSpacing": "1px", "margin": "0 0 16px 0"}),
            html.Div([
                btn("🔄 Retrain on Stream Data",   "btn-retrain"),
                btn("📈 Start Incremental Training", "btn-incremental"),
                btn("↩ Rollback to Previous",       "btn-rollback"),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "marginBottom": "12px"}),
            html.Div(id="ctrl-status", style={"fontSize": "13px", "color": MUTED}),
        ], style={"marginBottom": "24px"}),

        # Model performance + incremental learning
        html.Div([
            card(dcc.Graph(id="retrain-chart",     figure=empty_fig("MODEL PERFORMANCE OVER TRAINING SIZE"),        config={"displayModeBar": False}, style={"height": "320px"})),
            card(dcc.Graph(id="incremental-chart", figure=empty_fig("INCREMENTAL LEARNING CURVE (90/10, 10 ROUNDS)"), config={"displayModeBar": False}, style={"height": "320px"})),
        ], style=GRID2),

        # PSI drift + airline delay rate
        html.Div([
            card(dcc.Graph(id="psi-chart",     figure=empty_fig("PSI DRIFT PER FEATURE (LATEST)"),         config={"displayModeBar": False}, style={"height": "300px"})),
            card(dcc.Graph(id="airline-chart", figure=empty_fig("DELAY RATE BY AIRLINE (TOP 12 BY VOLUME)"), config={"displayModeBar": False}, style={"height": "300px"})),
        ], style=GRID2),

        # Delay by day + by hour
        html.Div([
            card(dcc.Graph(id="dow-chart",  figure=empty_fig("DELAY RATE BY DAY OF WEEK"),  config={"displayModeBar": False}, style={"height": "280px"})),
            card(dcc.Graph(id="hour-chart", figure=empty_fig("DELAY RATE BY DEPARTURE HOUR"), config={"displayModeBar": False}, style={"height": "280px"})),
        ], style=GRID2),

        html.P("DelayPredict · AIOps SoSe 2026",
               style={"color": MUTED, "fontSize": "12px", "marginTop": "8px"}),
    ],
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_json(path):
    try:
        return json.loads(path.read_text()) if path.exists() else []
    except Exception:
        return []


# ── Button callbacks ──────────────────────────────────────────────────────────
@app.callback(
    Output("ctrl-status", "children"),
    Output("ctrl-status", "style"),
    Input("btn-retrain",     "n_clicks"),
    Input("btn-incremental", "n_clicks"),
    Input("btn-rollback",    "n_clicks"),
    prevent_initial_call=True,
)
def handle_buttons(n_retrain, n_incremental, n_rollback):
    from dash import ctx
    triggered = ctx.triggered_id

    try:
        if triggered == "btn-retrain":
            res = http.post(f"{API_URL}/retrain", timeout=5).json()
            if res.get("status") == "already_running":
                msg, color = "⏳ Retrain is already running — check the charts below", YELLOW
            else:
                msg, color = "✅ Retrain started — charts will update live every 10 seconds", GREEN

        elif triggered == "btn-incremental":
            res = http.post(f"{API_URL}/incremental-training", timeout=5).json()
            if res.get("status") == "already_running":
                msg, color = f"⏳ Incremental training already running — round {res.get('rounds_done', '?')}/10", YELLOW
            else:
                msg, color = "✅ Incremental training started — 10 rounds, charts will update live", GREEN

        elif triggered == "btn-rollback":
            res = http.post(f"{API_URL}/rollback", timeout=5).json()
            msg, color = "↩ Rolled back to previous model version", ORANGE

        else:
            return no_update, no_update

    except Exception as e:
        msg, color = f"❌ Could not reach API: {e}", RED

    style = {"fontSize": "13px", "color": color, "fontWeight": "bold",
             "background": CARD, "padding": "8px 12px", "borderRadius": "6px",
             "border": f"1px solid {color}33", "display": "inline-block"}
    return msg, style


# ── Live KPI callback ─────────────────────────────────────────────────────────
@app.callback(Output("live-kpis", "children"), Input("interval", "n_intervals"))
def update_live_kpis(_):
    history     = load_json(RETRAIN_HISTORY_PATH)
    inc_history = load_json(INCREMENTAL_HISTORY_PATH)
    last        = history[-1]     if history     else None
    last_inc    = inc_history[-1] if inc_history else None

    return [
        kpi("Current ROC-AUC",  f"{last['roc_auc']:.4f}"    if last     else "N/A", "after last retrain"),
        kpi("Current F1",       f"{last['f1']:.4f}"          if last     else "N/A", "after last retrain"),
        kpi("Retrain Rounds",   str(len(history)),                                   "progressive retrains logged"),
        kpi("Incremental ROC",  f"{last_inc['roc_auc']:.4f}" if last_inc else "N/A", "round 10 of 10"),
    ]


# ── Chart callbacks ───────────────────────────────────────────────────────────
@app.callback(Output("retrain-chart", "figure"), Input("interval", "n_intervals"))
def update_retrain_chart(_):
    history = load_json(RETRAIN_HISTORY_PATH)
    fig = go.Figure(layout=base_layout("MODEL PERFORMANCE OVER TRAINING SIZE"))

    if not history:
        fig.add_annotation(text="No retrain data yet", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(color=MUTED, size=14))
        return fig

    sorted_h = sorted(history, key=lambda r: r.get("train_size", 0))
    labels   = [f"{r['train_size'] // 1000}k" for r in sorted_h]

    for metric, color, name in [
        ("roc_auc",  BLUE,   "ROC-AUC"),
        ("f1",       GREEN,  "F1"),
        ("accuracy", YELLOW, "Accuracy"),
    ]:
        fig.add_trace(go.Scatter(
            x=labels, y=[r[metric] for r in sorted_h],
            name=name, mode="lines+markers",
            line=dict(color=color, width=2), marker=dict(size=6),
        ))

    fig.update_layout(yaxis=dict(range=[0.5, 1.0]), xaxis_title="Training size", yaxis_title="Score")
    return fig


@app.callback(Output("incremental-chart", "figure"), Input("interval", "n_intervals"))
def update_incremental_chart(_):
    history = load_json(INCREMENTAL_HISTORY_PATH)
    fig = go.Figure(layout=base_layout("INCREMENTAL LEARNING CURVE (90/10 SPLIT, 10 ROUNDS)"))

    if not history:
        fig.add_annotation(text="No incremental training data yet", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(color=MUTED, size=14))
        return fig

    labels = [f"Round {r['round']}" for r in history]

    for metric, color, name in [
        ("roc_auc",  BLUE,   "ROC-AUC"),
        ("f1",       GREEN,  "F1"),
        ("accuracy", YELLOW, "Accuracy"),
    ]:
        fig.add_trace(go.Scatter(
            x=labels, y=[r[metric] for r in history],
            name=name, mode="lines+markers",
            line=dict(color=color, width=2), marker=dict(size=6),
        ))

    fig.update_layout(yaxis=dict(range=[0.5, 1.0]), xaxis_title="Round", yaxis_title="Score")
    return fig


@app.callback(Output("psi-chart", "figure"), Input("interval", "n_intervals"))
def update_psi_chart(_):
    history = load_json(RETRAIN_HISTORY_PATH)
    fig = go.Figure(layout=base_layout("PSI DRIFT PER FEATURE (LATEST)"))
    psi_scores = history[-1].get("psi_scores") or {} if history else {}

    if not psi_scores:
        fig.add_annotation(text="No drift data yet — stream data first",
                           x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font=dict(color=MUTED, size=14))
        return fig

    features = list(psi_scores.keys())
    values   = list(psi_scores.values())
    colors   = [RED if v > 0.2 else YELLOW if v > 0.1 else GREEN for v in values]

    fig.add_trace(go.Bar(
        x=values, y=features, orientation="h", marker_color=colors,
        showlegend=False,
        text=[f"{v:.5f}" for v in values], textposition="outside",
        textfont=dict(color=TEXT, size=11),
    ))
    fig.add_vline(x=0.1, line=dict(color=YELLOW, dash="dash", width=1))
    fig.add_vline(x=0.2, line=dict(color=RED,    dash="dash", width=1))
    fig.update_layout(
        xaxis=dict(range=[0, max(max(values) * 1.4, 0.25)], gridcolor=BORDER, zerolinecolor=BORDER),
        xaxis_title="PSI Score",
        annotations=[
            dict(x=0.1, y=1.05, xref="x", yref="paper", text="monitor",
                 showarrow=False, font=dict(color=YELLOW, size=10)),
            dict(x=0.2, y=1.05, xref="x", yref="paper", text="retrain",
                 showarrow=False, font=dict(color=RED, size=10)),
        ],
    )
    return fig


@app.callback(Output("airline-chart", "figure"), Input("interval", "n_intervals"))
def update_airline_chart(_):
    colors = [RED if v > 0.5 else YELLOW if v > 0.4 else GREEN for v in df_airline["Delay"]]
    fig = go.Figure(layout=base_layout("DELAY RATE BY AIRLINE (TOP 12 BY VOLUME)"))
    fig.add_trace(go.Bar(
        x=df_airline["Delay"] * 100, y=df_airline["Airline"],
        orientation="h", marker_color=colors, showlegend=False,
        text=[f"{v*100:.1f}%" for v in df_airline["Delay"]],
        textposition="outside", textfont=dict(color=TEXT, size=11),
    ))
    fig.update_layout(xaxis=dict(range=[0, 80]), xaxis_title="Delay rate (%)")
    return fig


@app.callback(Output("dow-chart", "figure"), Input("interval", "n_intervals"))
def update_dow_chart(_):
    colors = [RED if v > 0.5 else YELLOW if v > 0.4 else GREEN for v in df_dow["Delay"]]
    fig = go.Figure(layout=base_layout("DELAY RATE BY DAY OF WEEK"))
    fig.add_trace(go.Bar(
        x=df_dow["day"], y=df_dow["Delay"] * 100,
        marker_color=colors, showlegend=False,
        text=[f"{v*100:.1f}%" for v in df_dow["Delay"]],
        textposition="outside", textfont=dict(color=TEXT, size=11),
    ))
    fig.update_layout(yaxis=dict(range=[0, 60]), yaxis_title="Delay rate (%)")
    return fig


@app.callback(Output("hour-chart", "figure"), Input("interval", "n_intervals"))
def update_hour_chart(_):
    fig = go.Figure(layout=base_layout("DELAY RATE BY DEPARTURE HOUR"))
    fig.add_trace(go.Scatter(
        x=df_hour["DepartureHour"], y=df_hour["Delay"] * 100,
        mode="lines+markers",
        line=dict(color=BLUE, width=2), marker=dict(size=5),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.1)", showlegend=False,
    ))
    fig.update_layout(
        xaxis=dict(tickvals=list(range(0, 24, 2))),
        xaxis_title="Departure hour", yaxis_title="Delay rate (%)",
        yaxis=dict(range=[0, 80]),
    )
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)

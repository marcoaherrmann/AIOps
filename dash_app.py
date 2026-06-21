"""
DelayPredict — Plotly Dash Data Analytics Dashboard
-----------------------------------------------------
Focus: static dataset analytics (from raw CSV, 539k flights) +
       training controls (Retrain, Incremental, Rollback).

Model performance history, learning curves, and PSI drift are in Metabase (port 3000).

Runs on port 8050.
"""

import os
import requests as http
from pathlib import Path
from dash import Dash, html, dcc, Output, Input, no_update
import plotly.graph_objects as go
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
API_URL      = os.getenv("API_URL", "http://api:8000")
RAW_DATA_PATH = Path("/app/data/raw/airlines_delay.csv")

# ── Theme ─────────────────────────────────────────────────────────────────────
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

# ── Pre-compute static dataset aggregations (loaded once at startup) ──────────
df = pd.read_csv(RAW_DATA_PATH)
df["DepartureHour"] = df["Time"] // 60
df["Route"]         = df["AirportFrom"] + "-" + df["AirportTo"]

DAY_NAMES  = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
df_dow     = (
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

def btn(label, btn_id, tooltip=""):
    return html.Div([
        html.Button(label, id=btn_id, n_clicks=0, style={
            "background": "#1d4ed8", "color": "#fff", "border": "none",
            "padding": "10px 20px", "borderRadius": "8px", "cursor": "pointer",
            "fontSize": "14px", "fontWeight": "bold",
        }),
        html.Div([
            html.Span("?", style={
                "display": "inline-flex", "alignItems": "center", "justifyContent": "center",
                "width": "18px", "height": "18px", "borderRadius": "50%",
                "background": BORDER, "color": MUTED, "fontSize": "11px",
                "cursor": "help", "fontWeight": "bold", "flexShrink": "0",
            }),
            html.Div(tooltip, className="tooltip-box"),
        ], className="tooltip-wrap", style={"marginLeft": "6px"}),
    ], style={"display": "inline-flex", "alignItems": "center", "marginRight": "16px"})

GRID2 = {"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginBottom": "24px"}
GRID3 = {"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "16px", "marginBottom": "24px"}
GRID4 = {"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "16px", "marginBottom": "24px"}

# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={"backgroundColor": BG, "minHeight": "100vh",
           "padding": "24px", "fontFamily": "Arial, sans-serif"},
    children=[
        html.H1("✈️ DelayPredict — Data Analytics",
                style={"color": BLUE, "marginBottom": "4px"}),
        html.P("Dataset statistics (539k US flights) · Training controls",
               style={"color": MUTED, "fontSize": "14px", "marginBottom": "12px"}),

        # ── Navigation links ──────────────────────────────────────────────────
        html.Div([
            html.A("🗃️ Metabase BI Dashboard",
                   href="http://localhost:3000", target="_blank",
                   style={"color": BLUE, "fontSize": "13px", "fontWeight": "bold",
                          "background": "#0f2d47", "border": f"1px solid {BLUE}33",
                          "padding": "7px 14px", "borderRadius": "6px",
                          "textDecoration": "none", "marginRight": "10px"}),
            html.A("✈️ Streamlit Prediction UI",
                   href="http://localhost:8501", target="_blank",
                   style={"color": TEXT, "fontSize": "13px",
                          "background": CARD, "border": f"1px solid {BORDER}",
                          "padding": "7px 14px", "borderRadius": "6px",
                          "textDecoration": "none", "marginRight": "10px"}),
            html.A("🔬 MLflow Experiments",
                   href="http://localhost:5001", target="_blank",
                   style={"color": TEXT, "fontSize": "13px",
                          "background": CARD, "border": f"1px solid {BORDER}",
                          "padding": "7px 14px", "borderRadius": "6px",
                          "textDecoration": "none", "marginRight": "10px"}),
            html.A("📄 API Docs",
                   href="http://localhost:8000/docs", target="_blank",
                   style={"color": TEXT, "fontSize": "13px",
                          "background": CARD, "border": f"1px solid {BORDER}",
                          "padding": "7px 14px", "borderRadius": "6px",
                          "textDecoration": "none"}),
        ], style={"marginBottom": "32px", "display": "flex", "flexWrap": "wrap", "gap": "8px"}),

        dcc.Interval(id="interval", interval=10_000, n_intervals=0),

        # ── Rollback banner (hidden by default) ───────────────────────────────
        html.Div(id="rollback-banner"),

        # ── Static dataset KPIs ───────────────────────────────────────────────
        html.Div([
            kpi("Total Flights",  f"{TOTAL_ROWS:,}",    "in dataset"),
            kpi("Delay Rate",     f"{DELAY_RATE:.1f}%", "flights delayed > 15 min"),
            kpi("Airlines",       str(TOTAL_AIRLINES),  "unique carriers"),
            kpi("Routes",         f"{TOTAL_ROUTES:,}",  "unique A→B pairs"),
        ], style=GRID4),

        # ── Training Controls ─────────────────────────────────────────────────
        card([
            html.H3("Training Controls",
                    style={"color": MUTED, "fontSize": "13px", "textTransform": "uppercase",
                           "letterSpacing": "1px", "margin": "0 0 16px 0"}),
            html.Div([
                btn("🔄 Retrain on Stream Data",    "btn-retrain",
                    "Trains a baseline on the first 50% of data, then progressively adds "
                    "the remaining 50% in 5 steps to simulate incoming stream data. "
                    "Shows how model performance changes as more data arrives "
                    "and whether data drift affects prediction quality."),
                btn("📈 Start Incremental Training", "btn-incremental",
                    "Runs 10 cumulative training rounds on a 90/10 split. "
                    "Round 1 uses 10% of the training pool, Round 10 uses 100%. "
                    "Shows how model performance improves as more data is added."),
                btn("↩ Rollback to Previous",        "btn-rollback",
                    "Replaces the current model with the last backup. "
                    "Use this if a retrain produced worse results. "
                    "A new retrain will overwrite the backup again."),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "marginBottom": "12px"}),
            html.Div(id="ctrl-status", style={"fontSize": "13px", "color": MUTED}),
        ], style={"marginBottom": "24px"}),

        # ── Dataset analytics ─────────────────────────────────────────────────
        html.Div([
            card(dcc.Graph(id="airline-chart", config={"displayModeBar": False}, style={"height": "340px"})),
            card(dcc.Graph(id="dow-chart",     config={"displayModeBar": False}, style={"height": "340px"})),
        ], style=GRID2),

        html.Div([
            card(dcc.Graph(id="hour-chart", config={"displayModeBar": False}, style={"height": "300px"})),
        ], style={"marginBottom": "24px"}),

        html.P("DelayPredict · AIOps SoSe 2026",
               style={"color": MUTED, "fontSize": "12px", "marginTop": "8px"}),
    ],
)


# ── Rollback banner callback ──────────────────────────────────────────────────
@app.callback(Output("rollback-banner", "children"), Input("interval", "n_intervals"))
def update_rollback_banner(_):
    try:
        status = http.get(f"{API_URL}/status", timeout=3).json()
    except Exception:
        return None

    if not status.get("is_rolled_back"):
        return None

    return html.Div([
        html.Strong("↩ Rollback active"),
        html.Span(" — the model was rolled back to the previous version. "
                  "Run a new retrain to replace it.",
                  style={"fontWeight": "normal"}),
    ], style={
        "background": "#431407", "border": f"1px solid {ORANGE}",
        "color": ORANGE, "padding": "12px 18px", "borderRadius": "8px",
        "marginBottom": "16px", "fontSize": "14px",
    })


# ── Button callback ───────────────────────────────────────────────────────────
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
            msg, color = ("⏳ Retrain already running", YELLOW) if res.get("status") == "already_running" \
                    else ("✅ Retrain started — check Metabase for live metrics", GREEN)
        elif triggered == "btn-incremental":
            res = http.post(f"{API_URL}/incremental-training", timeout=5).json()
            msg, color = (f"⏳ Already running — round {res.get('rounds_done','?')}/10", YELLOW) \
                    if res.get("status") == "already_running" \
                    else ("✅ Incremental training started — 10 rounds", GREEN)
        elif triggered == "btn-rollback":
            http.post(f"{API_URL}/rollback", timeout=5)
            msg, color = "↩ Rolled back to previous model", ORANGE
        else:
            return no_update, no_update
    except Exception as e:
        msg, color = f"❌ API not reachable: {e}", RED

    style = {"fontSize": "13px", "color": color, "fontWeight": "bold",
             "background": CARD, "padding": "8px 12px", "borderRadius": "6px",
             "border": f"1px solid {color}33", "display": "inline-block"}
    return msg, style


# ── Static chart callbacks (data never changes, interval just keeps Dash happy)
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

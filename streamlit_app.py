"""
DelayPredict — Streamlit Demo UI
---------------------------------
Standalone prediction interface that talks to the FastAPI backend.

Start via Docker (automatic) or locally:
    streamlit run streamlit_app.py
"""

import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DelayPredict",
    page_icon="✈️",
    layout="centered",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("✈️ DelayPredict")
st.markdown("*Will your flight be delayed by more than 15 minutes?*")

col_links = st.columns(3)
with col_links[0]:
    st.link_button("📊 Data Analytics (Dash)", "http://localhost:8050", use_container_width=True)
with col_links[1]:
    st.link_button("🗃️ BI Dashboard (Metabase)", "http://localhost:3000", use_container_width=True)
with col_links[2]:
    st.link_button("🔬 Experiment Tracking (MLflow)", "http://localhost:5001", use_container_width=True)

st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    airline = st.text_input(
        "Airline", value="WN", max_chars=3,
        help="IATA airline code — e.g. WN, DL, AA, UA",
    ).upper().strip()
with col2:
    airport_from = st.text_input(
        "Airport From", value="LAX", max_chars=3,
        help="IATA departure airport code",
    ).upper().strip()
with col3:
    airport_to = st.text_input(
        "Airport To", value="JFK", max_chars=3,
        help="IATA arrival airport code",
    ).upper().strip()

col4, col5, col6 = st.columns(3)
with col4:
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_of_week = st.selectbox(
        "Day of Week",
        options=list(range(1, 8)),
        format_func=lambda x: day_names[x - 1],
        index=4,  # Friday
    )
with col5:
    departure_hour = st.slider("Departure Hour", min_value=0, max_value=23, value=8)
with col6:
    length = st.number_input("Duration (min)", min_value=1, max_value=1200, value=120)

st.divider()

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("🔍 Predict Delay", type="primary", use_container_width=True):
    if not airline or not airport_from or not airport_to:
        st.warning("Please fill in Airline, Airport From, and Airport To.")
    else:
        with st.spinner("Predicting..."):
            try:
                res = requests.post(
                    f"{API_URL}/predict",
                    json={
                        "Airline"      : airline,
                        "AirportFrom"  : airport_from,
                        "AirportTo"    : airport_to,
                        "DayOfWeek"    : day_of_week,
                        "DepartureHour": departure_hour,
                        "Length"       : length,
                    },
                    timeout=10,
                )

                if res.status_code == 200:
                    data    = res.json()
                    delayed = data["delay_predicted"]
                    prob    = data["delay_probability"] * 100

                    if delayed:
                        st.error(f"⚠️ **Delay likely** — {prob:.1f}% probability")
                    else:
                        st.success(f"✅ **On time likely** — {prob:.1f}% probability")

                    st.progress(data["delay_probability"], text=f"Delay probability: {prob:.1f}%")

                elif res.status_code == 422:
                    detail = res.json().get("detail", "")
                    st.error(f"Invalid input — make sure codes are uppercase (e.g. WN, LAX). Detail: {detail}")
                else:
                    st.error(f"API error {res.status_code}: {res.json().get('detail', 'Unknown error')}")

            except requests.exceptions.ConnectionError:
                st.error("Could not reach the API. Make sure the api container is running.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")


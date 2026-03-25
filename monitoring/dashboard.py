# monitoring/dashboard.py
#
# Reads all game sessions from MLflow and displays
# a monitoring dashboard using Streamlit.
#
# HOW TO RUN:
#   streamlit run monitoring/dashboard.py --server.port 8502
#
# WHAT IT SHOWS:
#   - Total games played
#   - Average questions per game
#   - Most guessed dishes
#   - Confidence over time
#   - Hardest dishes to guess

import streamlit as st
import mlflow
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

st.set_page_config(
    page_title = "Flavinator Monitor",
    layout     = "wide"
)

st.title("Flavinator - MLflow Monitoring Dashboard")
st.markdown("Track model performance across all game sessions.")
st.markdown("---")

# connect to mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

@st.cache_data(ttl=30)
def load_runs():
    """
    Loads all completed game runs from MLflow.

    WHY CACHE:
      Reading from database on every page refresh is slow.
      st.cache_data caches the result for 30 seconds.
      After 30 seconds it refreshes automatically.
    """
    try:
        runs = mlflow.search_runs(
            experiment_names = ["flavinator_game_sessions"],
            filter_string    = "tags.game_status = 'completed'"
        )
        return runs
    except Exception as e:
        st.error(f"Could not connect to MLflow: {e}")
        return pd.DataFrame()


runs = load_runs()

if runs.empty:
    st.info("No completed games yet. Play some games first then refresh.")
    st.markdown("Start the game at: http://localhost:8501")
    st.stop()

# ── SUMMARY METRICS ───────────────────────────────────────────────────────────

st.markdown("## Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label = "Total Games Played",
        value = len(runs)
    )

with col2:
    if "metrics.total_questions" in runs.columns:
        avg_q = runs["metrics.total_questions"].mean()
        st.metric(
            label = "Avg Questions Per Game",
            value = f"{avg_q:.1f}"
        )

with col3:
    if "metrics.final_confidence" in runs.columns:
        avg_conf = runs["metrics.final_confidence"].mean()
        st.metric(
            label = "Avg Final Confidence",
            value = f"{avg_conf:.1f}%"
        )

with col4:
    if "metrics.total_questions" in runs.columns:
        min_q = runs["metrics.total_questions"].min()
        st.metric(
            label = "Fewest Questions Needed",
            value = int(min_q)
        )

st.markdown("---")

# ── MOST GUESSED DISHES ───────────────────────────────────────────────────────

st.markdown("## Most Guessed Dishes")

if "params.final_guess" in runs.columns:
    guess_counts = (
        runs["params.final_guess"]
        .value_counts()
        .reset_index()
    )
    guess_counts.columns = ["Dish", "Times Guessed"]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.bar_chart(
            guess_counts.set_index("Dish")["Times Guessed"]
        )

    with col2:
        st.dataframe(guess_counts, use_container_width=True)

st.markdown("---")

# ── QUESTIONS PER GAME ────────────────────────────────────────────────────────

st.markdown("## Questions Per Game Over Time")

if "metrics.total_questions" in runs.columns:
    chart_data = runs[["start_time", "metrics.total_questions"]].copy()
    chart_data = chart_data.sort_values("start_time")
    chart_data.columns = ["Time", "Questions"]
    chart_data = chart_data.set_index("Time")
    st.line_chart(chart_data)

st.markdown("---")

# ── CONFIDENCE DISTRIBUTION ───────────────────────────────────────────────────

st.markdown("## Final Confidence Distribution")

if "metrics.final_confidence" in runs.columns:
    conf_data = runs["metrics.final_confidence"].dropna()

    bins = [0, 50, 70, 85, 95, 100]
    labels = ["0-50%", "50-70%", "70-85%", "85-95%", "95-100%"]
    binned = pd.cut(conf_data, bins=bins, labels=labels)
    dist   = binned.value_counts().sort_index()

    st.bar_chart(dist)

st.markdown("---")

# ── RAW DATA ──────────────────────────────────────────────────────────────────

st.markdown("## All Game Sessions")

display_cols = []
for col in ["params.final_guess", "metrics.total_questions",
            "metrics.final_confidence", "start_time"]:
    if col in runs.columns:
        display_cols.append(col)

if display_cols:
    display_df = runs[display_cols].copy()
    display_df.columns = [
        c.replace("params.", "").replace("metrics.", "")
        for c in display_cols
    ]
    st.dataframe(display_df, use_container_width=True)

st.markdown("---")
st.caption("Refreshes every 30 seconds. Data stored in mlflow.db")
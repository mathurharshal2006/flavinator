# src/app.py
#
# Streamlit frontend for Flavinator.
#
# WHY STREAMLIT:
#   Streamlit turns Python scripts into web UIs.
#   No HTML, no CSS, no JavaScript needed.
#   Perfect for ML projects and demos.
#
# HOW IT WORKS:
#   Streamlit reruns the entire script top to bottom
#   every time the user clicks a button or interacts.
#   st.session_state stores data between reruns.
#
# HOW TO RUN:
#   streamlit run src/app.py

import streamlit as st
import requests
import json
import os 

# ── CONFIG ────────────────────────────────────────────────────────────────────

API_URL = os.getenv("API_URL", "http://localhost:8000")

# page config must be the first streamlit command
st.set_page_config(
    page_title = "Flavinator",
    page_icon  = ":fork_and_knife:",
    layout     = "centered"
)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
#
# WHY SESSION STATE:
#   Streamlit reruns the whole script on every interaction.
#   Without session_state, all variables reset on every click.
#   session_state persists data across reruns - like memory for the app.
#
# Think of it like global variables that survive page refreshes.

if "game_started"    not in st.session_state:
    st.session_state.game_started     = False

if "game_over"       not in st.session_state:
    st.session_state.game_over        = False

if "current_feature" not in st.session_state:
    st.session_state.current_feature  = None

if "current_question" not in st.session_state:
    st.session_state.current_question = None

if "current_options" not in st.session_state:
    st.session_state.current_options  = []

if "question_number" not in st.session_state:
    st.session_state.question_number  = 0

if "history"         not in st.session_state:
    st.session_state.history          = []

if "final_guess"     not in st.session_state:
    st.session_state.final_guess      = None

if "confidence"      not in st.session_state:
    st.session_state.confidence       = 0.0

if "top_5"           not in st.session_state:
    st.session_state.top_5            = []


# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def start_new_game():
    """Calls /new-game endpoint and updates session state."""
    try:
        response = requests.get(f"{API_URL}/new-game")
        data     = response.json()

        st.session_state.game_started     = True
        st.session_state.game_over        = False
        st.session_state.current_feature  = data["feature"]
        st.session_state.current_question = data["question"]
        st.session_state.current_options  = data["options"]
        st.session_state.question_number  = 1
        st.session_state.history          = []
        st.session_state.final_guess      = None
        st.session_state.confidence       = 0.0
        st.session_state.top_5            = []

    except Exception as e:
        st.error(f"Could not connect to API. Is the server running? Error: {e}")


def submit_answer(feature, answer):
    """
    Sends answer to /answer endpoint.
    Updates session state with response.
    """
    try:
        response = requests.post(
            f"{API_URL}/answer",
            json = {"feature": feature, "answer": answer}
        )
        data = response.json()

        # save this answer to history
        st.session_state.history.append({
            "question": st.session_state.current_question,
            "answer"  : answer
        })

        st.session_state.confidence = data["confidence"]
        st.session_state.top_5      = data["top_5"]

        if data["game_over"]:
            st.session_state.game_over    = True
            st.session_state.final_guess  = data["final_guess"]

        else:
            nq = data["next_question"]
            st.session_state.current_feature  = nq["feature"]
            st.session_state.current_question = nq["question"]
            st.session_state.current_options  = nq["options"]
            st.session_state.question_number  = nq["question_number"]

    except Exception as e:
        st.error(f"Error sending answer: {e}")


# ── UI ────────────────────────────────────────────────────────────────────────

# title
st.title("Flavinator")
st.markdown("Think of any dish from anywhere in the world. I will guess it.")
st.markdown("---")

# ── NOT STARTED ───────────────────────────────────────────────────────────────

if not st.session_state.game_started:
    st.markdown("### How to play")
    st.markdown("1. Click **Start Game**")
    st.markdown("2. Think of any dish")
    st.markdown("3. Answer the questions honestly")
    st.markdown("4. Flavinator will guess your dish")

    if st.button("Start Game", type="primary", use_container_width=True):
        start_new_game()
        st.rerun()

# ── GAME OVER ─────────────────────────────────────────────────────────────────

elif st.session_state.game_over:
    st.markdown("## My guess is...")
    st.markdown(f"# {st.session_state.final_guess}")

    confidence = st.session_state.confidence
    st.progress(int(confidence) / 100)
    st.markdown(f"Confidence: **{confidence}%**")

    st.markdown("---")

    # show question history
    if st.session_state.history:
        st.markdown("### Questions asked")
        for i, item in enumerate(st.session_state.history, 1):
            st.markdown(f"**Q{i}:** {item['question']}  →  {item['answer']}")

    st.markdown("---")

    # show top 5 dishes considered
    if st.session_state.top_5:
        st.markdown("### Top candidates considered")
        for dish, prob in st.session_state.top_5:
            st.progress(int(prob * 100))
            st.caption(f"{dish}  {round(prob * 100, 1)}%")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Play Again", type="primary", use_container_width=True):
            start_new_game()
            st.rerun()
    with col2:
        if st.button("Reset", use_container_width=True):
            st.session_state.game_started = False
            st.rerun()

# ── GAME IN PROGRESS ──────────────────────────────────────────────────────────

else:
    # progress indicator
    st.markdown(f"Question **{st.session_state.question_number}** of 8")
    progress_val = (st.session_state.question_number - 1) / 8
    st.progress(progress_val)

    st.markdown("---")

    # current question
    st.markdown(f"### {st.session_state.current_question}")

    # answer buttons
    # WHY BUTTONS NOT DROPDOWN:
    #   Buttons are faster to click and more game-like.
    #   Each button click calls submit_answer and reruns the page.
    options = st.session_state.current_options
    cols    = st.columns(min(len(options), 3))

    for i, option in enumerate(options):
        col = cols[i % len(cols)]
        with col:
            if st.button(
                str(option),
                key              = f"opt_{i}_{option}",
                use_container_width = True
            ):
                submit_answer(
                    st.session_state.current_feature,
                    str(option)
                )
                st.rerun()

    st.markdown("---")

    # show history of answers so far
    if st.session_state.history:
        st.markdown("### Answered so far")
        for i, item in enumerate(st.session_state.history, 1):
            st.markdown(f"**Q{i}:** {item['question']}  →  **{item['answer']}**")
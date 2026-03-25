# src/api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
import mlflow

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game.engine import FlavinatorEngine

app = FastAPI(
    title       = "Flavinator API",
    description = "AI food guessing game using Naive Bayes + Decision Tree",
    version     = "1.0.0"
)

# mlflow experiment
# WHY SET TRACKING URI:
#   By default MLflow stores data in ./mlruns folder
#   We explicitly set it so Docker can find it correctly
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("flavinator_game_sessions")

# global engine instance
engine = FlavinatorEngine()

# global mlflow run id
# WHY GLOBAL: one MLflow run per game session
# we start the run on /new-game and end it when game is over
current_run_id = None


class AnswerRequest(BaseModel):
    feature : str
    answer  : str


@app.get("/")
def health_check():
    return {
        "status" : "running",
        "game"   : "Flavinator",
        "version": "1.0.0"
    }


@app.get("/new-game")
def new_game():
    """
    Starts a new game and a new MLflow run.

    WHY START MLFLOW RUN HERE:
      Each game = one MLflow run.
      We track the entire game as one unit of observation.
      This lets us ask questions like:
        "What is the average number of questions per game?"
        "Which dishes take the most questions to guess?"
    """
    global current_run_id

    # end previous run if exists
    if current_run_id:
        try:
            mlflow.end_run()
        except:
            pass

    engine.reset()

    # start a new MLflow run for this game session
    with mlflow.start_run() as run:
        current_run_id = run.info.run_id
        mlflow.set_tag("game_status", "in_progress")
        mlflow.log_param("max_questions", engine.MAX_QUESTIONS)
        mlflow.log_param("total_dishes",  len(engine.df))

    feature, question, ig = engine.get_next_question()
    options               = engine.get_options_for_feature(feature)

    return {
        "message"        : "New game started! Think of a dish.",
        "question_number": 1,
        "feature"        : feature,
        "question"       : question,
        "options"        : options,
        "ig_score"       : round(ig, 4),
        "run_id"         : current_run_id
    }


@app.post("/answer")
def process_answer(request: AnswerRequest):
    """
    Processes answer and logs to MLflow.

    WHAT WE LOG:
      - Each question and answer as a parameter
      - Running confidence after each answer
      - Final result when game ends
    """
    global current_run_id

    if engine.game_over:
        raise HTTPException(
            status_code = 400,
            detail      = "Game is over. Call /new-game to start again."
        )

    # log this answer to mlflow
    # WHY LOG EACH ANSWER:
    #   Later we can analyze which features were most commonly asked
    #   and which answers led to correct guesses
    if current_run_id:
        with mlflow.start_run(run_id=current_run_id):
            mlflow.log_param(
                f"q{engine.question_count + 1}_feature",
                request.feature
            )
            mlflow.log_param(
                f"q{engine.question_count + 1}_answer",
                request.answer
            )

    state = engine.process_answer(request.feature, request.answer)

    # log confidence after this answer
    if current_run_id:
        with mlflow.start_run(run_id=current_run_id):
            mlflow.log_metric(
                "confidence",
                state["confidence"],
                step=state["question_count"]
            )
            mlflow.log_metric(
                "dishes_remaining",
                state["remaining"],
                step=state["question_count"]
            )

    # game over - log final results
    if state["game_over"]:
        if current_run_id:
            with mlflow.start_run(run_id=current_run_id):
                mlflow.log_metric("total_questions",  state["question_count"])
                mlflow.log_metric("final_confidence", state["confidence"])
                mlflow.log_param("final_guess",       state["final_guess"])
                mlflow.set_tag("game_status",         "completed")

        return {
            **state,
            "next_question": None,
            "message"      : f"I guess it is {state['final_guess']}!"
        }

    feature, question, ig = engine.get_next_question()

    if feature is None:
        engine.game_over   = True
        engine.final_guess = state["top_dish"]
        return {
            **state,
            "game_over"    : True,
            "final_guess"  : state["top_dish"],
            "next_question": None,
            "message"      : f"I guess it is {state['top_dish']}!"
        }

    options = engine.get_options_for_feature(feature)

    return {
        **state,
        "next_question": {
            "feature"        : feature,
            "question"       : question,
            "options"        : options,
            "question_number": state["question_count"] + 1
        },
        "message": "Keep answering!"
    }


@app.get("/reset")
def reset_game():
    engine.reset()
    return {"message": "Game reset successfully."}


@app.get("/stats")
def get_stats():
    remaining           = engine.dt.get_remaining_dishes()
    dish, confidence, _ = engine.nb.predict()
    return {
        "questions_asked" : engine.question_count,
        "dishes_remaining": len(remaining),
        "top_guess"       : dish,
        "confidence"      : round(confidence * 100, 1),
        "remaining_dishes": remaining,
        "game_over"       : engine.game_over
    }
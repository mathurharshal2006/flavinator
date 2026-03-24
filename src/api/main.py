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

# mlflow setup
mlflow.set_experiment("flavinator_game_sessions")

# global engine
engine = FlavinatorEngine()


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
    Resets engine completely and returns first question.
    Called every time user starts a new game.
    """
    # reset engine state fully
    engine.reset()

    # verify reset worked
    if engine.game_over:
        raise HTTPException(status_code=500,
                            detail="Engine failed to reset")

    feature, question, ig = engine.get_next_question()
    options               = engine.get_options_for_feature(feature)

    return {
        "message"        : "New game started! Think of a dish.",
        "question_number": 1,
        "feature"        : feature,
        "question"       : question,
        "options"        : options,
        "ig_score"       : round(ig, 4)
    }


@app.post("/answer")
def process_answer(request: AnswerRequest):
    """
    Processes user answer.
    Returns next question or final guess.
    """
    if engine.game_over:
        raise HTTPException(
            status_code = 400,
            detail      = "Game is over. Call /new-game to start again."
        )

    # log to mlflow
    with mlflow.start_run(run_name=f"q_{engine.question_count + 1}",
                          nested=True):
        mlflow.log_param("feature", request.feature)
        mlflow.log_param("answer",  request.answer)
        mlflow.log_metric("question_number", engine.question_count + 1)

    state = engine.process_answer(request.feature, request.answer)

    if state["game_over"]:
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
    remaining            = engine.dt.get_remaining_dishes()
    dish, confidence, _  = engine.nb.predict()
    return {
        "questions_asked" : engine.question_count,
        "dishes_remaining": len(remaining),
        "top_guess"       : dish,
        "confidence"      : round(confidence * 100, 1),
        "remaining_dishes": remaining,
        "game_over"       : engine.game_over
    }
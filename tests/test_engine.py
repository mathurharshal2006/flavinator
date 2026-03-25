# tests/test_engine.py
#
# Unit tests for Flavinator.
#
# WHY TESTS:
#   Tests verify your code does what you think it does.
#   Without tests, you only know something broke AFTER it broke.
#   With tests, you know immediately when something breaks.
#
# HOW PYTEST WORKS:
#   Any function starting with test_ is automatically run by pytest.
#   If the function raises no error -> test passes (green)
#   If the function raises an error -> test fails (red)
#
# RUN TESTS:
#   pytest tests/

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import torch
from data.dishes          import get_dishes_dataset
from models.naive_bayes   import FlavinatorNaiveBayes, encode_dataset
from models.decision_tree import DecisionTreeSelector, entropy, information_gain
from game.engine          import FlavinatorEngine


# ── DATASET TESTS ─────────────────────────────────────────────────────────────

def test_dataset_loads():
    """Dataset must load without errors."""
    df = get_dishes_dataset()
    assert df is not None


def test_dataset_has_40_dishes():
    """We expect exactly 40 dishes."""
    df = get_dishes_dataset()
    assert len(df) == 40, f"Expected 40 dishes, got {len(df)}"


def test_dataset_has_correct_columns():
    """Dataset must have all required columns."""
    df       = get_dishes_dataset()
    required = [
        "name", "cuisine", "is_vegetarian", "is_spicy",
        "main_ingredient", "cooking_method", "is_sweet",
        "served_hot", "has_sauce", "texture", "meal_type"
    ]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_dataset_no_missing_values():
    """No dish should have empty values."""
    df = get_dishes_dataset()
    assert df.isnull().sum().sum() == 0, "Dataset has missing values"


def test_dataset_contains_biryani():
    """Biryani must be in the dataset."""
    df = get_dishes_dataset()
    assert "Biryani" in df["name"].values


# ── ENCODING TESTS ────────────────────────────────────────────────────────────

def test_encode_dataset_returns_numbers():
    """
    After encoding, all feature columns must contain only numbers.
    WHY: PyTorch cannot work with text, only numbers.
    """
    df           = get_dishes_dataset()
    encoded, _   = encode_dataset(df)

    for col in encoded.columns:
        if col == "name":
            continue
        assert encoded[col].dtype in ["int64", "float64"], \
            f"Column {col} is not numeric after encoding"


def test_encoders_contain_string_keys():
    """
    Encoders must map strings to numbers.
    WHY: user answers come as strings like 'True', 'Indian'
    """
    df         = get_dishes_dataset()
    _, encoders = encode_dataset(df)

    for feature, mapping in encoders.items():
        for key in mapping.keys():
            assert isinstance(key, str), \
                f"Encoder key for {feature} is not a string: {key}"


# ── NAIVE BAYES TESTS ─────────────────────────────────────────────────────────

def test_naive_bayes_trains():
    """Model must train without errors."""
    df    = get_dishes_dataset()
    model = FlavinatorNaiveBayes()
    model.train(df)
    assert model.log_prior is not None


def test_prior_probabilities_equal():
    """
    Before any questions, all dishes must have equal probability.
    WHY: we have no information yet so all dishes are equally likely.
    """
    df    = get_dishes_dataset()
    model = FlavinatorNaiveBayes()
    model.train(df)

    probs = model.get_probabilities()

    # all probabilities should be equal = 1/40 = 0.025
    expected = 1.0 / len(df)
    for p in probs:
        assert abs(p.item() - expected) < 0.001, \
            f"Prior probability {p.item()} != expected {expected}"


def test_probabilities_sum_to_one():
    """
    Probabilities must always sum to 1.0.
    WHY: they represent a probability distribution.
         if they do not sum to 1.0, the math is wrong.
    """
    df    = get_dishes_dataset()
    model = FlavinatorNaiveBayes()
    model.train(df)

    probs = model.get_probabilities()
    total = probs.sum().item()

    assert abs(total - 1.0) < 0.001, \
        f"Probabilities sum to {total}, expected 1.0"


def test_update_increases_matching_dish_probability():
    """
    After answering a question, matching dishes must have higher probability.
    WHY: this is the core of Bayes Theorem.
         evidence should increase belief in matching hypotheses.
    """
    df    = get_dishes_dataset()
    model = FlavinatorNaiveBayes()
    model.train(df)

    # get biryani index
    biryani_idx = model.dish_names.index("Biryani")

    # probability before
    probs_before = model.get_probabilities()
    prob_before  = probs_before[biryani_idx].item()

    # biryani is spicy - after answering True, biryani prob should go up
    model.update("is_spicy", "True")

    probs_after = model.get_probabilities()
    prob_after  = probs_after[biryani_idx].item()

    assert prob_after > prob_before, \
        f"Biryani prob did not increase after spicy=True: {prob_before} -> {prob_after}"


def test_reset_restores_prior():
    """
    After reset, probabilities must return to equal distribution.
    WHY: reset starts a new game, all dishes are equal again.
    """
    df    = get_dishes_dataset()
    model = FlavinatorNaiveBayes()
    model.train(df)

    # update then reset
    model.update("is_spicy", "True")
    model.reset()

    probs    = model.get_probabilities()
    expected = 1.0 / len(df)

    for p in probs:
        assert abs(p.item() - expected) < 0.001, \
            "Probabilities did not reset to equal distribution"


# ── DECISION TREE TESTS ───────────────────────────────────────────────────────

def test_entropy_of_uniform_distribution():
    """
    Uniform distribution has maximum entropy.
    WHY: when all possibilities are equal, uncertainty is maximum.
    """
    # 40 equal probabilities
    probs = torch.ones(40) / 40
    h     = entropy(probs)

    # entropy of uniform dist over 40 items = log2(40) = 5.32
    expected = torch.log2(torch.tensor(40.0)).item()
    assert abs(h - expected) < 0.01, \
        f"Entropy {h} != expected {expected}"


def test_entropy_of_certain_distribution():
    """
    Certain distribution (one dish at 100%) has zero entropy.
    WHY: no uncertainty = no information needed = zero entropy.
    """
    probs    = torch.zeros(40)
    probs[0] = 1.0
    h        = entropy(probs)

    assert abs(h - 0.0) < 0.001, \
        f"Entropy of certain distribution should be 0, got {h}"


def test_decision_tree_returns_best_question():
    """Decision tree must return a valid feature name."""
    df       = get_dishes_dataset()
    selector = DecisionTreeSelector()
    selector.fit(df)

    feature, ig, _ = selector.get_best_question()

    assert feature is not None
    assert feature in df.columns
    assert ig > 0


def test_cuisine_has_highest_information_gain():
    """
    Cuisine should have the highest information gain.
    WHY: it splits 40 dishes into 13 groups immediately.
         more groups = more information = higher IG.
    """
    df       = get_dishes_dataset()
    selector = DecisionTreeSelector()
    selector.fit(df)

    feature, ig, all_ig = selector.get_best_question()

    assert feature == "cuisine", \
        f"Expected cuisine to have highest IG, got {feature}"


def test_update_reduces_remaining_dishes():
    """
    After answering cuisine=Indian, only Indian dishes should remain.
    """
    df       = get_dishes_dataset()
    selector = DecisionTreeSelector()
    selector.fit(df)

    initial_count = len(selector.remaining_dish_idx)
    selector.update("cuisine", "Indian")
    after_count   = len(selector.remaining_dish_idx)

    assert after_count < initial_count, \
        "Remaining dishes did not decrease after answering cuisine"


# ── ENGINE TESTS ──────────────────────────────────────────────────────────────

def test_engine_loads():
    """Engine must initialize without errors."""
    engine = FlavinatorEngine()
    assert engine is not None
    assert not engine.game_over


def test_engine_guesses_biryani():
    """
    Engine must correctly guess Biryani given correct answers.
    This is an integration test - tests all components together.
    """
    engine = FlavinatorEngine()

    biryani_answers = {
        "cuisine"        : "Indian",
        "is_vegetarian"  : "False",
        "is_spicy"       : "True",
        "main_ingredient": "rice",
        "cooking_method" : "steamed",
        "is_sweet"       : "False",
        "served_hot"     : "True",
        "has_sauce"      : "False",
        "texture"        : "soft",
        "meal_type"      : "lunch"
    }

    for _ in range(engine.MAX_QUESTIONS):
        if engine.game_over:
            break
        feature, _, _ = engine.get_next_question()
        if feature is None:
            break
        answer = biryani_answers[feature]
        engine.process_answer(feature, answer)

    assert engine.game_over, "Game should be over"
    assert engine.final_guess == "Biryani", \
        f"Expected Biryani, got {engine.final_guess}"


def test_engine_reset_works():
    """After reset, engine must be ready for a new game."""
    engine = FlavinatorEngine()

    # play a few moves
    feature, _, _ = engine.get_next_question()
    engine.process_answer(feature, "Indian")

    # reset
    engine.reset()

    assert engine.question_count == 0
    assert not engine.game_over
    assert engine.final_guess is None
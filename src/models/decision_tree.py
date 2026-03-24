# src/models/decision_tree.py
#
# Decides which question to ask next during the game.
#
# Uses Information Gain to find the question that
# eliminates the most dishes with each answer.
#
# HOW TO READ THIS FILE:
#   1. entropy()           - measures uncertainty in a set of dishes
#   2. information_gain()  - measures how much a question reduces uncertainty
#   3. DecisionTreeSelector- picks the best question at each game step

import torch
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dishes import get_dishes_dataset
from models.naive_bayes import encode_dataset


# ── STEP 1: ENTROPY ───────────────────────────────────────────────────────────
#
# Entropy measures how uncertain we are about which dish it is.
#
# INTUITION:
#   Imagine you have a bag with 40 different colored balls.
#   If all 40 colors are different → maximum uncertainty → high entropy
#   If 39 balls are red, 1 is blue → almost certain → low entropy
#   If all 40 balls are red → completely certain → zero entropy
#
# FORMULA:
#   H = -sum( p * log2(p) ) for each dish
#
# WHY LOG2:
#   We use log base 2 so entropy is measured in "bits"
#   1 bit = one yes/no question needed to identify something
#   5.32 bits = about 5-6 yes/no questions needed (which matches our game!)

def entropy(probabilities):
    """
    Calculates entropy of a probability distribution.

    Args:
        probabilities: torch tensor of probabilities, must sum to 1.0
                       example: tensor([0.025, 0.025, ... 0.025]) for 40 dishes

    Returns:
        float: entropy value. Higher = more uncertain.

    Example:
        40 equal dishes  → entropy = 5.32 (very uncertain)
        2 equal dishes   → entropy = 1.0  (50/50, one question needed)
        1 dish at 100%   → entropy = 0.0  (certain, no questions needed)
    """
    # remove dishes with zero probability
    # WHY: log(0) is undefined (negative infinity), causes errors
    # dishes already eliminated have prob ~0, safe to ignore them
    probs = probabilities[probabilities > 1e-9]

    if len(probs) == 0:
        return 0.0

    # entropy formula: H = -sum(p * log2(p))
    # torch.log2 computes log base 2
    h = -torch.sum(probs * torch.log2(probs))
    return h.item()


# ── STEP 2: INFORMATION GAIN ──────────────────────────────────────────────────
#
# Information Gain tells us how much a question reduces entropy.
#
# FORMULA:
#   IG(feature) = entropy_before - weighted_average_entropy_after
#
# The "weighted average" means:
#   If asking "meal_type" splits 40 dishes into:
#     breakfast: 7 dishes  (7/40 = 17.5% of all dishes)
#     lunch:    11 dishes  (11/40 = 27.5% of all dishes)
#     dinner:    8 dishes  (8/40 = 20.0% of all dishes)
#     snack:     7 dishes  (7/40 = 17.5% of all dishes)
#     dessert:   7 dishes  (7/40 = 17.5% of all dishes)
#
#   Weighted entropy = (7/40)*H(breakfast) + (11/40)*H(lunch) + ...
#
#   Groups with more dishes carry more weight in the average.

def information_gain(df, feature, current_dish_indices, encoded_df):
    """
    Calculates how much information we gain by asking about this feature.

    Args:
        df                  : original dataset
        feature             : column name to evaluate e.g. "is_spicy"
        current_dish_indices: list of dish indices still in the game
                              (dishes not yet eliminated)
        encoded_df          : dataset with text converted to numbers

    Returns:
        float: information gain. Higher = better question to ask.
    """
    # get only the dishes still in consideration
    current_dishes = encoded_df.iloc[current_dish_indices]
    n_current      = len(current_dishes)

    if n_current == 0:
        return 0.0

    # entropy before asking this question
    # all remaining dishes are equally likely so prob = 1/n for each
    uniform_probs  = torch.ones(n_current) / n_current
    h_before       = entropy(uniform_probs)

    # entropy after asking this question
    # we look at each possible answer value and calculate:
    #   - how many dishes have that value
    #   - entropy of that group
    # then take the weighted average

    unique_values    = current_dishes[feature].unique()
    weighted_h_after = 0.0

    for value in unique_values:
        # dishes that have this value for this feature
        subset      = current_dishes[current_dishes[feature] == value]
        n_subset    = len(subset)

        if n_subset == 0:
            continue

        # weight = proportion of current dishes in this group
        weight      = n_subset / n_current

        # entropy of this group (uniform distribution within group)
        subset_probs = torch.ones(n_subset) / n_subset
        h_subset     = entropy(subset_probs)

        weighted_h_after += weight * h_subset

    # information gain = how much entropy was reduced
    ig = h_before - weighted_h_after
    return ig


# ── STEP 3: THE QUESTION SELECTOR ─────────────────────────────────────────────

class DecisionTreeSelector:
    """
    At each step of the game, picks the best question to ask next.

    HOW IT WORKS:
      1. Keep track of which dishes are still possible
      2. For each feature not yet asked, calculate Information Gain
      3. Ask the feature with highest Information Gain
      4. After user answers, eliminate dishes that don't match
      5. Repeat until confident

    WHY THIS WORKS WELL WITH NAIVE BAYES:
      Naive Bayes handles the probability math (scoring dishes)
      Decision Tree handles the strategy (choosing questions)
      Together they form a complete guessing system.
    """

    def __init__(self):
        self.df                  = None
        self.encoded_df          = None
        self.encoders            = None
        self.feature_columns     = []
        self.asked_features      = []
        self.remaining_dish_idx  = []


    def fit(self, df):
        """
        Prepares the selector with the dataset.

        Args:
            df: pandas DataFrame from get_dishes_dataset()
        """
        self.df              = df
        self.encoded_df, self.encoders = encode_dataset(df)
        self.feature_columns = [c for c in df.columns if c != "name"]
        self.reset()


    def reset(self):
        """Resets to start of a new game."""
        self.asked_features     = []
        # all dishes are candidates at the start
        self.remaining_dish_idx = list(range(len(self.df)))


    def get_best_question(self):
        """
        Finds the feature with highest Information Gain
        among features not yet asked.

        Returns:
            best_feature    : string, name of best feature to ask
            best_ig         : float, its information gain score
            all_ig_scores   : dict of all features and their IG scores
                              (useful for understanding and debugging)
        """
        # features we have not asked yet
        remaining_features = [
            f for f in self.feature_columns
            if f not in self.asked_features
        ]

        if not remaining_features:
            return None, 0.0, {}

        ig_scores = {}

        for feature in remaining_features:
            ig = information_gain(
                self.df,
                feature,
                self.remaining_dish_idx,
                self.encoded_df
            )
            ig_scores[feature] = round(ig, 4)

        # sort by information gain descending
        ig_scores    = dict(
            sorted(ig_scores.items(), key=lambda x: x[1], reverse=True)
        )
        best_feature = list(ig_scores.keys())[0]
        best_ig      = ig_scores[best_feature]

        return best_feature, best_ig, ig_scores


    def update(self, feature, answer_str):
        """
        After user answers a question, eliminate dishes
        that do not match the answer.

        Args:
            feature    : which feature was answered e.g. "is_spicy"
            answer_str : user's answer as string e.g. "True"
        """
        self.asked_features.append(feature)

        encoder = self.encoders[feature]

        if answer_str not in encoder:
            print(f"Unknown answer '{answer_str}' for '{feature}'")
            return

        answer_encoded = encoder[answer_str]

        # keep only dishes where this feature matches the answer
        new_remaining = []
        for idx in self.remaining_dish_idx:
            dish_value = int(self.encoded_df.iloc[idx][feature])
            if dish_value == answer_encoded:
                new_remaining.append(idx)

        self.remaining_dish_idx = new_remaining

        remaining_names = [self.df.iloc[i]["name"]
                           for i in self.remaining_dish_idx]
        print(f"  Dishes remaining: {len(remaining_names)}")
        if len(remaining_names) <= 5:
            print(f"  Candidates: {remaining_names}")


    def get_remaining_dishes(self):
        """Returns names of dishes still in consideration."""
        return [self.df.iloc[i]["name"] for i in self.remaining_dish_idx]


# ── STEP 4: TEST ──────────────────────────────────────────────────────────────
# run: python3 src/models/decision_tree.py

if __name__ == "__main__":

    df       = get_dishes_dataset()
    selector = DecisionTreeSelector()
    selector.fit(df)

    print("=" * 55)
    print("DECISION TREE - QUESTION SELECTOR TEST")
    print("=" * 55)

    # show information gain for all features at the start
    print("\nINFORMATION GAIN for all features (before any question):")
    print("Higher score = better question to ask first\n")

    _, _, all_ig = selector.get_best_question()
    for feature, ig in all_ig.items():
        bar = "#" * int(ig * 40)
        print(f"  {feature:<20} {bar} {ig:.4f}")

    print("\n" + "=" * 55)
    print("SIMULATING SMART GAME - Thinking of: BIRYANI")
    print("=" * 55)

    # answers for Biryani
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

    question_num = 1

    while len(selector.remaining_dish_idx) > 1 and question_num <= 10:
        # get the smartest question to ask right now
        best_feature, best_ig, _ = selector.get_best_question()

        if best_feature is None:
            break

        answer = biryani_answers[best_feature]

        print(f"\nQ{question_num}: {best_feature.replace('_', ' ').upper()}?")
        print(f"     Best question (IG = {best_ig:.4f})")
        print(f"     Answer: {answer}")

        selector.update(best_feature, answer)
        question_num += 1

    remaining = selector.get_remaining_dishes()
    print("\n" + "=" * 55)
    if len(remaining) == 1:
        print(f"IDENTIFIED: {remaining[0].upper()}")
        print(f"Questions needed: {question_num - 1}")
    else:
        print(f"Narrowed to: {remaining}")
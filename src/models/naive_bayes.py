# src/models/naive_bayes.py
#
# The brain of Flavinator.
# Uses Naive Bayes with PyTorch tensors to guess the dish.
#
# HOW TO READ THIS FILE:
#   1. encode_dataset()       - converts text to numbers
#   2. FlavinatorNaiveBayes   - the main class
#      - train()              - learns from dataset
#      - update()             - updates belief after each answer
#      - predict()            - returns best guess
#      - reset()              - starts a new game

import torch
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dishes import get_dishes_dataset


# ── STEP 1: ENCODING ──────────────────────────────────────────────────────────
#
# WHY: PyTorch only works with numbers, not words.
#      We need to convert every text value to an integer.
#
# EXAMPLE:
#   cuisine column has: Indian, Italian, Mexican, Japanese ...
#   After encoding:     0,      1,       2,       3        ...
#
# We also save the mapping so we can decode later
# (convert number back to word when showing results)

def encode_dataset(df):
    """
    Converts all text columns in the dataset to integers.

    Args:
        df: pandas DataFrame from get_dishes_dataset()

    Returns:
        encoded_df : same table but all numbers
        encoders   : dict mapping column -> {word: number}
    """
    encoded_df = df.copy()
    encoders   = {}

    for column in df.columns:
        if column == "name":
            continue

        # FIX: convert every value to string first
        # This makes True -> "True", False -> "False"
        # So when the user passes "True" as a string it will match
        encoded_df[column] = df[column].astype(str)

        unique_values = encoded_df[column].unique()
        mapping       = {val: idx for idx, val in enumerate(unique_values)}

        encoders[column]    = mapping
        encoded_df[column]  = encoded_df[column].map(mapping)

    return encoded_df, encoders


# ── STEP 2: THE MODEL ─────────────────────────────────────────────────────────

class FlavinatorNaiveBayes:
    """
    Naive Bayes classifier for guessing food dishes.

    WHY A CLASS:
      The model needs to remember things between questions
      during a game session - current probabilities, what was
      already asked, the encoders. A class stores all of this
      as self.something so it persists between function calls.

    HOW NAIVE BAYES WORKS IN THIS CLASS:
      1. train()  - reads dataset, computes probability tables
      2. update() - after each user answer, multiplies probabilities
      3. predict()- finds dish with highest probability
    """

    def __init__(self):
        # filled during training
        self.dish_names      = []
        self.num_dishes      = 0
        self.feature_columns = []
        self.encoders        = {}

        # the core probability tensors
        # WHY TENSORS: PyTorch tensors allow us to do math on
        # all 40 dishes simultaneously instead of looping one by one
        self.log_prior       = None   # shape: (40,)
        self.feature_probs   = {}     # shape per feature: (40, num_unique_values)

        # game state - changes every question
        self.current_log_probs = None
        self.questions_asked   = []


    def train(self, df):
        """
        Reads the dataset and learns two things:

        1. PRIOR: how likely is each dish before any questions?
           Since all 40 dishes are equally represented,
           P(any dish) = 1/40 = 0.025

        2. LIKELIHOOD: given a dish, how likely is each feature value?
           Example:
             P(is_spicy=True  | Biryani)     = 1.0
             P(is_spicy=False | Biryani)     = 0.0
             P(is_spicy=True  | Gulab Jamun) = 0.0
             P(is_spicy=False | Gulab Jamun) = 1.0

        WHY LOG PROBABILITIES:
          If we multiply many small numbers together, the result
          becomes so tiny that computers round it to zero.
          Example: 0.025 * 0.025 * 0.025 * 0.025 = 0.00000039
          After 10 features this becomes 0.000000000000000001

          Logarithms fix this. Instead of multiplying, we add:
            log(a * b) = log(a) + log(b)

          This is mathematically identical but numerically stable.
        """
        print("Training Flavinator...")

        encoded_df, self.encoders = encode_dataset(df)

        self.dish_names      = df["name"].tolist()
        self.num_dishes      = len(self.dish_names)
        self.feature_columns = [c for c in df.columns if c != "name"]

        print(f"  Dishes   : {self.num_dishes}")
        print(f"  Features : {len(self.feature_columns)}")

        # ── PRIOR PROBABILITY ─────────────────────────────────────────────
        # each dish has equal probability = 1/40
        # torch.ones(40) creates tensor([1., 1., 1., ... 1.]) with 40 ones
        # dividing by num_dishes gives tensor([0.025, 0.025, ... 0.025])
        # torch.log converts to log space

        prior          = torch.ones(self.num_dishes) / self.num_dishes
        self.log_prior = torch.log(prior)

        # ── LIKELIHOOD ────────────────────────────────────────────────────
        # for every feature, build a matrix of shape (num_dishes, num_values)
        # each cell = P(feature_value | dish)
        #
        # EXAMPLE for is_spicy (2 possible values: True=1, False=0):
        #
        #               False   True
        #   Biryani   [  0.0    1.0  ]   <- biryani is always spicy
        #   Gulab Jamun[ 1.0    0.0  ]   <- gulab jamun is never spicy
        #   Sushi     [  1.0    0.0  ]   <- sushi is never spicy
        #   Tacos     [  0.0    1.0  ]   <- tacos are always spicy

        for feature in self.feature_columns:
            num_vals     = len(self.encoders[feature])
            # create empty matrix filled with zeros
            # shape: (40 dishes, num unique values for this feature)
            prob_matrix  = torch.zeros(self.num_dishes, num_vals)

            for dish_idx in range(self.num_dishes):
                # what value does this dish have for this feature?
                dish_feature_val = int(encoded_df.iloc[dish_idx][feature])

                # LAPLACE SMOOTHING:
                # set all values to a tiny number instead of 0
                # WHY: if any probability is exactly 0.0, it will
                # wipe out the entire dish score no matter what else matches
                # A small fallback (1e-6) prevents this harsh punishment
                prob_matrix[dish_idx]                  = 1e-6
                prob_matrix[dish_idx][dish_feature_val] = 1.0

            # convert to log probabilities and store
            self.feature_probs[feature] = torch.log(prob_matrix)

        # initialize game state
        self.current_log_probs = self.log_prior.clone()

        print("Training complete.\n")


    def update(self, feature, value_str):
        """
        Called after user answers one question.

        This is where Bayes Theorem happens:
          new_score = old_score * P(answer | dish)

        In log space (addition instead of multiplication):
          new_log_score = old_log_score + log_P(answer | dish)

        So dishes whose features match the answer get higher scores.
        Dishes whose features do not match get scores near negative infinity.

        Args:
            feature   : string, which feature was asked e.g. "is_spicy"
            value_str : string, what user answered e.g. "True" or "False"
        """
        encoder = self.encoders[feature]

        # convert the user's text answer to a number
        # example: "True" -> 1, "Indian" -> 0
        if value_str not in encoder:
            print(f"Unknown answer '{value_str}' for feature '{feature}'")
            return

        value_idx = encoder[value_str]

        # get one column from the feature probability matrix
        # this gives us P(this answer | dish) for all 40 dishes at once
        # shape: (40,)
        log_likelihoods = self.feature_probs[feature][:, value_idx]

        # BAYES UPDATE - this is the core of the algorithm
        # add log likelihood to current log probabilities
        # in normal space this is: score = score * likelihood
        self.current_log_probs = self.current_log_probs + log_likelihoods

        self.questions_asked.append((feature, value_str))


    def get_probabilities(self):
        """
        Converts log scores back to real probabilities using softmax.

        WHAT IS SOFTMAX:
          Takes any list of numbers and converts them to probabilities
          that sum to exactly 1.0

          Example:
            scores  = [-10.2, -8.1, -3.4, -15.0]
            softmax -> [0.003, 0.025, 0.970, 0.001]

          The highest score gets the highest probability.
          All values sum to 1.0 so they are proper probabilities.

        WHY NOT JUST USE THE RAW SCORES:
          Raw log scores are negative numbers like -38.4, -42.1 etc.
          Softmax converts them to readable percentages.
        """
        probs = torch.softmax(self.current_log_probs, dim=0)
        return probs


    def predict(self):
        """
        Returns the current best guess.

        Returns:
            dish_name  : string, name of most likely dish
            confidence : float 0.0 to 1.0, how sure we are
            all_probs  : dict of all dishes sorted by probability
        """
        probs    = self.get_probabilities()

        # torch.argmax finds index of highest value in tensor
        best_idx   = torch.argmax(probs).item()
        confidence = probs[best_idx].item()
        dish_name  = self.dish_names[best_idx]

        # build readable dictionary of all dish probabilities
        all_probs = {
            self.dish_names[i]: round(probs[i].item(), 4)
            for i in range(self.num_dishes)
        }
        # sort highest probability first
        all_probs = dict(
            sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        )

        return dish_name, confidence, all_probs


    def reset(self):
        """Resets game state for a new round."""
        self.current_log_probs = self.log_prior.clone()
        self.questions_asked   = []
        print("Game reset. Think of a new dish!\n")


# ── STEP 3: TEST ──────────────────────────────────────────────────────────────
# run: python3 src/models/naive_bayes.py

if __name__ == "__main__":

    df    = get_dishes_dataset()
    model = FlavinatorNaiveBayes()
    model.train(df)

    print("=" * 55)
    print("SIMULATING A GAME")
    print("Thinking of: BIRYANI")
    print("=" * 55)

    # simulate user answering questions about Biryani
    questions = [
        ("is_spicy",        "True"),
        ("main_ingredient", "rice"),
        ("is_vegetarian",   "False"),
        ("served_hot",      "True"),
        ("has_sauce",       "False"),
        ("cooking_method",  "steamed"),
    ]

    for feature, answer in questions:
        print(f"\nQ: Is it {feature.replace('_', ' ')}?")
        print(f"A: {answer}")

        model.update(feature, answer)

        dish, confidence, all_probs = model.predict()

        print(f"\nTop guess : {dish} ({confidence*100:.1f}% confident)")
        print("Top 5 candidates:")

        # show top 5 with a simple bar
        for i, (name, prob) in enumerate(list(all_probs.items())[:5]):
            bar = "#" * int(prob * 40)
            print(f"  {name:<22} {bar} {prob*100:.1f}%")

    print("\n" + "=" * 55)
    print(f"FINAL GUESS: {dish.upper()}")
    print(f"Confidence : {confidence*100:.1f}%")
    print(f"Questions asked: {len(model.questions_asked)}")
# src/game/engine.py
#
# The game engine that combines:
#   - Decision Tree  : picks the smartest question
#   - Naive Bayes    : tracks dish probabilities
#
# This is the single class the frontend will talk to.
# It manages the full game flow from start to guess.

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dishes             import get_dishes_dataset
from models.naive_bayes      import FlavinatorNaiveBayes
from models.decision_tree    import DecisionTreeSelector


class FlavinatorEngine:
    """
    Combines Naive Bayes + Decision Tree into one game session.

    Naive Bayes  tracks probability of each dish after every answer.
    Decision Tree picks which question to ask next.

    WHY COMBINE BOTH:
      Decision Tree alone just eliminates dishes with hard yes/no logic.
      It breaks if the user gives a slightly wrong answer.

      Naive Bayes alone asks questions in a fixed order.
      It wastes questions on things that do not help much.

      Together:
        Decision Tree  = smart question ordering
        Naive Bayes    = soft probability scoring (handles uncertainty)
    """

    # guess when confidence crosses this threshold
    CONFIDENCE_THRESHOLD = 0.85

    # always guess after this many questions even if not confident
    MAX_QUESTIONS        = 8

    def __init__(self):
        self.df       = get_dishes_dataset()
        self.nb       = FlavinatorNaiveBayes()
        self.dt       = DecisionTreeSelector()

        # train both models on the same dataset
        self.nb.train(self.df)
        self.dt.fit(self.df)

        # game state
        self.question_count  = 0
        self.game_over       = False
        self.final_guess     = None


    def get_next_question(self):
        """
        Uses Decision Tree to pick the best question to ask right now.

        Returns:
            feature     : string, the feature to ask about
            question_str: human readable question
            ig_score    : information gain of this question
        """
        feature, ig, _ = self.dt.get_best_question()

        if feature is None:
            return None, None, 0.0

        # convert feature name to a readable question
        question_map = {
            "cuisine"        : "Which cuisine is it from?",
            "is_vegetarian"  : "Is it vegetarian?",
            "is_spicy"       : "Is it spicy?",
            "main_ingredient": "What is the main ingredient?",
            "cooking_method" : "How is it cooked?",
            "is_sweet"       : "Is it sweet?",
            "served_hot"     : "Is it served hot?",
            "has_sauce"      : "Does it have a sauce?",
            "texture"        : "What is the texture?",
            "meal_type"      : "What type of meal is it?"
        }

        question_str = question_map.get(feature, f"What is the {feature}?")
        return feature, question_str, ig


    def get_options_for_feature(self, feature):
        """
        Returns all possible answers for a given feature.

        WHY: The frontend needs to show buttons for each possible answer.
        Example: for "texture" show [soft, crunchy, creamy, chewy]

        Args:
            feature: string, feature name

        Returns:
            list of strings, all possible values
        """
        return list(self.dt.encoders[feature].keys())


    def process_answer(self, feature, answer):
        """
        Processes the user's answer.
        Updates both Naive Bayes and Decision Tree.

        Args:
            feature: string, which feature was answered
            answer : string, user's answer

        Returns:
            dict with current game state
        """
        # update Naive Bayes probabilities
        self.nb.update(feature, answer)

        # update Decision Tree (eliminate non-matching dishes)
        self.dt.update(feature, answer)

        self.question_count += 1

        # get current prediction
        dish, confidence, all_probs = self.nb.predict()

        # check if we should guess now
        remaining = self.dt.get_remaining_dishes()

        should_guess = (
            confidence >= self.CONFIDENCE_THRESHOLD or
            self.question_count >= self.MAX_QUESTIONS or
            len(remaining) == 1
        )

        if should_guess:
            self.game_over   = True
            self.final_guess = dish

        return {
            "top_dish"      : dish,
            "confidence"    : round(confidence * 100, 1),
            "remaining"     : len(remaining),
            "question_count": self.question_count,
            "game_over"     : self.game_over,
            "final_guess"   : self.final_guess,
            "top_5"         : list(all_probs.items())[:5]
        }


    def reset(self):
        """Resets the game for a new round."""
        self.nb.reset()
        self.dt.reset()
        self.question_count = 0
        self.game_over      = False
        self.final_guess    = None


# ── TEST ──────────────────────────────────────────────────────────────────────
# run: python3 src/game/engine.py

if __name__ == "__main__":

    engine = FlavinatorEngine()

    print("=" * 55)
    print("FLAVINATOR ENGINE TEST")
    print("Thinking of: SAMOSA")
    print("=" * 55)

    # answers for Samosa
    samosa_answers = {
        "cuisine"        : "Indian",
        "is_vegetarian"  : "True",
        "is_spicy"       : "True",
        "main_ingredient": "bread",
        "cooking_method" : "fried",
        "is_sweet"       : "False",
        "served_hot"     : "True",
        "has_sauce"      : "False",
        "texture"        : "crunchy",
        "meal_type"      : "snack"
    }

    while not engine.game_over:
        # get best question
        feature, question, ig = engine.get_next_question()

        if feature is None:
            break

        # get options for this feature
        options = engine.get_options_for_feature(feature)
        answer  = samosa_answers.get(feature, options[0])

        print(f"\nQ{engine.question_count + 1}: {question}")
        print(f"     Options : {options}")
        print(f"     Answer  : {answer}")

        # process the answer
        state = engine.process_answer(feature, answer)

        print(f"     Remaining dishes : {state['remaining']}")
        print(f"     Top guess        : {state['top_dish']} "
              f"({state['confidence']}%)")

    print("\n" + "=" * 55)
    print(f"FINAL GUESS : {engine.final_guess.upper()}")
    print(f"Questions   : {engine.question_count}")
    print("=" * 55)
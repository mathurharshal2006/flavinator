# src/game/engine.py

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dishes          import get_dishes_dataset
from models.naive_bayes   import FlavinatorNaiveBayes
from models.decision_tree import DecisionTreeSelector


class FlavinatorEngine:

    CONFIDENCE_THRESHOLD = 0.85
    MAX_QUESTIONS        = 8

    def __init__(self):
        self.df      = get_dishes_dataset()
        self.nb      = FlavinatorNaiveBayes()
        self.dt      = DecisionTreeSelector()
        self.nb.train(self.df)
        self.dt.fit(self.df)
        self.question_count = 0
        self.game_over      = False
        self.final_guess    = None

    def get_next_question(self):
        feature, ig, _ = self.dt.get_best_question()
        if feature is None:
            return None, None, 0.0
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
        return list(self.dt.encoders[feature].keys())

    def process_answer(self, feature, answer):
        # update both models
        self.nb.update(feature, answer)
        self.dt.update(feature, answer)
        self.question_count += 1

        # get naive bayes scores
        dish, confidence, all_probs = self.nb.predict()

        # get dishes still remaining after decision tree filtering
        remaining_names = self.dt.get_remaining_dishes()

        # CRITICAL FIX: only score dishes that decision tree allows
        filtered = {
            name: prob
            for name, prob in all_probs.items()
            if name in remaining_names
        }

        if filtered:
            # normalize filtered probabilities so they sum to 1
            total    = sum(filtered.values())
            filtered = {k: v / total for k, v in filtered.items()}
            filtered = dict(sorted(filtered.items(),
                                   key=lambda x: x[1], reverse=True))
            top_dish = list(filtered.keys())[0]
            top_conf = list(filtered.values())[0]
            top_5    = list(filtered.items())[:5]
        else:
            top_dish = dish
            top_conf = confidence
            top_5    = list(all_probs.items())[:5]

        should_guess = (
            top_conf >= self.CONFIDENCE_THRESHOLD or
            self.question_count >= self.MAX_QUESTIONS or
            len(remaining_names) == 1
        )

        if should_guess:
            self.game_over   = True
            self.final_guess = top_dish

        return {
            "top_dish"      : top_dish,
            "confidence"    : round(top_conf * 100, 1),
            "remaining"     : len(remaining_names),
            "question_count": self.question_count,
            "game_over"     : self.game_over,
            "final_guess"   : self.final_guess,
            "top_5"         : top_5
        }

    def reset(self):
        self.nb.reset()
        self.dt.reset()
        self.question_count = 0
        self.game_over      = False
        self.final_guess    = None


if __name__ == "__main__":
    engine = FlavinatorEngine()
    print("Engine loaded successfully")
    feature, question, ig = engine.get_next_question()
    print(f"First question: {question}")

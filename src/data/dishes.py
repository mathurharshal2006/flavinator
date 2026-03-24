# src/data/dishes.py
#
# This file is our knowledge base.
# Every dish is one row. Every column is one feature.
# The Naive Bayes and Decision Tree algorithms will learn from this data.

import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_dishes_dataset():
    """
    Returns a pandas DataFrame where:
      - Each ROW    = one dish
      - Each COLUMN = one feature (a property of the dish)

    WHY PANDAS DATAFRAME:
      A DataFrame is like an Excel table in Python.
      It lets us easily filter, sort, and feed data into ML algorithms.

    WHY THESE FEATURES:
      Each feature is something you can ask as a yes/no or
      multiple choice question. This is exactly how the game works.
      The better the features, the smarter the guessing.
    """

    dishes = [

        # ── INDIAN DISHES ──────────────────────────────────────────────────

        {
            "name": "Biryani",
            "cuisine": "Indian",
            "is_vegetarian": False,
            "is_spicy": True,
            "main_ingredient": "rice",
            "cooking_method": "steamed",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": False,
            "texture": "soft",
            "meal_type": "lunch"
        },
        {
            "name": "Butter Chicken",
            "cuisine": "Indian",
            "is_vegetarian": False,
            "is_spicy": True,
            "main_ingredient": "meat",
            "cooking_method": "grilled",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": True,
            "texture": "creamy",
            "meal_type": "dinner"
        },
        {
            "name": "Dal Tadka",
            "cuisine": "Indian",
            "is_vegetarian": True,
            "is_spicy": True,
            "main_ingredient": "lentils",
            "cooking_method": "boiled",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": True,
            "texture": "soft",
            "meal_type": "lunch"
        },
        {
            "name": "Masala Dosa",
            "cuisine": "Indian",
            "is_vegetarian": True,
            "is_spicy": True,
            "main_ingredient": "rice",
            "cooking_method": "fried",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": False,
            "texture": "crunchy",
            "meal_type": "breakfast"
        },
        {
            "name": "Gulab Jamun",
            "cuisine": "Indian",
            "is_vegetarian": True,
            "is_spicy": False,
            "main_ingredient": "dairy",
            "cooking_method": "fried",
            "is_sweet": True,
            "served_hot": False,
            "has_sauce": True,
            "texture": "soft",
            "meal_type": "dessert"
        },
        {
            "name": "Samosa",
            "cuisine": "Indian",
            "is_vegetarian": True,
            "is_spicy": True,
            "main_ingredient": "bread",
            "cooking_method": "fried",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": False,
            "texture": "crunchy",
            "meal_type": "snack"
        },
        {
            "name": "Paneer Tikka",
            "cuisine": "Indian",
            "is_vegetarian": True,
            "is_spicy": True,
            "main_ingredient": "dairy",
            "cooking_method": "grilled",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": False,
            "texture": "soft",
            "meal_type": "snack"
        },
        {
            "name": "Chole Bhature",
            "cuisine": "Indian",
            "is_vegetarian": True,
            "is_spicy": True,
            "main_ingredient": "bread",
            "cooking_method": "fried",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": True,
            "texture": "soft",
            "meal_type": "breakfast"
        },
        {
            "name": "Rasmalai",
            "cuisine": "Indian",
            "is_vegetarian": True,
            "is_spicy": False,
            "main_ingredient": "dairy",
            "cooking_method": "boiled",
            "is_sweet": True,
            "served_hot": False,
            "has_sauce": True,
            "texture": "soft",
            "meal_type": "dessert"
        },
        {
            "name": "Pav Bhaji",
            "cuisine": "Indian",
            "is_vegetarian": True,
            "is_spicy": True,
            "main_ingredient": "bread",
            "cooking_method": "fried",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": True,
            "texture": "soft",
            "meal_type": "snack"
        },
        {
            "name": "Idli",
            "cuisine": "Indian",
            "is_vegetarian": True,
            "is_spicy": False,
            "main_ingredient": "rice",
            "cooking_method": "steamed",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": False,
            "texture": "soft",
            "meal_type": "breakfast"
        },
        {
            "name": "Rogan Josh",
            "cuisine": "Indian",
            "is_vegetarian": False,
            "is_spicy": True,
            "main_ingredient": "meat",
            "cooking_method": "boiled",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": True,
            "texture": "soft",
            "meal_type": "dinner"
        },
        {
            "name": "Aloo Paratha",
            "cuisine": "Indian",
            "is_vegetarian": True,
            "is_spicy": True,
            "main_ingredient": "bread",
            "cooking_method": "fried",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": False,
            "texture": "soft",
            "meal_type": "breakfast"
        },
        {
            "name": "Kheer",
            "cuisine": "Indian",
            "is_vegetarian": True,
            "is_spicy": False,
            "main_ingredient": "rice",
            "cooking_method": "boiled",
            "is_sweet": True,
            "served_hot": False,
            "has_sauce": False,
            "texture": "creamy",
            "meal_type": "dessert"
        },
        {
            "name": "Vada Pav",
            "cuisine": "Indian",
            "is_vegetarian": True,
            "is_spicy": True,
            "main_ingredient": "bread",
            "cooking_method": "fried",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": True,
            "texture": "soft",
            "meal_type": "snack"
        },
        {
            "name": "Palak Paneer",
            "cuisine": "Indian",
            "is_vegetarian": True,
            "is_spicy": True,
            "main_ingredient": "dairy",
            "cooking_method": "boiled",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": True,
            "texture": "creamy",
            "meal_type": "dinner"
        },
        {
            "name": "Tandoori Chicken",
            "cuisine": "Indian",
            "is_vegetarian": False,
            "is_spicy": True,
            "main_ingredient": "meat",
            "cooking_method": "grilled",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": False,
            "texture": "soft",
            "meal_type": "dinner"
        },
        {
            "name": "Jalebi",
            "cuisine": "Indian",
            "is_vegetarian": True,
            "is_spicy": False,
            "main_ingredient": "bread",
            "cooking_method": "fried",
            "is_sweet": True,
            "served_hot": True,
            "has_sauce": True,
            "texture": "crunchy",
            "meal_type": "dessert"
        },
        {
            "name": "Upma",
            "cuisine": "Indian",
            "is_vegetarian": True,
            "is_spicy": False,
            "main_ingredient": "grains",
            "cooking_method": "boiled",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": False,
            "texture": "soft",
            "meal_type": "breakfast"
        },
        {
            "name": "Fish Curry",
            "cuisine": "Indian",
            "is_vegetarian": False,
            "is_spicy": True,
            "main_ingredient": "seafood",
            "cooking_method": "boiled",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": True,
            "texture": "soft",
            "meal_type": "lunch"
        },

        # ── WORLD DISHES ───────────────────────────────────────────────────

        {
            "name": "Sushi",
            "cuisine": "Japanese",
            "is_vegetarian": False,
            "is_spicy": False,
            "main_ingredient": "rice",
            "cooking_method": "raw",
            "is_sweet": False,
            "served_hot": False,
            "has_sauce": True,
            "texture": "chewy",
            "meal_type": "lunch"
        },
        {
            "name": "Pizza",
            "cuisine": "Italian",
            "is_vegetarian": True,
            "is_spicy": False,
            "main_ingredient": "bread",
            "cooking_method": "baked",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": True,
            "texture": "crunchy",
            "meal_type": "dinner"
        },
        {
            "name": "Tacos",
            "cuisine": "Mexican",
            "is_vegetarian": False,
            "is_spicy": True,
            "main_ingredient": "meat",
            "cooking_method": "grilled",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": True,
            "texture": "crunchy",
            "meal_type": "lunch"
        },
        {
            "name": "Ramen",
            "cuisine": "Japanese",
            "is_vegetarian": False,
            "is_spicy": False,
            "main_ingredient": "noodles",
            "cooking_method": "boiled",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": True,
            "texture": "chewy",
            "meal_type": "lunch"
        },
        {
            "name": "Croissant",
            "cuisine": "French",
            "is_vegetarian": True,
            "is_spicy": False,
            "main_ingredient": "bread",
            "cooking_method": "baked",
            "is_sweet": False,
            "served_hot": False,
            "has_sauce": False,
            "texture": "crunchy",
            "meal_type": "breakfast"
        },
        {
            "name": "Pad Thai",
            "cuisine": "Thai",
            "is_vegetarian": False,
            "is_spicy": True,
            "main_ingredient": "noodles",
            "cooking_method": "fried",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": True,
            "texture": "chewy",
            "meal_type": "dinner"
        },
        {
            "name": "Tiramisu",
            "cuisine": "Italian",
            "is_vegetarian": True,
            "is_spicy": False,
            "main_ingredient": "dairy",
            "cooking_method": "raw",
            "is_sweet": True,
            "served_hot": False,
            "has_sauce": False,
            "texture": "creamy",
            "meal_type": "dessert"
        },
        {
            "name": "Burger",
            "cuisine": "American",
            "is_vegetarian": False,
            "is_spicy": False,
            "main_ingredient": "meat",
            "cooking_method": "grilled",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": True,
            "texture": "soft",
            "meal_type": "lunch"
        },
        {
            "name": "Falafel",
            "cuisine": "Middle Eastern",
            "is_vegetarian": True,
            "is_spicy": False,
            "main_ingredient": "lentils",
            "cooking_method": "fried",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": True,
            "texture": "crunchy",
            "meal_type": "snack"
        },
        {
            "name": "Paella",
            "cuisine": "Spanish",
            "is_vegetarian": False,
            "is_spicy": False,
            "main_ingredient": "rice",
            "cooking_method": "fried",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": False,
            "texture": "soft",
            "meal_type": "lunch"
        },
        {
            "name": "Baklava",
            "cuisine": "Middle Eastern",
            "is_vegetarian": True,
            "is_spicy": False,
            "main_ingredient": "bread",
            "cooking_method": "baked",
            "is_sweet": True,
            "served_hot": False,
            "has_sauce": True,
            "texture": "crunchy",
            "meal_type": "dessert"
        },
        {
            "name": "Kimchi Fried Rice",
            "cuisine": "Korean",
            "is_vegetarian": False,
            "is_spicy": True,
            "main_ingredient": "rice",
            "cooking_method": "fried",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": False,
            "texture": "soft",
            "meal_type": "lunch"
        },
        {
            "name": "Churros",
            "cuisine": "Spanish",
            "is_vegetarian": True,
            "is_spicy": False,
            "main_ingredient": "bread",
            "cooking_method": "fried",
            "is_sweet": True,
            "served_hot": True,
            "has_sauce": True,
            "texture": "crunchy",
            "meal_type": "dessert"
        },
        {
            "name": "Pho",
            "cuisine": "Vietnamese",
            "is_vegetarian": False,
            "is_spicy": False,
            "main_ingredient": "noodles",
            "cooking_method": "boiled",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": True,
            "texture": "chewy",
            "meal_type": "lunch"
        },
        {
            "name": "Moussaka",
            "cuisine": "Greek",
            "is_vegetarian": False,
            "is_spicy": False,
            "main_ingredient": "meat",
            "cooking_method": "baked",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": True,
            "texture": "soft",
            "meal_type": "dinner"
        },
        {
            "name": "Tom Yum Soup",
            "cuisine": "Thai",
            "is_vegetarian": False,
            "is_spicy": True,
            "main_ingredient": "seafood",
            "cooking_method": "boiled",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": True,
            "texture": "soft",
            "meal_type": "lunch"
        },
        {
            "name": "Crepes",
            "cuisine": "French",
            "is_vegetarian": True,
            "is_spicy": False,
            "main_ingredient": "bread",
            "cooking_method": "fried",
            "is_sweet": True,
            "served_hot": True,
            "has_sauce": True,
            "texture": "soft",
            "meal_type": "breakfast"
        },
        {
            "name": "Peking Duck",
            "cuisine": "Chinese",
            "is_vegetarian": False,
            "is_spicy": False,
            "main_ingredient": "meat",
            "cooking_method": "baked",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": True,
            "texture": "crunchy",
            "meal_type": "dinner"
        },
        {
            "name": "Empanadas",
            "cuisine": "Mexican",
            "is_vegetarian": False,
            "is_spicy": True,
            "main_ingredient": "bread",
            "cooking_method": "baked",
            "is_sweet": False,
            "served_hot": True,
            "has_sauce": False,
            "texture": "crunchy",
            "meal_type": "snack"
        },
        {
            "name": "Gelato",
            "cuisine": "Italian",
            "is_vegetarian": True,
            "is_spicy": False,
            "main_ingredient": "dairy",
            "cooking_method": "raw",
            "is_sweet": True,
            "served_hot": False,
            "has_sauce": False,
            "texture": "creamy",
            "meal_type": "dessert"
        },
    ]

    return pd.DataFrame(dishes)


# run this file directly to verify the dataset
# In terminal: python3 src/data/dishes.py

if __name__ == "__main__":
    df = get_dishes_dataset()

    print("Dataset loaded successfully")
    print(f"Total dishes  : {len(df)}")
    print(f"Total features: {len(df.columns) - 1}")
    print(f"\nCuisines: {sorted(df['cuisine'].unique())}")
    print(f"\nFirst 5 rows:")
    print(df.head())
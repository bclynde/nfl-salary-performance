"""Visualization helpers for exploratory data analysis (EDA).

This module loads the modeling dataset created in clean.py and provides
starter plots. Add more charts/functions as the dataset and questions evolve.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# bring the final modeling dataset in; clean.py takes care of
# loading + transforming the raw data.  
from clean import load_salary_data, load_game_data, build_modeling_dataset


def load_model_dataset():
    """Load and return the final modeling dataset used for visualization."""
    salary = load_salary_data()
    games = load_game_data() 
    df = build_modeling_dataset(salary, games)
    return df


def basic_histogram(df: pd.DataFrame):
    """Draw a simple histogram of seasons in the modeling dataset.

    This is just a starter visualization; explore the dataframe and add
    more functions below
    """

    if df is None or df.empty:
        print("Dataset is empty; nothing to plot yet.")
        return
    plt.figure(figsize=(10, 6))
    # df["season"], may change if named differently in clean.py
    seasons = df["season"].dropna().astype(int)
    plt.hist(seasons, bins=range(seasons.min(), seasons.max() + 2), edgecolor="black")
    plt.title("Team-Games per Season")
    plt.xlabel("Season")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def main():
    df = load_model_dataset()
    print("Model dataset shape:", df.shape)
    basic_histogram(df)


if __name__ == "__main__":
    main()

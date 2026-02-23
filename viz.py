"""Visualization helpers for exploratory data analysis (EDA).

This module loads the modeling dataset created in clean.py and provides
starter plots. Add more charts/functions as the dataset and questions evolve.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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
    seasons = df["year"].dropna().astype(int)
    plt.hist(seasons, bins=range(seasons.min(), seasons.max() + 2), edgecolor="black")
    plt.title("Team-Games per Season")
    plt.xlabel("Season")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

### Can make scatter plots, box plots, line plots,
#   heatmaps, and/or more histograms
#   meant to explore relationships between 
#   salary variables and performance variables

def plot_spending_vs_wins(df, position_pct_col='QB_P'):
    """Corrected interactive scatter plot using your actual columns"""
    if df is None or df.empty:
        print("No data available to plot.")
        return

    fig = px.scatter(
        df, 
        x=position_pct_col, 
        y='W_PCT',  # Changed from win_pct
        color='team', # Changed from Team (lowercase in your clean.py)
        hover_data=['year', 'W'], # Changed from Season and total_wins
        title=f'NFL {position_pct_col} vs. Win Percentage (2013-2022)',
        labels={position_pct_col: f'Percent of Cap Spent on {position_pct_col}', 
                'W_PCT': 'Season Win Percentage'}
    )

    fig.show()


def main():
    df = load_model_dataset()
    
    if not df.empty:
        print("Generating EDA visualizations...")
        basic_histogram(df)
        plot_spending_vs_wins(df, position_pct_col='QB_P')
        
        # PASTE THE NEW CODE HERE
        correlation = df['QB_P'].corr(df['W_PCT'])
        print(f"\n--- FINANCIAL ANALYSIS ---")
        print(f"Correlation between QB Spending and Win %: {correlation:.4f}")
    else:
        print("Dataset is empty.")

if __name__ == "__main__":
    main()
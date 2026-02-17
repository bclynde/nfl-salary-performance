import pandas as pd

SALARY_PATH = "data/NFL_Salary_By_Position_Group.csv"
GAMES_PATH  = "data/spreadspoke_scores.csv"

def load_salary_data():
    """Load raw salary dataset"""
    df_salary = pd.read_csv(SALARY_PATH)
    return df_salary

def load_game_data():
    """Load raw game dataset"""
    df_games = pd.read_csv(GAMES_PATH)
    return df_games

def build_modeling_dataset(df_salary, df_games):
    """
    Build a team-season modeling dataset.

    TODO:
    1. Filter seasons to 2013–2022
    2. Aggregate weekly games into team-season metrics
       - wins
       - losses
       - points for
       - points against
       - point differential
    3. Merge on (team, season)
    """

    # Placeholder so file runs
    df_final = pd.DataFrame()
    return df_final


if __name__ == "__main__":
    print("Running clean.py...\n")

    salary = load_salary_data()
    games = load_game_data()

    print("Salary shape:", salary.shape)
    print("Games shape:", games.shape)

    print("\nSalary columns:")
    print(salary.columns)

    print("\nGame columns:")
    print(games.columns)

    final_df = build_modeling_dataset(salary, games)
    print("\nFinal modeling dataset shape:", final_df.shape)

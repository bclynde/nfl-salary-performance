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
    # 1. Standardize column names
    df_salary = df_salary.rename(columns={'Season': 'year', 'Team': 'team'})

    # 2. THE FIX: Strip city names so 'New England Patriots' becomes 'Patriots'
    # This ensures it matches the 'team_home' column in the games file
    df_salary['team'] = df_salary['team'].str.split().str[-1]
    df_games['team_home'] = df_games['team_home'].str.split().str[-1]

    # 3. Filter years
    df_salary = df_salary[(df_salary['year'] >= 2013) & (df_salary['year'] <= 2022)]

    # 4. Merge
    df_final = pd.merge(
        df_salary, 
        df_games, 
        left_on=['year', 'team'], 
        right_on=['schedule_season', 'team_home'], 
        how='inner'
    )
    
    print(f"--- DEBUG: Found {len(df_final)} matching rows ---")
    return df_final
    
    # 3. Filter years (2013-2022)
    df_salary = df_salary[(df_salary['year'] >= 2013) & (df_salary['year'] <= 2022)]
    
    # 4. Merge and Debug
    df_final = pd.merge(df_salary, df_games, left_on=['year', 'team'], 
                        right_on=['schedule_season', 'team_home'], how='inner')
    
    print(f"--- DEBUG: Found {len(df_final)} matching rows ---")
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

    print("\nUnique Teams in Salary Data:")
    print(sorted(salary['team'].unique()))

    print("\nUnique Teams in Games Data:")
    print(sorted(games['team_home'].unique()))


import pandas as pd

SALARY_PATH = "data/NFL_Salary_By_Position_Group.csv"
GAMES_PATH  = "data/spreadspoke_scores.csv"

TEAM_ABBR_TO_NAME = {
    "Arizona Cardinals": "Cardinals",
    "Atlanta Falcons": "Falcons",
    "Baltimore Ravens": "Ravens",
    "Buffalo Bills": "Bills",
    "Carolina Panthers": "Panthers",
    "Chicago Bears": "Bears",
    "Cincinnati Bengals": "Bengals",
    "Cleveland Browns": "Browns",
    "Dallas Cowboys": "Cowboys",
    "Denver Broncos": "Broncos",
    "Detroit Lions": "Lions",
    "Green Bay Packers": "Packers",
    "Houston Texans": "Texans",
    "Indianapolis Colts": "Colts",
    "Jacksonville Jaguars": "Jaguars",
    "Kansas City Chiefs": "Chiefs",
    "Las Vegas Raiders": "Raiders",
    "Los Angeles Chargers": "Chargers",
    "Los Angeles Rams": "Rams",
    "Miami Dolphins": "Dolphins",
    "Minnesota Vikings": "Vikings",
    "New England Patriots": "Patriots",
    "New Orleans Saints": "Saints",
    "New York Giants": "Giants",
    "New York Jets": "Jets",
    "Oakland Raiders": "Raiders",
    "Philadelphia Eagles": "Eagles",
    "Pittsburgh Steelers": "Steelers",
    "San Diego Chargers": "Chargers",
    "San Francisco 49ers": "49ers",
    "Seattle Seahawks": "Seahawks",
    "St. Louis Rams": "Rams",
    "Tampa Bay Buccaneers": "Buccaneers",
    "Tennessee Titans": "Titans",
    "Washington Commanders": "Commanders",
    "Washington Football Team": "Commanders",
    "Washington Redskins": "Commanders",
}

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
    """
    df_games = df_games.copy()
    df_games["schedule_season"] = pd.to_numeric(df_games["schedule_season"], errors="coerce")
    df_games = df_games[
        (df_games["schedule_season"] >= 2013) &
        (df_games["schedule_season"] <= 2022)
    ].copy()

    # Drop rows with missing scores
    df_games = df_games.dropna(subset=["score_home", "score_away"])
    df_games["score_home"] = pd.to_numeric(df_games["score_home"], errors="coerce")
    df_games["score_away"] = pd.to_numeric(df_games["score_away"], errors="coerce")
    df_games = df_games.dropna(subset=["score_home", "score_away"])
    """
    2. Aggregate weekly games into team-season metrics
       - wins
       - losses
       - points for
       - points against
       - point differential
    """
    base_cols = ["schedule_date", "schedule_season", "schedule_week",
                 "schedule_playoff", "score_home", "score_away",
                 "team_home", "team_away"]

    # Keep only columns that exist in the dataframe
    base_cols = [c for c in base_cols if c in df_games.columns]
    games = df_games[base_cols].copy()

    # --- Home team rows ---
    home = games.rename(columns={
        "team_home": "team_abbr",
        "team_away": "opponent_abbr",
        "score_home": "points_for",
        "score_away": "points_against",
    }).copy()
    home["is_home"] = True

    # --- Away team rows ---
    away = games.rename(columns={
        "team_away": "team_abbr",
        "team_home": "opponent_abbr",
        "score_away": "points_for",
        "score_home": "points_against",
    }).copy()
    away["is_home"] = False

    # Combine
    df_team_game = pd.concat([home, away], ignore_index=True)

    df_team_game["point_diff"] = df_team_game["points_for"] - df_team_game["points_against"]
    df_team_game["win"]  = (df_team_game["point_diff"] > 0).astype(int)
    df_team_game["loss"] = (df_team_game["point_diff"] < 0).astype(int)
    df_team_game["tie"]  = (df_team_game["point_diff"] == 0).astype(int)

    """
    3. Merge on (team, season)
    """
    df_team_game["team_name"] = df_team_game["team_abbr"].map(TEAM_ABBR_TO_NAME)

    unmapped = df_team_game[df_team_game["team_name"].isna()]["team_abbr"].unique()
    if len(unmapped) > 0:
        print(f"Warning: unmapped team abbreviations (rows will be dropped): {unmapped}")
    df_team_game = df_team_game.dropna(subset=["team_name"])

    df_team_game = df_team_game.rename(columns={"schedule_season": "Season"})

    salary_cols = [
        "Team", "Season",
        "QB_P", "RB_P", "WR_P", "TE_P", "OL_P",
        "IDL_P", "EDGE_P", "LB_P", "S_P", "CB_P",
        "Offense_P", "Defense_P",
        "QB", "RB", "WR", "TE", "OL",
        "IDL", "EDGE", "LB", "S", "CB",
        "Cap", "W_PCT", "Playoffs", "SB",
        ]
    salary_cols = [c for c in salary_cols if c in df_salary.columns]
    df_sal = df_salary[salary_cols].copy()
    df_sal["Season"] = pd.to_numeric(df_sal["Season"], errors="coerce")

    df_final = df_team_game.merge(
        df_sal,
        left_on=["team_name", "Season"],
        right_on=["Team", "Season"],
        how="left",
    )

    df_final = df_final.drop(columns=["Team"], errors="ignore")
    df_final = df_final.sort_values(["Season", "schedule_date", "team_name"]).reset_index(drop=True)

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

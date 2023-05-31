# %%
import pandas as pd
import numpy as np
import sqlite3 as db
import datetime
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from importlib import reload
sys.path.append(r'\..')
import External_functions
reload(External_functions)
from External_functions import run_bayesian_opt_lightgbm
# %%
conn = db.connect(r'..\database.sqlite')
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table';")
category_names = c.fetchall()

df_sqlite_seq = pd.read_sql_query('select * from sqlite_sequence', conn)
df_player_attributes = pd.read_sql_query('select * from Player_Attributes', conn)
df_player = pd.read_sql_query('select * from Player', conn)
df_match = pd.read_sql_query('select * from Match', conn)
df_league = pd.read_sql_query('select * from League', conn)
df_country = pd.read_sql_query('select * from Country', conn)
df_team = pd.read_sql_query('select * from Team', conn)
df_team_attributes = pd.read_sql_query('select * from Team_Attributes', conn)
# %%
for key in df_match.keys():
    print(key)
# %%
player_keys = []
for key in df_match.keys():
    if "player" in key:
        player_keys.append(key)
odds_keys = np.append(np.array(["match_api_id"]), df_match.keys()[85:])
df_match_odds = df_match[odds_keys]
display(df_match_odds)
non_stats_keys = np.append(np.array(player_keys), df_match.keys()[77:])
df_match_small = df_match.drop(non_stats_keys, axis = 1)
display(df_match_small)
print(int(df_match_small["date"][0].split(" ")[0].replace("-", "")))
def convert_date(date_array):
    return [int(date.split(" ")[0].replace("-", "")) for date in date_array]

df_match_small["date"] = convert_date(df_match_small["date"].to_numpy())
display(df_match_small)
# %%
print(np.array([True, True, False], dtype = int))
# %%
def create_outcome(home_goal, away_goal):
    diff = home_goal - away_goal
    outcome = np.concatenate([[diff > 0], [diff == 0], [diff < 0]], axis = 0).T.astype(int)
    return pd.DataFrame(outcome, columns=["home_team_win", "draw", "away_team_win"])

df_match_with_outcome = pd.concat([df_match_small, create_outcome(df_match_small["home_team_goal"].to_numpy(), df_match_small["away_team_goal"].to_numpy())], axis = 1)

# %%
print(np.unique(df_match_with_outcome["stage"]))
# %%
def running_score(df):
    
    leagues = np.unique(df["league_id"])
    seasons = np.unique(df["season"])
    for league in leagues[:1]:
        for season in seasons[:1]:
            print(df_league["name"][df_league["id"] == league].values[0])
            print(season)
            temp = df[np.logical_and(df["league_id"] == league, df["season"] == season)]

            teams = np.unique(temp["home_team_api_id"])
            display(temp)
            for team in teams[:1]:
                print(df_team["team_long_name"][df_team["team_api_id"] == team].values[0])
                outcomes_home = temp[temp["home_team_api_id"] == team][["date", "match_api_id", "home_team_win", "draw", "away_team_win",'home_team_goal', 'away_team_goal']]
                outcomes_away = temp[temp["away_team_api_id"] == team][["date", "match_api_id", "away_team_win", "draw", "home_team_win", 'away_team_goal', 'home_team_goal']]
                outcomes_home.columns = ["date", "match_api_id", "W", "D", "L", "GF", "GA"]
                outcomes_away.columns = ["date", "match_api_id", "W", "D", "L", "GF", "GA"]
                outcomes = pd.concat([outcomes_home, outcomes_away]).sort_values("date")
                outcomes.iloc[:,2:] = np.cumsum(outcomes.iloc[:,2:])
                stats = {}
                stats["GD"] = outcomes["GF"]-outcomes["GA"]
                pld = np.sum(outcomes.iloc[:,2:5], axis = 1)
                stats["GFpm"] = outcomes["GF"]/pld
                stats["GApm"] = outcomes["GA"]/pld
                stats["GDpm"] = outcomes["GD"]/pld
                stats["Form_messure1"]
                # stats["Form_messure2"]
                # stats["Form_messure3"]
                # stats["Form_messure4"]
                pts = outcomes.iloc[:,2:5].values@np.array([3,1,0])
                print(pd.concat([outcomes, pd.DataFrame(stats)], axis = 1))


            # Look for incomplete data
            # temp = df[np.logical_and(df["league_id"] == league, df["season"] == season)]
            # length = len(np.unique(temp["home_team_api_id"].values))
            # if (length*(length-1) > temp.shape[0] or length < 10):
            #     print(df_league["name"][df_league["id"] == league].values[0])
            #     print(season)
            #     print(f"number of teams:{length}")
            #     print(f"number of matches:{temp.shape[0]}")
            #     print("  ")


print(df_match_with_outcome.keys())
display(df_match_with_outcome)
running_score(df_match_with_outcome)
    
# %%

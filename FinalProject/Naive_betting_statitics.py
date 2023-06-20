# %%
import pandas as pd
import numpy as np
import sqlite3 as db
import datetime
from lightgbm import LGBMClassifier
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay, mean_absolute_error, accuracy_score
from bayes_opt import BayesianOptimization
import seaborn as sns
import sys
from importlib import reload
import functools as ft
sys.path.append(r'\..')
import External_functions
reload(External_functions)
from External_functions import run_bayesian_opt_lightgbm
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
def create_outcome(home_goal, away_goal, match_api_id_df):
    diff = home_goal - away_goal
    outcome = np.concatenate([[diff > 0], [diff == 0], [diff < 0]], axis = 0).T.astype(int)
    temp_outcome = pd.DataFrame(outcome, columns=["home_team_win", "draw", "away_team_win"])
    return pd.concat([match_api_id_df, temp_outcome], axis = 1)

# df_outcomes = pd.concat([df_match["match_api_id"], create_outcome(df_match["home_team_goal"].to_numpy(), df_match["away_team_goal"].to_numpy())], axis = 1)
# df_bets = pd.concat([df_match["match_api_id"], df_match.iloc[:,85:]], axis = 1)
# df_merged = pd.merge(df_outcomes, df_bets, on = "match_api_id", how = "inner")
df_outcomes = create_outcome(df_match["home_team_goal"].to_numpy(), df_match["away_team_goal"].to_numpy(), df_match["match_api_id"])

# %%
def caloutcome(outcome, bets, guess, dfs = True):
    # Outcome should be [0,1] or [1,0] for 2d 
    join_key = "match_api_id"
    # Just to make sure they are lined up:
    if dfs:
        guess.columns = [join_key]+[str(i) for i in range(len(guess.keys()[1:]))]
        merged = outcome.merge(bets,on=join_key, how = "inner").merge(guess, on=join_key, how = "inner")
        if merged.shape[0] < guess.shape[0]:
            print("removed guesses. Not good")
        outcome = merged[outcome.keys().drop(join_key)]
        bets = merged[bets.keys().drop(join_key)]
        guess = merged[guess.keys().drop(join_key)]

    # Extent guess and outcome to match bets
    guess = np.tile(guess, bets.shape[1] // guess.shape[1])
    outcome = np.tile(outcome, bets.shape[1] // outcome.shape[1])

    return outcome*bets*guess

df_outcomes = create_outcome(df_match["home_team_goal"].to_numpy(), df_match["away_team_goal"].to_numpy(), df_match["match_api_id"])
df_bets = df_match[np.append(np.array(["match_api_id"]), df_match.keys()[85:])]

guess_dict ={}

guess_H = np.repeat([[1,0,0]], df_match.shape[0], axis = 0)
df_H = pd.concat([df_match["match_api_id"], pd.DataFrame(guess_H)], axis = 1)
guess_dict["Always Home"] = df_H

guess_D = np.repeat([[0,1,0]], df_match.shape[0], axis = 0)
df_D = pd.concat([df_match["match_api_id"], pd.DataFrame(guess_D)], axis = 1)
guess_dict["Always Draw"] = df_D

guess_A = np.repeat([[0,0,1]], df_match.shape[0], axis = 0)
df_A = pd.concat([df_match["match_api_id"], pd.DataFrame(guess_A)], axis = 1)
guess_dict["Always Away"] = df_A

guess_low = df_bets["match_api_id"]
for bookmaker_keys in np.split(df_bets.keys()[1:], len(df_bets.keys()[1:])//3):
    df = df_bets[bookmaker_keys]
    guess_l_temp = np.zeros_like(df)
    lowest_indices = df.idxmin(axis=1).dropna().apply(lambda x: df.columns.get_loc(x))
    guess_l_temp[lowest_indices.index,lowest_indices.values] = 1
    guess_l_temp = pd.concat([df_bets["match_api_id"], pd.DataFrame(guess_l_temp, columns=bookmaker_keys)],axis = 1)
    guess_low = pd.merge(guess_low, guess_l_temp, on = "match_api_id", how = "outer")
guess_dict["Always Lowest"] = guess_low

guess_high = df_bets["match_api_id"]
for bookmaker_keys in np.split(df_bets.keys()[1:], len(df_bets.keys()[1:])//3):
    df = df_bets[bookmaker_keys]
    guess_h_temp = np.zeros_like(df)
    highest_indices = df.idxmax(axis=1).dropna().apply(lambda x: df.columns.get_loc(x))
    guess_h_temp[highest_indices.index,highest_indices.values] = 1
    guess_h_temp = pd.concat([df_bets["match_api_id"], pd.DataFrame(guess_h_temp, columns=bookmaker_keys)],axis = 1)
    guess_high = pd.merge(guess_high, guess_h_temp, on = "match_api_id", how = "outer")
guess_dict["Always Highest"] = guess_high

guess_coin = df_bets["match_api_id"]

guess_c = np.zeros((df_match.shape[0],3))
print(guess_c)
indices = np.random.choice([0, 1, 2], df_match.shape[0])
guess_c[np.arange(df_match.shape[0]), indices] = 1
print(guess_c)
guess_coin = pd.concat([df_match["match_api_id"], pd.DataFrame(guess_c)], axis = 1)
guess_dict["Coin flip"] = guess_coin

stats_df = pd.DataFrame()

for guess,guess_df in guess_dict.items(): 
    roi_dict = {}
    money_out = caloutcome(df_outcomes,df_bets,guess_df)
    for i, bookmaker in enumerate(money_out.keys()[::3]):
        bookmaker = bookmaker[:-1]
        bookmaker_keys = np.array([bookmaker in name[:-1] for name in money_out.keys()])
        df = money_out.iloc[:,bookmaker_keys].dropna()
        payout = df.sum(axis = 1).sum()
        roi = (payout - df.shape[0])/df.shape[0]
        roi_dict[bookmaker] = roi
    stats_df = pd.concat([stats_df, pd.DataFrame(roi_dict, index=[guess])],axis = 0, join = "outer")
display(stats_df)
# %%
fig, ax = plt.subplots(figsize = (14,8))
heat = sns.heatmap(stats_df, annot = True, ax = ax, cbar_kws={'label': "Roi"})
# plt.savefig("roi_for_bettors.pdf", dpi = 300)
ax.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, left = False, labeltop=True)
ax.set_xlabel("Bettors", fontsize = 16)
ax.set_ylabel("Bet strategy", fontsize = 16) 
ax.xaxis.set_label_position('top') 
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(fontsize=20)

plt.savefig("roi_for_bettors.pdf", dpi = 400)
# %%

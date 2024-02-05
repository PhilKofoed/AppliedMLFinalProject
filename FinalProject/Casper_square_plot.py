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
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization
import seaborn as sns
import sys
from importlib import reload
sys.path.append(r'\..')
import External_functions
reload(External_functions)
from External_functions import run_bayesian_opt_lightgbm
import warnings
warnings.filterwarnings("ignore")


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

def heads_tails_betting(df_bookie, outcomes, plot=True):
    bookie_names = ['B365', 'BW', 'IW', 'LB', 'PS', 'WH', 'SJ', 'VC', 'GB', 'BS']
    coin_flip_outcomes = np.random.choice(['H', 'XA'], p=[0.5,0.5], size=len(df_bookie.index))
    h_outcomes = np.array(['H']*len(df_bookie.index))
    xa_outcomes = np.array(['XA']*len(df_bookie.index))

    true_outcomes = np.array(['H' if outcome==0 else 'XA' for outcome in outcomes])
    end_money_coin = {k: [] for k in bookie_names}
    end_money_h = {k: [] for k in bookie_names}
    end_money_xa = {k: [] for k in bookie_names}
    end_money_low = {k: [] for k in bookie_names}
    end_money_high = {k: [] for k in bookie_names}
    master_roi = {}
    errors = np.zeros((2,len(bookie_names)))
    for idx, row in df_bookie.iterrows():
        true_outcome = true_outcomes[idx]
        # Coin flip
        our_outcome_coin = coin_flip_outcomes[idx]
        if our_outcome_coin == true_outcome:
            for i, name in enumerate(bookie_names):
                string = name+our_outcome_coin
                end_money_coin[name].append(row[string])
        else:
            for i, name in enumerate(bookie_names):
                string = name+our_outcome_coin
                end_money_coin[name].append(0*row[string])
        # Always H
        our_outcome_h = h_outcomes[idx]
        if our_outcome_h == true_outcome:
            for i, name in enumerate(bookie_names):
                string = name+our_outcome_h
                end_money_h[name].append(row[string])
        else:
            for i, name in enumerate(bookie_names):
                string = name+our_outcome_h
                end_money_h[name].append(0*row[string])
        # Always XA
        our_outcome_xa = xa_outcomes[idx]
        if our_outcome_xa == true_outcome:
            for i, name in enumerate(bookie_names):
                string = name+our_outcome_xa
                end_money_xa[name].append(row[string])
        else:
            for i, name in enumerate(bookie_names):
                string = name+our_outcome_xa
                end_money_xa[name].append(0*row[string])
        # Lowest Odds
        for i, name in enumerate(bookie_names):
            if row[name+'H'] < row[name+'XA']:
                our_outcome_low = 'H'
            else:
                our_outcome_low = 'XA'
            if our_outcome_low==true_outcome:
                string = name+our_outcome_low
                end_money_low[name].append(row[string])
            else:
               string = name+our_outcome_low
               end_money_low[name].append(0*row[string]) 
        # Highest Odds
        for i, name in enumerate(bookie_names):
            if row[name+'H'] > row[name+'XA']:
                our_outcome_high = 'H'
            else:
                our_outcome_high = 'XA'
            if our_outcome_high==true_outcome:
                string = name+our_outcome_high
                end_money_high[name].append(row[string])
            else:
               string = name+our_outcome_high
               end_money_high[name].append(0*row[string])
    
    # Evaluate ROI's
    roi_coin = {k: [] for k in bookie_names}
    for key, value in list(end_money_coin.items()):
        value = np.array(value)
        value = value[~np.isnan(value)]
        roi_coin[key] = ((np.sum(value) - len(value)) / len(value)) 
    #master_roi['Coin flip'] = np.mean(roi_coin)
    #errors[:,0] = np.quantile(roi_coin, [0.25 , 0.75])
    roi_h = {k: [] for k in bookie_names}
    for key, value in end_money_h.items():
        value = np.array(value)
        value = value[~np.isnan(value)]
        roi_h[key] = ((np.sum(value) - len(value)) / len(value)) 
    #master_roi['Always H'] = np.mean(roi_h)
    #errors[:,1] = np.quantile(roi_h, [0.25 , 0.75])

    roi_xa = {k: [] for k in bookie_names}
    for key, value in end_money_xa.items():
        value = np.array(value)
        value = value[~np.isnan(value)]
        roi_xa[key] = ((np.sum(value) - len(value)) / len(value)) 
    #master_roi['Always XA'] = np.mean(roi_xa)
    #errors[:,2] = np.quantile(roi_xa, [0.25 , 0.75])

    roi_low = {k: [] for k in bookie_names}
    for key, value in end_money_low.items():
        value = np.array(value)
        value = value[~np.isnan(value)]
        roi_low[key] = ((np.sum(value) - len(value)) / len(value)) 
    #master_roi['Always Lowest Odds'] = np.mean(roi_low)
    #errors[:,3] = np.quantile(roi_low, [0.25 , 0.75])

    roi_high = {k: [] for k in bookie_names}
    for key, value in end_money_high.items():
        value = np.array(value)
        value = value[~np.isnan(value)]
        roi_high[key] = ((np.sum(value) - len(value)) / len(value)) 
    #master_roi['Always Highest Odds'] = np.mean(roi_high)
    #errors[:,4] = np.quantile(roi_high, [0.25 , 0.75])
    if plot:
        basic_horizontal_barplot(master_roi.values(), errors.T, master_roi.keys(), "Return on investment (ROI) in %", 'hej')
    return [pd.DataFrame(roi_coin,index=[0]), pd.DataFrame(roi_h,index=[0]), pd.DataFrame(roi_xa,index=[0]), pd.DataFrame(roi_low,index=[0]), pd.DataFrame(roi_high, index=[0])]

# %%
df = pd.read_csv(r'.\bookielist.csv', index_col=0)
df = pd.concat([df['match_api_id'], 1/df.drop(['outcome', 'match_api_id', 'outcome'], axis=1), df['outcome']], axis=1)
# %%
x = heads_tails_betting(df.drop(['outcome', 'match_api_id'], axis=1), df['outcome'], plot=False)

# %%
df_plot = pd.concat(x)
df_plot.index = ['Coinflip', 'Always H', 'Always XA', 'Always Lowest Odds', 'Always Highest Odds']
df_plot['Mean'] = df_plot.mean(axis=1)
df_plot.loc['Mean'] = df_plot.mean(axis=0)
# %%
display(df_plot)
#%% Naive plots
import matplotlib.ticker as mtick
fig, ax = plt.subplots(figsize=(9,4.5))
ax.grid(False)
ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,  
    left=False,    # ticks along the top edge are off
    labelbottom=False,
    labeltop = True) # labels along the bottom edge are off
sns.heatmap(df_plot, ax=ax, fmt='.1%', annot=True, annot_kws={"size": 8}, cmap = "Greens_r", vmax = 0)
cbar = ax.collections[0].colorbar
cbar.set_ticks([-0.12, -0.1, -0.08, -0.06, -0.04, -0.02, 0])
cbar.set_ticklabels(['- 12%', '- 10%', '- 8%', '- 6%', '- 4%', '- 2%', "0 %"])
ax.set_xlabel("Bookmakers", fontsize = 14)
ax.set_ylabel("Betting Strategy", fontsize = 14, rotation=0)
ax.xaxis.set_label_position('top') 
cbar.ax.set_title(label='ROI', size = 14)
ax.yaxis.set_label_coords(-0.17,0.98)
plt.tight_layout()
plt.savefig("Naive_roi_v1.png", dpi = 400)
print(df_match.shape)


# row_totals = df_plot.mean(axis=1)
# fig, ax = plt.subplots()
# sns.barplot(x=100*row_totals, y=df_plot.index, ax=ax)
# ax.xaxis.set_major_formatter(mtick.PercentFormatter())
# fig.show()

# column_totals = df_plot.mean(axis=0)
# fig, ax = plt.subplots()
# ax.xaxis.set_major_formatter(mtick.PercentFormatter())
# sns.barplot(x=100*column_totals, y=df_plot.columns, ax=ax)

# %%
outcome_array = np.concatenate([[df["outcome"].values==0], [df["outcome"].values==1]], axis = 0).T.astype(int)
outcome_array = np.tile(outcome_array, 10)
df_odds = df.drop(["match_api_id", "outcome"], axis =1)
guess_dict ={}

guess_H = np.repeat([[1,0]], df.shape[0], axis = 0)
guess_H = np.tile(guess_H, df_odds.shape[1]//guess_H.shape[1])
guess_dict["Always Home"] = guess_H

guess_XA = np.repeat([[0,1]], df.shape[0], axis = 0)
guess_XA = np.tile(guess_XA, df_odds.shape[1]//guess_XA.shape[1])
guess_dict["Always Draw/Away"] = guess_XA

guess_l_indices = df_odds.iloc[:,::2].idxmin(axis=1).apply(lambda x: df_odds.columns.get_loc(x))
guess_low = np.zeros_like(df_odds)
guess_low[np.arange(df_odds.shape[0]), guess_l_indices] = 1
guess_dict["Always Lowest"] = guess_low


guess_h_indices = df_odds.idxmax(axis=1).apply(lambda x: df_odds.columns.get_loc(x))
guess_high = np.zeros_like(df_odds)
guess_high[np.arange(df_odds.shape[0]), guess_h_indices] = 1
guess_dict["Always Highest"] = guess_high 

guess_coin = np.zeros((df_odds.shape[0], 2))
indices = np.random.choice([0, 1], df_odds.shape[0])
guess_coin[np.arange(df_odds.shape[0]), indices] = 1
guess_coin = np.tile(guess_coin, df_odds.shape[1]//guess_coin.shape[1])
guess_dict["Coin Flip"] = guess_coin

resulting_roi = {}
for key, guess in guess_dict.items():
    resulting_roi[key] = (np.sum((outcome_array*df_odds*guess).max(axis = 1).values)-df_odds.shape[0])/df_odds.shape[0]

plt.style.use("../casper_style.mplstyle")
fig, ax = plt.subplots(figsize = (9,6))
ax.grid(False)
plots = sns.barplot(pd.DataFrame(resulting_roi, index = [0]), ax = ax)
ax.set_ylim(-0.12, 0)

ax.set_xticklabels(resulting_roi.keys(), rotation = -7)
ax.xaxis.set_label_position('top') 
ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,  
    left=True,    # ticks along the top edge are off
    labelbottom=False,
    labeltop = True)
ax.set_xlabel("Best bet per match - Betting Strategies")
ax.set_ylabel("ROI")
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2%'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')
    
old_best = np.roll(df_plot.max(axis = 1).drop("Mean"),-1)
for i, value in enumerate(old_best):
    ax.hlines(value, i-0.4,i+0.4, color = (0.1,0.1,0.1), linestyles = "--")
    ax.annotate("Best Naive", (i, value), ha='center', va='center',
                   size=10, xytext=(0, 8),
                   textcoords='offset points')

ax.annotate("uses casper_style", (4.2,-0.12), ha='center', va='center',
                   size=8, xytext=(0, 8),
                   textcoords='offset points')
ax.set_yticklabels(["-12%", "-10%", "-8%","-6%", "-4%", "-2%", "0%"])
plt.tight_layout()
plt.savefig("BetterNaive_roi_v1.png", dpi =400)

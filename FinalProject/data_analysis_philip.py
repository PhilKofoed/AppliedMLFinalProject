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
print(df_match.shape)
player_keys = []
for key in df_match.keys():
    if "player" in key:
        player_keys.append(key)
odds_keys = np.append(np.array(["match_api_id"]), df_match.keys()[85:])
df_match_odds = df_match[odds_keys]
non_stats_keys = np.append(np.array(player_keys), df_match.keys()[77:])
df_match_small = df_match.drop(non_stats_keys, axis = 1)

display(df_match)

def convert_date(date_array):
    # replaces yyyy-mm-dd hour:min:sec with yyyymmdd
    return [int(date.split(" ")[0].replace("-", "")) for date in date_array]

df_match_small["date"] = convert_date(df_match_small["date"].to_numpy())
def create_outcome(home_goal, away_goal):
    diff = home_goal - away_goal
    outcome = np.concatenate([[diff > 0], [diff == 0], [diff < 0]], axis = 0).T.astype(int)
    return pd.DataFrame(outcome, columns=["home_team_win", "draw", "away_team_win"])

df_match_with_outcome = pd.concat([df_match_small, create_outcome(df_match_small["home_team_goal"].to_numpy(), df_match_small["away_team_goal"].to_numpy())], axis = 1)
print(np.sum(df_match_with_outcome[["home_team_win", "draw", "away_team_win"]])/df_match_with_outcome.shape[0])
# %%
def running_score(df):
    def form_messure1(df):
        pass
    leagues = np.unique(df["league_id"])[:1]
    display(leagues)
    seasons = np.unique(df["season"])[:1]
    df_to_concat = pd.DataFrame()
    for league in leagues:
        for season in seasons:
            print(df_league["name"][df_league["id"] == league].values[0])
            print(season)
            temp = df[np.logical_and(df["league_id"] == league, df["season"] == season)]

            teams = np.unique(temp["home_team_api_id"])
            team_stats = {}
            for team in teams:
                # print(df_team["team_long_name"][df_team["team_api_id"] == team].values[0])
                outcomes_home = temp[temp["home_team_api_id"] == team][["date", "match_api_id", "home_team_win", "draw", "away_team_win",'home_team_goal', 'away_team_goal']]
                outcomes_away = temp[temp["away_team_api_id"] == team][["date", "match_api_id", "away_team_win", "draw", "home_team_win", 'away_team_goal', 'home_team_goal']]
                outcomes_home.columns = ["date", "match_api_id", "W", "D", "L", "GF", "GA"]
                outcomes_away.columns = ["date", "match_api_id", "W", "D", "L", "GF", "GA"]
                outcomes_unaltered = pd.concat([outcomes_home, outcomes_away]).sort_values("date")
                outcomes = outcomes_unaltered.copy()
                outcomes.iloc[:,2:] = np.cumsum(outcomes.iloc[:,2:])
                stats = {}  
                stats["GD"] = outcomes["GF"]-outcomes["GA"]
                stats["pld"] = np.sum(outcomes.iloc[:,2:5], axis = 1)
                stats["pld left"] = stats["pld"].values[::-1]-1
                stats["pts"] = outcomes.iloc[:,2:5].values@np.array([3,1,0])

                stats["GFpm"] = outcomes["GF"]/stats["pld"]
                stats["GApm"] = outcomes["GA"]/stats["pld"]
                stats["GDpm"] = stats["GD"]/stats["pld"]
                stats["ptspm"] = stats["pts"]/stats["pld"]
                
                # Stats are for after the game and not before. 1 is a win, 0.5 is a draw and 0 is a lose and -1 if unknown
                stats["-1 match"] = outcomes_unaltered.iloc[:,2:5].values@np.array([1,0.5,0])
                stats["-2 match"] = np.append(np.array([-1 for i in range(1)]), outcomes_unaltered.iloc[:-1,2:5].values@np.array([1,0.5,0]))
                stats["-3 match"] = np.append(np.array([-1 for i in range(2)]), outcomes_unaltered.iloc[:-2,2:5].values@np.array([1,0.5,0]))
                stats["-4 match"] = np.append(np.array([-1 for i in range(3)]), outcomes_unaltered.iloc[:-3,2:5].values@np.array([1,0.5,0]))
                stats["-5 match"] = np.append(np.array([-1 for i in range(4)]), outcomes_unaltered.iloc[:-4,2:5].values@np.array([1,0.5,0]))
                stats["Form_m_1"] = np.sum([stats["-1 match"], stats["-2 match"], stats["-3 match"], stats["-4 match"], stats["-5 match"]])
                outcomes = pd.concat([outcomes, pd.DataFrame(stats)], axis = 1)
                first_match_zero_data = np.array([0 for i in outcomes.keys()[2:]])
                first_match_zero_data[-5:] = np.array([-1 for i in outcomes.keys()[-5:]])
                df_first_match = pd.DataFrame({key:[value] for key,value in zip(outcomes.keys()[2:], first_match_zero_data)}, index = [0])
                df_first_match["pld left"] = outcomes.shape[0]
                outcomes_moved_up = pd.concat([df_first_match, outcomes.iloc[:-1,2:]], ignore_index=True,axis = 0)
                outcomes_moved_up.index = outcomes.index
                outcomes = pd.concat([outcomes.iloc[:,:2], outcomes_moved_up], axis = 1) 
                team_stats[team] = outcomes
            
            running_pts = {}
            running_gd = {}
            for team in teams:
                running_pts[team] = 0
                running_gd[team] = 0
            running_pts_df = pd.DataFrame(running_pts, index = [0])
            running_gd_df = pd.DataFrame(running_gd, index = [0])

            for match, home, away, home_goal, away_goal in temp.sort_values("date")[["match_api_id", "home_team_api_id", "away_team_api_id", 'home_team_goal', 'away_team_goal']].values:
                home_stats_df = team_stats[home]
                away_stats_df = team_stats[away]
                home_stat = home_stats_df.loc[home_stats_df["match_api_id"].values == match]
                away_stat = away_stats_df.loc[away_stats_df["match_api_id"].values == match]

                if home_stat["pld"].values == 0:
                    home_stat["pos"] = 0
                else:
                    pos_gd = np.sum(running_gd_df[running_pts_df.keys()[running_pts_df.iloc[0].values == home_stat["pts"].values]].iloc[0].values > home_stat["GD"].values)
                    pos_pts = np.sum(running_pts_df.iloc[0].values > home_stat["pts"].values)
                    home_stat["pos"] = pos_gd + pos_pts + 1

                if away_stat["pld"].values == 0:
                    away_stat["pos"] = 0
                else:
                    pos_gd = np.sum(running_gd_df[running_pts_df.keys()[running_pts_df.iloc[0].values == away_stat["pts"].values]].iloc[0].values > away_stat["GD"].values)
                    pos_pts = np.sum(running_pts_df.iloc[0].values > away_stat["pts"].values)
                    away_stat["pos"] = pos_gd + pos_pts + 1
                 
                diff = home_goal - away_goal
                if diff > 0:
                    running_pts_df[home] +=3
                elif diff == 0:
                    running_pts_df[home] +=1
                    running_pts_df[away] +=1
                else:
                    running_pts_df[away] +=3
                running_gd_df[home] += diff
                running_gd_df[away] -= diff

                match_stat = pd.merge(home_stat, away_stat, on = ["date", "match_api_id"], how = "inner", suffixes=["_home", "_away"])
                diff_temp = (home_stat.iloc[:,2:]-away_stat.iloc[:,2:]).add_suffix("_diff")
                diff_df = pd.concat([home_stat.iloc[:,:2], diff_temp], axis = 1)
                match_stat_final = pd.merge(match_stat, diff_df, on = ["date", "match_api_id"], how = "inner", suffixes=["", ""])
                df_to_concat = pd.concat([df_to_concat, match_stat_final], join ="outer")

            # Look for incomplete data
            # temp = df[np.logical_and(df["league_id"] == league, df["season"] == season)]
            # length = len(np.unique(temp["home_team_api_id"].values))
            # if (length*(length-1) > temp.shape[0] or length < 10):
            #     print(df_league["name"][df_league["id"] == league].values[0])
            #     print(season)
            #     print(f"number of teams:{length}")
            #     print(f"number of matches:{temp.shape[0]}")
            #     print("  ")
    full = pd.merge(df, df_to_concat, on = ["date", "match_api_id"], suffixes=["", ""])
    return full


# print(df_match_with_outcome.keys())
# display(df_match_with_outcome)
df_match_full = running_score(df_match_with_outcome)
display(df_match_full.drop(['id', "country_id", "league_id", 'season', 'stage', 'date',
       'home_team_api_id', 'away_team_api_id', 'home_team_goal',
       'away_team_goal', 'home_team_win', 'draw', 'away_team_win'], axis = 1))
# # %%
# display(df_match_full)
# X = df_match_full.drop(['id', "country_id", "league_id", 'season', 'stage', 'date',
#        'home_team_api_id', 'away_team_api_id', 'home_team_goal',
#        'away_team_goal', 'home_team_win', 'draw', 'away_team_win'], axis = 1)
# X.to_csv("match_data_with_stats_v2.csv", index = None)
# %%
# df_match_full.to_csv("match_data_with_stats.csv", index = None)
# %% 
# display(df_match)
outcomes = create_outcome(df_match_small["home_team_goal"].to_numpy(), df_match_small["away_team_goal"].to_numpy())
outcomes = outcomes@[0,1,1]
outcomes = pd.concat([df_match_small["match_api_id"], outcomes], axis = 1, keys = ["match_api_id", "outcome"])
df_match_stats = pd.read_csv("match_data_with_stats_v2.csv")
scale = StandardScaler().fit(df_match_stats.drop("match_api_id",axis = 1))
df_match_with_id = pd.merge(df_match[["match_api_id", "home_team_api_id", "away_team_api_id"]], df_match_stats, on = "match_api_id").sort_values("match_api_id")
away_full_df = pd.concat([df_match_with_id["match_api_id"], df_match_with_id.filter(like = "away")], axis = 1)
away_full_df.columns = away_full_df.columns.str.replace('_away', '')
home_full_df = pd.concat([df_match_with_id["match_api_id"], df_match_with_id.filter(like = "home")], axis = 1)
home_full_df.columns = home_full_df.columns.str.replace('_home', '')

final_list = []
final_outcome = []
for match, home, away in df_match_with_id[["match_api_id", "home_team_api_id", "away_team_api_id"]].values:
    number_of_d = 9
    matches_with_home = np.logical_or(df_match_with_id["home_team_api_id"] == home, df_match_with_id["away_team_api_id"] == home)
    matches_with_away = np.logical_or(df_match_with_id["home_team_api_id"] == away, df_match_with_id["away_team_api_id"] == away)
    home_before = np.logical_and(matches_with_home, df_match_with_id["match_api_id"] < match)
    away_before = np.logical_and(matches_with_away, df_match_with_id["match_api_id"] < match)
    if np.sum(home_before) < number_of_d or np.sum(away_before) < number_of_d:
        continue
    df_home_away = away_full_df[home_before][-number_of_d:].query(f"away_team_api_id == {home}")
    df_home_home = home_full_df[home_before][-number_of_d:].query(f"home_team_api_id == {home}")
    home_stats = pd.concat([df_home_away, df_home_home]).sort_values("match_api_id").drop(["match_api_id", "home_team_api_id", "away_team_api_id"], axis = 1)
    df_away_away = away_full_df[away_before][-number_of_d:].query(f"away_team_api_id == {away}")
    df_away_home = home_full_df[away_before][-number_of_d:].query(f"home_team_api_id == {away}")
    away_stats = pd.concat([df_away_away, df_away_home]).sort_values("match_api_id").drop(["match_api_id", "home_team_api_id", "away_team_api_id"], axis = 1)
    diff_stats = home_stats.values - away_stats.values
    total_pre = np.concatenate([home_stats.values, away_stats.values, diff_stats], axis = 1)
    total_pre = np.concatenate([total_pre, df_match_stats.query(f"match_api_id == {match}").drop(["match_api_id"], axis = 1)])
    total_pre = scale.transform(total_pre).reshape(1,-1,10)
    final_list.append(total_pre)
    final_outcome.append(outcomes.query(f"match_api_id == {match}").values)


final_array = np.concatenate(final_list, axis = 0)
np.save("RNN_data_v2.npy", final_array)
final_outcome = np.concatenate(final_outcome, axis = 0)
np.save("RNN_outcome_v2.npy", final_outcome)
# np.save(final_array

# %%

# Create a sample NumPy array
arr = np.zeros((25500, 70, 10), dtype=np.float32)

# Save the array to a binary file
np.save('array_data.npy', arr)
# %%


    # home_df = df_match_with_id[np.logical_and(matches_with_home, df_match_with_id["match_api_id"] <= match)]

    # away_df = df_match_with_id[np.logical_and(matches_with_home, df_match_with_id["match_api_id"] <= match)]
    # df_match_with_id[np.logical_and(matches_with_away, df_match_with_id["match_api_id"] < match)].iloc[-number_of_d:])
#        'home_team_api_id', 'away_team_api_id', 'home_team_goal',
#        'away_team_goal', 'home_team_win', 'draw', 'away_team_win'], axis = 1)
# X.to_csv("match_stats_with_api.csv", index = None)
# %%

def create_outcome(home_goal, away_goal):
    diff = home_goal - away_goal
    outcome = np.concatenate([[diff > 0], [diff == 0], [diff < 0]], axis = 0).T.astype(int)
    return pd.DataFrame(outcome, columns=["home_team_win", "draw", "away_team_win"])
df_outcomes = create_outcome(df_match["home_team_goal"].to_numpy(), df_match["away_team_goal"].to_numpy())


y = df_outcomes[['home_team_win', 'draw', 'away_team_win']]
y = y@[0, 1, 2]
display(y)

# X = df_match_stats.drop(['id', 'season', 'stage', 'date', 'match_api_id',
#        'home_team_api_id', 'away_team_api_id', 'home_team_goal',
#        'away_team_goal', 'home_team_win', 'draw', 'away_team_win'], axis = 1)
X = df_match_stats.iloc[:,30:]
display(X)
display(X)
display(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Converting the dataset in proper LGB format
# d_train=lgb.Dataset(X_train, label=y_train)
# #setting up the parameters
# params={}
# params['learning_rate']=0.03
# params['boosting_type']='gbdt' #GradientBoostingDecisionTree
# params['objective']='multiclass' #Multi-class target feature
# params['metric']='multi_logloss' #metric for multi-class
# params['max_depth']=10
# params['num_class']=3 #no.of unique values in the target class not inclusive of the end value
# #training the model
# model=lgb.train(params,d_train,100)  #training the model on 100 epocs
# #prediction on the test dataset
def bayes_optimize_params_cv(X, y, classifier, param_dict_range, rng):
    
    def opt_func(**kwargs):
        
        for key, value in kwargs.items():
            if value > 1:
                kwargs[key] = int(value)
        score = cross_val_score(classifier(**kwargs), X, y, scoring = "roc_auc_ovo", cv = 5).mean()
        return score
    
    optimizer = BayesianOptimization(f = opt_func,
                                     pbounds = param_dict_range,
                                     random_state = rng,
                                     verbose=2)
    optimizer.maximize(n_iter = 30)    
    params = {}
    for key, value in optimizer.max["params"].items():
        if value > 1:
            params[key] = int(value)      
    return params
param_dict = {
    "min_child_samples": (0, 400), 
    "num_leaves": (2,300)
    }
# params = bayes_optimize_params_cv(X, y, LGBMClassifier, param_dict, 42)
# model = LGBMClassifier(**params, objective = "multiclass")
# model.fit(X_train, y_train)
# print(model.score(X_train, y_train))
# print(model.score(X_test, y_test))
# %%
print(sorted(list(zip(model.feature_importances_, X.keys())), reverse = True))
# %%
y_pred = model.predict(X_test)
print(np.sum(y_pred == 0)/y_pred.shape)
print(np.sum(y_pred == 1)/y_pred.shape)
print(np.sum(y_pred == 2)/y_pred.shape)
print(model.predict_proba(X_test))
print(y_test)
# %%
df_match_stats = pd.read_csv("match_data_with_stats_v2.csv")
df_bookies = pd.read_csv(r'.\bookielist.csv', index_col=0)
join_key = "match_api_id"
out = create_outcome(df_match_small["home_team_goal"].to_numpy(), df_match_small["away_team_goal"].to_numpy())
y = pd.concat([df_match_small[join_key], pd.DataFrame(out@[0, 1, 1],columns = ["y"])], axis = 1)

X = df_match_stats.merge(y, on = join_key)
print(X.shape[0]*0.2)
X_train_only = X[~X[join_key].isin(df_bookies[join_key]).values]

y_train_only = X_train_only["y"]
y_train_only.name = "y"
y_train_only = pd.DataFrame(y_train_only)
X_train_only = X_train_only.drop([join_key, "y"], axis = 1)

bookies_keys = df_bookies.keys().to_numpy()
X =  X.merge(df_bookies, on = join_key)
df_bookies = X[bookies_keys]
y = X[[join_key, "y"]]
X = X.drop([*bookies_keys, "y"], axis = 1)
split_pro = 0.2*out.shape[0]/X.shape[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_pro)
y_test_matches = y_test[join_key]
y_train = y_train.drop(join_key, axis = 1)
y_test = y_test.drop(join_key, axis = 1)
X_train = pd.concat([X_train, X_train_only], axis = 0)
y_train = pd.concat([y_train, y_train_only], axis = 0)
print(y_train)
print(y_test)
param_dict = {
    "min_child_samples": (10, 700), 
    "num_leaves": (2,100),
    "n_estimators": (10, 300),
    }
params = bayes_optimize_params_cv(X, y.drop([join_key], axis =1), LGBMClassifier, param_dict, 40)
model = LGBMClassifier(**params)
model.fit(X_train, y_train)
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))
y_pred = model.predict(X_test)
print(np.sum(y_pred == 0)/y_pred.shape)
print(np.sum(y_pred == 1)/y_pred.shape)
 # %%


def roi_loss(y_true, proba, df_bookies, plot=False, cutoff=0, betting_strat=0):
    df_bookies = df_bookies.drop(['match_api_id', 'outcome'], axis=1)
    proba = pd.DataFrame(proba)
    proba = np.tile(proba, int(df_bookies.shape[1] / 2))
    df_new = (proba / (df_bookies))
    # Check for cutoff
    df_conf_check = df_new.max(axis=1).values
    df_new = df_new.loc[df_conf_check>cutoff]
    df_bookies = df_bookies.loc[df_conf_check>cutoff]

    df_idx = df_new.idxmax(axis=1).apply(lambda x: df_bookies.columns.get_loc(x))
    y_guess = np.zeros_like(df_bookies)
    y_guess[np.arange(df_bookies.shape[0]), df_idx.values] = (df_new.to_numpy()[np.arange(df_bookies.shape[0]), df_idx.values])**betting_strat

    y_true['H'] = np.abs(y_true['outcome']-1)
    y_true['XA'] = y_true['outcome']
    y_true = y_true.drop(['outcome'], axis=1)

    y_true = y_true.loc[df_conf_check>cutoff]
    
    y_true = np.tile(y_true, int(df_bookies.shape[1] / 2))
    
    df_money = y_true*(1/df_bookies)*y_guess
    #df_final = df_money.sum(axis=1).sum(axis=0)
    y = list((df_money.sum(axis=1)-(df_new.to_numpy()[np.arange(df_bookies.shape[0]), df_idx.values])**betting_strat).to_numpy())
    roi = np.cumsum(y)[-1]/len(y)
    if plot:
        return roi, y
    return roi

y_pred = model.predict_proba(X_test)
test_book = pd.DataFrame(y_test_matches).merge(df_bookies, on = join_key, how = "inner")

y_test_corruption = y_test.copy()
roi, y = roi_loss(y_test_corruption, y_pred, test_book, plot = True)
print(roi)
# %%
plt.plot(np.cumsum(y))
# %%
print(model.feature_importances_)
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.show()
y_pred = model.predict_proba(X_test)
sns.histplot(y_pred[:,0][y_test==1], color = "r", label = "1", alpha = 0.7)
sns.histplot(y_pred[:,0][y_test==0], color = "k", label = "0", alpha = 0.7)
plt.legend()
plt.show()
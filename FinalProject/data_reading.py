#%%
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
import warnings
warnings.filterwarnings('ignore')
#%%
conn = db.connect(r'..\database.sqlite')
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table';")
category_names = c.fetchall()
# %%
df_sqlite_seq = pd.read_sql_query('select * from sqlite_sequence', conn)
df_player_attributes = pd.read_sql_query('select * from Player_Attributes', conn)
df_player = pd.read_sql_query('select * from Player', conn)
df_match = pd.read_sql_query('select * from Match', conn)
df_league = pd.read_sql_query('select * from League', conn)
df_country = pd.read_sql_query('select * from Country', conn)
df_team = pd.read_sql_query('select * from Team', conn)
df_team_attributes = pd.read_sql_query('select * from Team_Attributes', conn)

# %%
def match_outcome(match):
    outcome = pd.DataFrame()
    outcome.loc[0, 'match_api_id'] = match['match_api_id']

    home_goal = match['home_team_goal']
    away_goal = match['away_team_goal']
    if home_goal > away_goal:
        outcome.loc[0, 'outcome'] = 1
    if home_goal < away_goal:
        outcome.loc[0, 'outcome'] = 0
    if home_goal == away_goal:
        outcome.loc[0, 'outcome'] = 0
    return outcome

def all_match_outcome(match_df):
    outcomes = []
    for home, away in zip(match_df['home_team_goal'], match_df['away_team_goal']):
        if home > away:
            outcomes.append(1)
        if home < away:
            outcomes.append(0)
        if home == away:
            outcomes.append(0)
    df_outcome = pd.DataFrame({'outcome': outcomes})  
    df_outcome['match_api_id'] = match_df['match_api_id'] 
    return df_outcome

def convert_time_to_unix(date):
    formated_date = datetime.datetime.strptime(date,"%Y-%m-%d %H:%M:%S")
    Unix_timestamp = datetime.datetime.timestamp(formated_date)
    return Unix_timestamp

def find_features_for_match(match_api):
    # Find relevant match
    match = df_match.loc[df_match['match_api_id'] == match_api]
    # Find the date and format as Unix
    date = convert_time_to_unix(match['date'][0])
    # Find home away id's
    home_api = match['home_team_api_id'][0]
    away_api = match['away_team_api_id'][0]
    # Find all stats of home team
    home_stats = df_team_attributes.loc[df_team_attributes['team_api_id'] == home_api]
    home_stat_date_idx = np.argmin([np.abs(date - convert_time_to_unix(d)) for d in home_stats['date']])
    home_stats = pd.DataFrame(home_stats.iloc[home_stat_date_idx]).T
    print(home_stats.shape)
    # Find all stats of away team
    away_stats = df_team_attributes.loc[df_team_attributes['team_api_id'] == away_api]
    away_stat_date_idx = np.argmin([np.abs(date - convert_time_to_unix(d)) for d in away_stats['date']])
    away_stats = pd.DataFrame(away_stats.iloc[away_stat_date_idx]).T
    print(away_stats.shape)

    df = pd.concat([home_stats.add_suffix('_home'), away_stats.add_suffix('_away')], axis=1)
    return df

def find_features_for_all_match(match_df):
    # Find relevant match
    dates_unix = match_df['date'].apply(convert_time_to_unix).to_numpy()
    home_api = match_df['home_team_api_id'].to_numpy()
    away_api = match_df['away_team_api_id'].to_numpy()
    match_api = match_df['match_api_id'].to_numpy()
    error_match_api_idx = []
    df_home = pd.DataFrame()
    df_away = pd.DataFrame()
    df_team_attributes_stripped = df_team_attributes.loc[:, ~df_team_attributes.columns.str.contains('Class')].drop(['team_fifa_api_id', 'id'], axis=1)
    for i, (home, away) in enumerate(zip(home_api, away_api)):
        try:
            home_stats = df_team_attributes_stripped.loc[df_team_attributes_stripped['team_api_id'] == home]
            home_stat_date_idx = np.argmin([np.abs(dates_unix[i] - convert_time_to_unix(d)) for d in home_stats['date']])
            home_stats = pd.DataFrame(home_stats.iloc[home_stat_date_idx]).T

            away_stats = df_team_attributes_stripped.loc[df_team_attributes_stripped['team_api_id'] == away]
            away_stat_date_idx = np.argmin([np.abs(dates_unix[i] - convert_time_to_unix(d)) for d in away_stats['date']])
            away_stats = pd.DataFrame(away_stats.iloc[away_stat_date_idx]).T

            df_home = df_home.append(home_stats) 
            df_away = df_away.append(away_stats) 
        except:
            error_match_api_idx.append(np.where(match_api==match_api[i])[0][0])
            pass
    match_api = np.delete(match_api, error_match_api_idx)
    #display(df_home)
    df_all = pd.concat([df_home.add_suffix('_home').reset_index(), df_away.add_suffix('_away').reset_index()], axis=1)
    df_all['match_api_id'] = match_api
    df_all = df_all.drop(['index', 'date_home', 'date_away'], axis=1)
    #df_all = df_home.merge(df_away, left_on=list(df_home.columns), right_on=list(df_away.columns), suffixes=('_home', '_away'))
    return df_all

# %%
#df_x = find_features_for_all_match(df_match)
#df_x.to_csv(r'C:\Users\caspe\Desktop\AppliedML2023\FinalProject\features_classification')
df_x = pd.read_csv(r'.\features_classification', index_col=0)
# %%
df_y = all_match_outcome(df_match)
df_y = df_y[df_y.match_api_id.isin(df_x.match_api_id)]

# %%
df_x = df_x.apply(pd.to_numeric, errors='coerce')
X_train, X_test, y_train, y_test = train_test_split(df_x.drop(['match_api_id', 'team_api_id_home', 'team_api_id_away'], axis=1).reset_index(drop=True), df_y.drop(['match_api_id'], axis=1).reset_index(drop=True), test_size=0.2, random_state=42)

params = run_bayesian_opt_lightgbm(df_x.drop(['match_api_id', 'team_api_id_home', 'team_api_id_away'], axis=1).reset_index(drop=True), df_y.drop(['match_api_id'], axis=1).reset_index(drop=True), (20,100), (1,200), (0.01,0.5), (50,200), 50)

lgb = LGBMClassifier(**params)
model = lgb.fit(X_train, y_train)
y_pred = model.predict(X_train)
proba = model.predict_proba(X_train).T[1]
model.get_params()
train_acc = accuracy_score(y_train, y_pred)

fig, ax = plt.subplots()
df_train_result = pd.DataFrame({'Class':proba,'outcome':y_train['outcome']})
sns.histplot(data=df_train_result, x='Class',hue='outcome',alpha=1,ax=ax,multiple='stack')
ax.vlines(x=0.5, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], linestyle='--', label='Separator', color='k')
ax.set_xlabel('Classification Score')
ax.set_title(f'Stacked classification histogram of revenue\nTrain set - Accuracy = {train_acc:.2%}')
#%%
# Testing
y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
proba = model.predict_proba(X_test).T[1]
print("Accuracy:",test_acc)
#%%
# Plot Testing
fig, ax = plt.subplots()
df_test_result = pd.DataFrame({'Class':proba,'Truth':y_test['Truth']})
sns.histplot(data=df_test_result, x='Class',hue='Truth',alpha=1,ax=ax,multiple='stack')
ax.vlines(x=0.5, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], linestyle='--', label='Separator', color='k')
ax.set_xlabel('Classification Score')
ax.set_title(f'Stacked classification histogram of revenue\nTest set - Accuracy = {test_acc:.2%}')
#%%
import optuna
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


dtrain = lgb.Dataset(X_train, label=y_train)

param_distributions = {
    "lambda_l1": (1e-8, 10.0),
    "lambda_l2": (1e-8, 10.0),
    "num_leaves": (2, 256),
    "feature_fraction": (0.4, 1.0),
    "bagging_fraction": (0.4, 1.0),
    "bagging_freq": (1, 7),
    "min_child_samples": (5, 100),
    "learning_rate": (0.01, 0.5),
}

def objective(trial):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_loguniform("lambda_l1", *param_distributions["lambda_l1"]),
        "lambda_l2": trial.suggest_loguniform("lambda_l2", *param_distributions["lambda_l2"]),
        "num_leaves": trial.suggest_int("num_leaves", *param_distributions["num_leaves"]),
        "feature_fraction": trial.suggest_uniform("feature_fraction", *param_distributions["feature_fraction"]),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", *param_distributions["bagging_fraction"]),
        "bagging_freq": trial.suggest_int("bagging_freq", *param_distributions["bagging_freq"]),
        "min_child_samples": trial.suggest_int("min_child_samples", *param_distributions["min_child_samples"]),
        "learning_rate": trial.suggest_loguniform("learning_rate", *param_distributions["learning_rate"]),
        "feature_pre_filter": False,  # Set feature_pre_filter to False
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:, 1]
    logloss = log_loss(y_test, y_pred)

    return logloss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

params = trial.params
params["objective"] = "binary"
params["metric"] = "binary_logloss"
params["verbosity"] = -1
params["boosting_type"] = "gbdt"

model_optuna = lgb.LGBMClassifier(**params)
model_optuna.fit(X_train, y_train)

print('Training score', model_optuna.score(X_train, y_train))
print('Validation score', model_optuna.score(X_test, y_test))
print(model_optuna.feature_importances_)
# %%

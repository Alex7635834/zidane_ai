# %% Entraînement modèles de score
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
import optuna
import joblib

features = [
    'neutral', 'year', 'tournament_score',
    'cum_titles_home', 'cum_runner_up_home', 'cum_third_home',
    'cum_titles_away', 'cum_runner_up_away', 'cum_third_away',
    'elo_home', 'elo_away', 'elo_diff',
    'home_last5_wins', 'home_last5_losses', 'home_last5_draws',
    'away_last5_wins', 'away_last5_losses', 'away_last5_draws',
    'h2h_home_wins', 'h2h_away_wins', 'h2h_draw', 'h2h_diff'
]

df = pd.read_csv("../../data/data.csv")
df = df.dropna(subset=['home_score', 'away_score'])

train = df[df['year'] < 2018]
test  = df[df['year'] >= 2018]

X_train, X_test = train[features], test[features]

model_home = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
model_home.fit(X_train, train['home_score'])

model_away = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
model_away.fit(X_train, train['away_score'])

print(f"MAE home : {mean_absolute_error(test['home_score'], model_home.predict(X_test)):.4f}")
print(f"MAE away : {mean_absolute_error(test['away_score'], model_away.predict(X_test)):.4f}")

joblib.dump(model_home, '../models/model_home.pkl')
joblib.dump(model_away, '../models/model_away.pkl')
print("Modèles score sauvegardés !")

def objective_home(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
    }
    m = XGBRegressor(**params)
    m.fit(X_train, train['home_score'])
    return mean_absolute_error(test['home_score'], m.predict(X_test))

def objective_away(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
    }
    m = XGBRegressor(**params)
    m.fit(X_train, train['away_score'])
    return mean_absolute_error(test['away_score'], m.predict(X_test))


optuna.logging.set_verbosity(optuna.logging.INFO)

study_home = optuna.create_study(direction='minimize')
study_home.optimize(objective_home, n_trials=50)

study_away = optuna.create_study(direction='minimize')
study_away.optimize(objective_away, n_trials=50)

print(f"Meilleur MAE home : {study_home.best_value:.4f}")
print(f"Meilleur MAE away : {study_away.best_value:.4f}")


model_home = XGBRegressor(**study_home.best_params, random_state=42, objective='count:poisson', max_delta=0.7)
model_home.fit(X_train, train['home_score'])

model_away = XGBRegressor(**study_away.best_params, random_state=42, objective='count:poisson', max_delta=0.7)
model_away.fit(X_train, train['away_score'])

joblib.dump(model_home, '../models/model_home.pkl')
joblib.dump(model_away, '../models/model_away.pkl')
print("Modèles score sauvegardés !")


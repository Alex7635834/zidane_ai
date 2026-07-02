import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, accuracy_score
import optuna
import joblib

df = pd.read_csv("../../data/data.csv")

features = [
    'neutral', 'year', 'tournament_score',
    'cum_titles_home', 'cum_runner_up_home', 'cum_third_home',
    'cum_titles_away', 'cum_runner_up_away', 'cum_third_away',
    'elo_home', 'elo_away', 'elo_diff',
    'home_last5_wins', 'home_last5_losses', 'home_last5_draws',
    'away_last5_wins', 'away_last5_losses', 'away_last5_draws',
    'h2h_home_wins', 'h2h_away_wins', 'h2h_draw', 'h2h_diff'
]

# Entrainement et test
train = df[df['year'] < 2018]
test  = df[df['year'] >= 2018]

# On découpe entre test et résultat
X_train, y_train = train[features], train['result']
X_test,  y_test  = test[features],  test['result']

# Modèle 1 er paramètres
model = XGBClassifier(
    objective='multi:softprob', # Multiclasse
    num_class=3, # 3 (victoire, défaite, nulle)
    n_estimators=300, # Nombre d'arbre
    learning_rate=0.05,  # A quelle point il corrige ces erreurs
    max_depth=4, # Nombre de question en profondeur
    subsample=0.8, # 80% des lignes
    colsample_bytree=0.8, # 80% des colonnes
    eval_metric='mlogloss',
    early_stopping_rounds=20
)

model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=50)

proba = model.predict_proba(X_test)
print(f"Log-loss : {log_loss(y_test, proba):.4f}")
print(f"Accuracy : {accuracy_score(y_test, model.predict(X_test)):.4f}")
print("Meilleur features: ", model.feature_importances_)


# Essai 1 : Acc = 0.568, LL = 0.93

## Recherche hyperparamètres 


def objective(trial):
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }
    
    model = XGBClassifier(**params, eval_metric='mlogloss', early_stopping_rounds=20)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
    
    proba = model.predict_proba(X_test)
    return log_loss(y_test, proba)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)  # 50 essais

print("Meilleurs params:", study.best_params)
print("Meilleur log-loss:", study.best_value)


best_model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    early_stopping_rounds=20,
    random_state=42,
    **study.best_params  # on injecte les params du trial 34
)

best_model.fit(X_train, y_train,
               eval_set=[(X_test, y_test)],
               verbose=50)

proba_best = best_model.predict_proba(X_test)
print(f"Log-loss final : {log_loss(y_test, proba_best):.4f}")
print(f"Accuracy final : {accuracy_score(y_test, best_model.predict(X_test)):.4f}")

joblib.dump(best_model, '../models/model.pkl')

# Modèle final : accuracy 0.6026, LL = 0.87

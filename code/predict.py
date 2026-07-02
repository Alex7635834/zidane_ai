import pandas as pd
import numpy as np
import joblib
import sys
from traitement_clean import run_pipeline
from scipy.stats import poisson

model = joblib.load('../models/model.pkl')
model_home = joblib.load('../models/model_home.pkl')
model_away = joblib.load('../models/model_away.pkl')

features = [
    'neutral', 'year', 'tournament_score',
    'cum_titles_home', 'cum_runner_up_home', 'cum_third_home',
    'cum_titles_away', 'cum_runner_up_away', 'cum_third_away',
    'elo_home', 'elo_away', 'elo_diff',
    'home_last5_wins', 'home_last5_losses', 'home_last5_draws',
    'away_last5_wins', 'away_last5_losses', 'away_last5_draws',
    'h2h_home_wins', 'h2h_away_wins', 'h2h_draw', 'h2h_diff'
]

def predict(home_team, away_team, neutral=True):
    results = pd.read_csv("../data/results.csv")
    
    results = results[
        (results["home_team"] == home_team) | 
        (results["away_team"] == home_team) |
        (results["home_team"] == away_team) | 
        (results["away_team"] == away_team)
    ].copy()

    # Injection du match fictif
    fake = pd.DataFrame([{
        "date": "2026-07-01",
        "home_team": home_team,
        "away_team": away_team,
        "home_score": np.nan,
        "away_score": np.nan,
        "tournament": "FIFA World Cup",
        "city": "Unknown",
        "country": "Unknown",
        "neutral": neutral,
    }])

    results = pd.concat([results, fake], ignore_index=True)

    # Pipeline complet
    results = run_pipeline(results)

    # Dernière ligne = notre match
    match = results.iloc[[-1]][features]
    pd.set_option('display.max_columns', None)  # toutes les colonnes
    pd.set_option('display.max_rows', None)     # toutes les lignes
    pd.set_option('display.width', None)        # pas de retour à la ligne
    pd.set_option('display.float_format', '{:.2f}'.format)  # 2 décimales

    print(match.to_string())
    proba = model.predict_proba(match)[0] # Prédiction des probas de gagner

    print(f"\n{'='*38}")
    print(f"  {home_team} vs {away_team}")
    print(f"{'='*38}")
    print(f"  {home_team} gagne : {proba[2]*100:.1f}%")
    print(f"  Match nul        : {proba[1]*100:.1f}%")
    print(f"  {away_team} gagne : {proba[0]*100:.1f}%")
    print(f"{'='*38}")

    if proba[2] > proba[0] and proba[2] > proba[1]:
        winner = home_team
    elif proba[0] > proba[2] and proba[0] > proba[1]:
        winner = away_team
    else:
        winner = "Match nul"

    print(f"  Pronostic → {winner}")
    print(f"{'='*38}\n")
    print(f"Scores poisson\n")
    print(f"{'='*38}\n")


    lambda_home = model_home.predict(match)[0] 
    lambda_away = model_away.predict(match)[0] 

    # Distribution des scores probables
    for h in range(6):
        for a in range(6):
            p = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
            if p > 0.05:
                print(f"{h}-{a} : {p*100:.1f}%")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage : python predict.py <home> <away>")
        sys.exit(1)
    predict(sys.argv[1], sys.argv[2])
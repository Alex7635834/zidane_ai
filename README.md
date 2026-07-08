# Zidane AI - World Cup Predictor

Modèle de machine learning pour prédire l'issue et le score de matchs internationaux de football, entraîné sur **49 000 matchs depuis 1872**.

---

## Ce que ça fait

Pour deux équipes données, le modèle prédit :
- **Le gagnant** — avec les probabilités (victoire home / nul / victoire away)
- **Le score probable** — via une distribution de Poisson sur les combinaisons de scores

```bash
python predict.py France Brazil

  France vs Brazil
  ======================================
  France gagne  : 45.2%
  Match nul     : 27.1%
  Brazil gagne  : 27.7%
  ======================================
  Pronostic → France

  Scores Poisson
  1-0 : 13.2%
  1-1 : 11.4%
  0-0 : 10.8%
  2-1 :  8.3%
  ...
```

---

## Architecture

```
results.csv  (49 000 matchs bruts depuis 1872)
     │
     ▼
traitement_clean.py         ← pipeline de feature engineering
     │
     ├── Titres cumulés WC  (cum_titles, cum_runner_up, cum_third)
     ├── Score d'importance du tournoi  (Friendly=1 → FIFA WC=5)
     ├── ELO dynamique      (mis à jour match par match)
     ├── Forme last 5       (wins / losses / draws sur 5 derniers matchs)
     └── H2H                (historique des confrontations directes, 10 derniers)
     │
     ▼
data.csv  (dataset enrichi avec toutes les features)
     │
     ├──▶ training.py        ← XGBoost Classifier  →  model.pkl
     └──▶ training_score.py  ← XGBoost Regressor x2 →  model_home.pkl / model_away.pkl
```

---

## Features utilisées

| Feature | Description |
|---|---|
| `elo_home / elo_away / elo_diff` | Score ELO de chaque équipe au moment du match |
| `home_last5_wins/losses/draws` | Forme récente de l'équipe à domicile |
| `away_last5_wins/losses/draws` | Forme récente de l'équipe à l'extérieur |
| `h2h_home_wins / away_wins / draw` | Historique des confrontations directes |
| `h2h_diff` | Différence de victoires en H2H |
| `cum_titles/runner_up/third` | Palmarès historique en Coupe du Monde |
| `tournament_score` | Importance de la compétition (1 à 5) |
| `neutral` | Terrain neutre ou non |
| `year` | Année du match |

---

## Modèles

### Prédiction du gagnant — `model.pkl`
- **Algorithme** : XGBoost Classifier (`multi:softprob`, 3 classes)
- **Optimisation** : Optuna (50 trials, recherche bayésienne)
- **Métrique** : Log-loss / Accuracy
- **Résultats** : ~0.87 log-loss / ~60% accuracy sur données post-2018

### Prédiction du score — `model_home.pkl` / `model_away.pkl`
- **Algorithme** : XGBoost Regressor × 2 (un par équipe)
- **Optimisation** : Optuna (50 trials)
- **Métrique** : MAE (Mean Absolute Error)
- **Résultats** : ~1.04 buts MAE home / ~0.86 buts MAE away
- **Post-processing** : Distribution de Poisson sur les λ prédits pour obtenir les probas de chaque score

---

## Comment ça marche au niveau la prédiction

Au lieu de chercher les stats dans le dataset, `predict.py` **rejoue le pipeline** sur les matchs des deux équipes :

```
1. Charger results.csv
2. Filtrer uniquement les matchs impliquant l'une des deux équipes
3. Injecter le match fictif en dernière ligne (score = NaN)
4. Lancer traitement_clean.py → toutes les features se calculent naturellement
5. Récupérer la dernière ligne = notre match avec ses stats à jour
6. Donner au modèle → probas + scores Poisson
```

Ainsi le ELO, le last5 et le H2H sont **toujours à jour** par rapport au dernier match connu.

---

## Installation

```bash
git clone https://github.com/Alex7635834/Zidane_AI
cd zidane_ai
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
xgboost
scikit-learn
optuna
scipy
joblib
```

### Entraîner les modèles

```bash
# 1. Générer le dataset enrichi
python traitement_clean.py

# 2. Entraîner le modèle gagnant
python training.py

# 3. Entraîner les modèles de score
python training_score.py
```

### Prédire un match

```bash
python predict.py <home_team> <away_team>

# Exemples
python predict.py France Brazil
python predict.py Argentina Germany
python predict.py Mexico Ecuador
```

> Les noms d'équipes doivent correspondre exactement à ceux du dataset (ex: `Brazil` et non `Brasil`).

---

## Performances

| Modèle | Métrique | Score | Baseline |
|---|---|---|---|
| Gagnant | Accuracy | 60.1% | 33% (aléatoire) |
| Gagnant | Log-loss | 0.87 | 1.10 |
| Score home | MAE | 1.04 buts | — |
| Score away | MAE | 0.86 buts | — |

> Le foot est par nature difficile à prédire — 60% d'accuracy sur des matchs internationaux est dans le haut de ce qu'on peut espérer avec des données tabulaires.

---

## Résultats

Après simulation de monte-carlo, (predict_winner.py) voici les résultats du modèle.
```bash
=== Sur 20 000 simulations ===

France          :  20.0% de titres
Spain           :  19.6% de titres
Argentina       :  19.5% de titres
England         :  14.4% de titres
Morocco         :  11.7% de titres
Norway          :   3.9% de titres
Belgium         :   3.6% de titres
Portugal        :   3.0% de titres
Colombia        :   1.9% de titres
Switzerland     :   1.8% de titres
Egypt           :   0.5% de titres
United States   :   0.1% de titres

=== Podiums (champion, finaliste, troisième) les plus fréquents ===

Spain - Argentina - France : 2.42%
Spain - England - France : 2.17%
France - England - Spain : 2.00%
Argentina - France - England : 2.00%
Argentina - Spain - France : 1.98%
France - Argentina - England : 1.90%
France - England - Argentina : 1.82%
Spain - Argentina - Morocco : 1.77%
Spain - England - Morocco : 1.68%
France - Argentina - Spain : 1.65%

```

## Auteur

Ruiz Alexandre - élève ingénieur à l'école des Mines de Douai
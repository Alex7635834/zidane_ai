# formers_name = vrai nom des pays. current,former,start_date,end_date
# goalsoccers = buteurs. date,home_team,away_team,team,scorer,minute,own_goal,penalty
# results = resultat domicile / ext, type de championnat. date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
# shootsout = gagnant premier buteur. date,home_team,away_team,winner,first_shooter
# stadium = key_id,stadium_id,stadium_name,city_name,country_name,stadium_capacity,stadium_wikipedia_link,city_wikipedia_link
# players = key_id,player_id,family_name,given_name,birth_date,goal_keeper,defender,midfielder,forward,count_tournaments,list_tournaments,player_wikipedia_link
# awards winner = key_id,tournament_id,tournament_name,award_id,award_name,shared,player_id,family_name,given_name


## ETAPE 2 qui a gagné quoi au moment du match

import pandas as pd
from retrieval import retrieve_team_data

retrieve_team_data()

results = pd.read_csv("../data/results.csv")
former_name = pd.read_csv("../data/former_names.csv")
wc = pd.read_csv("../data/tournament_standings.csv")

mapping = dict(zip(former_name["former"], former_name["current"])) # dictionnaire former : current

print(mapping)

wc["team_name"] = wc["team_name"].replace(mapping)

results["date"] = pd.to_datetime(results["date"])
results["year"] = results["date"].dt.year
wc["wc_year"] = wc["tournament_name"].str[:4].astype(int)


wc = wc.sort_values(["team_name", "wc_year"]) # On trie

wc["title"] = (wc["position"] == 1).astype(int) # On nomme les colonnes 1er 2eme 3eme
wc["runner_up"] = (wc["position"] == 2).astype(int)
wc["third"] = (wc["position"] == 3).astype(int)

wc["cum_titles"] = wc.groupby("team_name")["title"].cumsum() # On les agrège
wc["cum_runner_up"] = wc.groupby("team_name")["runner_up"].cumsum()
wc["cum_third"] = wc.groupby("team_name")["third"].cumsum()

wc.to_csv("history_winners.csv", index = False)


def get_wc_hist(team, year): # combien de fois premier, 2 eme et 3eme
    history = wc[
        (wc["team_name"] == team) & (wc["wc_year"] < year)
    ]

    if history.empty:
        return (0, 0, 0)
    return (history.iloc[-1]["cum_titles"], history.iloc[-1]["cum_runner_up"], history.iloc[-1]["cum_third"])

def build_cumulative():
    for index, row in results.iterrows():

        team_home = row["home_team"]
        team_away = row["away_team"]
        year = row["year"]
        titles, runners, thirds = get_wc_hist(team_home, year)
        a_titles, a_runners, a_thirds = get_wc_hist(team_away, year)

        results.loc[index, "cum_titles_home"] = titles
        results.loc[index, "cum_runner_up_home"] = runners
        results.loc[index, "cum_third_home"] = thirds

        results.loc[index, "cum_titles_away"] = a_titles
        results.loc[index, "cum_runner_up_away"] = a_runners
        results.loc[index, "cum_third_away"] = a_thirds

build_cumulative() 

print("après build cumulative")
print(results.tail())

## Mapping de l'importance de la compétition

competition_importance = {

    # 5 = Compétitions majeures
    "FIFA World Cup": 5,
    "UEFA Euro": 5,
    "Copa América": 5,
    "African Cup of Nations": 5,
    "AFC Asian Cup": 5,
    "CONCACAF Gold Cup": 5,
    "Gold Cup": 5,
    "Oceania Nations Cup": 5,
    "Confederations Cup": 5,

    # 4 = Qualifications majeures + Nations League
    "FIFA World Cup qualification": 4,
    "UEFA Euro qualification": 4,
    "African Cup of Nations qualification": 4,
    "AFC Asian Cup qualification": 4,
    "Gold Cup qualification": 4,
    "CONCACAF Championship qualification": 4,
    "Oceania Nations Cup qualification": 4,
    "COSAFA Cup": 4,
    "CECAFA Cup": 4,
    "UNCAF Cup": 4,
    "AFF Championship": 4,
    "ASEAN Championship": 4,
    "EAFF Championship": 4,
    "WAFF Championship": 4,
    "CONCACAF Nations League": 4,
    "UEFA Nations League": 4,
    "CAFA Nations Cup": 4,

    # 3 = Jeux olympiques et compétitions continentales secondaires
    "Olympic Games": 3,
    "Asian Games": 3,
    "Pan American Championship": 3,
    "Central American and Caribbean Games": 3,
    "South Asian Games": 3,
    "Pacific Games": 3,
    "All-African Games": 3,
    "Bolivarian Games": 3,
    "Intercontinental Cup": 3,

    # 2 = Tournois régionaux ou invitation
    "Kirin Cup": 2,
    "Kirin Challenge Cup": 2,
    "King's Cup": 2,
    "Gulf Cup": 2,
    "Merdeka Tournament": 2,
    "Rous Cup": 2,
    "Tournoi de France": 2,
    "Four Nations Tournament": 2,
    "Three Nations Cup": 2,
    "Tri Nation Tournament": 2,
    "Tri-Nations Cup": 2,
    "Tri-Nations Series": 2,
    "Mauritius Four Nations Cup": 2,
    "Cyprus International Tournament": 2,
    "Nehru Cup": 2,
    "Merlion Cup": 2,
    "Simba Tournament": 2,
    "Prime Minister's Cup": 2,

    # 1 = Matchs amicaux
    "Friendly": 1,
}

results["tournament_score"] = results["tournament"].map(competition_importance).fillna(1)

print("après mapping tournoi")
print(results.tail())

#calcul de l'elo

default_elo = 1500
HOME_ADVANTAGE = 70  # points virtuels ajoutés à l'équipe à domicile pour le calcul de la proba attendue
elo = {}

def expected(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))  # Proba de gagner

def get_result(home, away):
    if home > away:
        return 1
    elif home < away:
        return 0
    else:
        return 0.5

def goal_diff_multiplier(goal_diff): # Multiplicateur de but de difference marqué cf elorating.net
    goal_diff = abs(goal_diff)
    if goal_diff <= 1:
        return 1
    elif goal_diff == 2:
        return 1.5
    else:
        return (11 + goal_diff) / 8

def k_factor(tournament_score): # Varie selon importance du tournoi
    base_k = {
        1: 20,   # Friendly
        2: 25,
        3: 30,
        4: 35,
        5: 40,
        6: 45,
        7: 50,
        8: 55,
        9: 60,
        10: 65,  # FIFA World Cup
    }

    score = int(round(tournament_score))
    score = max(1, min(10, score))
    return base_k[score]

def boucle():

    results["elo_home"] = 0.0
    results["elo_away"] = 0.0

    for i, row in results.iterrows():  # On boucle sur les matchs

        home = row["home_team"]
        away = row["away_team"]

        elo_home = elo.get(home, default_elo)  # On prend son elo si il en a un sinon on prend elo par defaut
        elo_away = elo.get(away, default_elo)

        results.at[i, "elo_home"] = elo_home  # On enregistre l'elo
        results.at[i, "elo_away"] = elo_away

        exp_home = expected(elo_home + HOME_ADVANTAGE, elo_away) # Probabilité attendue, avec avantage terrain pour l'équipe à domicile
        exp_away = 1 - exp_home

        home_score = row["home_score"]
        away_score = row["away_score"]
        goal_diff = home_score - away_score

        result = get_result(home_score, away_score)  # On observe qui a gagné en vrai

        G = goal_diff_multiplier(goal_diff)
        K = k_factor(row["tournament_score"])

        elo[home] = elo_home + K * G * (result - exp_home)
        elo[away] = elo_away + K * G * ((1 - result) - exp_away)

    results["elo_diff"] = results["elo_home"] - results["elo_away"]

boucle()

print("après build")
print(results.tail())

results.to_csv("../data/results_elo_last5.csv", index = False)

ranking = (
    pd.DataFrame({
        "team": elo.keys(),
        "elo": elo.values()
    })
    .sort_values("elo", ascending=False)
)

print(ranking.head(20))

from collections import defaultdict, deque

form = defaultdict(lambda: deque(maxlen=5))

def stats(hist):
    wins = sum(x == 1 for x in hist)
    losses = sum(x == -1 for x in hist)
    draws = sum(x == 0 for x in hist)
    return wins, losses, draws

def last5():

    for i, row in results.iterrows():

        home = row["home_team"]
        away = row["away_team"]

        hw, hl, hd = stats(form[home]) #scores wins loss draw des 5 derniers match
        aw, al, ad = stats(form[away])

        results.at[i, "home_last5_wins"] = hw # dans la ligne i colonne ca met hw
        results.at[i, "home_last5_losses"] = hl
        results.at[i, "home_last5_draws"] = hd

        results.at[i, "away_last5_wins"] = aw
        results.at[i, "away_last5_losses"] = al
        results.at[i, "away_last5_draws"] = ad

        # update AFTER match
        if row["home_score"] > row["away_score"]:
            form[home].append(1)
            form[away].append(-1)

        elif row["home_score"] < row["away_score"]:
            form[home].append(-1)
            form[away].append(1)

        else:
            form[home].append(0)
            form[away].append(0)



last5()

results.to_csv("../data/results_final.csv", index = False)

form = defaultdict(lambda: deque(maxlen=10))

def stats(hist, home, away):
    wins = sum(x == home for x in hist)
    losses = sum(x == away for x in hist)
    draws = sum(x == "draw" for x in hist)
    return wins, losses, draws

results=pd.read_csv("../data/results_final.csv")

def h2h():
    rencontres={}
    for i, row in results.iterrows():

        
        home = row["home_team"]
        away = row["away_team"]

        couple = tuple(sorted([home,away]))

        if couple in rencontres.keys() :
            results.at[i,"h2h_home_wins"]=stats(rencontres[couple], home, away)[0]
            results.at[i,"h2h_away_wins"]=stats(rencontres[couple], home, away)[1]
            results.at[i,"h2h_draw"]=stats(rencontres[couple], home, away)[2]
            results["h2h_diff"] = (results["h2h_home_wins"] - results["h2h_away_wins"])
        else:

            rencontres[couple]=deque(maxlen=10)
            results.at[i,"h2h_home_wins"]=stats(rencontres[couple], home, away)[0]
            results.at[i,"h2h_away_wins"]=stats(rencontres[couple], home, away)[1]
            results.at[i,"h2h_draw"]=stats(rencontres[couple], home, away)[2]
            results["h2h_diff"] = (results["h2h_home_wins"] - results["h2h_away_wins"])
        
        if row["home_score"] > row["away_score"]:
            rencontres[couple].append(home)

        elif row["home_score"] < row["away_score"]:
            rencontres[couple].append(away)

        else:
            rencontres[couple].append("draw")
            
h2h()

results.to_csv("../data/results_final_v2.csv", index = False)

results=pd.read_csv("../data/results_final_v2.csv")
## Définition de la target 2 = home gagne, 1 = égalité, 0 = away gagne

def get_result(row):
    if row['home_score'] > row['away_score']: return 2
    elif row['home_score'] < row['away_score']: return 0
    else: return 1

results['result'] = results.apply(get_result, axis=1)

results.to_csv("../data/data.csv", index = False)

## ETAPE 1 les noms

"""
mapping = dict(zip(former_name["former"], former_name["current"])) # dictionnaire former : current

print(mapping)

results["home_team"] = results["home_team"].replace(mapping)
results["away_team"] = results["away_team"].replace(mapping)
results["country"] = results["country"].replace(mapping)

results.to_csv("../data/results.csv", index=False)

results_world_cup_titles = results.merge(
    former_name.add_suffix("_home"),
    left_on="home_team",
    right_on="former_home",
    how="left"
)

results_former_name = results_world_cup_titles.merge(
    former_name.add_suffix("_away"),
    left_on="away_team",
    right_on="former_away",
    how="left"
)

results_world_cup_titles["home_team"] = results_former_name["current_home"].fillna(0)

print(results_world_cup_titles.head())

results_world_cup_titles.to_csv("../data/result_world_cup_titles.csv", index=False)


results_former_name = results.merge(
    former_name.add_suffix("_home"),
    left_on="home_team",
    right_on="former_home",
    how="left"
)

results_former_name = results_former_name.merge(
    former_name.add_suffix("_away"),
    left_on="away_team",
    right_on="former_away",
    how="left"
)

results_former_name = results_former_name.merge(
    former_name.add_suffix("_country"),
    left_on="country",
    right_on="former_country",
    how="left"
)

results_former_name["home_team"] = results_former_name["current_home"].fillna(results_former_name["home_team"])

print(results_former_name.head())

results_former_name.to_csv("../data/match2.csv", index=False)

"""
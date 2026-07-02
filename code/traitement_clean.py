import pandas as pd
from collections import defaultdict, deque

def run_pipeline(results):
    print("Début pipeline")
    former_name = pd.read_csv("../data/former_names.csv")
    wc = pd.read_csv("../data/tournament_standings.csv")

    mapping = dict(zip(former_name["former"], former_name["current"]))
    wc["team_name"] = wc["team_name"].replace(mapping)

    results["date"] = pd.to_datetime(results["date"])
    results["year"] = results["date"].dt.year
    wc["wc_year"] = wc["tournament_name"].str[:4].astype(int)

    wc = wc.sort_values(["team_name", "wc_year"])
    wc["title"]     = (wc["position"] == 1).astype(int)
    wc["runner_up"] = (wc["position"] == 2).astype(int)
    wc["third"]     = (wc["position"] == 3).astype(int)
    wc["cum_titles"]    = wc.groupby("team_name")["title"].cumsum()
    wc["cum_runner_up"] = wc.groupby("team_name")["runner_up"].cumsum()
    wc["cum_third"]     = wc.groupby("team_name")["third"].cumsum()

    print("Titres cumulés")
    def get_wc_hist(team, year):
        history = wc[(wc["team_name"] == team) & (wc["wc_year"] < year)]
        if history.empty:
            return (0, 0, 0)
        return (history.iloc[-1]["cum_titles"], history.iloc[-1]["cum_runner_up"], history.iloc[-1]["cum_third"])

    for index, row in results.iterrows():
        t, r, th   = get_wc_hist(row["home_team"], row["year"])
        at, ar, ath = get_wc_hist(row["away_team"], row["year"])
        results.loc[index, "cum_titles_home"]    = t
        results.loc[index, "cum_runner_up_home"] = r
        results.loc[index, "cum_third_home"]     = th
        results.loc[index, "cum_titles_away"]    = at
        results.loc[index, "cum_runner_up_away"] = ar
        results.loc[index, "cum_third_away"]     = ath


    competition_importance = {
        "FIFA World Cup": 5, "UEFA Euro": 5, "Copa América": 5,
        "African Cup of Nations": 5, "AFC Asian Cup": 5,
        "CONCACAF Gold Cup": 5, "Gold Cup": 5, "Oceania Nations Cup": 5,
        "Confederations Cup": 5, "FIFA World Cup qualification": 4,
        "UEFA Euro qualification": 4, "African Cup of Nations qualification": 4,
        "AFC Asian Cup qualification": 4, "Gold Cup qualification": 4,
        "CONCACAF Championship qualification": 4, "Oceania Nations Cup qualification": 4,
        "COSAFA Cup": 4, "CECAFA Cup": 4, "UNCAF Cup": 4,
        "AFF Championship": 4, "ASEAN Championship": 4, "EAFF Championship": 4,
        "WAFF Championship": 4, "CONCACAF Nations League": 4, "UEFA Nations League": 4,
        "CAFA Nations Cup": 4, "Olympic Games": 3, "Asian Games": 3,
        "Pan American Championship": 3, "Central American and Caribbean Games": 3,
        "South Asian Games": 3, "Pacific Games": 3, "All-African Games": 3,
        "Bolivarian Games": 3, "Intercontinental Cup": 3, "Kirin Cup": 2,
        "Kirin Challenge Cup": 2, "King's Cup": 2, "Gulf Cup": 2,
        "Merdeka Tournament": 2, "Rous Cup": 2, "Tournoi de France": 2,
        "Four Nations Tournament": 2, "Three Nations Cup": 2,
        "Tri Nation Tournament": 2, "Tri-Nations Cup": 2, "Tri-Nations Series": 2,
        "Mauritius Four Nations Cup": 2, "Cyprus International Tournament": 2,
        "Nehru Cup": 2, "Merlion Cup": 2, "Simba Tournament": 2,
        "Prime Minister's Cup": 2, "Friendly": 1,
    }
    results["tournament_score"] = results["tournament"].map(competition_importance).fillna(1)

    print("Calcul ELO")
    default_elo    = 1500
    HOME_ADVANTAGE = 70
    elo = {}

    def expected(elo_a, elo_b):
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    def goal_diff_multiplier(goal_diff):
        goal_diff = abs(goal_diff)
        if goal_diff <= 1:   return 1
        elif goal_diff == 2: return 1.5
        else:                return (11 + goal_diff) / 8

    def k_factor(tournament_score):
        base_k = {1:20, 2:25, 3:30, 4:35, 5:40, 6:45, 7:50, 8:55, 9:60, 10:65}
        score = max(1, min(10, int(round(tournament_score))))
        return base_k[score]

    results["elo_home"] = 0.0
    results["elo_away"] = 0.0

    for i, row in results.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        elo_home = elo.get(home, default_elo)
        elo_away = elo.get(away, default_elo)

        results.at[i, "elo_home"] = elo_home
        results.at[i, "elo_away"] = elo_away

        # Match fictif → on lit l'elo mais on ne le met pas à jour
        if pd.isna(row["home_score"]):
            continue

        exp_home  = expected(elo_home + HOME_ADVANTAGE, elo_away)
        goal_diff = row["home_score"] - row["away_score"]
        result    = 1 if goal_diff > 0 else (0 if goal_diff < 0 else 0.5)
        G = goal_diff_multiplier(goal_diff)
        K = k_factor(row["tournament_score"])

        elo[home] = elo_home + K * G * (result - exp_home)
        elo[away] = elo_away + K * G * ((1 - result) - (1 - exp_home))

    results["elo_diff"] = results["elo_home"] - results["elo_away"]

    print("Last 5")
    form = defaultdict(lambda: deque(maxlen=5))

    def stats_last5(hist):
        return (
            sum(x == 1  for x in hist),
            sum(x == -1 for x in hist),
            sum(x == 0  for x in hist)
        )

    for i, row in results.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        hw, hl, hd = stats_last5(form[home])
        aw, al, ad = stats_last5(form[away])

        results.at[i, "home_last5_wins"]   = hw
        results.at[i, "home_last5_losses"] = hl
        results.at[i, "home_last5_draws"]  = hd
        results.at[i, "away_last5_wins"]   = aw
        results.at[i, "away_last5_losses"] = al
        results.at[i, "away_last5_draws"]  = ad

        if pd.isna(row["home_score"]):
            continue

        if row["home_score"] > row["away_score"]:
            form[home].append(1);  form[away].append(-1)
        elif row["home_score"] < row["away_score"]:
            form[home].append(-1); form[away].append(1)
        else:
            form[home].append(0);  form[away].append(0)

    print("H2H")
    rencontres = {}

    def stats_h2h(hist, home, away):
        return (
            sum(x == home   for x in hist),
            sum(x == away   for x in hist),
            sum(x == "draw" for x in hist)
        )

    for i, row in results.iterrows():
        home   = row["home_team"]
        away   = row["away_team"]
        couple = tuple(sorted([home, away]))

        if couple not in rencontres:
            rencontres[couple] = deque(maxlen=10)

        hw, aw, dw = stats_h2h(rencontres[couple], home, away)
        results.at[i, "h2h_home_wins"] = hw
        results.at[i, "h2h_away_wins"] = aw
        results.at[i, "h2h_draw"]      = dw
        results.at[i, "h2h_diff"]      = hw - aw

        if pd.isna(row["home_score"]):
            continue

        if row["home_score"] > row["away_score"]:
            rencontres[couple].append(home)
        elif row["home_score"] < row["away_score"]:
            rencontres[couple].append(away)
        else:
            rencontres[couple].append("draw")

    def get_result(row):
        if pd.isna(row["home_score"]): return None
        if row["home_score"] > row["away_score"]: return 2
        elif row["home_score"] < row["away_score"]: return 0
        else: return 1

    results["result"] = results.apply(get_result, axis=1)

    results.to_csv("../data/data_limit.csv", index=False)

    return results


if __name__ == "__main__":
    results = pd.read_csv("../data/results.csv")
    results = run_pipeline(results)
    results.to_csv("../data/data_limit.csv", index=False)
    print("data.csv généré !")
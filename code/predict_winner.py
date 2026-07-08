from predict import predict, create_dataset_tournament, predict_tournament
from collections import Counter
import numpy as np

teams = ["France", "Morocco", "Belgium", "United States", "Portugal", "Spain",
         "Switzerland", "Colombia", "Argentina", "Egypt", "Norway", "England"]

df = create_dataset_tournament(teams)

_cache = {}

def get_proba(home, away):
    key = (home, away)
    if key not in _cache:
        winner, proba, lam_h, lam_a = predict_tournament(home, away, df)
        _cache[key] = proba  # [p_away, p_draw, p_home]
    return _cache[key]

def draw_winner(home, away):
    p_away, p_draw, p_home = get_proba(home, away)
    # phase à élimination directe : on redistribue la proba de nul au prorata de la force relative
    total = p_home + p_away
    p_home_ko = p_home + p_draw * (p_home / total) # prorata : produit en croix
    return home if np.random.random() < p_home_ko else away

def simulate_tournament():
    # 8èmes - le 06/07 
    w1 = draw_winner("Portugal", "Spain")
    w2 = draw_winner("Belgium", "United States")
    w3 = draw_winner("Switzerland", "Colombia")
    w4 = draw_winner("Argentina", "Egypt")

    # 1/4
    qA1 = draw_winner(w1, w2)
    qA2 = draw_winner("France", "Morocco")
    qB1 = draw_winner(w3, w4)
    qB2 = draw_winner("Norway", "England")

    # 1/2
    sA = draw_winner(qA1, qA2)
    sB = draw_winner(qB1, qB2)

    # Finale
    champion = draw_winner(sA, sB)
    finaliste = sB if champion == sA else sA
    demi_finalistes = {qA1, qA2, qB1, qB2} - {champion, finaliste}

    # Petite finale
    troisieme = draw_winner(list(demi_finalistes)[0], list(demi_finalistes)[1])
     
    return champion, finaliste, troisieme, tuple(sorted(demi_finalistes))

N = 20000

champions = Counter()
podiums = Counter()

for i in range(N):
    champ, finaliste, troisieme, autres = simulate_tournament()
    key = (champ, finaliste, troisieme)
    champions[champ] += 1
    podiums[key] += 1

print(f"\n=== Sur {N} simulations ===\n")
for team, count in champions.most_common():
    print(f"{team:15s} : {count/N*100:5.1f}% de titres")

print("\n=== Podiums (champion, finaliste, troisième) les plus fréquents ===\n")
for (champ, fin, ter), count in podiums.most_common(10):
    print(f"{champ} - {fin} - {ter} : {count/N*100:.2f}%")



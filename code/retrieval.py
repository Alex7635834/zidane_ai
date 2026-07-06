import kagglehub
import shutil
from pathlib import Path

def retrieve_team_data():

    DATA_DIR = Path(__file__).parent.parent / "data"
    DATA_DIR.mkdir(exist_ok=True)

    downloaded = kagglehub.dataset_download(
        "martj42/international-football-results-from-1872-to-2017",
        path="results.csv"
    )

    shutil.copy(downloaded, DATA_DIR / "results.csv")
    print(f"Copié dans {DATA_DIR / 'results.csv'}")
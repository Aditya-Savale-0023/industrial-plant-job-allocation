from pathlib import Path
import pandas as pd


# Project root: industrial-allocation/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


from pathlib import Path
import pandas as pd

DATA_DIR = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "processed"
)

def load_data():
    plants = pd.read_csv(DATA_DIR / "hubs.csv")
    jobs = pd.read_csv(DATA_DIR / "jobs.csv")
    compatibility = pd.read_csv(
        DATA_DIR / "compatibility.csv"
    )

    return plants, jobs, compatibility
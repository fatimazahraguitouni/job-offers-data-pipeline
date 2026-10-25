import pandas as pd
from sqlalchemy import text

from src.db import get_engine, save_postings

CLEAN_PATH = "data/processed/postings_clean.csv"

if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH, parse_dates=["listed_time"])
    save_postings(df)

    with get_engine().connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM postings")).scalar()
    print(f"{n} lignes chargées dans PostgreSQL")
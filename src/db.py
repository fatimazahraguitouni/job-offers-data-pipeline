import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

load_dotenv()


def get_engine():
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB"),
    )
    return create_engine(url)


def save_postings(df: pd.DataFrame, table: str = "postings") -> None:
    df.to_sql(
        table, get_engine(), if_exists="replace",
        index=False, chunksize=1000, method="multi",
    )


def read_postings(table: str = "postings") -> pd.DataFrame:
    return pd.read_sql(f"SELECT * FROM {table}", get_engine())
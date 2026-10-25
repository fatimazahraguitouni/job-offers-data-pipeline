import re
import pandas as pd

RAW_PATH = "data/raw/postings.csv"
CLEAN_PATH = "data/processed/postings_clean.csv"

COLUMNS_TO_KEEP = [
    "job_id", "company_name", "title", "description", "location",
    "formatted_work_type", "formatted_experience_level",
    "remote_allowed", "views", "applies", "normalized_salary",
    "listed_time",
]


def clean_text(text: str) -> str:
    """Met en minuscules, retire le HTML et les espaces en trop."""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_postings(df: pd.DataFrame) -> pd.DataFrame:
    df = df[COLUMNS_TO_KEEP].copy()

    # Doublons
    df = df.drop_duplicates(subset="job_id")

    # On a besoin du texte et de la cible
    df = df.dropna(subset=["title", "description", "formatted_experience_level"])

    # remote_allowed : vide = non précisé -> 0
    df["remote_allowed"] = df["remote_allowed"].fillna(0).astype(int)

    # Dates : timestamp en millisecondes -> date
    df["listed_time"] = pd.to_datetime(df["listed_time"], unit="ms")

    # Texte nettoyé + colonne combinée pour le modèle
    df["title"] = df["title"].map(clean_text)
    df["description"] = df["description"].map(clean_text)
    df["text"] = df["title"] + " " + df["description"]

    return df.reset_index(drop=True)


if __name__ == "__main__":
    import os

    raw = pd.read_csv(RAW_PATH)
    clean = clean_postings(raw)

    os.makedirs("data/processed", exist_ok=True)
    clean.to_csv(CLEAN_PATH, index=False)

    print("Avant :", raw.shape)
    print("Après :", clean.shape)
    print()
    print(clean["formatted_experience_level"].value_counts())
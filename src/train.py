import json
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.db import read_postings

TARGET = "formatted_experience_level"
MODEL_PATH = "models/experience_model.joblib"
METRICS_PATH = "models/metrics.json"


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2),
            min_df=5,
            stop_words="english",
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=300,
        )),
    ])


if __name__ == "__main__":
    df = read_postings()
    X, y = df["text"], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    print(classification_report(y_test, y_pred))
    print(f"F1 macro : {f1_macro:.3f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    report = classification_report(y_test, y_pred, output_dict=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"f1_macro": f1_macro, "report": report}, f, indent=2)
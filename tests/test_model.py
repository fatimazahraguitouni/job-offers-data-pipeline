from src.train import build_pipeline


def test_pipeline_fits_and_predicts():
    # min_df=5 dans le vectoriseur : on répète les exemples
    X = (["senior director leadership strategy team management"] * 10
         + ["intern student summer internship learning"] * 10)
    y = ["Director"] * 10 + ["Internship"] * 10

    model = build_pipeline()
    model.fit(X, y)

    pred = model.predict(["summer internship for a student"])[0]
    proba = model.predict_proba(["summer internship for a student"])[0]

    assert pred == "Internship"
    assert abs(proba.sum() - 1.0) < 1e-6
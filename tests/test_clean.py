import pandas as pd

from src.clean import COLUMNS_TO_KEEP, clean_postings, clean_text


def make_df(**overrides) -> pd.DataFrame:
    """Petit DataFrame de test avec toutes les colonnes attendues."""
    row = {
        "job_id": 1,
        "company_name": "ACME",
        "title": "Data Engineer",
        "description": "<p>Build   pipelines</p>",
        "location": "Paris",
        "formatted_work_type": "Full-time",
        "formatted_experience_level": "Entry level",
        "remote_allowed": 1.0,
        "views": 10,
        "applies": 2,
        "normalized_salary": 50000.0,
        "listed_time": 1_700_000_000_000,
    }
    row.update(overrides)
    return pd.DataFrame([row], columns=COLUMNS_TO_KEEP)


def test_clean_text_removes_html_and_spaces():
    assert clean_text("<p>Hello   WORLD</p>") == "hello world"


def test_clean_text_handles_non_string():
    assert clean_text(None) == "none"


def test_duplicates_are_removed():
    df = pd.concat([make_df(), make_df()], ignore_index=True)
    assert len(clean_postings(df)) == 1


def test_rows_without_target_are_dropped():
    df = pd.concat(
        [make_df(job_id=1), make_df(job_id=2, formatted_experience_level=None)],
        ignore_index=True,
    )
    result = clean_postings(df)
    assert result["job_id"].tolist() == [1]


def test_rows_without_description_are_dropped():
    df = make_df(description=None)
    assert len(clean_postings(df)) == 0


def test_remote_allowed_missing_becomes_zero():
    result = clean_postings(make_df(remote_allowed=None))
    assert result.loc[0, "remote_allowed"] == 0


def test_text_column_combines_title_and_description():
    result = clean_postings(make_df())
    assert result.loc[0, "text"] == "data engineer build pipelines"


def test_listed_time_is_converted_to_datetime():
    result = clean_postings(make_df())
    assert pd.api.types.is_datetime64_any_dtype(result["listed_time"])
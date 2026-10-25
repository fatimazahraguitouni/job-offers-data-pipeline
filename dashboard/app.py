import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# Permet d'importer le dossier src depuis dashboard/
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.clean import clean_text  # noqa: E402
from src.db import get_engine  # noqa: E402

MODEL_PATH = ROOT / "models" / "experience_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

st.set_page_config(page_title="Offres d'emploi LinkedIn", page_icon="📊", layout="wide")


@st.cache_data(ttl=600)
def query(sql: str) -> pd.DataFrame:
    return pd.read_sql(sql, get_engine())


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("📊 Offres d'emploi LinkedIn 2023-2024")
st.caption("Pipeline : CSV Kaggle → pandas → PostgreSQL → scikit-learn → Streamlit")

tab_overview, tab_model, tab_predict = st.tabs(["Aperçu", "Modèle", "Prédire un niveau"])

# ---------------------------------------------------------------- Aperçu
with tab_overview:
    kpis = query("""
        SELECT COUNT(*) AS offres,
               COUNT(DISTINCT company_name) AS entreprises,
               ROUND(100.0 * AVG(remote_allowed), 1) AS pct_remote,
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY normalized_salary) AS salaire_median
        FROM postings
    """).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Offres", f"{int(kpis['offres']):,}".replace(",", " "))
    c2.metric("Entreprises", f"{int(kpis['entreprises']):,}".replace(",", " "))
    c3.metric("Télétravail autorisé", f"{kpis['pct_remote']} %")
    c4.metric("Salaire annuel médian (normalisé)", f"{kpis['salaire_median']:,.0f}".replace(",", " "))

    left, right = st.columns(2)

    levels = query("""
        SELECT formatted_experience_level AS niveau, COUNT(*) AS offres
        FROM postings GROUP BY 1 ORDER BY 2 DESC
    """)
    left.plotly_chart(
        px.bar(levels, x="niveau", y="offres", title="Offres par niveau d'expérience"),
        use_container_width=True,
    )

    work_types = query("""
        SELECT formatted_work_type AS contrat, COUNT(*) AS offres
        FROM postings GROUP BY 1 ORDER BY 2 DESC
    """)
    right.plotly_chart(
        px.bar(work_types, x="contrat", y="offres", title="Offres par type de contrat"),
        use_container_width=True,
    )

    timeline = query("""
        SELECT DATE_TRUNC('day', listed_time) AS jour, COUNT(*) AS offres
        FROM postings GROUP BY 1 ORDER BY 1
    """)
    st.plotly_chart(
        px.line(timeline, x="jour", y="offres", title="Offres publiées par jour"),
        use_container_width=True,
    )

    left2, right2 = st.columns(2)

    locations = query("""
        SELECT location AS lieu, COUNT(*) AS offres
        FROM postings GROUP BY 1 ORDER BY 2 DESC LIMIT 15
    """)
    left2.plotly_chart(
        px.bar(locations.sort_values("offres"), x="offres", y="lieu",
               orientation="h", title="Top 15 des lieux"),
        use_container_width=True,
    )

    salaries = query("""
        SELECT formatted_experience_level AS niveau,
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY normalized_salary) AS salaire_median
        FROM postings WHERE normalized_salary IS NOT NULL
        GROUP BY 1 ORDER BY 2
    """)
    right2.plotly_chart(
        px.bar(salaries, x="niveau", y="salaire_median",
               title="Salaire médian par niveau (offres avec salaire)"),
        use_container_width=True,
    )

# ---------------------------------------------------------------- Modèle
with tab_model:
    if not METRICS_PATH.exists():
        st.warning("Aucun modèle trouvé. Lance d'abord : python -m src.train")
    else:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        report = metrics["report"]

        m1, m2 = st.columns(2)
        m1.metric("F1 macro", f"{metrics['f1_macro']:.3f}")
        m2.metric("Accuracy", f"{report['accuracy']:.3f}")

        classes = {k: v for k, v in report.items()
                   if k not in ("accuracy", "macro avg", "weighted avg")}
        table = pd.DataFrame(classes).T.rename(columns={"support": "nb exemples test"})
        st.dataframe(table.round(3), use_container_width=True)

        st.plotly_chart(
            px.bar(table.reset_index(), x="index", y="f1-score",
                   title="F1-score par niveau", labels={"index": "niveau"}),
            use_container_width=True,
        )
        st.caption(
            "Classes déséquilibrées : le modèle utilise class_weight='balanced' "
            "et un découpage stratifié ; la métrique principale est le F1 macro."
        )

# ---------------------------------------------------------------- Prédire
with tab_predict:
    if not MODEL_PATH.exists():
        st.warning("Aucun modèle trouvé. Lance d'abord : python -m src.train")
    else:
        model = load_model()
        title = st.text_input("Titre du poste", "Senior Data Engineer")
        description = st.text_area(
            "Description de l'offre", height=220,
            placeholder="Colle ici la description d'une offre d'emploi...",
        )

        if st.button("Prédire le niveau"):
            if not description.strip():
                st.error("Ajoute une description.")
            else:
                full_text = clean_text(title) + " " + clean_text(description)
                proba = model.predict_proba([full_text])[0]
                classes = model.named_steps["clf"].classes_
                result = (pd.DataFrame({"niveau": classes, "probabilité": proba})
                          .sort_values("probabilité", ascending=False))

                st.success(f"Niveau prédit : **{result.iloc[0]['niveau']}**")
                st.plotly_chart(
                    px.bar(result, x="niveau", y="probabilité"),
                    use_container_width=True,
                )
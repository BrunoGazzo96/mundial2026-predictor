import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Mundial 2026 Predictor",
    page_icon="🏆",
    layout="wide",
)

OUTPUTS = Path(__file__).parent.parent / "outputs"
DATA_RAW = Path(__file__).parent.parent / "data" / "raw"


@st.cache_data
def load_predictions():
    path = OUTPUTS / "predicciones_2026.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_teams():
    return pd.read_csv(DATA_RAW / "mundial2026_equipos.csv")


def fmt_pct(val: float) -> str:
    return f"{val*100:.1f}%"


teams_df = load_teams()
predictions = load_predictions()

st.title("🏆 Mundial 2026 — Predictor")
st.caption("Análisis predictivo basado en 45.000+ partidos históricos de fútbol internacional")

page = st.sidebar.radio(
    "Navegación",
    ["Dashboard", "Simulador", "Grupos", "EDA Highlights"],
    index=0,
)

# ── Dashboard ──────────────────────────────────────────────────────────────
if page == "Dashboard":
    st.header("Probabilidades de campeonato")

    if predictions is not None:
        cols = ["equipo", "grupos", "16vos", "cuartos", "semis", "final", "campeon"]
        df_show = predictions[cols].copy()
        for col in ["grupos", "16vos", "cuartos", "semis", "final", "campeon"]:
            df_show[col] = df_show[col].map(fmt_pct)

        st.dataframe(df_show, use_container_width=True, hide_index=True)

        fig = px.bar(
            predictions.head(10),
            x="equipo", y="campeon",
            title="Top 10 candidatos al título",
            labels={"campeon": "Prob. campeonato", "equipo": "Selección"},
            color="campeon",
            color_continuous_scale="Greens",
        )
        fig.update_traces(text=predictions.head(10)["campeon"].map(fmt_pct), textposition="outside")
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aún no hay predicciones generadas. Corré primero el notebook `03_modelo_simulacion.ipynb`.")

# ── Simulador ───────────────────────────────────────────────────────────────
elif page == "Simulador":
    st.header("Simulador de partidos")
    all_teams = sorted(teams_df["equipo"].tolist())
    col1, col2 = st.columns(2)
    with col1:
        equipo_a = st.selectbox("Selección A", all_teams, index=0)
    with col2:
        equipo_b = st.selectbox("Selección B", all_teams, index=1)

    if equipo_a == equipo_b:
        st.warning("Elegí dos selecciones diferentes.")
    else:
        st.info("El simulador estará disponible una vez entrenado el modelo.")

# ── Grupos ──────────────────────────────────────────────────────────────────
elif page == "Grupos":
    st.header("Grupos del Mundial 2026")
    for grupo in sorted(teams_df["grupo"].unique()):
        st.subheader(f"Grupo {grupo}")
        sub = teams_df[teams_df["grupo"] == grupo][
            ["equipo", "confederacion", "ranking_fifa", "mundiales_previos"]
        ].sort_values("ranking_fifa")
        st.dataframe(sub, use_container_width=True, hide_index=True)

# ── EDA Highlights ──────────────────────────────────────────────────────────
elif page == "EDA Highlights":
    st.header("EDA Highlights")
    st.info("Los gráficos del análisis exploratorio aparecerán aquí una vez completado el notebook `01_eda.ipynb`.")
    graficos = list((Path(__file__).parent.parent / "outputs" / "graficos").glob("*.png"))
    if graficos:
        for img_path in graficos:
            st.image(str(img_path), caption=img_path.stem)
    else:
        st.write("Sin gráficos exportados aún.")

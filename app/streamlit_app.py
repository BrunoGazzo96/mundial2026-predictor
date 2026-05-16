import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Mundial 2026 Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── ISO codes para el mapa ──────────────────────────────────────────────────
ISO_MAP = {
    "Argentina": "ARG", "Brasil": "BRA", "Francia": "FRA", "España": "ESP",
    "Inglaterra": "GBR", "Portugal": "PRT", "Países Bajos": "NLD",
    "Bélgica": "BEL", "Alemania": "DEU", "Uruguay": "URY", "Colombia": "COL",
    "Marruecos": "MAR", "Croacia": "HRV", "Suiza": "CHE", "Japón": "JPN",
    "México": "MEX", "Senegal": "SEN", "Ecuador": "ECU", "Australia": "AUS",
    "Corea del Sur": "KOR", "Estados Unidos": "USA", "Irán": "IRN",
    "Turquía": "TUR", "Austria": "AUT", "Canadá": "CAN", "Noruega": "NOR",
    "Escocia": "GBR", "Suecia": "SWE", "Túnez": "TUN",
    "Costa de Marfil": "CIV", "Arabia Saudita": "SAU", "Ghana": "GHA",
    "Egipto": "EGY", "Panamá": "PAN", "Paraguay": "PRY", "Argelia": "DZA",
    "Irak": "IRQ", "Jordania": "JOR", "Congo DR": "COD", "Uzbekistán": "UZB",
    "Sudáfrica": "ZAF", "Chequia": "CZE", "Catar": "QAT",
    "Bosnia Herzegovina": "BIH", "Haití": "HTI", "Curazao": "CUW",
    "Nueva Zelanda": "NZL", "Cabo Verde": "CPV",
}

CONF_COLORS = {
    "UEFA": "#1565C0", "CONMEBOL": "#2E7D32", "CONCACAF": "#F9A825",
    "CAF": "#B71C1C", "AFC": "#6A1B9A", "OFC": "#00838F",
}

RONDAS_ES = {
    "clasifica": "Clasifica", "16avos": "16avos", "octavos": "Octavos",
    "cuartos": "Cuartos", "semis": "Semis", "final": "Final", "campeon": "Campeón",
}


# ── Carga de datos (cacheada) ───────────────────────────────────────────────
@st.cache_data
def load_predictions():
    p = ROOT / "outputs" / "predicciones_2026.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_teams():
    return pd.read_csv(ROOT / "data" / "raw" / "mundial2026_equipos.csv")


@st.cache_data
def load_features_2026():
    p = ROOT / "data" / "processed" / "features_2026.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource
def load_probs_dict():
    """Carga el modelo y pre-calcula probabilidades de todos los enfrentamientos."""
    model_path = ROOT / "outputs" / "model_gbm.joblib"
    feat_path  = ROOT / "data" / "processed" / "features_2026.csv"
    if not model_path.exists() or not feat_path.exists():
        return None
    import joblib
    from src.features import FEATURE_COLS
    from src.simulator import precompute_match_probs
    data = joblib.load(model_path)
    model = data["model"]
    features_2026 = pd.read_csv(feat_path)
    return precompute_match_probs(model, features_2026)


# ── Helpers ─────────────────────────────────────────────────────────────────
def pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def fmt_table(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = out[c].map(pct)
    return out


# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("🏆 Mundial 2026")
st.sidebar.caption("Análisis predictivo basado en 45.000+ partidos históricos")
page = st.sidebar.radio(
    "Navegación",
    ["Dashboard", "Simulador de partidos", "Grupos", "EDA Highlights"],
)
st.sidebar.divider()
st.sidebar.markdown(
    "**Modelo:** GradientBoosting  \n"
    "**Simulaciones:** 10.000 torneos  \n"
    "**Features:** ELO · Forma reciente · Ranking FIFA  \n"
    "**Datos:** Kaggle · FIFA  \n"
)

# ── Carga ───────────────────────────────────────────────────────────────────
predictions = load_predictions()
teams_df    = load_teams()
probs_dict  = load_probs_dict()

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.title("🏆 Mundial 2026 — Predictor")
    st.markdown(
        "Simulación de **10.000 torneos** completos usando un modelo "
        "GradientBoosting entrenado sobre partidos históricos de alta competencia."
    )

    if predictions is None:
        st.error("No se encontró `outputs/predicciones_2026.csv`. Corré el notebook 03.")
        st.stop()

    pred = predictions.merge(teams_df[["equipo", "confederacion", "grupo"]], on="equipo")

    # Métricas clave
    top1, top2, top3, top4 = pred.head(4)["equipo"].values
    c1, c2, c3, c4 = st.columns(4)
    for col, eq, label in [
        (c1, top1, "🥇 Favorito"),
        (c2, top2, "🥈 2do"),
        (c3, top3, "🥉 3ro"),
        (c4, top4, "4to"),
    ]:
        row = pred[pred["equipo"] == eq].iloc[0]
        col.metric(label, eq, f"{row['campeon']*100:.1f}% prob. título")

    st.divider()

    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        # Bar chart top 15
        top15 = pred.head(15).copy()
        top15["color"] = top15["confederacion"].map(CONF_COLORS).fillna("#607D8B")
        top15["pct_label"] = top15["campeon"].map(pct)

        fig_bar = go.Figure(go.Bar(
            x=top15["campeon"] * 100,
            y=top15["equipo"],
            orientation="h",
            marker_color=top15["color"],
            text=top15["pct_label"],
            textposition="outside",
        ))
        fig_bar.update_layout(
            title="Top 15 candidatos al título",
            xaxis_title="Probabilidad de campeonato (%)",
            yaxis=dict(autorange="reversed"),
            height=480,
            margin=dict(l=10, r=60, t=40, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        # Mapa mundial
        pred["iso"] = pred["equipo"].map(ISO_MAP)
        fig_map = px.choropleth(
            pred,
            locations="iso",
            color="campeon",
            hover_name="equipo",
            hover_data={"campeon": ":.1%", "iso": False},
            color_continuous_scale="YlOrRd",
            title="Probabilidad de título por país",
        )
        fig_map.update_layout(
            height=380,
            margin=dict(l=0, r=0, t=40, b=0),
            coloraxis_colorbar=dict(title="Prob. título", tickformat=".0%"),
            geo=dict(showframe=False, showcoastlines=True),
        )
        st.plotly_chart(fig_map, use_container_width=True)

    st.divider()
    st.subheader("Tabla completa — 48 selecciones")

    rondas_cols = ["clasifica", "16avos", "octavos", "cuartos", "semis", "final", "campeon"]
    display = pred[["equipo", "confederacion", "grupo"] + rondas_cols].copy()
    display = display.rename(columns=RONDAS_ES)
    for c in RONDAS_ES.values():
        if c in display.columns:
            display[c] = display[c].map(pct)

    st.dataframe(
        display.sort_values("Campeón", ascending=False),
        use_container_width=True,
        hide_index=True,
        height=500,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2: SIMULADOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Simulador de partidos":
    st.title("⚽ Simulador de partidos")
    st.markdown("Elegí dos selecciones y el modelo predice las probabilidades.")

    all_teams = sorted(teams_df["equipo"].tolist())

    col1, col_vs, col2 = st.columns([2, 0.5, 2])
    with col1:
        team_a = st.selectbox("Selección A", all_teams, index=all_teams.index("Argentina"))
    with col_vs:
        st.markdown("<br><h2 style='text-align:center'>VS</h2>", unsafe_allow_html=True)
    with col2:
        team_b = st.selectbox("Selección B", all_teams, index=all_teams.index("Francia"))

    if team_a == team_b:
        st.warning("Elegí dos selecciones diferentes.")
        st.stop()

    if probs_dict is None:
        st.error("Modelo no disponible. Corré el notebook 03 para generar `model_gbm.joblib`.")
        st.stop()

    probs = probs_dict.get((team_a, team_b))
    if probs is None:
        st.error(f"No hay datos para {team_a} vs {team_b}.")
        st.stop()

    # sklearn: clase 0=B gana, 1=empate, 2=A gana
    p_b, p_e, p_a = float(probs[0]), float(probs[1]), float(probs[2])

    st.divider()

    # Probabilidades principales
    c1, c2, c3 = st.columns(3)
    c1.metric(f"🏅 Gana {team_a}", pct(p_a))
    c2.metric("🤝 Empate", pct(p_e))
    c3.metric(f"🏅 Gana {team_b}", pct(p_b))

    # Gauge doble
    fig_gauge = go.Figure()
    fig_gauge.add_trace(go.Bar(
        x=[p_a * 100, p_e * 100, p_b * 100],
        y=[team_a, "Empate", team_b],
        orientation="h",
        marker_color=["#2E7D32", "#9E9E9E", "#1565C0"],
        text=[pct(p_a), pct(p_e), pct(p_b)],
        textposition="outside",
    ))
    fig_gauge.update_layout(
        height=200,
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis=dict(range=[0, 100], title="Probabilidad (%)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Info adicional de los equipos
    def team_row(name):
        row = teams_df[teams_df["equipo"] == name]
        if row.empty:
            return {}
        return row.iloc[0].to_dict()

    info_a, info_b = team_row(team_a), team_row(team_b)
    if info_a and info_b:
        st.divider()
        ca, cb = st.columns(2)
        with ca:
            st.markdown(f"**{team_a}**")
            st.write(f"Confederación: {info_a['confederacion']}")
            st.write(f"Ranking FIFA: #{info_a['ranking_fifa']}")
            st.write(f"Mundiales previos: {info_a['mundiales_previos']}")
        with cb:
            st.markdown(f"**{team_b}**")
            st.write(f"Confederación: {info_b['confederacion']}")
            st.write(f"Ranking FIFA: #{info_b['ranking_fifa']}")
            st.write(f"Mundiales previos: {info_b['mundiales_previos']}")

    st.divider()

    # Simulación N veces
    st.subheader("Simulación de múltiples partidos")
    n_sim = st.slider("Cantidad de partidos a simular", 100, 5000, 1000, step=100)

    if st.button("▶ Simular", type="primary"):
        resultados_sim = np.random.choice(
            [f"Gana {team_a}", "Empate", f"Gana {team_b}"],
            size=n_sim,
            p=[p_a, p_e, p_b],
        )
        conteo = pd.Series(resultados_sim).value_counts().reset_index()
        conteo.columns = ["resultado", "count"]
        conteo["pct"] = conteo["count"] / n_sim

        fig_sim = px.bar(
            conteo,
            x="resultado", y="count",
            text=conteo["pct"].map(pct),
            title=f"Resultado de {n_sim:,} simulaciones: {team_a} vs {team_b}",
            color="resultado",
            color_discrete_map={
                f"Gana {team_a}": "#2E7D32",
                "Empate": "#9E9E9E",
                f"Gana {team_b}": "#1565C0",
            },
        )
        fig_sim.update_traces(textposition="outside")
        fig_sim.update_layout(showlegend=False, yaxis_title="Partidos",
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sim, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3: GRUPOS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Grupos":
    st.title("🗂️ Grupos del Mundial 2026")
    st.markdown(
        "Probabilidad de cada equipo de clasificar a los **16avos de final** "
        "(top-2 garantizados + mejores 8 terceros)."
    )

    if predictions is None:
        st.error("Corré el notebook 03 primero.")
        st.stop()

    pred = predictions.merge(teams_df[["equipo", "confederacion", "grupo"]], on="equipo")
    grupos_list = sorted(pred["grupo"].unique())

    for row_idx in range(0, len(grupos_list), 4):
        cols = st.columns(4)
        for col_idx, grupo in enumerate(grupos_list[row_idx:row_idx + 4]):
            with cols[col_idx]:
                g_df = (
                    pred[pred["grupo"] == grupo]
                    .sort_values("clasifica", ascending=False)
                    .reset_index(drop=True)
                )
                st.markdown(f"### Grupo {grupo}")

                fig_g = go.Figure(go.Bar(
                    x=g_df["clasifica"] * 100,
                    y=g_df["equipo"],
                    orientation="h",
                    marker_color=[
                        CONF_COLORS.get(c, "#607D8B") for c in g_df["confederacion"]
                    ],
                    text=g_df["clasifica"].map(pct),
                    textposition="outside",
                ))
                fig_g.add_vline(x=50, line_dash="dash", line_color="red", opacity=0.4)
                fig_g.update_layout(
                    height=160,
                    margin=dict(l=5, r=50, t=5, b=5),
                    xaxis=dict(range=[0, 115], title=""),
                    yaxis=dict(title=""),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                )
                st.plotly_chart(fig_g, use_container_width=True, key=f"grupo_{grupo}")

                # Mini tabla
                mini = g_df[["equipo", "clasifica", "campeon"]].copy()
                mini["clasifica"] = mini["clasifica"].map(pct)
                mini["campeon"]   = mini["campeon"].map(pct)
                mini.columns      = ["Equipo", "Clasifica", "Título"]
                st.dataframe(mini, hide_index=True, use_container_width=True)

        if row_idx + 4 < len(grupos_list):
            st.divider()

    # Ranking de grupos por dificultad
    st.divider()
    st.subheader("Ranking de grupos por dificultad media")
    st.caption("Promedio de ranking FIFA de los 4 equipos de cada grupo (menor = más difícil)")

    dif = teams_df.groupby("grupo").agg(
        ranking_medio=("ranking_fifa", "mean"),
        equipos=("equipo", lambda x: " · ".join(x)),
    ).reset_index().sort_values("ranking_medio")

    fig_dif = px.bar(
        dif, x="grupo", y="ranking_medio",
        title="Dificultad por grupo (ranking FIFA promedio)",
        labels={"ranking_medio": "Ranking FIFA promedio (menor = más difícil)", "grupo": "Grupo"},
        text="ranking_medio",
        color="ranking_medio",
        color_continuous_scale="RdYlGn_r",
    )
    fig_dif.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_dif.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
    st.plotly_chart(fig_dif, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4: EDA HIGHLIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "EDA Highlights":
    st.title("📊 EDA Highlights")
    st.markdown(
        "Los hallazgos más interesantes del análisis exploratorio sobre "
        "**45.000+ partidos** de fútbol internacional (1872–2026)."
    )

    GRAFICOS = ROOT / "outputs" / "graficos"

    charts = [
        ("goles_por_anio.png",           "Goles promedio por partido en Mundiales (1930–2022)"),
        ("campeones_historicos.png",      "Campeones mundiales 1930–2022"),
        ("confederaciones_por_anio.png",  "Equipos por confederación en cada Mundial"),
        ("resultados_por_contexto.png",   "Factor local: cancha propia vs neutral vs Mundial"),
        ("resultados_por_era.png",        "Resultados en Mundiales por era"),
        ("top_victorias_mundiales.png",   "Top 15 selecciones por victorias en Mundiales"),
        ("elo_wc2026.png",               "ELO Rating — 48 equipos clasificados al WC2026"),
        ("features_analisis.png",         "Correlación de features con el resultado"),
        ("feature_importance.png",        "Importancia de features — GradientBoosting"),
        ("predicciones_campeon.png",      "Probabilidades de título — resultado final"),
        ("heatmap_probabilidades.png",    "Probabilidades por ronda — Top 20"),
        ("clasificacion_grupos.png",      "Probabilidad de clasificar a 16avos por grupo"),
    ]

    for i in range(0, len(charts), 2):
        cols = st.columns(2)
        for j, (fname, title) in enumerate(charts[i:i + 2]):
            path = GRAFICOS / fname
            if path.exists():
                with cols[j]:
                    st.markdown(f"**{title}**")
                    st.image(str(path), use_container_width=True)
            else:
                cols[j].info(f"Gráfico pendiente: `{fname}`")

    # Scatter interactivo (HTML)
    html_path = GRAFICOS / "ranking_vs_winrate.html"
    if html_path.exists():
        st.divider()
        st.markdown("**Ranking FIFA vs Win Rate histórico en Mundiales (interactivo)**")
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=550, scrolling=False)

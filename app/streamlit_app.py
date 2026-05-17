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

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
h1 { font-weight: 800; letter-spacing: -0.04em; }
h2 { font-weight: 700; letter-spacing: -0.02em; }
h3 { font-weight: 600; }

[data-testid="stSidebar"] {
    background: #080c14;
    border-right: 1px solid rgba(255,255,255,0.05);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #8d99ae;
    font-size: 0.82rem;
    line-height: 1.8;
}

hr { border-color: rgba(255,255,255,0.07) !important; }

[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    overflow: hidden;
}

[data-testid="stMetricDelta"] svg { display: none; }
[data-testid="stMetricDelta"] { color: #8d99ae !important; font-size: 0.78rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Códigos ISO-2 para flagcdn.com ───────────────────────────────────────────
ISO2_MAP = {
    "Argentina": "ar",    "Brasil": "br",         "Francia": "fr",
    "España": "es",       "Inglaterra": "gb-eng",  "Portugal": "pt",
    "Países Bajos": "nl", "Bélgica": "be",         "Alemania": "de",
    "Uruguay": "uy",      "Colombia": "co",         "Marruecos": "ma",
    "Croacia": "hr",      "Suiza": "ch",            "Japón": "jp",
    "México": "mx",       "Senegal": "sn",          "Ecuador": "ec",
    "Australia": "au",    "Corea del Sur": "kr",    "Estados Unidos": "us",
    "Irán": "ir",         "Turquía": "tr",          "Austria": "at",
    "Canadá": "ca",       "Noruega": "no",          "Escocia": "gb-sct",
    "Suecia": "se",       "Túnez": "tn",            "Costa de Marfil": "ci",
    "Arabia Saudita": "sa", "Ghana": "gh",          "Egipto": "eg",
    "Panamá": "pa",       "Paraguay": "py",         "Argelia": "dz",
    "Irak": "iq",         "Jordania": "jo",         "Congo DR": "cd",
    "Uzbekistán": "uz",   "Sudáfrica": "za",        "Chequia": "cz",
    "Catar": "qa",        "Bosnia Herzegovina": "ba", "Haití": "ht",
    "Curazao": "cw",      "Nueva Zelanda": "nz",    "Cabo Verde": "cv",
}

# ── ISO-3 para el mapa choropleth ────────────────────────────────────────────
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


# ── Carga de datos ───────────────────────────────────────────────────────────
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


# ── Helpers ──────────────────────────────────────────────────────────────────
def pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def flag_url(team: str, size: int = 20) -> str:
    code = ISO2_MAP.get(team, "")
    return f"https://flagcdn.com/w{size}/{code}.png" if code else ""


def flag_img(team: str, height: int = 22) -> str:
    url = flag_url(team, size=40)
    if not url:
        return ""
    return (
        f'<img src="{url}" '
        f'style="height:{height}px; border-radius:3px; '
        f'box-shadow:0 1px 4px rgba(0,0,0,0.5); display:block; margin:0 auto 6px;">'
    )


def stat_card(label: str, team: str, value: str, accent: str) -> str:
    flag = flag_img(team, height=24)
    return f"""
    <div style="
        background: linear-gradient(145deg, {accent}1a 0%, {accent}08 100%);
        border: 1px solid {accent}40;
        border-radius: 14px;
        padding: 18px 14px;
        text-align: center;
        min-height: 120px;
        display: flex; flex-direction: column; justify-content: center; gap: 4px;
    ">
        <span style="font-size:10px; color:{accent}cc; text-transform:uppercase;
                     letter-spacing:.12em; font-weight:700;">{label}</span>
        {flag}
        <span style="font-size:17px; font-weight:800; color:#eef2ff; line-height:1.2;">{team}</span>
        <span style="font-size:22px; font-weight:800; color:{accent}; line-height:1;">{value}</span>
    </div>
    """


def prob_bar(team_a: str, team_b: str, p_a: float, p_e: float, p_b: float) -> str:
    fa = flag_img(team_a, height=26)
    fb = flag_img(team_b, height=26)
    return f"""
    <div style="display:flex; gap:10px; margin:18px 0 24px;">
        <div style="flex:{p_a:.3f}; background:linear-gradient(135deg,#1b5e20,#2e7d32);
                    border-radius:12px; padding:22px 12px; text-align:center; min-width:90px;">
            {fa}
            <div style="font-size:10px; color:#a5d6a7; text-transform:uppercase;
                        letter-spacing:.1em; font-weight:700; margin-bottom:6px;">{team_a}</div>
            <div style="font-size:32px; font-weight:800; color:#fff;">{p_a*100:.1f}%</div>
        </div>
        <div style="flex:{p_e:.3f}; background:linear-gradient(135deg,#1e2130,#2a2f45);
                    border-radius:12px; padding:22px 12px; text-align:center; min-width:80px;
                    border:1px solid rgba(255,255,255,0.08);">
            <div style="font-size:10px; color:#8d99ae; text-transform:uppercase;
                        letter-spacing:.1em; font-weight:700; margin-bottom:6px;">Empate</div>
            <div style="font-size:32px; font-weight:800; color:#cdd6f4;">{p_e*100:.1f}%</div>
        </div>
        <div style="flex:{p_b:.3f}; background:linear-gradient(135deg,#0d47a1,#1565c0);
                    border-radius:12px; padding:22px 12px; text-align:center; min-width:90px;">
            {fb}
            <div style="font-size:10px; color:#90caf9; text-transform:uppercase;
                        letter-spacing:.1em; font-weight:700; margin-bottom:6px;">{team_b}</div>
            <div style="font-size:32px; font-weight:800; color:#fff;">{p_b*100:.1f}%</div>
        </div>
    </div>
    """


def add_flag_col(df: pd.DataFrame, equipo_col: str = "Equipo") -> pd.DataFrame:
    """Inserta columna de URLs de banderas al inicio del DataFrame."""
    df = df.copy()
    df.insert(0, " ", df[equipo_col].map(lambda e: flag_url(e, size=20)))
    return df


FLAG_COL_CFG = st.column_config.ImageColumn(" ", width="small")


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🏆 Mundial 2026")
st.sidebar.caption("Análisis predictivo basado en 49.000+ partidos históricos")
page = st.sidebar.radio(
    "Navegación",
    ["Dashboard", "Simulador de partidos", "Grupos", "EDA Highlights"],
)
st.sidebar.divider()
st.sidebar.link_button(
    "Ver código en GitHub",
    "https://github.com/BrunoGazzo96/mundial2026-predictor",
    use_container_width=True,
)
st.sidebar.divider()
st.sidebar.markdown("**Stack técnico**")
st.sidebar.markdown("""
<div style="display:flex; flex-wrap:wrap; gap:5px; margin-top:4px;">
  <span style="background:#1565C0;color:#fff;padding:3px 9px;border-radius:20px;font-size:11px;font-weight:600;">Python</span>
  <span style="background:#2E7D32;color:#fff;padding:3px 9px;border-radius:20px;font-size:11px;font-weight:600;">scikit-learn</span>
  <span style="background:#6A1B9A;color:#fff;padding:3px 9px;border-radius:20px;font-size:11px;font-weight:600;">Streamlit</span>
  <span style="background:#B71C1C;color:#fff;padding:3px 9px;border-radius:20px;font-size:11px;font-weight:600;">pandas</span>
  <span style="background:#E65100;color:#fff;padding:3px 9px;border-radius:20px;font-size:11px;font-weight:600;">Plotly</span>
  <span style="background:#37474F;color:#fff;padding:3px 9px;border-radius:20px;font-size:11px;font-weight:600;">NumPy</span>
</div>
""", unsafe_allow_html=True)
st.sidebar.divider()
st.sidebar.markdown("**Rendimiento del modelo**")
st.sidebar.markdown("""
<div style="font-size:12px; color:#8d99ae; line-height:2;">
  Accuracy (3 clases)&nbsp;&nbsp;<b style="color:#cdd6f4;">59.4%</b><br>
  Log-loss&nbsp;&nbsp;<b style="color:#cdd6f4;">0.91</b><br>
  Partidos de entrenamiento&nbsp;&nbsp;<b style="color:#cdd6f4;">10.636</b><br>
  Partidos históricos analizados&nbsp;&nbsp;<b style="color:#cdd6f4;">49.287</b>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.markdown("""
<div style="text-align:center; padding:6px 0 2px; font-size:12px; color:#8d99ae; line-height:1.9;">
  Hecho por<br>
  <b style="color:#cdd6f4; font-size:13px;">Bruno Gazzo</b><br>
  <a href="https://www.linkedin.com/in/bruno-gazzo-909161195"
     style="color:#1565C0; text-decoration:none;">LinkedIn</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/BrunoGazzo96"
     style="color:#1565C0; text-decoration:none;">GitHub</a>
</div>
""", unsafe_allow_html=True)

predictions = load_predictions()
teams_df    = load_teams()
probs_dict  = load_probs_dict()


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.title("Mundial 2026 — Predictor")

    # Snapshot banner
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0f1f3d,#1a2e52);
                border:1px solid #2d4a7a; border-left:4px solid #1565C0;
                border-radius:8px; padding:14px 18px; margin:6px 0 18px;">
      <span style="color:#90caf9;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;">
        Snapshot pre-torneo
      </span>
      <p style="color:#cdd6f4;margin:6px 0 0;font-size:13px;line-height:1.6;">
        Predicciones calculadas el <b>16 de mayo de 2026</b>, antes del inicio del Mundial
        (11 jun 2026). El modelo hizo su apuesta antes de que arranque el primer partido —
        los resultados actualizarán una vez terminada la fase de grupos.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Métricas del proyecto
    cm1, cm2, cm3, cm4 = st.columns(4)
    cm1.metric("Partidos analizados", "49.287", "1872 – 2026")
    cm2.metric("Simulaciones MC", "10.000", "27 segundos de cómputo")
    cm3.metric("Accuracy del modelo", "59.4 %", "GradientBoosting 3 clases")
    cm4.metric("Features", "10", "ELO · forma · ranking FIFA")

    st.divider()

    if predictions is None:
        st.error("No se encontró `outputs/predicciones_2026.csv`. Corré el notebook 03.")
        st.stop()

    pred = predictions.merge(teams_df[["equipo", "confederacion", "grupo"]], on="equipo")

    # Cards top 4
    ACCENTS = ["#FFD700", "#9E9E9E", "#CD7F32", "#1565C0"]
    LABELS  = ["1er favorito", "2do favorito", "3er favorito", "4to favorito"]
    cols_cards = st.columns(4)
    for i, (col, (_, row)) in enumerate(zip(cols_cards, pred.head(4).iterrows())):
        with col:
            st.markdown(
                stat_card(LABELS[i], row["equipo"], pct(row["campeon"]), ACCENTS[i]),
                unsafe_allow_html=True,
            )

    st.divider()
    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        top15 = pred.head(15).copy()
        top15["color"]     = top15["confederacion"].map(CONF_COLORS).fillna("#607D8B")
        top15["pct_label"] = top15["campeon"].map(pct)

        fig_bar = go.Figure(go.Bar(
            x=top15["campeon"] * 100,
            y=top15["equipo"],
            orientation="h",
            marker_color=top15["color"],
            marker_line_width=0,
            text=top15["pct_label"],
            textposition="outside",
            textfont=dict(size=11, color="#cdd6f4"),
        ))
        for conf, color in CONF_COLORS.items():
            if conf in top15["confederacion"].values:
                fig_bar.add_trace(go.Scatter(
                    x=[None], y=[None], name=conf,
                    mode="markers",
                    marker=dict(color=color, size=10, symbol="square"),
                    showlegend=True,
                ))
        fig_bar.update_layout(
            title=dict(text="Top 15 candidatos al título", font=dict(size=15, color="#eef2ff")),
            xaxis=dict(
                title="Probabilidad de campeonato (%)",
                gridcolor="rgba(255,255,255,0.06)",
                zerolinecolor="rgba(255,255,255,0.1)",
                color="#8d99ae",
            ),
            yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)", color="#cdd6f4"),
            height=490,
            margin=dict(l=10, r=80, t=44, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8d99ae", size=11),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=10), bgcolor="rgba(0,0,0,0)",
            ),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
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
            height=390,
            margin=dict(l=0, r=0, t=40, b=0),
            coloraxis=dict(colorbar=dict(
                title=dict(text="Prob.", font=dict(color="#8d99ae")),
                tickformat=".0%",
                bgcolor="rgba(0,0,0,0)",
                tickcolor="#8d99ae",
            )),
            geo=dict(
                showframe=False,
                showcoastlines=True,
                bgcolor="rgba(0,0,0,0)",
                landcolor="#1e2540",
                oceancolor="#0d1117",
                showocean=True,
                coastlinecolor="rgba(255,255,255,0.12)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8d99ae"),
            title=dict(
                text="Probabilidad de título por país",
                font=dict(color="#eef2ff", size=14),
            ),
        )
        st.plotly_chart(fig_map, use_container_width=True)

    st.divider()
    st.subheader("Tabla completa — 48 selecciones")

    rondas_cols = ["clasifica", "16avos", "octavos", "cuartos", "semis", "final", "campeon"]
    display = pred[["equipo", "confederacion", "grupo"] + rondas_cols].copy()
    for c in rondas_cols:
        display[c] = (display[c] * 100).round(1)
    display = display.rename(columns={
        **RONDAS_ES, "equipo": "Equipo", "confederacion": "Conf.", "grupo": "Gr.",
    })
    display = add_flag_col(display)

    col_cfg = {
        " ":          FLAG_COL_CFG,
        "Clasifica":  st.column_config.ProgressColumn("Clasifica",  format="%.1f%%", min_value=0, max_value=100),
        "16avos":     st.column_config.ProgressColumn("16avos",     format="%.1f%%", min_value=0, max_value=100),
        "Octavos":    st.column_config.ProgressColumn("Octavos",    format="%.1f%%", min_value=0, max_value=100),
        "Cuartos":    st.column_config.ProgressColumn("Cuartos",    format="%.1f%%", min_value=0, max_value=100),
        "Semis":      st.column_config.ProgressColumn("Semis",      format="%.1f%%", min_value=0, max_value=100),
        "Final":      st.column_config.ProgressColumn("Final",      format="%.1f%%", min_value=0, max_value=100),
        "Campeón":    st.column_config.ProgressColumn("Campeón",    format="%.1f%%", min_value=0, max_value=15),
    }
    st.dataframe(
        display.sort_values("Campeón", ascending=False),
        use_container_width=True,
        hide_index=True,
        height=500,
        column_config=col_cfg,
    )

    st.divider()
    with st.expander("¿Cómo funciona el modelo?"):
        st.markdown("""
**Pipeline completo: datos históricos → predicción deployada**

**1. Datos**
49.287 partidos internacionales (1872–2026) más el ranking FIFA histórico.
Filtrado a partidos de alta competencia (Mundiales, torneos continentales, amistosos relevantes) para reducir ruido.

**2. Feature engineering**
- **ELO Rating** calculado cronológicamente sobre toda la historia — captura jerarquía real mejor que el ranking FIFA porque pondera la dificultad del rival.
- **Forma reciente** — promedio de goles (ventana 10) y win rate (ventana 30). Siempre con `shift(1)` para evitar data leakage: el modelo nunca ve el resultado del partido que predice.
- **ELO diff** — diferencia entre ELO local y visitante como feature sintético.

**3. Modelo**
GradientBoosting (`n_estimators=300, max_depth=4, lr=0.05`) vs. Logistic Regression como baseline.
Clasificación 3 clases: Victoria / Empate / Derrota del equipo local.
Accuracy en test set: **59.4%** | Log-loss: **0.91**.
Para fútbol, ~60% en 3 clases es el techo real — el deporte tiene ruido estructural que ningún modelo elimina.

**4. Optimización de la simulación Monte Carlo**
El cuello de botella inicial era llamar a `predict_proba()` partido por partido dentro del loop (≈950.000 llamadas al modelo para 10.000 torneos). Solución: pre-calcular los 2.256 enfrentamientos posibles en **un solo batch call** y guardarlos en un dict. La simulación solo hace `np.random.choice`. Resultado: de 10 minutos a **27 segundos**.

**5. Formato WC2026**
48 equipos · 12 grupos de 4 · Clasifican top-2 de cada grupo + mejores 8 terceros (32 al bracket) · 16avos → Octavos → Cuartos → Semis → Final → Campeón.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2: SIMULADOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Simulador de partidos":
    st.title("Simulador de partidos")
    st.markdown("Elegí dos selecciones y el modelo estima las probabilidades del enfrentamiento.")

    all_teams = sorted(teams_df["equipo"].tolist())

    col1, col_vs, col2 = st.columns([5, 1, 5])
    with col1:
        team_a = st.selectbox("Selección local", all_teams, index=all_teams.index("Argentina"))
    with col_vs:
        st.markdown(
            "<div style='text-align:center; padding-top:28px; color:#8d99ae; "
            "font-weight:700; font-size:14px;'>VS</div>",
            unsafe_allow_html=True,
        )
    with col2:
        team_b = st.selectbox("Selección visitante", all_teams, index=all_teams.index("Francia"))

    if team_a == team_b:
        st.warning("Elegí dos selecciones diferentes.")
        st.stop()

    if probs_dict is None:
        st.error("Modelo no disponible. Corré el notebook 03 para generar `model_gbm.joblib`.")
        st.stop()

    probs = probs_dict.get((team_a, team_b))
    if probs is None:
        st.error(f"No hay datos pre-calculados para {team_a} vs {team_b}.")
        st.stop()

    # sklearn: clase 0=B gana, 1=empate, 2=A gana
    p_b, p_e, p_a = float(probs[0]), float(probs[1]), float(probs[2])

    st.markdown(prob_bar(team_a, team_b, p_a, p_e, p_b), unsafe_allow_html=True)

    # Info de los equipos
    def team_info(name):
        row = teams_df[teams_df["equipo"] == name]
        return row.iloc[0].to_dict() if not row.empty else {}

    info_a, info_b = team_info(team_a), team_info(team_b)
    if info_a and info_b:
        pred_row_a = predictions[predictions["equipo"] == team_a].iloc[0] if predictions is not None else None
        pred_row_b = predictions[predictions["equipo"] == team_b].iloc[0] if predictions is not None else None

        ca, cb = st.columns(2)
        with ca:
            st.markdown(f"**{team_a}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Confederación", info_a["confederacion"])
            c2.metric("Ranking FIFA",  f"#{info_a['ranking_fifa']}")
            c3.metric("Prob. título",  pct(pred_row_a["campeon"]) if pred_row_a is not None else "—")
        with cb:
            st.markdown(f"**{team_b}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Confederación", info_b["confederacion"])
            c2.metric("Ranking FIFA",  f"#{info_b['ranking_fifa']}")
            c3.metric("Prob. título",  pct(pred_row_b["campeon"]) if pred_row_b is not None else "—")

    st.divider()
    st.subheader("Simulación de múltiples partidos")
    n_sim = st.slider("Cantidad de partidos", 100, 5000, 1000, step=100)

    if st.button("Simular", type="primary"):
        resultados_sim = np.random.choice(
            [f"Gana {team_a}", "Empate", f"Gana {team_b}"],
            size=n_sim,
            p=[p_a, p_e, p_b],
        )
        conteo = pd.Series(resultados_sim).value_counts().reset_index()
        conteo.columns = ["resultado", "count"]
        conteo["pct"] = conteo["count"] / n_sim

        color_map = {
            f"Gana {team_a}": "#2E7D32",
            "Empate":         "#546E7A",
            f"Gana {team_b}": "#1565C0",
        }
        fig_sim = px.bar(
            conteo, x="resultado", y="count",
            text=conteo["pct"].map(pct),
            title=f"{n_sim:,} simulaciones: {team_a} vs {team_b}",
            color="resultado",
            color_discrete_map=color_map,
        )
        fig_sim.update_traces(textposition="outside", marker_line_width=0, textfont=dict(size=13))
        fig_sim.update_layout(
            showlegend=False,
            yaxis=dict(title="Partidos"),
            xaxis=dict(title=""),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8d99ae"),
            title=dict(font=dict(color="#eef2ff", size=14)),
        )
        st.plotly_chart(fig_sim, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3: GRUPOS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Grupos":
    st.title("Grupos del Mundial 2026")
    st.markdown(
        "Probabilidad de clasificar a **16avos de final** "
        "(top-2 de cada grupo + mejores 8 terceros)."
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
                    marker_color=[CONF_COLORS.get(c, "#607D8B") for c in g_df["confederacion"]],
                    marker_line_width=0,
                    text=g_df["clasifica"].map(pct),
                    textposition="outside",
                    textfont=dict(size=10, color="#cdd6f4"),
                ))
                fig_g.add_vline(x=50, line_dash="dash", line_color="#ef5350", opacity=0.5)
                fig_g.update_layout(
                    height=165,
                    margin=dict(l=5, r=55, t=5, b=5),
                    xaxis=dict(range=[0, 120], title="", gridcolor="rgba(255,255,255,0.05)", color="#8d99ae"),
                    yaxis=dict(title="", color="#cdd6f4"),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                )
                st.plotly_chart(fig_g, use_container_width=True, key=f"grupo_{grupo}")

                mini = g_df[["equipo", "clasifica", "campeon"]].copy()
                mini["clasifica"] = (mini["clasifica"] * 100).round(1)
                mini["campeon"]   = (mini["campeon"]   * 100).round(1)
                mini.columns      = ["Equipo", "Clasifica %", "Título %"]
                mini = add_flag_col(mini)

                st.dataframe(
                    mini, hide_index=True, use_container_width=True,
                    column_config={
                        " ": FLAG_COL_CFG,
                        "Clasifica %": st.column_config.ProgressColumn(
                            "Clasifica %", format="%.1f%%", min_value=0, max_value=100,
                        ),
                        "Título %": st.column_config.ProgressColumn(
                            "Título %", format="%.1f%%", min_value=0, max_value=15,
                        ),
                    },
                )

        if row_idx + 4 < len(grupos_list):
            st.divider()

    st.divider()
    st.subheader("Dificultad por grupo")
    st.caption("Ranking FIFA promedio de los 4 equipos — menor número = grupo más difícil")

    dif = (
        teams_df.groupby("grupo")
        .agg(ranking_medio=("ranking_fifa", "mean"), equipos=("equipo", lambda x: " · ".join(x)))
        .reset_index()
        .sort_values("ranking_medio")
    )
    fig_dif = px.bar(
        dif, x="grupo", y="ranking_medio",
        title="Dificultad por grupo (ranking FIFA promedio)",
        labels={"ranking_medio": "Ranking FIFA promedio", "grupo": "Grupo"},
        text="ranking_medio",
        color="ranking_medio",
        color_continuous_scale="RdYlGn_r",
        hover_data={"equipos": True, "ranking_medio": ":.1f"},
    )
    fig_dif.update_traces(
        texttemplate="%{text:.1f}", textposition="outside",
        marker_line_width=0, textfont=dict(color="#cdd6f4"),
    )
    fig_dif.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis=dict(showscale=False),
        font=dict(color="#8d99ae"),
        title=dict(font=dict(color="#eef2ff", size=14)),
        xaxis=dict(gridcolor="rgba(0,0,0,0)", color="#8d99ae"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", color="#8d99ae"),
    )
    st.plotly_chart(fig_dif, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4: EDA HIGHLIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "EDA Highlights":
    st.title("EDA Highlights")
    st.markdown(
        "Los hallazgos más relevantes del análisis exploratorio sobre "
        "**49.000+ partidos** de fútbol internacional (1872–2026)."
    )

    GRAFICOS = ROOT / "outputs" / "graficos"

    charts = [
        ("goles_por_anio.png",          "Goles promedio por partido en Mundiales (1930–2022)"),
        ("campeones_historicos.png",     "Campeones mundiales 1930–2022"),
        ("confederaciones_por_anio.png", "Equipos por confederación en cada Mundial"),
        ("resultados_por_contexto.png",  "Factor local: cancha propia vs neutral vs Mundial"),
        ("resultados_por_era.png",       "Resultados en Mundiales por era"),
        ("top_victorias_mundiales.png",  "Top 15 selecciones por victorias en Mundiales"),
        ("elo_wc2026.png",              "ELO Rating — 48 equipos clasificados al WC2026"),
        ("features_analisis.png",        "Correlación de features con el resultado"),
        ("feature_importance.png",       "Importancia de features — GradientBoosting"),
        ("predicciones_campeon.png",     "Probabilidades de título — resultado final"),
        ("heatmap_probabilidades.png",   "Heatmap de probabilidades por ronda — Top 20"),
        ("clasificacion_grupos.png",     "Probabilidad de clasificar a 16avos por grupo"),
    ]

    for i in range(0, len(charts), 2):
        cols = st.columns(2)
        for j, (fname, title) in enumerate(charts[i:i + 2]):
            path = GRAFICOS / fname
            with cols[j]:
                if path.exists():
                    st.markdown(f"**{title}**")
                    st.image(str(path), use_container_width=True)
                else:
                    st.info(f"Gráfico pendiente: `{fname}`")

    html_path = GRAFICOS / "ranking_vs_winrate.html"
    if html_path.exists():
        st.divider()
        st.markdown("**Ranking FIFA vs Win Rate histórico en Mundiales (interactivo)**")
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=550, scrolling=False)

import pandas as pd
import numpy as np


def compute_elo(df: pd.DataFrame, k: int = 32, base: int = 1500) -> dict:
    """
    Calcula ELO de cada selección a lo largo del tiempo.
    Retorna dict {team: elo_final} y agrega columnas elo_home/elo_away al df.
    """
    elo = {}
    df = df.sort_values("date").reset_index(drop=True)
    elo_home_list, elo_away_list = [], []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        elo.setdefault(h, base)
        elo.setdefault(a, base)

        elo_h, elo_a = elo[h], elo[a]
        expected_h = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))

        if row["home_score"] > row["away_score"]:
            score_h = 1
        elif row["home_score"] == row["away_score"]:
            score_h = 0.5
        else:
            score_h = 0

        elo[h] = elo_h + k * (score_h - expected_h)
        elo[a] = elo_a + k * ((1 - score_h) - (1 - expected_h))

        elo_home_list.append(elo_h)
        elo_away_list.append(elo_a)

    df["elo_home"] = elo_home_list
    df["elo_away"] = elo_away_list
    return df, elo


def compute_rolling_stats(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Agrega win_rate y promedios de goles por equipo en ventana móvil."""
    df = df.sort_values("date").copy()
    # Implementación en el notebook 02 — placeholder
    return df


def build_match_features(df: pd.DataFrame, wc2026_teams: pd.DataFrame) -> pd.DataFrame:
    """Construye el dataset de features para entrenar el modelo."""
    df = df.copy()
    df["result"] = np.where(
        df["home_score"] > df["away_score"], 2,
        np.where(df["home_score"] == df["away_score"], 1, 0)
    )
    return df

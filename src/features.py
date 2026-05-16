import pandas as pd
import numpy as np


def compute_elo(df: pd.DataFrame, k: int = 32, base: int = 1500) -> tuple[pd.DataFrame, dict]:
    """
    Calcula ELO de cada equipo procesando los partidos en orden cronológico.
    Registra el ELO *previo* al partido en las columnas elo_home / elo_away.
    Retorna (df_con_elo, dict_elo_final).
    """
    elo: dict[str, float] = {}
    df = df.sort_values("date").reset_index(drop=True)
    elo_home_list, elo_away_list = [], []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        elo.setdefault(h, base)
        elo.setdefault(a, base)

        elo_h, elo_a = elo[h], elo[a]
        expected_h = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))

        if row["home_score"] > row["away_score"]:
            score_h = 1.0
        elif row["home_score"] == row["away_score"]:
            score_h = 0.5
        else:
            score_h = 0.0

        elo[h] = elo_h + k * (score_h - expected_h)
        elo[a] = elo_a + k * ((1 - score_h) - (1 - expected_h))

        elo_home_list.append(elo_h)
        elo_away_list.append(elo_a)

    df = df.copy()
    df["elo_home"] = elo_home_list
    df["elo_away"] = elo_away_list
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    return df, elo


def compute_rolling_stats(
    df: pd.DataFrame, window: int = 10, long_window: int = 30
) -> pd.DataFrame:
    """
    Para cada partido calcula, usando solo información PREVIA (shift 1):
      - gf_avg_home/away  : promedio goles convertidos últimos `window` partidos
      - ga_avg_home/away  : promedio goles recibidos últimos `window` partidos
      - win_rate_home/away: % victorias últimos `long_window` partidos

    Estrategia: transforma el dataset a vista por-equipo (cada partido
    aparece 2 veces), computa las rolling en ese formato y vuelve a
    pivotar al formato original via merge.
    """
    df = df.sort_values("date").reset_index(drop=True)

    home_view = df[["date", "home_team", "home_score", "away_score"]].copy()
    home_view.columns = ["date", "team", "gf", "ga"]

    away_view = df[["date", "away_team", "away_score", "home_score"]].copy()
    away_view.columns = ["date", "team", "gf", "ga"]

    th = pd.concat([home_view, away_view], ignore_index=True)
    th = th.sort_values(["team", "date"]).reset_index(drop=True)
    th["win"] = (th["gf"] > th["ga"]).astype(int)

    g = th.groupby("team", sort=False)
    th["gf_avg"] = g["gf"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=3).mean()
    )
    th["ga_avg"] = g["ga"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=3).mean()
    )
    th["win_rate"] = g["win"].transform(
        lambda x: x.shift(1).rolling(long_window, min_periods=5).mean()
    )

    # Si un equipo juega dos partidos el mismo día, tomamos el primer registro
    # (el shift ya excluye ese partido del cómputo)
    th = th.drop_duplicates(subset=["team", "date"], keep="first")

    home_stats = th[["date", "team", "gf_avg", "ga_avg", "win_rate"]].rename(
        columns={
            "team": "home_team",
            "gf_avg": "gf_avg_home",
            "ga_avg": "ga_avg_home",
            "win_rate": "win_rate_home",
        }
    )
    away_stats = th[["date", "team", "gf_avg", "ga_avg", "win_rate"]].rename(
        columns={
            "team": "away_team",
            "gf_avg": "gf_avg_away",
            "ga_avg": "ga_avg_away",
            "win_rate": "win_rate_away",
        }
    )

    df = df.merge(home_stats, on=["date", "home_team"], how="left")
    df = df.merge(away_stats, on=["date", "away_team"], how="left")
    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columna 'result': 2=victoria home, 1=empate, 0=derrota home."""
    df = df.copy()
    df["result"] = np.where(
        df["home_score"] > df["away_score"], 2,
        np.where(df["home_score"] == df["away_score"], 1, 0),
    )
    return df


FEATURE_COLS = [
    "elo_home", "elo_away", "elo_diff",
    "gf_avg_home", "ga_avg_home", "win_rate_home",
    "gf_avg_away", "ga_avg_away", "win_rate_away",
    "neutral",
]


def build_wc2026_features(
    teams_df: pd.DataFrame,
    elo_final: dict,
    team_stats_latest: pd.DataFrame,
    nombre_map: dict,
) -> pd.DataFrame:
    """
    Genera features para todos los posibles enfrentamientos entre los
    48 equipos del WC2026. Usado por el simulador.

    team_stats_latest: DataFrame con columnas [team, gf_avg, ga_avg, win_rate]
                       con los stats más recientes de cada equipo.
    nombre_map: dict {nombre_es -> nombre_en} para cruzar con el dataset histórico.
    """
    teams = teams_df["equipo"].tolist()
    rows = []
    for i, ta in enumerate(teams):
        for tb in teams[i + 1:]:
            ta_en = nombre_map.get(ta, ta)
            tb_en = nombre_map.get(tb, tb)

            elo_a = elo_final.get(ta_en, 1500)
            elo_b = elo_final.get(tb_en, 1500)

            def get_stat(team_en, col):
                mask = team_stats_latest["team"] == team_en
                return team_stats_latest.loc[mask, col].values[0] if mask.any() else np.nan

            rows.append({
                "team_a": ta, "team_b": tb,
                "team_a_en": ta_en, "team_b_en": tb_en,
                "elo_home": elo_a, "elo_away": elo_b, "elo_diff": elo_a - elo_b,
                "gf_avg_home": get_stat(ta_en, "gf_avg"),
                "ga_avg_home": get_stat(ta_en, "ga_avg"),
                "win_rate_home": get_stat(ta_en, "win_rate"),
                "gf_avg_away": get_stat(tb_en, "gf_avg"),
                "ga_avg_away": get_stat(tb_en, "ga_avg"),
                "win_rate_away": get_stat(tb_en, "win_rate"),
                "neutral": 1,
            })
            # También el partido inverso (B como "home")
            rows.append({
                "team_a": tb, "team_b": ta,
                "team_a_en": tb_en, "team_b_en": ta_en,
                "elo_home": elo_b, "elo_away": elo_a, "elo_diff": elo_b - elo_a,
                "gf_avg_home": get_stat(tb_en, "gf_avg"),
                "ga_avg_home": get_stat(tb_en, "ga_avg"),
                "win_rate_home": get_stat(tb_en, "win_rate"),
                "gf_avg_away": get_stat(ta_en, "gf_avg"),
                "ga_avg_away": get_stat(ta_en, "ga_avg"),
                "win_rate_away": get_stat(ta_en, "win_rate"),
                "neutral": 1,
            })

    return pd.DataFrame(rows)

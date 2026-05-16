import pandas as pd
from pathlib import Path

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"


def load_historical_results() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / "resultados_historicos.csv", parse_dates=["date"])
    return df


def load_fifa_ranking() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / "ranking_fifa_historico.csv", parse_dates=["rank_date"])


def load_wc2026_teams() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / "mundial2026_equipos.csv")


def filter_world_cups(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["tournament"] == "FIFA World Cup"].copy()


def filter_high_stakes(df: pd.DataFrame) -> pd.DataFrame:
    tournaments = [
        "FIFA World Cup",
        "UEFA Euro",
        "Copa América",
        "AFC Asian Cup",
        "Africa Cup of Nations",
        "FIFA World Cup qualification",
        "Confederations Cup",
    ]
    return df[df["tournament"].isin(tournaments)].copy()

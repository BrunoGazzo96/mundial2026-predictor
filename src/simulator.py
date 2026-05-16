import numpy as np
import pandas as pd
from src.features import FEATURE_COLS

# Distribuciones de marcadores por tipo de resultado
_WIN_SCORES = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2)]
_WIN_PROBS  = [0.22,   0.23,   0.26,   0.09,   0.09,   0.04,  0.03,  0.02,  0.02]
_DRAW_SCORES = [(0, 0), (1, 1), (2, 2), (3, 3)]
_DRAW_PROBS  = [0.22,   0.55,   0.20,   0.03]


def _sample_score(result: str) -> tuple[int, int]:
    if result == "A":
        idx = np.random.choice(len(_WIN_SCORES), p=_WIN_PROBS)
        return _WIN_SCORES[idx]
    elif result == "E":
        idx = np.random.choice(len(_DRAW_SCORES), p=_DRAW_PROBS)
        return _DRAW_SCORES[idx]
    else:
        idx = np.random.choice(len(_WIN_SCORES), p=_WIN_PROBS)
        gf, ga = _WIN_SCORES[idx]
        return ga, gf


def build_features_dict(features_2026: pd.DataFrame) -> dict:
    """Dict {(team_a, team_b): array_de_features} para build_probs_dict."""
    return {
        (row["team_a"], row["team_b"]): row[FEATURE_COLS].values.astype(float)
        for _, row in features_2026.iterrows()
    }


def precompute_match_probs(model, features_2026: pd.DataFrame) -> dict:
    """
    Pre-calcula probabilidades de todos los enfrentamientos en un solo
    batch call. Hace la simulación 1000x más rápida.

    Retorna dict {(team_a, team_b): array([p_B_gana, p_empate, p_A_gana])}.
    """
    X = features_2026[FEATURE_COLS]
    all_probs = model.predict_proba(X)  # una sola llamada batch

    return {
        (row["team_a"], row["team_b"]): all_probs[i]
        for i, (_, row) in enumerate(features_2026.iterrows())
    }


def simular_partido(
    team_a: str,
    team_b: str,
    probs_dict: dict,
    knockout: bool = False,
) -> str:
    """
    Samplea el resultado de un partido desde probabilidades pre-calculadas.
    Retorna 'A' (gana team_a), 'B' (gana team_b) o 'E' (empate, solo grupos).
    En knockout=True la prob de empate se divide 50/50 (va a penales).
    """
    probs = probs_dict.get((team_a, team_b))
    if probs is None:
        if knockout:
            return "A" if np.random.random() < 0.5 else "B"
        return np.random.choice(["A", "E", "B"], p=[0.42, 0.24, 0.34])

    # sklearn ordena clases: 0=B gana, 1=empate, 2=A gana
    p_b, p_e, p_a = probs[0], probs[1], probs[2]

    if knockout:
        p_a_adj = p_a + p_e * 0.5
        return "A" if np.random.random() < p_a_adj else "B"
    else:
        return np.random.choice(["A", "E", "B"], p=[p_a, p_e, p_b])


def simular_grupo(
    equipos: list[str],
    probs_dict: dict,
) -> tuple[list, dict]:
    """Round-robin de 6 partidos. Retorna (tabla_ordenada, stats)."""
    puntos = {e: 0 for e in equipos}
    gf     = {e: 0 for e in equipos}
    gc     = {e: 0 for e in equipos}

    for i, ea in enumerate(equipos):
        for eb in equipos[i + 1:]:
            resultado = simular_partido(ea, eb, probs_dict)
            score_a, score_b = _sample_score(resultado)
            gf[ea] += score_a;  gc[ea] += score_b
            gf[eb] += score_b;  gc[eb] += score_a

            if resultado == "A":
                puntos[ea] += 3
            elif resultado == "E":
                puntos[ea] += 1; puntos[eb] += 1
            else:
                puntos[eb] += 3

    tabla = sorted(
        equipos,
        key=lambda e: (puntos[e], gf[e] - gc[e], gf[e]),
        reverse=True,
    )
    stats = {e: {"pts": puntos[e], "gf": gf[e], "gc": gc[e], "gd": gf[e] - gc[e]}
             for e in equipos}
    return tabla, stats


def simular_fase_grupos(
    grupos: dict[str, list[str]],
    probs_dict: dict,
) -> tuple[dict, list]:
    """
    Simula los 12 grupos.
    Retorna (tabla_grupos, lista_de_stats_de_terceros).
    """
    tabla_grupos: dict[str, list] = {}
    terceros: list[dict] = []

    for grupo, equipos in grupos.items():
        tabla, stats = simular_grupo(equipos, probs_dict)
        tabla_grupos[grupo] = tabla
        tercero = tabla[2]
        terceros.append({"equipo": tercero, "grupo": grupo, **stats[tercero]})

    return tabla_grupos, terceros


def _seleccionar_mejores_terceros(terceros: list[dict], n: int = 8) -> list[str]:
    """Selecciona los n mejores 3eros por puntos → GD → GF."""
    return [
        t["equipo"]
        for t in sorted(terceros, key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)[:n]
    ]


def _crear_bracket_r32(
    tabla_grupos: dict[str, list],
    terceros_clasificados: list[str],
) -> list[str]:
    """
    Bracket de 32 equipos (16avos de final).

    Pares de grupos (A-B, C-D, E-F, G-H, I-J, K-L):
      Slot 1: 1ro_X vs 2do_Y
      Slot 2: 1ro_Y vs 2do_X
    → Los 8 mejores 3eros ocupan 4 cruces adicionales entre sí.
    """
    bracket: list[str] = []
    for gx, gy in [("A","B"), ("C","D"), ("E","F"), ("G","H"), ("I","J"), ("K","L")]:
        bracket.append(tabla_grupos[gx][0])  # 1ro X
        bracket.append(tabla_grupos[gy][1])  # 2do Y
        bracket.append(tabla_grupos[gy][0])  # 1ro Y
        bracket.append(tabla_grupos[gx][1])  # 2do X

    bracket.extend(terceros_clasificados)    # 8 mejores 3eros
    return bracket                           # 24 + 8 = 32


def simular_torneo(
    grupos: dict[str, list[str]],
    probs_dict: dict,
    n: int = 10_000,
) -> pd.DataFrame:
    """
    Simula el torneo WC2026 N veces (Monte Carlo).

    Fases: Grupos → 16avos → Octavos → Cuartos → Semis → Final → Campeón.
    Clasifican al bracket: top-2 de cada grupo + 8 mejores 3eros (32 equipos).

    Retorna DataFrame con probabilidades por equipo y ronda.
    """
    equipos = [e for g in grupos.values() for e in g]
    rondas  = ["clasifica", "16avos", "octavos", "cuartos", "semis", "final", "campeon"]
    conteo: dict[str, dict[str, int]] = {e: {r: 0 for r in rondas} for e in equipos}

    for _ in range(n):
        tabla_grupos, terceros = simular_fase_grupos(grupos, probs_dict)

        # Clasificados a 16avos (top-2 de cada grupo + mejores 8 terceros)
        for tabla in tabla_grupos.values():
            conteo[tabla[0]]["clasifica"] += 1
            conteo[tabla[1]]["clasifica"] += 1

        mejores_terceros = _seleccionar_mejores_terceros(terceros, n=8)
        for eq in mejores_terceros:
            conteo[eq]["clasifica"] += 1

        # Bracket 16avos de final (32 equipos)
        bracket = _crear_bracket_r32(tabla_grupos, mejores_terceros)
        for eq in bracket:
            conteo[eq]["16avos"] += 1  # = clasifica (idéntico, útil para verificar)

        # Eliminatorias: 16avos → Octavos → Cuartos → Semis → Final → Campeón
        ronda_actual = bracket[:]
        for nombre_ronda in ["octavos", "cuartos", "semis", "final", "campeon"]:
            siguiente: list[str] = []
            for i in range(0, len(ronda_actual), 2):
                flag = simular_partido(ronda_actual[i], ronda_actual[i + 1],
                                       probs_dict, knockout=True)
                siguiente.append(ronda_actual[i] if flag == "A" else ronda_actual[i + 1])
            ronda_actual = siguiente
            for eq in ronda_actual:
                conteo[eq][nombre_ronda] += 1

    rows = [{"equipo": e, **{r: conteo[e][r] / n for r in rondas}} for e in equipos]
    return pd.DataFrame(rows).sort_values("campeon", ascending=False).reset_index(drop=True)

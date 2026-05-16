import numpy as np
import pandas as pd


def simular_partido(equipo_a: str, equipo_b: str, model, features_dict: dict) -> str:
    """
    Retorna 'A', 'B' o 'empate' sampleando las probabilidades del modelo.
    features_dict debe tener las features precalculadas por par de equipos.
    """
    from src.features import build_match_features  # import lazy para evitar ciclos
    feats = features_dict.get((equipo_a, equipo_b))
    if feats is None:
        raise ValueError(f"No hay features para {equipo_a} vs {equipo_b}")
    probs = model.predict_proba([feats])[0]  # [derrota, empate, victoria]
    return np.random.choice(["B", "empate", "A"], p=probs)


def simular_fase_grupos(grupos: dict, model, features_dict: dict) -> list:
    """Simula la fase de grupos y retorna los 32 clasificados."""
    clasificados = []
    for grupo, equipos in grupos.items():
        puntos = {e: 0 for e in equipos}
        dif_goles = {e: 0 for e in equipos}

        for i, ea in enumerate(equipos):
            for eb in equipos[i + 1:]:
                resultado = simular_partido(ea, eb, model, features_dict)
                if resultado == "A":
                    puntos[ea] += 3
                elif resultado == "empate":
                    puntos[ea] += 1
                    puntos[eb] += 1
                else:
                    puntos[eb] += 3

        tabla = sorted(equipos, key=lambda e: puntos[e], reverse=True)
        clasificados.extend(tabla[:2])  # top 2 por grupo
    return clasificados


def simular_eliminatorias(clasificados: list, model, features_dict: dict) -> str:
    """Simula eliminación directa a partir de los 32 clasificados."""
    ronda = clasificados[:]
    while len(ronda) > 1:
        siguiente = []
        for i in range(0, len(ronda), 2):
            ea, eb = ronda[i], ronda[i + 1]
            resultado = simular_partido(ea, eb, model, features_dict)
            ganador = ea if resultado == "A" else eb
            siguiente.append(ganador)
        ronda = siguiente
    return ronda[0]


def simular_torneo(grupos: dict, model, features_dict: dict, n: int = 10_000) -> pd.DataFrame:
    """
    Simula el torneo N veces.
    Retorna DataFrame con probabilidades por equipo y por ronda.
    """
    equipos = [e for g in grupos.values() for e in g]
    conteo = {e: {"grupos": 0, "16vos": 0, "cuartos": 0,
                  "semis": 0, "final": 0, "campeon": 0} for e in equipos}

    for _ in range(n):
        clasificados = simular_fase_grupos(grupos, model, features_dict)
        for e in clasificados:
            conteo[e]["16vos"] += 1
        campeon = simular_eliminatorias(clasificados, model, features_dict)
        conteo[campeon]["campeon"] += 1

    rows = []
    for equipo, rondas in conteo.items():
        row = {"equipo": equipo}
        row.update({k: v / n for k, v in rondas.items()})
        rows.append(row)

    return pd.DataFrame(rows).sort_values("campeon", ascending=False).reset_index(drop=True)

# Mundial 2026 Predictor

Análisis predictivo del Mundial 2026 usando datos históricos de 45.000+ partidos internacionales.

## ¿Qué incluye?

- Análisis exploratorio de mundiales 1930–2022
- Modelo GBM entrenado sobre resultados históricos de fútbol internacional
- Simulación del torneo completo (10.000 iteraciones Monte Carlo)
- App interactiva deployada: _link próximamente_

## Stack

Python · pandas · scikit-learn · Streamlit · Plotly

## Estructura

```
data/          # Datasets raw y procesados
notebooks/     # EDA, feature engineering, modelo
src/           # Módulos Python (loader, features, model, simulator)
app/           # App Streamlit
outputs/       # Gráficos y predicciones exportadas
```

## Cómo correrlo localmente

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Datos

- **International football results (1872–2024):** [Kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
- **Ranking FIFA histórico:** FIFA.com
- **Equipos y grupos 2026:** elaboración propia

## Autor

Bruno Gazzo — [LinkedIn](https://www.linkedin.com/in/bruno-gazzo-909161195)

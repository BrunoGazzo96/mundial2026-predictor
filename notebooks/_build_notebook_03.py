"""Genera notebooks/03_modelo_simulacion.ipynb con JSON válido."""
import nbformat

cells = []
def md(s): return nbformat.v4.new_markdown_cell(s)
def code(s): return nbformat.v4.new_code_cell(s)

cells.append(md(
    "# Parte 3 — Modelo + Simulación Monte Carlo\n\n"
    "1. Logistic Regression (baseline) vs GradientBoosting\n"
    "2. Importancia de features\n"
    "3. Simulación Monte Carlo — 10.000 torneos\n"
    "4. Exporta `outputs/predicciones_2026.csv`\n\n"
    "**Fases WC2026:** Grupos → 16avos de final → Octavos → Cuartos → Semis → Final"
))

cells.append(code(
    'import sys\n'
    'sys.path.append("..")\n'
    'import pandas as pd\n'
    'import numpy as np\n'
    'import matplotlib.pyplot as plt\n'
    'import matplotlib.ticker as mtick\n'
    'import seaborn as sns\n'
    'from pathlib import Path\n'
    'from src.features import FEATURE_COLS\n'
    'from src.model import train, feature_importance, save_model\n'
    'from src.simulator import precompute_match_probs, simular_torneo\n'
    'from src.data_loader import load_wc2026_teams\n'
    'sns.set_theme(style="whitegrid")\n'
    'plt.rcParams["figure.dpi"] = 120\n'
    'OUTPUTS = Path("../outputs")\n'
    'GRAFICOS = OUTPUTS / "graficos"\n'
))

cells.append(md("## 1. Dataset de entrenamiento"))
cells.append(code(
    'df = pd.read_csv("../data/processed/features_train.csv")\n'
    'X, y = df[FEATURE_COLS], df["result"]\n'
    'print(f"Partidos: {len(df):,}  |  Features: {len(FEATURE_COLS)}")\n'
    'print(y.value_counts().rename({0:"Derrota", 1:"Empate", 2:"Victoria"}))\n'
))

cells.append(md("## 2. Baseline — Logistic Regression"))
cells.append(code('lr_model, X_test_lr, y_test_lr, scaler_lr = train(X, y, model_type="lr")\n'))

cells.append(md("## 3. GradientBoosting"))
cells.append(code(
    'gbm_model, X_test_gbm, y_test_gbm, _ = train(X, y, model_type="gbm")\n'
    'save_model(gbm_model, name="model_gbm.joblib")\n'
))

cells.append(md("## 4. Importancia de features (GBM)"))
cells.append(code(
    'imp_df = feature_importance(gbm_model, FEATURE_COLS)\n'
    'fig, ax = plt.subplots(figsize=(8, 5))\n'
    'colors = ["#1a73e8" if i < 3 else "#90CAF9" for i in range(len(imp_df))]\n'
    'ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color=colors[::-1], edgecolor="white")\n'
    'ax.set_title("Importancia de features — GradientBoosting", fontsize=13, fontweight="bold")\n'
    'ax.set_xlabel("Importancia (Gini)")\n'
    'ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))\n'
    'plt.tight_layout()\n'
    'plt.savefig(GRAFICOS / "feature_importance.png", dpi=150, bbox_inches="tight")\n'
    'plt.show()\n'
    'print(imp_df.to_string(index=False))\n'
))

cells.append(md("## 5. Matrices de confusión — LR vs GBM"))
cells.append(code(
    'from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix\n'
    'fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n'
    'labels = ["Derrota", "Empate", "Victoria"]\n'
    'X_lr_sc = scaler_lr.transform(X_test_lr) if scaler_lr else X_test_lr\n'
    'for ax, model, Xt, yt, name in [\n'
    '    (axes[0], lr_model, X_lr_sc, y_test_lr, "LR (baseline)"),\n'
    '    (axes[1], gbm_model, X_test_gbm, y_test_gbm, "GradientBoosting"),\n'
    ']:\n'
    '    cm = confusion_matrix(yt, model.predict(Xt))\n'
    '    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, colorbar=False, cmap="Blues")\n'
    '    ax.set_title(f"Confusión — {name}", fontweight="bold")\n'
    'plt.tight_layout()\n'
    'plt.savefig(GRAFICOS / "confusion_matrix.png", dpi=150, bbox_inches="tight")\n'
    'plt.show()\n'
))

cells.append(md(
    "## 6. Simulación Monte Carlo — 10.000 torneos\n\n"
    "**Optimización:** Pre-calculamos las probabilidades de los 2.256 enfrentamientos "
    "posibles en un solo batch call. La simulación solo hace `np.random.choice` — sin "
    "llamadas al modelo durante el loop."
))
cells.append(code(
    'features_2026 = pd.read_csv("../data/processed/features_2026.csv")\n'
    'print("Pre-calculando probabilidades...")\n'
    'probs_dict = precompute_match_probs(gbm_model, features_2026)\n'
    'print(f"  {len(probs_dict)} pares pre-calculados")\n\n'
    'equipos_df = load_wc2026_teams()\n'
    'grupos = {g: equipos_df[equipos_df["grupo"]==g]["equipo"].tolist()\n'
    '          for g in sorted(equipos_df["grupo"].unique())}\n'
    'print("\\nGrupos WC2026:")\n'
    'for g, eq in grupos.items():\n'
    '    print(f"  Grupo {g}: {eq}")\n'
))

cells.append(code(
    'N_SIMS = 10_000\n'
    'print(f"Simulando {N_SIMS:,} torneos...")\n'
    'resultados = simular_torneo(grupos, probs_dict, n=N_SIMS)\n'
    'resultados.to_csv(OUTPUTS / "predicciones_2026.csv", index=False)\n'
    'print("Guardado: outputs/predicciones_2026.csv")\n'
))

cells.append(md("## 7. Tabla completa de resultados"))
cells.append(code(
    'cols = ["equipo","clasifica","16avos","octavos","cuartos","semis","final","campeon"]\n'
    'display_df = resultados[cols].copy()\n'
    'for c in cols[1:]:\n'
    '    display_df[c] = display_df[c].map(lambda x: f"{x*100:.1f}%")\n'
    'print("TOP 20:")\n'
    'print(display_df.head(20).to_string(index=False))\n'
))

cells.append(md("## 8. Candidatos al título — Top 15"))
cells.append(code(
    'conf_colors = {"UEFA":"#1565C0","CONMEBOL":"#2E7D32","CONCACAF":"#F9A825",\n'
    '               "CAF":"#B71C1C","AFC":"#6A1B9A","OFC":"#00838F"}\n'
    'equipo_conf = dict(zip(equipos_df["equipo"], equipos_df["confederacion"]))\n'
    'top15 = resultados.head(15)\n'
    'colors_b = [conf_colors.get(equipo_conf.get(e,""),"#607D8B") for e in top15["equipo"]]\n'
    'fig, ax = plt.subplots(figsize=(11, 7))\n'
    'bars = ax.barh(top15["equipo"][::-1], top15["campeon"][::-1]*100,\n'
    '               color=colors_b[::-1], edgecolor="white", height=0.7)\n'
    'for bar, val in zip(bars, top15["campeon"][::-1]*100):\n'
    '    ax.text(val+0.2, bar.get_y()+bar.get_height()/2,\n'
    '            f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")\n'
    'ax.set_title(f"Probabilidad de ganar el Mundial 2026\\n({N_SIMS:,} simulaciones)",\n'
    '             fontsize=14, fontweight="bold")\n'
    'ax.set_xlabel("Probabilidad de campeonato (%)")\n'
    'ax.margins(x=0.12)\n'
    'from matplotlib.patches import Patch\n'
    'ax.legend(handles=[Patch(facecolor=c, label=k) for k,c in conf_colors.items()],\n'
    '          loc="lower right", title="Confederación", fontsize=8)\n'
    'plt.tight_layout()\n'
    'plt.savefig(GRAFICOS / "predicciones_campeon.png", dpi=150, bbox_inches="tight")\n'
    'plt.show()\n'
))

cells.append(md("## 9. Heatmap — Probabilidades por ronda (Top 20)"))
cells.append(code(
    'rondas = ["clasifica","16avos","octavos","cuartos","semis","final","campeon"]\n'
    'labels_r = ["Clasifica","16avos","Octavos","Cuartos","Semis","Final","Campeón"]\n'
    'heat = resultados.head(20).set_index("equipo")[rondas] * 100\n'
    'fig, ax = plt.subplots(figsize=(12, 8))\n'
    'sns.heatmap(heat, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5, ax=ax,\n'
    '            cbar_kws={"label":"Probabilidad (%)"}, xticklabels=labels_r)\n'
    'ax.set_title("Probabilidades por ronda — Top 20", fontsize=13, fontweight="bold")\n'
    'ax.set_ylabel("")\n'
    'ax.tick_params(axis="x", rotation=0)\n'
    'plt.tight_layout()\n'
    'plt.savefig(GRAFICOS / "heatmap_probabilidades.png", dpi=150, bbox_inches="tight")\n'
    'plt.show()\n'
))

cells.append(md("## 10. Probabilidad de clasificar a 16avos — por grupo"))
cells.append(code(
    'res_g = resultados.merge(equipos_df[["equipo","grupo","confederacion"]], on="equipo")\n'
    'fig, axes = plt.subplots(3, 4, figsize=(16, 10))\n'
    'axes = axes.flatten()\n'
    'for i, grupo in enumerate(sorted(grupos.keys())):\n'
    '    g_df = res_g[res_g["grupo"]==grupo].sort_values("clasifica", ascending=False)\n'
    '    ax = axes[i]\n'
    '    cg = [conf_colors.get(c,"#607D8B") for c in g_df["confederacion"]]\n'
    '    bars2 = ax.bar(g_df["equipo"], g_df["clasifica"]*100, color=cg, edgecolor="white")\n'
    '    ax.axhline(50, color="red", linestyle="--", alpha=0.5, lw=1)\n'
    '    ax.set_title(f"Grupo {grupo}", fontweight="bold", fontsize=10)\n'
    '    ax.set_ylim(0, 110)\n'
    '    ax.yaxis.set_major_formatter(mtick.PercentFormatter())\n'
    '    ax.tick_params(axis="x", rotation=25, labelsize=7)\n'
    '    for bar, val in zip(bars2, g_df["clasifica"]*100):\n'
    '        ax.text(bar.get_x()+bar.get_width()/2, val+1,\n'
    '                f"{val:.0f}%", ha="center", fontsize=7)\n'
    'plt.suptitle("Prob. de clasificar a 16avos de final",\n'
    '             fontsize=14, fontweight="bold", y=1.01)\n'
    'plt.tight_layout()\n'
    'plt.savefig(GRAFICOS / "clasificacion_grupos.png", dpi=150, bbox_inches="tight")\n'
    'plt.show()\n'
))

cells.append(md("## 11. Resumen ejecutivo"))
cells.append(code(
    'print("="*55)\n'
    'print("   PREDICCIÓN MUNDIAL 2026")\n'
    'print("="*55)\n'
    'print(f"   Simulaciones: {N_SIMS:,} | Modelo: GradientBoosting")\n'
    'print()\n'
    'print("   TOP 10 CANDIDATOS AL TÍTULO:")\n'
    'for _, row in resultados.head(10).iterrows():\n'
    '    conf = equipo_conf.get(row["equipo"],"")\n'
    '    print(f\'   {row["equipo"]:<22} {row["campeon"]*100:5.1f}%  [{conf}]\')\n'
    'print()\n'
    'res_conf = resultados.merge(equipos_df[["equipo","confederacion"]], on="equipo")\n'
    'conf_ch = res_conf.groupby("confederacion")["campeon"].sum().sort_values(ascending=False)\n'
    'print("   PROB. DE TÍTULO POR CONFEDERACIÓN:")\n'
    'for conf, prob in conf_ch.items():\n'
    '    print(f"   {conf:<12} {prob*100:5.1f}%")\n'
))

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.0"}
}
with open("../notebooks/03_modelo_simulacion.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print("Notebook generado OK")

"""Graphiques du projet (palette validée, marques fines, étiquettes directes)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from config import RESULTS_DIR, SENSORS, HORIZON
from simulation import NOISE

BLUE, ORANGE, AQUA, VIOLET, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#8a8985"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MODEL_STYLE = {"Gradient Boosting": (BLUE, "-"), "Random Forest": (ORANGE, "-"),
               "Régression logistique": (AQUA, "-"), "Référence 1 variable": (GREY, "--")}
SENSOR_COLORS = dict(zip(SENSORS, [ORANGE, BLUE, AQUA, VIOLET, "#eda100"]))

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
})


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / name, dpi=160)
    plt.close(fig)


def fig_modes(clean_df, feat_df):
    """Un exemple de machine par mode de panne : capteurs normalisés vs état sain."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, mode in zip(axes, ["roulement", "surchauffe", "fuite"]):
        mid = feat_df[feat_df["mode_panne"] == mode]["machine_id"].iloc[0]
        d = feat_df[feat_df["machine_id"] == mid]
        for s in SENSORS:
            ax.plot(d["cycle"], d[f"{s}_vs_sain"] / NOISE[s], color=SENSOR_COLORS[s],
                    label=s, linewidth=1.6)
        ax.axhline(0, color=GREY, linewidth=1)
        ax.set_title(f"Panne « {mode} » (machine {mid})")
        ax.set_xlabel("Cycle")
    axes[0].set_ylabel("Écart à l'état sain (en écarts-types de bruit)")
    axes[0].legend(ncol=2, fontsize=8, loc="upper left")
    _save(fig, "fig1_modes_de_panne.png")


def fig_cleaning(raw, cleaned, sensor="vibration"):
    """Avant / après nettoyage autour du plus gros pic corrigé."""
    r = raw.sort_values(["machine_id", "cycle"]).reset_index(drop=True)
    diff = (r[sensor] - cleaned[sensor]).abs()
    i = int(diff.idxmax())
    mid = r.loc[i, "machine_id"]
    lo, hi = r.loc[i, "cycle"] - 40, r.loc[i, "cycle"] + 40
    a = r[(r.machine_id == mid) & r.cycle.between(lo, hi)]
    b = cleaned[(cleaned.machine_id == mid) & cleaned.cycle.between(lo, hi)]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(a["cycle"], a[sensor], color=ORANGE, marker="o", markersize=3, linewidth=1.2,
            label="Brut (pic + valeurs manquantes)")
    ax.plot(b["cycle"], b[sensor], color=BLUE, linewidth=2, label="Nettoyé")
    ax.set_title(f"Nettoyage du capteur « {sensor} » (machine {mid})")
    ax.set_xlabel("Cycle"); ax.set_ylabel(sensor); ax.legend()
    _save(fig, "fig2_nettoyage.png")


def fig_roc_pr(y_test, probas):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, p in probas.items():
        c, ls = MODEL_STYLE[name]
        fpr, tpr, _ = roc_curve(y_test, p)
        pr, rc, _ = precision_recall_curve(y_test, p)
        a1.plot(fpr, tpr, color=c, linestyle=ls, label=name)
        a2.plot(rc, pr, color=c, linestyle=ls, label=name)
    a1.plot([0, 1], [0, 1], color=GREY, linewidth=0.8, linestyle=":")
    a1.set_title("Courbes ROC (test final)"); a1.set_xlabel("Faux positifs (taux)")
    a1.set_ylabel("Vrais positifs (taux)"); a1.legend(loc="lower right")
    a2.set_title("Courbes précision-rappel (test final)"); a2.set_xlabel("Rappel")
    a2.set_ylabel("Précision")
    _save(fig, "fig3_roc_pr.png")


def fig_importance(imp):
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(imp.index, imp.values, color=BLUE, height=0.55)
    for y, v in enumerate(imp.values):
        ax.text(v + imp.max() * 0.01, y, f"{v:.3f}", va="center", color=INK2, fontsize=9)
    ax.set_title("Importance par capteur (baisse de PR-AUC si on mélange le capteur)")
    ax.set_xlabel("Baisse de PR-AUC"); ax.grid(axis="y", visible=False)
    _save(fig, "fig4_importance_capteurs.png")


def fig_threshold(curve, thr):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(curve["seuil"], curve["cout"], color=BLUE)
    ax.scatter([thr], [curve.loc[curve.seuil == thr, "cout"].iloc[0]], color=ORANGE,
               zorder=5, s=50)
    ax.annotate(f"seuil choisi = {thr:.2f}", (thr, curve.loc[curve.seuil == thr, "cout"].iloc[0]),
                textcoords="offset points", xytext=(-14, 40), ha="right", color=INK,
                arrowprops=dict(arrowstyle="-", color=GREY, linewidth=0.8))
    ax.set_title("Coût moyen par machine selon le seuil d'alerte")
    ax.set_xlabel("Seuil de probabilité"); ax.set_ylabel("Coût moyen (unités)")
    _save(fig, "fig5_seuil_cout.png")


def fig_strategies(costs):
    names, vals = list(costs), list(costs.values())
    cols = [GREY, GREY, BLUE]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.barh(names[::-1], vals[::-1], color=cols[::-1], height=0.5)
    for y, v in enumerate(vals[::-1]):
        ax.text(v + max(vals) * 0.01, y, f"{v:.1f}", va="center", color=INK, fontsize=10)
    ax.set_title("Coût moyen par machine selon la stratégie (test final)")
    ax.set_xlabel("Coût moyen (unités)"); ax.grid(axis="y", visible=False)
    _save(fig, "fig6_strategies_cout.png")


def fig_rul(df_test, pred, mids):
    fig, axes = plt.subplots(1, len(mids), figsize=(5.5 * len(mids), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, mid in zip(axes, mids):
        m = (df_test["machine_id"] == mid).values
        d = df_test[m]
        ax.plot(d["cycle"], d["RUL_clip"], color=GREY, label="RUL réel (plafonné)")
        ax.plot(d["cycle"], pred[m], color=BLUE, label="RUL prédit")
        ax.set_title(f"Machine {mid} — panne « {d['mode_panne'].iloc[0]} »")
        ax.set_xlabel("Cycle")
    axes[0].set_ylabel("RUL (cycles)"); axes[0].legend()
    _save(fig, "fig7_rul.png")


def fig_lead(outcomes):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ok = outcomes[outcomes["statut"] == "alerte utile"]["delai_alerte"]
    ax.hist(ok, bins=range(0, int(max(ok.max(), HORIZON * 2)) + 5, 4), color=BLUE,
            rwidth=0.92)
    n_miss = int((outcomes["statut"] == "panne ratée").sum())
    ax.set_title(f"Délai d'alerte avant la panne ({len(ok)} alertes utiles, {n_miss} panne(s) ratée(s))")
    ax.set_xlabel("Cycles restants au moment de l'alerte"); ax.set_ylabel("Nombre de machines")
    ax.grid(axis="x", visible=False)
    _save(fig, "fig8_delai_alerte.png")

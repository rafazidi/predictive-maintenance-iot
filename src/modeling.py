"""Modèles, validation croisée par machine, décision de maintenance (coût)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (SEED, N_FOLDS, ALERT_PERSIST, MIN_LEAD, C_FAIL, C_PLANNED,
                    C_WASTE, PREVENTIVE_AT)


# ---------------------------------------------------------------------------
# Classification : "panne dans les 24 prochains cycles ?"
# ---------------------------------------------------------------------------
def make_classifiers():
    return {
        "Référence 1 variable": Pipeline([
            ("sel", SelectKBest(f_classif, k=1)), ("sc", StandardScaler()),
            ("m", LogisticRegression(class_weight="balanced"))]),
        "Régression logistique": Pipeline([
            ("sc", StandardScaler()),
            ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, min_samples_leaf=5, class_weight="balanced",
            n_jobs=-1, random_state=SEED),
        "Gradient Boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=SEED),
    }


def classif_metrics(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {"Précision": precision_score(y, pred, zero_division=0),
            "Rappel": recall_score(y, pred), "F1": f1_score(y, pred),
            "AUC": roc_auc_score(y, proba), "PR-AUC": average_precision_score(y, proba)}


def cross_validate_models(X, y, groups):
    """GroupKFold : une machine n'est jamais à la fois en train et en validation.
    Retourne (tableau mean/std par modèle, prédictions out-of-fold)."""
    rows, oof = [], {}
    for name, proto in make_classifiers().items():
        from sklearn.base import clone
        proba = np.zeros(len(y))
        folds = []
        for tr, va in GroupKFold(N_FOLDS).split(X, y, groups):
            m = clone(proto).fit(X.iloc[tr], y.iloc[tr])
            proba[va] = m.predict_proba(X.iloc[va])[:, 1]
            folds.append(classif_metrics(y.iloc[va], proba[va]))
        f = pd.DataFrame(folds)
        rows.append({"Modèle": name, **{f"{c}": f[c].mean() for c in f},
                     **{f"{c}_std": f[c].std() for c in f}})
        oof[name] = proba
    return pd.DataFrame(rows), oof


# ---------------------------------------------------------------------------
# Décision de maintenance : alertes, délai d'alerte, coût
# ---------------------------------------------------------------------------
def first_alert_cycle(proba, cycles, thr, k=ALERT_PERSIST):
    """Premier cycle où proba >= thr pendant k cycles consécutifs (sinon None)."""
    ok = pd.Series(proba >= thr).rolling(k).sum().values == k
    idx = np.flatnonzero(ok)
    return None if len(idx) == 0 else int(cycles[idx[0]])


def machine_outcomes(df, proba, thr):
    """Un résultat par machine : alerte à quel cycle, délai avant panne, coût."""
    out = []
    for mid, grp in df.assign(p=proba).groupby("machine_id"):
        cycles = grp["cycle"].values
        life = int(cycles.max())
        c = first_alert_cycle(grp["p"].values, cycles, thr)
        lead = None if c is None else life - c
        if lead is not None and lead >= MIN_LEAD:
            cost = C_PLANNED + C_WASTE * lead
            status = "alerte utile"
        else:
            cost, status = C_FAIL, "panne ratée"
        out.append({"machine_id": mid, "vie": life, "cycle_alerte": c,
                    "delai_alerte": lead, "statut": status, "cout": cost})
    return pd.DataFrame(out)


def strategy_costs(df, proba, thr):
    """Compare 3 stratégies : réactive, préventive fixe, prédictive (modèle)."""
    lives = df.groupby("machine_id")["cycle"].max()
    reactive = np.full(len(lives), C_FAIL)
    preventive = C_PLANNED + C_WASTE * (lives.values - PREVENTIVE_AT)
    pred = machine_outcomes(df, proba, thr)
    return {"Réactive (attendre la panne)": reactive.mean(),
            f"Préventive (remplacer à {PREVENTIVE_AT} cycles)": preventive.mean(),
            "Prédictive (modèle ML)": pred["cout"].mean()}, pred


def tune_threshold(df, proba, grid=None):
    """Seuil qui minimise le coût moyen par machine (sur prédictions out-of-fold)."""
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2) if grid is None else grid
    costs = [machine_outcomes(df, proba, t)["cout"].mean() for t in grid]
    return float(grid[int(np.argmin(costs))]), pd.DataFrame({"seuil": grid, "cout": costs})


# ---------------------------------------------------------------------------
# Régression : RUL (durée de vie résiduelle)
# ---------------------------------------------------------------------------
def nasa_score(pred, true):
    """Score asymétrique (C-MAPSS) : prédire trop tard coûte plus cher que trop tôt."""
    d = np.asarray(pred) - np.asarray(true)
    return np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)


def fit_rul_models(Xtr, ytr):
    reg = RandomForestRegressor(n_estimators=150, min_samples_leaf=3, n_jobs=-1,
                                random_state=SEED).fit(Xtr, ytr)
    base = DummyRegressor(strategy="mean").fit(Xtr, ytr)
    return reg, base


def rul_metrics(model, X, y_clip, y_true, zone=50):
    pred = model.predict(X)
    crit = (y_true.values <= zone)           # zone critique : 50 derniers cycles
    d = pred - y_clip.values
    return {"MAE (tous)": mean_absolute_error(y_clip, pred),
            "RMSE (tous)": mean_squared_error(y_clip, pred) ** 0.5,
            f"MAE (RUL<={zone})": mean_absolute_error(y_clip[crit], pred[crit]),
            f"Biais (RUL<={zone})": float(d[crit].mean()),
            f"Score NASA moyen (RUL<={zone})": float(nasa_score(pred[crit], y_clip.values[crit]).mean())}, pred

"""Pipeline complet : simulation -> nettoyage -> features -> modèles -> décision.

Lancer :  python src/main.py        (environ 1 à 2 minutes)
"""
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit

import plots
from config import (DATA_DIR, RESULTS_DIR, SEED, SENSORS, TEST_SIZE, HORIZON)
from modeling import (classif_metrics, cross_validate_models, fit_rul_models,
                      make_classifiers, rul_metrics, strategy_costs, tune_threshold)
from preprocessing import add_features, clean, feature_columns, sensor_of
from simulation import simulate


def grouped_permutation_importance(model, X, y, n_repeats=5):
    """Mélange TOUTES les features d'un capteur ensemble (elles sont corrélées),
    mesure la baisse de PR-AUC."""
    rng = np.random.default_rng(SEED)
    base = average_precision_score(y, model.predict_proba(X)[:, 1])
    res = {}
    for s in SENSORS:
        cols = [c for c in X.columns if sensor_of(c) == s]
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[cols] = X[cols].values[rng.permutation(len(X))]
            drops.append(base - average_precision_score(y, model.predict_proba(Xp)[:, 1]))
        res[s] = float(np.mean(drops))
    return pd.Series(res)


def main():
    RESULTS_DIR.mkdir(exist_ok=True); DATA_DIR.mkdir(exist_ok=True)
    out = open(RESULTS_DIR / "resultats.txt", "w", encoding="utf-8")

    def log(*a):
        print(*a); print(*a, file=out)

    # 1. Simulation -------------------------------------------------------
    raw = simulate()
    n_nan = int(raw[SENSORS].isna().sum().sum())
    log(f"[1] Dataset brut : {raw.shape}, {raw.machine_id.nunique()} machines, "
        f"{n_nan} valeurs manquantes")
    log("    Modes de panne : " + raw.groupby("machine_id")["mode_panne"].first()
        .value_counts().to_dict().__repr__())
    raw.round(4).to_csv(DATA_DIR / "dataset_brut.csv", index=False)

    # 2. Pré-traitement ---------------------------------------------------
    cleaned = clean(raw)
    data = add_features(cleaned)
    feats = feature_columns(data)
    data.round(4).to_csv(DATA_DIR / "dataset_pretraite.csv", index=False)
    log(f"[2] Après pré-traitement : {len(feats)} features, "
        f"{data['panne_imminente'].mean():.1%} d'échantillons « panne imminente » (RUL <= {HORIZON})")

    # 3. Découpage PAR MACHINE : dev (80 %) / test final (20 %) -----------
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    dev_i, test_i = next(gss.split(data, groups=data["machine_id"]))
    dev, test = data.iloc[dev_i].reset_index(drop=True), data.iloc[test_i].reset_index(drop=True)
    Xd, yd, gd = dev[feats], dev["panne_imminente"], dev["machine_id"]
    Xt, yt = test[feats], test["panne_imminente"]
    log(f"[3] Dev : {dev.machine_id.nunique()} machines | Test final : {test.machine_id.nunique()} machines")

    # 4. Validation croisée par machine (sur dev) -------------------------
    cv, oof = cross_validate_models(Xd, yd, gd)
    cv.to_csv(RESULTS_DIR / "comparaison_modeles_cv.csv", index=False)
    log("\n[4] VALIDATION CROISÉE (5 plis, par machine) — moyenne ± écart-type")
    for _, r in cv.iterrows():
        log(f"    {r['Modèle']:<24} P={r['Précision']:.3f} R={r['Rappel']:.3f} "
            f"F1={r['F1']:.3f}±{r['F1_std']:.3f} AUC={r['AUC']:.3f} "
            f"PR-AUC={r['PR-AUC']:.3f}±{r['PR-AUC_std']:.3f}")
    best = cv.loc[cv["PR-AUC"].idxmax(), "Modèle"]
    log(f"    Meilleur modèle (PR-AUC) : {best}")

    # 5. Seuil d'alerte optimisé sur le COÛT (prédictions out-of-fold) ----
    thr, curve = tune_threshold(dev, oof[best])
    log(f"\n[5] Seuil d'alerte optimal (coût minimal, dev) : {thr:.2f}")

    # 6. Test final : modèles entraînés sur dev, évalués sur 30 machines --
    log("\n[6] TEST FINAL (machines jamais vues)")
    models, probas, rows = {}, {}, []
    for name, proto in make_classifiers().items():
        m = clone(proto).fit(Xd, yd)
        models[name] = m
        probas[name] = m.predict_proba(Xt)[:, 1]
        r = {"Modèle": name, **classif_metrics(yt, probas[name], 0.5)}
        rows.append(r)
        log(f"    {name:<24} P={r['Précision']:.3f} R={r['Rappel']:.3f} F1={r['F1']:.3f} "
            f"AUC={r['AUC']:.3f} PR-AUC={r['PR-AUC']:.3f}")
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "comparaison_modeles_test.csv", index=False)

    # 7. Décision de maintenance : stratégies comparées -------------------
    costs, outcomes = strategy_costs(test, probas[best], thr)
    outcomes.to_csv(RESULTS_DIR / "alertes_par_machine.csv", index=False)
    log("\n[7] COÛT MOYEN PAR MACHINE (test final)")
    for k, v in costs.items():
        log(f"    {k:<46} {v:6.1f}")
    react = list(costs.values())[0]
    log(f"    Économie vs réactive : {1 - costs['Prédictive (modèle ML)'] / react:.0%}")
    ok = outcomes[outcomes["statut"] == "alerte utile"]
    log(f"    Alertes utiles : {len(ok)}/{len(outcomes)} | pannes ratées : "
        f"{len(outcomes) - len(ok)} | délai d'alerte médian : {ok['delai_alerte'].median():.0f} cycles")

    # 8. Interprétation : importance par capteur --------------------------
    imp = grouped_permutation_importance(models[best], Xt, yt)
    log("\n[8] Importance par capteur (baisse de PR-AUC) :\n" + imp.sort_values(ascending=False)
        .round(4).to_string())

    # 9. Régression RUL ---------------------------------------------------
    reg, base = fit_rul_models(Xd, dev["RUL_clip"])
    m_reg, pred = rul_metrics(reg, Xt, test["RUL_clip"], test["RUL"])
    m_base, _ = rul_metrics(base, Xt, test["RUL_clip"], test["RUL"])
    log("\n[9] RÉGRESSION RUL (test final)")
    for k in m_reg:
        log(f"    {k:<32} Random Forest = {m_reg[k]:8.2f} | référence (moyenne) = {m_base[k]:8.2f}")

    # 10. Figures ---------------------------------------------------------
    plots.fig_modes(cleaned, data)
    plots.fig_cleaning(raw, cleaned)
    plots.fig_roc_pr(yt, probas)
    plots.fig_importance(imp)
    plots.fig_threshold(curve, thr)
    plots.fig_strategies(costs)
    mids = [test[test.mode_panne == m]["machine_id"].iloc[0] for m in ["roulement", "fuite"]]
    plots.fig_rul(test, pred, mids)
    plots.fig_lead(outcomes)

    summary = {"cv": cv.round(4).to_dict("records"), "test": pd.DataFrame(rows).round(4).to_dict("records"),
               "best_model": best, "threshold": thr, "costs": {k: round(v, 2) for k, v in costs.items()},
               "rul": {k: round(v, 3) for k, v in m_reg.items()},
               "rul_baseline": {k: round(v, 3) for k, v in m_base.items()},
               "importance": imp.round(4).to_dict(),
               "alerts": {"useful": int(len(ok)), "total": int(len(outcomes)),
                          "median_lead": float(ok["delai_alerte"].median())}}
    (RESULTS_DIR / "resume.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    out.close()


if __name__ == "__main__":
    main()

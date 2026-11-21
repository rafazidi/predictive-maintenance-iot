"""Nettoyage et construction des features.

Principe clé : tout est CAUSAL. La valeur calculée au cycle t n'utilise que les
cycles <= t, comme dans un système temps réel. Le test tests/test_pipeline.py
le vérifie automatiquement.
"""
import numpy as np
import pandas as pd

from config import (SENSORS, SPIKE_WINDOW, SPIKE_K, FEAT_WINDOW, DIFF_LAG,
                    BASELINE_CYCLES, HORIZON, RUL_CLIP)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Détecte les pics (médiane glissante causale + MAD), remplace par NaN,
    puis comble les trous par la dernière valeur connue (forward fill)."""
    data = raw.sort_values(["machine_id", "cycle"]).reset_index(drop=True).copy()
    g = data.groupby("machine_id")
    for s in SENSORS:
        med = g[s].transform(lambda x: x.rolling(SPIKE_WINDOW, min_periods=1).median())
        res = data[s] - med
        # Echelle du bruit : constante de calibration du capteur (MAD robuste)
        mad = 1.4826 * np.nanmedian(np.abs(res - np.nanmedian(res)))
        data.loc[res.abs() > SPIKE_K * mad, s] = np.nan
        # ffill = causal. Le bfill ne sert que pour un NaN au tout premier cycle.
        data[s] = data.groupby("machine_id")[s].transform(lambda x: x.ffill().bfill())
    return data


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """6 features par capteur : valeur, moyenne, écart-type, variation, écart à
    l'état sain, tendance. Plus les deux cibles."""
    data = data.copy()
    g = data.groupby("machine_id")
    w = FEAT_WINDOW
    for s in SENSORS:
        data[f"{s}_mean{w}"] = g[s].transform(lambda x: x.rolling(w, min_periods=1).mean())
        data[f"{s}_std{w}"] = g[s].transform(lambda x: x.rolling(w, min_periods=1).std()).fillna(0)
        data[f"{s}_diff{DIFF_LAG}"] = g[s].transform(lambda x: x.diff(DIFF_LAG)).fillna(0)
        # Écart à l'état sain : élimine l'offset propre à chaque machine
        base = g[s].transform(lambda x: x.expanding(min_periods=1).mean().where(
            np.arange(len(x)) < BASELINE_CYCLES).ffill())
        data[f"{s}_vs_sain"] = data[f"{s}_mean{w}"] - base
        data[f"{s}_trend{w}"] = g[f"{s}_mean{w}"].transform(lambda x: x.diff(w)).fillna(0)
    data["panne_imminente"] = (data["RUL"] <= HORIZON).astype(int)   # classification
    data["RUL_clip"] = data["RUL"].clip(upper=RUL_CLIP)               # régression
    return data


def feature_columns(data: pd.DataFrame) -> list:
    return [c for c in data.columns if any(c.startswith(s) for s in SENSORS)]


def sensor_of(feature: str) -> str:
    return next(s for s in SENSORS if feature.startswith(s))

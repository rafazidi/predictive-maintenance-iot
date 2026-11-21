"""Tests automatiques : lancer avec  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import SENSORS, HORIZON  # noqa: E402
from modeling import first_alert_cycle, nasa_score  # noqa: E402
from preprocessing import add_features, clean, feature_columns  # noqa: E402
from simulation import simulate  # noqa: E402

RAW = simulate(n_machines=12, seed=1)


def test_simulation_is_reproducible():
    a = simulate(n_machines=5, seed=7)
    b = simulate(n_machines=5, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_rul_reaches_zero_at_failure():
    assert (RAW.groupby("machine_id")["RUL"].min() == 0).all()


def test_features_are_causal():
    """Changer les 20 derniers cycles d'une machine ne doit PAS changer le passé."""
    f1 = add_features(clean(RAW))
    modified = RAW.copy()
    last = modified.groupby("machine_id")["cycle"].transform("max")
    mask = (modified["machine_id"] == 3) & (modified["cycle"] > last - 20)
    for s in SENSORS:
        modified.loc[mask, s] = modified.loc[mask, s].fillna(0) + 999
    f2 = add_features(clean(modified))
    past = (f1["machine_id"] == 3) & (f1["cycle"] <= f1.groupby("machine_id")["cycle"].transform("max") - 20)
    cols = feature_columns(f1)
    np.testing.assert_allclose(f1.loc[past, cols].values, f2.loc[past, cols].values, atol=1e-9)


def test_no_nan_in_features_after_cleaning():
    d = add_features(clean(RAW))
    assert not d[feature_columns(d)].isna().any().any()


def test_label_matches_horizon():
    d = add_features(clean(RAW))
    assert ((d["RUL"] <= HORIZON) == (d["panne_imminente"] == 1)).all()


def test_split_by_machine_has_no_overlap():
    d = add_features(clean(RAW))
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=0).split(d, groups=d["machine_id"]))
    assert set(d.iloc[tr].machine_id).isdisjoint(set(d.iloc[te].machine_id))


def test_first_alert_needs_persistence():
    cycles = np.arange(1, 11)
    p = np.array([0, 0.9, 0, 0.9, 0.9, 0.9, 0, 0, 0, 0])
    assert first_alert_cycle(p, cycles, 0.5, k=3) == 6      # 3 cycles consécutifs : 4, 5, 6
    assert first_alert_cycle(p * 0, cycles, 0.5, k=3) is None


def test_nasa_score_penalises_late_more_than_early():
    assert nasa_score([30], [20])[0] > nasa_score([10], [20])[0]

"""Simulation de données de capteurs IoT jusqu'à la panne (run-to-failure).

Pourquoi simuler ? Les données industrielles réelles sont rarement publiques.
Le simulateur reproduit les difficultés d'un vrai parc de machines :
  - 3 modes de panne qui n'affectent pas les mêmes capteurs ;
  - début de dégradation et vitesse différents d'une machine à l'autre ;
  - sévérité variable (certaines machines montrent peu de signes) ;
  - régimes de charge qui décalent les mesures (fausses alertes possibles) ;
  - dérive lente d'un capteur, sans lien avec la panne ;
  - valeurs manquantes (2 %) et pics aberrants (0,5 %).
"""
import numpy as np
import pandas as pd

from config import (SENSORS, LIFE_RANGE, MODES, SEED, N_MACHINES)

# Valeurs "saines" moyennes et bruit par capteur
BASE = {"temperature": 70.0, "vibration": 2.0, "pression": 5.0, "rpm": 1500.0, "courant": 12.0}
NOISE = {"temperature": 1.5, "vibration": 0.3, "pression": 0.15, "rpm": 15.0, "courant": 0.5}

# Effet du mode de panne sur chaque capteur (amplitude à la fin de vie)
MODE_EFFECT = {
    "roulement":  {"vibration": 3.0, "temperature": 6.0, "courant": 1.2},
    "surchauffe": {"temperature": 20.0, "courant": 3.0, "vibration": 0.6},
    "fuite":      {"pression": -1.1, "rpm": -70.0, "courant": 0.4},
}
# Effet d'un changement de charge (régime de fonctionnement)
LOAD_EFFECT = {"temperature": 5.0, "vibration": 0.25, "pression": 0.0, "rpm": -15.0, "courant": 1.5}


def simulate(n_machines: int = N_MACHINES, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for m in range(1, n_machines + 1):
        life = int(rng.integers(*LIFE_RANGE))
        t = np.arange(1, life + 1)
        mode = MODES[int(rng.integers(0, len(MODES)))]

        # Dégradation : début aléatoire, puis montée en puissance non linéaire
        onset = rng.uniform(0.30, 0.65)
        power = rng.uniform(1.5, 3.0)
        deg = np.clip((t / life - onset) / (1 - onset), 0, 1) ** power
        severity = rng.uniform(0.6, 1.4)

        # Régime de charge : paliers de 20 à 60 cycles (0.85, 1.0 ou 1.15)
        load = np.empty(life)
        i = 0
        while i < life:
            seg = int(rng.integers(20, 60))
            load[i:i + seg] = rng.choice([0.85, 1.0, 1.15])
            i += seg

        # Dérive lente d'un capteur choisi au hasard (30 % des machines)
        drift_sensor = rng.choice(SENSORS) if rng.random() < 0.3 else None

        cols = {"machine_id": m, "cycle": t, "mode_panne": mode}
        for s in SENSORS:
            off = rng.normal(0, 1) * NOISE[s] * 1.5          # variabilité entre machines
            x = BASE[s] + off + rng.normal(0, NOISE[s], life)
            x += LOAD_EFFECT[s] * (load - 1.0) / 0.15 * 0.5   # effet de charge
            x += MODE_EFFECT[mode].get(s, 0.0) * severity * deg
            if s == drift_sensor:
                x += rng.choice([-1, 1]) * 2.0 * NOISE[s] * (t / life)
            cols[s] = x
        df = pd.DataFrame(cols)
        df["RUL"] = life - t
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    # Imperfections des capteurs IoT : pics aberrants puis valeurs manquantes
    for s in SENSORS:
        n = len(data)
        spike = rng.random(n) < 0.005
        miss = rng.random(n) < 0.02
        data.loc[spike, s] += rng.choice([-1, 1], spike.sum()) * 8 * data[s].std()
        data.loc[miss, s] = np.nan
    return data

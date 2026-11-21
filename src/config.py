"""Paramètres globaux du projet (un seul endroit pour tout modifier)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

SEED = 42
SENSORS = ["temperature", "vibration", "pression", "rpm", "courant"]

# --- Simulation -------------------------------------------------------------
N_MACHINES = 150
LIFE_RANGE = (150, 350)          # durée de vie (cycles) tirée par machine
MODES = ["roulement", "surchauffe", "fuite"]   # 3 modes de panne

# --- Pré-traitement ---------------------------------------------------------
SPIKE_WINDOW = 7                 # médiane glissante (causale) pour détecter les pics
SPIKE_K = 6                      # seuil = 6 x MAD
FEAT_WINDOW = 10                 # fenêtre des moyennes / écarts-types
DIFF_LAG = 5                     # variation sur 5 cycles
BASELINE_CYCLES = 20             # les 20 premiers cycles = état "sain" de la machine

# --- Cibles -----------------------------------------------------------------
HORIZON = 24                     # panne "imminente" si RUL <= 24 cycles
RUL_CLIP = 125                   # plafonnement du RUL (régression)

# --- Évaluation -------------------------------------------------------------
TEST_SIZE = 0.2                  # 20 % des machines en test final
N_FOLDS = 5                      # validation croisée par machine

# --- Modèle de coût (décision de maintenance) -------------------------------
ALERT_PERSIST = 3                # une alerte = proba >= seuil pendant 3 cycles de suite
MIN_LEAD = 3                     # il faut au moins 3 cycles pour intervenir
C_FAIL = 100.0                   # coût d'une panne non planifiée
C_PLANNED = 10.0                 # coût d'une maintenance planifiée
C_WASTE = 0.5                     # coût par cycle de vie utile gaspillé (valeur machine ~125 sur ~250 cycles)
PREVENTIVE_AT = 150              # stratégie préventive : remplacement fixe à 150 cycles

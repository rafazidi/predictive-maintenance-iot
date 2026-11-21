# Maintenance prédictive industrielle (IoT)

Pipeline complet en Python : simulation de capteurs IoT, pré-traitement, modèles de machine learning, et **décision de maintenance chiffrée en coût**.

> Projet du Master Data Science & Software Development — sujet 2 de la liste de projets IoT.
> Les données sont **simulées** (voir [Limites](#limites)).

## Résultats principaux

Test final sur 30 machines jamais vues (120 machines pour le développement) :

| Stratégie de maintenance | Coût moyen par machine |
|---|---|
| Réactive (attendre la panne) | 100,0 |
| Préventive (remplacer à 150 cycles) | 60,0 |
| **Prédictive (modèle ML)** | **28,6** |

- **−71 %** de coût par rapport à la maintenance réactive ; 29 pannes anticipées sur 30, délai d'alerte médian de 29 cycles.
- Classification « panne dans les 24 prochains cycles », validation croisée par machine (PR-AUC) : référence à 1 variable 0,46, régression logistique 0,78, Random Forest 0,78, Gradient Boosting 0,79.
- Durée de vie résiduelle (RUL) : erreur moyenne de 15,5 cycles (référence « moyenne » : 35,2). Le modèle est **trop optimiste** en fin de vie (biais de +7,1 cycles).
- Les trois modèles sont proches entre eux (écart-type de la PR-AUC entre plis : 0,03 à 0,04) : le gain vient du machine learning en général, pas d'un modèle précis.

Détails chiffrés : [`results/resultats.txt`](results/resultats.txt).

## Installation et lancement

```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python src/main.py          # environ 2 minutes, régénère data/ et results/
pytest -q                   # 8 tests automatiques
```

La graine est fixée (`SEED = 42`) : les résultats sont reproductibles à l'identique.

## Structure du dépôt

```
src/
  config.py          tous les paramètres (horizon, coûts, fenêtres...)
  simulation.py      génération des données de 150 machines
  preprocessing.py   nettoyage et features (tout est causal)
  modeling.py        modèles, validation croisée, seuil, coûts, RUL
  plots.py           graphiques
  main.py            enchaîne toutes les étapes
tests/test_pipeline.py   tests automatiques
data/                dataset_brut.csv, dataset_pretraite.csv
results/             tableaux, graphiques, resultats.txt, resume.json
docs/Presentation_Maintenance_Predictive.pptx   présentation (18 slides, notes de l'orateur)
```

## Comprendre le projet de A à Z

1. **Simulation** (`simulation.py`) : 150 machines suivies jusqu'à la panne, 5 capteurs (température, vibration, pression, rotation, courant). Trois modes de panne (roulement, surchauffe, fuite), début et vitesse de dégradation propres à chaque machine, régimes de charge, dérive de capteur, 2 % de valeurs manquantes et 0,5 % de pics.
2. **Nettoyage** (`preprocessing.clean`) : pics détectés par médiane glissante causale et MAD, puis remplacement par la dernière valeur connue.
3. **Features** (`preprocessing.add_features`) : 6 par capteur (valeur, moyenne sur 10 cycles, écart-type, variation sur 5 cycles, écart à l'état sain, tendance), soit 30 features. Calculées sur le passé uniquement.
4. **Cibles** : `panne_imminente` (RUL ≤ 24 cycles, 9,8 % de positifs) pour la classification ; `RUL_clip` (plafonné à 125) pour la régression.
5. **Découpage par machine** : 120 machines de développement, 30 de test final. Une machine n'est jamais à la fois en entraînement et en test, sinon les scores sont gonflés.
6. **Validation croisée** (`GroupKFold`, 5 plis) sur les 120 machines : 4 modèles comparés, dont une **référence à 1 variable** qui sert à vérifier que le machine learning apporte quelque chose. Métrique principale : PR-AUC (classe rare).
7. **Décision** : alerte quand la probabilité dépasse un seuil pendant 3 cycles de suite. Le seuil minimise un coût (panne non planifiée 100 ; maintenance planifiée 10 plus 0,5 par cycle de vie utile gaspillé), choisi sur les prédictions out-of-fold, donc sans toucher aux machines de test.
8. **Interprétation** : importance par capteur (on mélange toutes les features d'un capteur ensemble, car elles sont corrélées). Pression, température et vibration portent l'essentiel du signal.
9. **RUL** : forêt aléatoire de régression, évaluée par MAE, biais et score asymétrique NASA (prédire trop tard coûte plus cher que trop tôt).

## Paramètres à modifier

Tout est dans [`src/config.py`](src/config.py) : `N_MACHINES`, `HORIZON` (24), `RUL_CLIP` (125), `ALERT_PERSIST`, et les coûts `C_FAIL`, `C_PLANNED`, `C_WASTE`, `PREVENTIVE_AT`. Les coûts sont des **hypothèses** : dans une vraie usine, on les remplace par les coûts réels.

## Limites

- **Données simulées** : les scores sur de vraies machines seront différents. À valider sur un jeu public (par exemple NASA C-MAPSS) ou des données d'usine.
- Seulement 30 machines de test : les chiffres de coût sont indicatifs (intervalles de confiance à calculer, par exemple par bootstrap).
- Le calcul de l'échelle de bruit du nettoyage (MAD) est global ; on le considère comme une constante de calibration du capteur. Le premier cycle d'une machine peut être comblé par la valeur suivante si elle manque.
- Le modèle de RUL est trop optimiste en fin de vie. Pistes : régression quantile, perte asymétrique, LSTM ou CNN 1D sur fenêtres de séries temporelles.
- Déploiement réel : passerelle edge, MQTT, base de séries temporelles, surveillance de la dérive des données et ré-entraînement périodique.

## Licence

MIT, voir [`LICENSE`](LICENSE).

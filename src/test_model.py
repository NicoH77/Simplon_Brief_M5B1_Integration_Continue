"""
test_model.py - Validation du modèle avant déploiement
=================================================

Fonction de validation du modèle de risque de diabète avant déploiement dans l'environnement cible.

Ce script est utilisé dans le workflow GitHub Actions comme étape de contrôle qualité du modèle.

L'objectif est de vérifier que les performances du modèle sauvegardé respectent le seuil minimal métier avant autorisation du déploiement de l'API.

Afin de tester, à appeler depuis le notebook après l'éxéction de :
- train_model.py

Règle de validation :
---------------------
- si le Recall < 0.60 :
    * le script retourne une erreur
    * le job GitHub Actions échoue
    * le workflow s'arrête
    * l'API n'est pas déployée

- si le Recall >= 0.60 :
    * le test est validé
    * le workflow continue
    * le déploiement de l'API est autorisé

Prérequis :
-----------
1. rendre ce script importable depuis le notebook
2. avoir exécuté train_model.py
3. disposer d'un modèle sérialisé au format pickle (.pkl)
4. disposer d'un dataset de référence pour la validation

Exécution depuis un notebook :
------------------------------
1. importer la fonction :
    from src.test_model import test_model

2. exécuter :
    recall = test_model()
  
Valeur retournée :
------------------
float :
    Recall calculé sur le dataset de référence.

Codes de sortie :
-----------------
0 : Validation réussie
1 : Validation échouée (Recall inférieur au seuil minimal)

"""


import sys
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import recall_score


MODEL_PATH = Path("app/api/diabetes_risk_model.pkl")
REFERENCE_DATA_PATH = Path("data/reference/diabetes_reference.csv")

MIN_RECALL = 0.60

def test_model() -> float:
    """
    Charge le modèle sauvegardé, exécute les prédictions sur le dataset de référence et retourne son recall.
    """
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    # features = artifact["features"]

    df = pd.read_csv(REFERENCE_DATA_PATH)

    FEATURES = [
        "pregnancies", "glucose", "blood_pressure", "skin_thickness",
        "insulin", "bmi", "diabetes_pedigree", "age",
        ]

    X = df[FEATURES]
    y_true = df["outcome"]

    y_pred = model.predict(X)

    recall = recall_score(y_true, y_pred)

    print(f"Recall obtenu : {recall:.4f}")

    if recall < MIN_RECALL:
        print(
            f"❌ Recall insuffisant : "
            f"{recall:.4f} < {MIN_RECALL}"
        )
        sys.exit(1)

    print(
        f"✅ Recall validé : "
        f"{recall:.4f} >= {MIN_RECALL}"
    )
    sys.exit(0)
        
    return recall


if __name__ == "__main__":
    score = test_model()

    print(f"Recall : {score:.3f}")
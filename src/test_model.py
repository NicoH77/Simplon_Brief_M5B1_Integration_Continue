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
    Charge le modèle sauvegardé,
    exécute les prédictions sur le dataset
    de référence et retourne son recall.
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
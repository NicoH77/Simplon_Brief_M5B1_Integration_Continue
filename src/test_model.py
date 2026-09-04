from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import recall_score


MODEL_PATH = Path("outputs/diabetes_risk_model.pkl")
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

    assert recall >= MIN_RECALL, (
        f"Recall insuffisant : "
        f"{recall:.3f} < {MIN_RECALL:.2f}"
    )

    return recall


if __name__ == "__main__":
    score = test_model()

    print(f"Recall : {score:.3f}")
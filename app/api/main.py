"""
===============================================================================
Nom du script : main.py
Auteur : Nico H
Projet : M5 - Prédiction de risque de diabète
Version : 1.0
===============================================================================

Exposition du modèle via l'API FastAPI.

Le modèle est récupéré automatiquement depuis MLflow à partir d'un Run ID
et utilisé pour réaliser des prédictions.

Fonctionnalités principales
---------------------------
- Chargement du meilleur modèle depuis MLflow.
- Vérification de l'état de l'API (health).
- Consultation des métadonnées du modèle déployé.
- Prédiction à partir de nouvelles données.

Pipeline de prédiction
----------------------

6. Prédiction via le modèle chargé depuis MLflow.

Endpoints
---------
GET /
    Endpoint racine.

GET /health
    Vérifie que l'API est opérationnelle et que le modèle est chargé.

GET /model-info
    Retourne les informations du modèle actuellement déployé
    (Run ID, URI MLflow, type du modèle, etc.).

POST /predict
    Réalise une prédiction à partir de données transmises
    sous forme de .


"""
import os

from fastapi import FastAPI, UploadFile, File, HTTPException

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from pathlib import Path


# ------------------------------------------------------------------------------
# Initialisation de l'environnement
# ------------------------------------------------------------------------------
MODEL_BACKEND = os.getenv(
    "MODEL_BACKEND",
    "mlflow"
)

# ------------------------------------------------------------------------------
# Initialisation de l'application FastAPI
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Diabete Risks Classifier API",
    version="1.0"
)


# ------------------------------------------------------------------
# Schéma des données d'entrée
# ------------------------------------------------------------------

class PatientData(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree: float
    age: int

# ------------------------------------------------------------------
# CONFIGURATION MLFLOW - Chargement du modèle selon le contexte
# ------------------------------------------------------------------
if MODEL_BACKEND == "pkl":
    MODEL_PATH = Path("/app/api/diabetes_risk_model.pkl")
    features = None
    client = None

    print("🚀 Chargement du modèle depuis un fichier PKL")
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    features = artifact["features"]
    print(f"✅ Modèle chargé : {MODEL_PATH}")

else:
    print("🚀 Chargement du modèle depuis MLFlow")
    
    MODEL_NAME = os.getenv("MODEL_NAME", "diabetes-risk-model")
    MODEL_STAGE = os.getenv("MODEL_STAGE", "Staging")
    MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

    # attention : faire référence à l'instance MLflow qui s'éxécute sur l'hote (en local)
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model = mlflow.sklearn.load_model(MODEL_URI)

    client = MlflowClient()

    print("✅  Modèle chargé")



# ------------------------------------------------------------------
# ENDPOINT RACINE
# ------------------------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Diabete Risks Classifier API"
    }

# ------------------------------------------------------------------
# ENDPOINT Health-Check
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "UP",
        "model_loaded": model is not None
    }


# ------------------------------------------------------------------
# ENDPOINT MODEL-INFO
# ------------------------------------------------------------------

@app.get("/model-info")
def get_model_info():
    """
    Retourne les informations du modèle actuellement utilisé.

    Returns:
        dict: Informations du modèle issu du MLflow Model Registry.
    """
    if MODEL_BACKEND == "pkl":
        
        return {
        "source": "joblib",
        "model_path": str(MODEL_PATH),
        "status": "loaded"
        }

    try:
        latest_model = client.get_latest_versions(
            MODEL_NAME,
            stages=[MODEL_STAGE]
        )[0]

        return {
            "model_name": latest_model.name,
            "version": latest_model.version,
            "stage": latest_model.current_stage,
            "run_id": latest_model.run_id,
            "source": latest_model.source,
            "status": "loaded"
        }

    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc)
        }

# ------------------------------------------------------------------
# ENDPOINT PREDICTION
# ------------------------------------------------------------------

@app.post("/predict")
def predict(data: PatientData):

    input_df = pd.DataFrame(
        [data.dict()]
    )

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 4)
    }
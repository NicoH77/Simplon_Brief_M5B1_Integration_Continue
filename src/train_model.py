"""
train_model.py - MLflow avec Scikit-Learn
=================================================

Fonction d'entraînement avec MLflow pour le modèle de risque de diabète.

À appeler depuis le notebook après la création :
- du pipeline
- du GridSearchCV
- des jeux X_train, X_test, y_train, y_test

Prérequis:
---------
1. rendre ce script importable depuis le notebook

Exécution:
---------
1. appeler la fonction train_model()
  

Interface MLflow:
----------------
Après exécution, consultez http://localhost:5000 pour voir :
- Les runs dans l'expérience
- Les datasets loggés (onglet Overview)
- Les métriques loggées 
- Les artefacts comparables (matrices de confusion)
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
import joblib
import matplotlib.pyplot as plt


def train_model(search,X_train,X_test,y_train,y_test,model_name="diabetes-risk-model",stage="Staging"):
    """
    Entraîne le GridSearchCV et log les résultats dans MLflow.
    
    Cette fonction démontre le logging complet avec MLflow :
    1. Paramètres : hyperparamètres du modèle
    2. Métriques : performances mesurées
    3. Tags : métadonnées descriptives
    4. Artefacts : fichiers (graphiques, rapports)
    5. Dataset : données d'entraînement (traçabilité)
    6. Modèle : modèle sérialisé réutilisable
    
    Args:
        search (GridSearchCV): Objet de recherche d'hyperparamètres configuré avec le pipeline
            de prétraitement et de modélisation.
        X_train (pd.DataFrame): Variables du jeu d'entraînement
        X_test (pd.DataFrame): Variables du jeu de test.
        y_train (pd.Series): Variable cible du jeu d'entraînement.
        y_test (pd.Series): Variable cible du jeu d'évaluation.
        model_name (str, optional): Nom du modèle dans le MLflow Model Registry.
            Par défaut : "diabetes-risk-model".
        stage (str, optional): Stage MLflow dans lequel enregistrer la version du modèle
            Par défaut : "Staging".

    Returns:
        sklearn.pipeline.Pipeline:
            Le meilleur modèle obtenu après recherche des
            hyperparamètres.

    """
    # ========================================================================
    # DÉMARRAGE D'UN RUN MLFLOW
    # ========================================================================
    # mlflow.start_run() crée un nouveau run dans l'expérience active
    # run_name : nom affiché dans l'interface (optionnel mais recommandé)
    # Le context manager (with) garantit que le run est fermé proprement
    
    with mlflow.start_run(run_name=f"eval_{model_name}"):

        # ====================================================================
        # ÉTAPE 1 : ENTRAÎNEMENT DU MODÈLE
        # ====================================================================
        # fit() entraîne le modèle sur les données d'entrainement
        # Après cette étape, le modèle peut faire des prédictions
        
        print(f"\n🔄 Entraînement: {model_name}")
        search.fit(X_train, y_train)

        best_model = search.best_estimator_        

        # Prédictions
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        # ====================================================================
        # ÉTAPE 2 : CALCUL DES MÉTRIQUES
        # ====================================================================
        # Les métriques quantifient la performance du modèle
        # 
        # accuracy  : % de prédictions correctes
        # precision : parmi les positifs prédits, combien sont vrais positifs
        # recall    : parmi les vrais positifs, combien sont détectés
        # f1_score  : moyenne harmonique de precision et recall
        
        # Capture des métriques d'évaluation        
        # accuracy = accuracy_score(y_test, y_pred)
        # recall = recall_score(y_test, y_pred)
        # f1 = f1_score(y_test, y_pred)
        # auc = roc_auc_score(y_test, y_proba)

        metriques = {
            "accuracy": accuracy_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1_score": f1_score(y_test, y_pred, average='weighted'),
            "auc": roc_auc_score(y_test, y_proba)
        }
        
        # ====================================================================
        # ÉTAPE 3 : LOG MLFLOW
        # ====================================================================

        # Log des paramètres
        mlflow.log_params(search.best_params_)
        
        # Log des métriques
        mlflow.log_metric("accuracy", metriques['accuracy'])
        mlflow.log_metric("recall", metriques['recall'])
        mlflow.log_metric("f1_score", metriques['f1_score'])
        mlflow.log_metric("roc_auc", metriques['auc'])
        mlflow.log_metric("best_cv_recall", search.best_score_)

        # ====================================================================
        # ÉTAPE 4 : MATRICE DE CONFUSION
        # ====================================================================
        # Capture et log de la matrice de confusion

        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)

        # Plot
        fig, ax = plt.subplots(figsize=(5, 5))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm
        )

        disp.plot(ax=ax)

        # Log de la matrice dans MLFLow
        mlflow.log_figure(fig, "confusion_matrix.png")

        plt.close()        


        # =========================
        # ÉTAPE 5 : Enregistrement du modèle
        # =========================

        model_info = mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            serialization_format="pickle"
        )


        # ====================================================================
        # AFFICHAGE DES RÉSULTATS
        # ====================================================================
        print(f"✅ Modèle {model_name} entraîné et logué avec succès dans MLflow.")
        print(f"   - Accuracy: {metriques['accuracy']:.4f}")
        print(f"   - recall: {metriques['recall']:.4f}")
        print(f"   - F1-Score: {metriques['f1_score']:.4f}")
        print(f"   - auc: {metriques['f1_score']:.4f}")        
        

        # =========================
        # Registry
        # =========================

        client = MlflowClient()

        model_version = mlflow.register_model(model_info.model_uri, model_name)

        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage=stage
        )

        print("Run ID :", mlflow.active_run().info.run_id)
        print("Model version :", model_version.version)
        print("Recall :", round(metriques['recall'], 4))

        return best_model



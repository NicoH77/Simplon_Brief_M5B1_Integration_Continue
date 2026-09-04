# Module 5 Brief 1

# 🩺 Diabetes Risk Prediction - MLOps & Continuous Integration

## 📖 Contexte

Ce projet a été réalisé dans le cadre du **Brief M5** consacré au suivi d'expériences Machine Learning et à l'intégration continue d'un modèle de classification.

L'objectif est de mettre en œuvre une démarche **MLOps** complète permettant de :

- préparer les données ;
- rechercher automatiquement les meilleurs hyperparamètres d'un modèle ;
- suivre les expérimentations avec MLflow ;
- implémenter une "quality gate" via GitHub Actions.

---

## 🎯 Objectif métier

Prédire le risque de diabète à partir de caractéristiques médicales de patients à l'aide d'un modèle de Machine Learning supervisé.

---

## 🏗 Architecture du projet

```text
.
├── .github/
│   └── workflows/         # GitHub Actions
├── app/                   # API et ML flow
│   └── api/               # api diabetes-api
│   └── mlflow/            # artefacts & données mlflow
├── data/                  # Jeux de données
├── docs/                  # Documentation
├── notebooks/             # Analyses exploratoires
│   └── entrainement_modele.ipynb
├── outputs/               # Modèles et artefacts générés
├── src/                   # Code source
│   └── train_model.py     # entrainement du modèle
│   └── test_model.py      # test du modèle
├── requirements.txt
└── README.md
```

---

## ⚙️ Principales Technologies utilisées

### Data Science & Machine Learning

- Python
- Scikit-Learn

### MLOps

- MLflow
- Joblib

### Qualité Logicielle

- GitHub Actions

---

## 🚀 Installation

```bash
pip install -r requirements.txt
```

---

## 📊 Entraînement du modèle

Lancer l'entraînement :

```bash
python src/train_model.py
```

Le pipeline réalise automatiquement :

1. le chargement des données ;
2. le prétraitement ;
3. la création du pipeline Scikit-Learn ;
4. l'optimisation par GridSearchCV ;
5. l'évaluation du modèle ;
6. la sauvegarde du meilleur modèle.

---

## 📈 Suivi des expérimentations avec MLflow

En environnement local, les expérimentations sont suivies avec MLflow :

- paramètres ;
- métriques ;
- artefacts ;
- modèles enregistrés.

Lancement de l'interface :

```bash
mlflow ui
```

Puis ouvrir :

```text
http://localhost:5000
```

---

## 💾 Sauvegarde du modèle

le script `train_model.py` :
- enregistre les métriques et la matrice de confusion dans MLflow ;
- le meilleur modèle est enregistré dans le MLflow Registry.

Le notebook `entrainement_modele.ipynb` enregistre le modèle :
- `diabetes_risk_model.pkl`


---

## 🚦 Integration continue et Validation automatique des performances

Le workflow GitHub Actions exécute automatiquement :

- installation des dépendances ;
- exécution des tests ;
- validation des performances ;
- Déploiement de l'api dans un docker

Une vérification automatique du **Recall** est effectuée après l'entraînement.

```python
RECALL_THRESHOLD = 0.60
```

Grâce à l'intégration d'une *quality gate* dans **GitHub Actions** :

- ✅ un modèle dont le Recall est supérieur ou égal au seuil défini est autorisé à être déployé ;
- ❌ un modèle dont le Recall est inférieur au seuil provoque l'échec du workflow ;
- 🚫 aucun déploiement de l'API n'est réalisé tant que les critères de qualité ne sont pas respectés.

Cette vérification permet de garantir un niveau minimal de performance avant toute mise à disposition du modèle.

---

# 📝 Conclusion et perspectives

Dans ce projet, deux stratégies de gestion des modèles ont été mises en œuvre en fonction de l'environnement d'exécution.

Lors d'une exécution **en local**, le suivi des expérimentations est réalisé avec **MLflow**. Cet outil permet d'enregistrer automatiquement :

- les paramètres d'entraînement ;
- les métriques obtenues ;
- les artefacts générés ;
- le meilleur modèle sélectionné par la phase de recherche d'hyperparamètres (*GridSearchCV*).

Cette approche facilite l'analyse comparative des expérimentations et assure une excellente traçabilité des résultats.

En revanche, lors de l'exécution du workflow dans **GitHub Actions**, l'utilisation de MLflow a volontairement été écartée. Bien qu'intéressant d'un point de vue MLOps, son déploiement dans un environnement CI/CD nécessite la mise en place et la maintenance d'un serveur MLflow accessible depuis les runners GitHub, ainsi que la gestion :

- de l'authentification ;
- du stockage des artefacts ;
- de la persistance des données ;
- du registre des modèles.

POur ce brief, afin de conserver un pipeline simple, robuste et facilement reproductible, le modèle entraîné est donc sauvegardé sous la forme d'un fichier **PKL** grâce à :

```python
joblib.dump(
    {
        "model": best_pipeline
    },
    "./outputs/diabetes_risk_model.pkl",
)
```

Cette solution permet de récupérer directement le modèle issu de l'entraînement et de l'utiliser ultérieurement sans dépendre d'une infrastructure externe.

## 🚀 Bonnes pratiques en environnement réel

Dans un contexte de production, il serait néanmoins recommandé d'adopter une architecture MLOps plus complète.

Les bonnes pratiques consistent généralement à :

- conserver un **serveur MLflow centralisé** pour le suivi des expérimentations ;
- stocker les modèles dans un **Model Registry** permettant de gérer les différentes versions ;
- utiliser un stockage d'artefacts persistant (Azure Blob Storage, AWS S3, Google Cloud Storage, etc.) ;
- automatiser la promotion des modèles entre les environnements **Development**, **Staging** et **Production** ;
- mettre en place des contrôles qualité et des validations de performance avant tout déploiement ;
- assurer la traçabilité complète entre le code source, les données d'entraînement, les métriques obtenues et le modèle déployé.

---

## 📜 Licence

Projet réalisé dans un cadre pédagogique dans le cadre de formation IA & Data.



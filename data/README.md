# Données — TP Module 5

- `raw/diabetes_train.csv` : dataset d'entraînement (700 profils patients), à utiliser pour réécrire le script d'entraînement à partir du notebook (`notebooks/entrainement_modele.ipynb`).
- `reference/diabetes_reference.csv` : jeu de référence labellisé (200 profils, même distribution que l'entraînement), à utiliser pour calculer le rappel du modèle dans le test de non-régression (cas nominal — le rappel doit rester au-dessus du seuil donné).
- `reference/diabetes_reference_drifted.csv` : même format et profils statistiquement proches du jeu de référence, mais où le lien entre les caractéristiques cliniques et le diabète a évolué (dérive de concept). Le rappel du modèle chute nettement dessus. À utiliser pour vérifier que votre test détecterait bien une dégradation si elle survenait.

## Dictionnaire des colonnes

| Colonne | Description |
|---|---|
| pregnancies | Nombre de grossesses |
| glucose | Glycémie plasmatique (test de tolérance au glucose) |
| blood_pressure | Tension artérielle diastolique (mm Hg) |
| skin_thickness | Épaisseur du pli cutané tricipital (mm) |
| insulin | Insuline sérique à 2h (mu U/ml) |
| bmi | Indice de masse corporelle |
| diabetes_pedigree | Fonction pedigree du diabète (facteur héréditaire) |
| age | Âge du patient |
| outcome | 1 = risque de diabète présent, 0 = absent |

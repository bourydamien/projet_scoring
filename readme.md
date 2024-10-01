# Crédit Scoring API

## Description

Ce projet consiste en la création d'un modèle de **scoring crédit** pour une société financière. L'objectif est de prédire si un client est éligible pour un crédit en utilisant des données historiques. Le projet inclut :

- L'entraînement du modèle sous **Google Colab** avec l'utilisation de GPU pour l'optimisation.
- L'utilisation d'une métrique personnalisée.
- La gestion du pipeline de machine learning, y compris le traitement des données et la normalisation.
- L'intégration de **CatBoost**, **H2O** et **Dummie_claddifier**.
- La gestion du déséquilibre des classes avec  la pondération des classes.
- Un test des différents seuils de décision afin de minimiser le cout pondéré.
- Le suivi des expérimentations via **MLflow**, stockant les artefacts dans **Google Drive**.
- Le déploiement d'une API permettant d'accéder au modèle.

## Contenu du projet

Le projet contient les éléments suivants :

1. **Notebook d'entraînement** : `scoring_colab_catboost_gpu.ipynb`
   - Entraînement et comparaison de plusieurs modèles sur Google Colab en utilisant un GPU.
   - Suivi des expérimentations via MLflow et enregistrement des artefacts (modèles et transformations) sur Google Drive.

2. **Notebook d'optimisation** : `Catboost_optimisation_seuil_scoring.ipynb`
   - Optimisation du seuil de décision pour le modèle **CatBoost** afin de minimiser le coût métier entre faux positifs et faux négatifs.

3. **Fichiers de configuration** :
   - `python-test.yaml` : Fichier de configuration de **Git Action**.

4. **API** : `API.py`
   - Implémentation de l'API pour prédire le scoring crédit à partir de nouvelles données.
   - Déploiement de l'API avec **Render**, et chargement des fichiers de modèles depuis Google Drive.

5. **Tests unitaires** : `test_api.py`
   - Tests unitaires pour l'API.

## Installation

Pour cloner et démarrer le projet localement, assurez-vous d'avoir les dépendances suivantes :

- Python 3.8 ou supérieur
- Google Colab pour l'entraînement du modèle
- Google Drive pour le stockage des artefacts et données.
- MLflow pour le suivi des expérimentations
- Git Action pour la CD/CI
- Render pour le cloud





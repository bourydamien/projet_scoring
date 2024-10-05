import os
import json
import numpy as np
import pandas as pd
import pickle
import joblib
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import sys

def check_dataframe(df, scaler):
    """
    Vérifie l'intégrité du DataFrame avant l'application du scaler.
    
    :param df: DataFrame à vérifier
    :param scaler: Scaler pré-entraîné
    :return: Dictionnaire contenant l'état de la vérification
    """
    issues = []

    # Vérifier les valeurs manquantes
    if df.isnull().values.any():
        issues.append("Le DataFrame contient des valeurs manquantes.")

    # Vérifier les valeurs infinies
    if np.isinf(df.values).any():
        issues.append("Le DataFrame contient des valeurs infinies.")

    # Vérifier les dimensions
    if df.shape[1] != scaler.n_features_in_:
        issues.append(f"Le DataFrame a {df.shape[1]} colonnes, alors que le scaler attend {scaler.n_features_in_} colonnes.")

    # Vérifier les colonnes
    if not all(col in df.columns for col in scaler.feature_names_in_):
        missing_cols = set(scaler.feature_names_in_) - set(df.columns)
        issues.append(f"Les colonnes suivantes sont manquantes dans le DataFrame : {', '.join(missing_cols)}")

    return issues

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'app/utils')))

from utils.inference import (
    load_preprocessing_objects,
    add_features_and_correct_anomaly,
    apply_label_encoding,
    apply_one_hot_encoding,
    apply_boxcox_transformations,
    apply_scaler,
    predict_with_catboost
)
MODEL_FOLDER = "app/models/"
app = FastAPI()
@app.get("/")
async def root():
    return {"message": "L'API de scoring fonctionne."}

# Point d'entrée pour prédiction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Lire le fichier uploadé
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Charger les objets de prétraitement
        try:
            (label_encoders, one_hot_columns, scaler, lambda_params, catboost_model, aligned_columns) = load_preprocessing_objects(MODEL_FOLDER)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur lors du chargement des objets de prétraitement : {str(e)}")

        # Étape 1 : Correction des anomalies et ajout de features dérivées
        try:
            df = add_features_and_correct_anomaly(df)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur dans add_features_and_correct_anomaly : {str(e)}")

        # Étape 2 : Appliquer le label encoding
        try:
            df = apply_label_encoding(df, label_encoders)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur dans apply_label_encoding : {str(e)}")
        
        # Étape 3 : Appliquer le One-Hot Encoding
        try:
            df = apply_one_hot_encoding(df, one_hot_columns)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur dans apply_one_hot_encoding : {str(e)}")

        # Étape 4 : Appliquer la transformation Box-Cox
        try:
            df = apply_boxcox_transformations(df, lambda_params)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur dans apply_boxcox_transformations : {str(e)}")
        issue = check_dataframe(df, scaler)
        return JSONResponse(content={"predictions": issue})
        '''# Étape 5 : Appliquer le scaler
        try:
            df = apply_scaler(df, scaler)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur dans apply_scaler : {str(e)}")

        # Étape 6 : Prédire avec le modèle CatBoost
        try:
            predictions = predict_with_catboost(catboost_model, df)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur dans predict_with_catboost : {str(e)}")

        # Retourner les résultats
        return JSONResponse(content={"predictions": predictions.tolist()})'''

    except HTTPException as e:
        # L'erreur a déjà été traitée avec une fonction spécifique, donc juste la relancer
        raise e

    except Exception as e:
        # Si une autre erreur inattendue se produit, la capturer ici
        raise HTTPException(status_code=500, detail=f"Erreur inattendue : {str(e)}")

# Démarrage de l'API
if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Utiliser la variable d'environnement PORT ou 8000 par défaut
    uvicorn.run(app, host='0.0.0.0', port=port)  # Écoute sur toutes les interfaces

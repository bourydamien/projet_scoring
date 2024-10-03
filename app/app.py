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


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))

from utils.inference import (
    load_preprocessing_objects,
    add_features_and_correct_anomaly,
    apply_label_encoding,
    apply_one_hot_encoding,
    apply_boxcox_transformations,
    apply_scaler,
    predict_with_catboost
)
MODEL_FOLDER = "utils/models/"
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
        # Convertir le contenu en DataFrame
        df = pd.read_csv(io.BytesIO(contents))
        
        # Charger les objets de prétraitement
        (label_encoder, label_encoded_columns, one_hot_columns, scaler, 
         transformed_columns, lambda_params, catboost_model) = load_preprocessing_objects(MODEL_FOLDER)

        # Appliquer les étapes de prétraitement
        df = add_features_and_correct_anomaly(df)  # Correction d'anomalies et ajout de features dérivées
        
        # Appliquer le label encoding
        df = apply_label_encoding(df, label_encoded_columns, label_encoder)
        
        # Appliquer le One-Hot Encoding en alignant avec les colonnes d'entraînement
        df = apply_one_hot_encoding(df, one_hot_columns)
        
        # Appliquer la transformation Box-Cox
        df = apply_boxcox_transformations(df, lambda_params)
        
        # Appliquer le scaler
        df = apply_scaler(df, scaler)
        
        # Faire des prédictions avec le modèle CatBoost
        predictions = predict_with_catboost(catboost_model, df)

        # Retourner les résultats
        return JSONResponse(content={"predictions": predictions.tolist()})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Démarrage de l'API
if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Utiliser la variable d'environnement PORT ou 8000 par défaut
    uvicorn.run(app, host='0.0.0.0', port=port)  # Écoute sur toutes les interfaces


import os
import gc
import json
import numpy as np
import pandas as pd
import pickle
import joblib
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import sys


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

    

        # Appliquer les étapes de prétraitement
        df = add_features_and_correct_anomaly(df)
        df = apply_label_encoding(df, label_encoders)
        df = apply_one_hot_encoding(df, aligned_columns)
        df = apply_boxcox_transformations(df, lambda_params)
        df = apply_scaler(df, scaler)
        
        # Faire les prédictions
        predictions = predict_with_catboost(catboost_model, df)
        #return predictions
        return JSONResponse(content={"predictions": predictions.tolist()})

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction : {e}")

        # Retourner les résultats
        #return JSONResponse(content={"predictions": predictions.tolist()})
        #return JSONResponse(content={"predictions": forma_ohc})

    except HTTPException as e:
        raise e

    except Exception as e:
        # Si une autre erreur inattendue se produit, la capturer ici
        raise HTTPException(status_code=500, detail=f"Erreur inattendue : {str(e)}")

# Démarrage de l'API
if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Utiliser la variable d'environnement PORT ou 8000 par défaut
    uvicorn.run(app, host='0.0.0.0', port=port)  # Écoute sur toutes les interfaces

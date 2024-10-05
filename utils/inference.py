import subprocess
import sys
import os
import pickle
import joblib
import numpy as np
import pandas as pd
import json
from scipy import stats
from catboost import CatBoostClassifier


def install_requirements(requirements_file='requirements.txt'):
    """
    Installe les dépendances listées dans le fichier requirements.txt.
    
    :param requirements_file: Chemin vers le fichier requirements.txt
    """
    try:
        # Exécuter la commande pip pour installer les dépendances depuis requirements.txt
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print(f"Toutes les dépendances ont été installées à partir de {requirements_file}.")
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'installation des dépendances : {e}")



def load_preprocessing_objects(model_folder):
    try:
        # Charger les encodeurs
        with open(os.path.join(model_folder, 'label_encoders.pkl'), 'rb') as f:
            label_encoders = pickle.load(f)

        print("Label encoders chargés avec succès.")

        # Charger les colonnes One-Hot
        with open(os.path.join(model_folder, 'one_hot_columns.pkl'), 'rb') as f:
            one_hot_columns = pickle.load(f)

        print("Colonnes One-Hot chargées avec succès.")

        # Charger le scaler
        with open(os.path.join(model_folder, 'scaler.pkl'), 'rb') as f:
            scaler = joblib.load(f)

        print("Scaler chargé avec succès.")

        # Charger les paramètres de transformation Box-Cox
        with open(os.path.join(model_folder, 'lambda_params.json'), 'rb') as f:
            lambda_params = json.load(f)

        print("Paramètres de transformation Box-Cox chargés avec succès.")

        # Charger le modèle CatBoost
        catboost_model = CatBoostClassifier()  # ou CatBoostRegressor
        catboost_model.load_model(os.path.join(model_folder, 'best_model_with_weights.cbm'))

        print("Modèle CatBoost chargé avec succès.")

        with open(os.path.join(model_folder, 'aligned_columns.json'), 'rb') as f:
            aligned_columns = json.load(f)

        return label_encoders, one_hot_columns, scaler, lambda_params, catboost_model, aligned_columns

    except FileNotFoundError as fnf_error:
        raise ValueError(f"Erreur lors du chargement des objets de prétraitement : Fichier non trouvé - {fnf_error}")

    except pickle.UnpicklingError as unpickling_error:
        raise ValueError(f"Erreur lors du chargement des objets de prétraitement : Erreur de dé-sérialisation - {unpickling_error}")

    except Exception as e:
        raise ValueError(f"Erreur lors du chargement des objets de prétraitement : {str(e)}")


def add_features_and_correct_anomaly(df):
    df['DAYS_EMPLOYED_ANOM'] = df["DAYS_EMPLOYED"] == 365243
    df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)

    # Ajouter les colonnes dérivées
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    return df



def apply_label_encoding(df, label_encoders):
    """
    Applique les LabelEncoders aux colonnes d'un DataFrame à partir d'un dictionnaire de LabelEncoders.
    Paramètres:
    df : pandas DataFrame
        Le DataFrame contenant les colonnes à encoder.
    label_encoders : dict
        Dictionnaire contenant un LabelEncoder pour chaque colonne encodée (clé = nom de colonne, valeur = LabelEncoder).

    Retourne:
    pandas DataFrame
        Le DataFrame avec les colonnes encodées.
    """
    # Boucle sur chaque colonne et son LabelEncoder dans le dictionnaire
    for column, le in label_encoders.items():
        if column in df.columns:
            # Appliquer le LabelEncoder à la colonne
            df[column] = le.transform(df[column])
        else:
            raise KeyError(f"La colonne {column} n'existe pas dans le DataFrame.")
    
    return df


#ohc


def apply_one_hot_encoding(test_data, aligned_columns):
    """
    Applique le One-Hot Encoding au fichier de test et s'assure que toutes les colonnes présentes dans aligned_columns
    sont également dans test_data. Ajoute les colonnes manquantes avec des zéros si elles n'existent pas.

    :param test_data: DataFrame des données de test
    :param aligned_columns: Liste des colonnes à utiliser pour l'alignement (déjà chargées depuis un JSON)
    :return: DataFrame des données de test transformées et alignées
    """
    # Appliquer le One-Hot Encoding aux données de test
    test_data_encoded = pd.get_dummies(test_data)
    
    # Ajouter les colonnes manquantes avec des zéros
    for col in aligned_columns:
        if col not in test_data_encoded.columns:
            test_data_encoded[col] = 0
    
    # Réordonner les colonnes pour correspondre à celles de aligned_columns
    test_data_encoded = test_data_encoded[aligned_columns]
    
    print(f"Forme des données de test après One-Hot Encoding et alignement : {test_data_encoded.shape}")
    
    return test_data_encoded





def apply_boxcox_transformations(df, lambda_param):
    """
    Applique la transformation Box-Cox aux colonnes spécifiées dans lambda_param avec leurs paramètres respectifs.

    :param df: DataFrame à transformer
    :param lambda_param: Dictionnaire contenant les colonnes et leurs paramètres lambda pour Box-Cox
    :return: DataFrame transformé
    """
    df_transformed = df.copy()  # Créer une copie pour ne pas modifier l'original

    for column, lambda_value in lambda_param.items():
        if column in df_transformed.columns and (df_transformed[column] > 0).all():
            df_transformed[column] = stats.boxcox(df_transformed[column], lmbda=lambda_value)  # Appliquer la transformation

    return df_transformed


def apply_scaler(df, scaler):
    """
    Applique un scaler pré-entraîné a la Df
    
    :param df: DataFrame à transformer
    :param scaler: Scaler pré-entraîné 
    :return: DataFrame transformé
    """
    # Appliquer le scaler sur les colonnes du DataFrame
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
    
    return df_scaled

def predict_with_catboost(model, df):
    """
    Applique un modèle CatBoost pré-entraîné sur un DataFrame prétraité pour faire des prédictions.
    
    :param model: Modèle CatBoost pré-entraîné
    :param df: DataFrame contenant les données déjà prétraitées (scalées, encodées, etc.)
    :return: Les prédictions faites par le modèle sur les données
    """
    # Faire des prédictions en utilisant le modèle CatBoost pré-entraîné
    predictions = model.predict(df)
    
    return predictions
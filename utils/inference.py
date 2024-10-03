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
    with open(os.path.join(model_folder, 'label_encoders.pkl'), 'rb') as f:
        label_encoders = pickle.load(f)
    
    with open(os.path.join(model_folder, 'one_hot_columns.pkl'), 'rb') as f:
        one_hot_columns = pickle.load(f)
    
    with open(os.path.join(model_folder, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    with open(os.path.join(model_folder, 'lambda_params.json'), 'rb') as f:
        lambda_params = pickle.load(f)
    
    with open(os.path.join(model_folder, 'catboost_model.pkl'), 'rb') as f:
        catboost_model = pickle.load(f)

    # Assurez-vous que vous retournez exactement 6 valeurs
    return label_encoders, one_hot_columns, scaler, None, lambda_params, catboost_model

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


def apply_one_hot_encoding(test_data, one_hot_encoded_columns):
    """
    Applique le One-Hot Encoding au fichier de test en s'assurant que les colonnes
    sont alignées avec celles du One-Hot Encoding utilisé lors de l'entraînement.

    :param test_data: DataFrame des données de test
    :param one_hot_encoded_columns: Liste des colonnes après One-Hot Encoding lors de l'entraînement
    :return: DataFrame des données de test transformées
    """
    # Appliquer le One-Hot Encoding aux données de test
    test_data_encoded = pd.get_dummies(test_data)
    
    # Aligner les colonnes du fichier de test avec celles utilisées à l'entraînement
    # Remplir les colonnes manquantes avec des zéros
    test_data_encoded = test_data_encoded.reindex(columns=one_hot_encoded_columns, fill_value=0)
    
    print(f"Forme des données de test après One-Hot Encoding et alignement: {test_data_encoded.shape}")
    
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
        # Vérifier que la colonne existe dans le DataFrame
        if column in df_transformed.columns:
            # Appliquer la transformation Box-Cox (si tous les valeurs de la colonne sont positives)
            if (df_transformed[column] > 0).all():
                df_transformed[column], _ = stats.boxcox(df_transformed[column], lmbda=lambda_value)
            else:
                print(f"Attention : La transformation Box-Cox ne peut être appliquée à {column} car elle contient des valeurs <= 0.")
    
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
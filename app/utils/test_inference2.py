import pandas as pd
import numpy as np
import sys
import os

# Ajoutez le chemin du dossier utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.inference import add_features_and_correct_anomaly  

def test_add_features_and_correct_anomaly():
    # Créer un DataFrame d'exemple avec des données simplifiées pour le test
    data = {
        'DAYS_EMPLOYED': [1000, 365243, 5000],
        'AMT_CREDIT': [200000, 300000, 150000],
        'AMT_INCOME_TOTAL': [50000, 60000, 45000],
        'AMT_ANNUITY': [5000, 10000, 8000],
        'DAYS_BIRTH': [-10000, -15000, -12000]
    }
    df = pd.DataFrame(data)

    # Appeler la fonction à tester
    df_transformed = add_features_and_correct_anomaly(df.copy())

    # Les assertions pour vérifier les transformations
    assert (df_transformed['DAYS_EMPLOYED_ANOM'] == [False, True, False]).all(), "Erreur dans la détection des anomalies sur DAYS_EMPLOYED."
    assert np.isnan(df_transformed.loc[1, 'DAYS_EMPLOYED']), "La valeur 365243 n'a pas été remplacée par NaN dans la colonne DAYS_EMPLOYED."
    assert 'CREDIT_INCOME_PERCENT' in df_transformed.columns, "La colonne CREDIT_INCOME_PERCENT n'a pas été créée."
    assert df_transformed.loc[0, 'CREDIT_INCOME_PERCENT'] == 200000 / 50000, "Erreur dans le calcul de CREDIT_INCOME_PERCENT."
    assert 'ANNUITY_INCOME_PERCENT' in df_transformed.columns, "La colonne ANNUITY_INCOME_PERCENT n'a pas été créée."
    assert df_transformed.loc[0, 'ANNUITY_INCOME_PERCENT'] == 5000 / 50000, "Erreur dans le calcul de ANNUITY_INCOME_PERCENT."
    assert 'CREDIT_TERM' in df_transformed.columns, "La colonne CREDIT_TERM n'a pas été créée."
    assert df_transformed.loc[0, 'CREDIT_TERM'] == 5000 / 200000, "Erreur dans le calcul de CREDIT_TERM."
    assert 'DAYS_EMPLOYED_PERCENT' in df_transformed.columns, "La colonne DAYS_EMPLOYED_PERCENT n'a pas été créée."
    assert df_transformed.loc[0, 'DAYS_EMPLOYED_PERCENT'] == 1000 / -10000, "Erreur dans le calcul de DAYS_EMPLOYED_PERCENT."

if __name__ == "__main__":
    import pytest
    pytest.main()

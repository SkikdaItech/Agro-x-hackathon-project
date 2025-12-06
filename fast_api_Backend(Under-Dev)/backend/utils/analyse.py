import pandas as pd
import numpy as np

df = pd.read_csv("genus_data.csv")   # adjust path

def create_analysis(include_correlation=False):
    analysis = {}

    # 1) STATISTIQUES NUMÉRIQUES
    numeric_df = df.select_dtypes(include=[np.number])
    analysis["numeric_statistics"] = numeric_df.describe().round(3).to_dict()

    # 2) LISTE DES COLONNES DU DATASET
    analysis["columns"] = df.columns.tolist()

    # 3) ANALYSE DES VALEURS MANQUANTES
    analysis["missing_values"] = (
        df.isnull().sum().to_dict()
    )

    # 4) AFFICHAGE STRUCTURÉ DES COLONNES ET TYPES
    analysis["column_types"] = [
        {"column": col, "dtype": str(df[col].dtype)}
        for col in df.columns
    ]

    # 5) HISTOGRAMMES DES VARIABLES NUMÉRIQUES
    histograms = {}
    for col in numeric_df.columns:
        hist, bin_edges = np.histogram(df[col].dropna(), bins=20)
        histograms[col] = {
            "hist": hist.tolist(),
            "bins": bin_edges.tolist()
        }
    analysis["numeric_histograms"] = histograms

    # 6) MATRICE DE CORRÉLATION (PREMIUM VERSION)
    if include_correlation:
        corr = numeric_df.corr().round(3)
        analysis["correlation_matrix"] = corr.to_dict()

    return analysis

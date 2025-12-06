import pandas as pd
import numpy as np
from utils.load_model import model, preprocessor

# Load genus data
df = pd.read_csv("genus_data.csv")
numeric_cols = ['C_value', 'tavg', 'perc_ag', 'perc_per', 'HybProp', 'Hyb_Ratio']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col].fillna(df[col].median(), inplace=True)

# Columns used for the model
feature_names = [
    'C_value_ABS_DIFF',
    'C_value_RATIO',
    'Same_Family',
    'tavg_ABS_DIFF',
    'perc_ag_SUM',
    'perc_per_SUM'
]


def predict_pair_probability(genusA: str, genusB: str) -> float:
    """
    Predict hybridization probability for a pair of genera.
    """
    if genusA not in df['Genus'].values or genusB not in df['Genus'].values:
        raise ValueError("One or both genera not found in dataset.")

    rowA = df[df['Genus'] == genusA].iloc[0]
    rowB = df[df['Genus'] == genusB].iloc[0]

    features = pd.DataFrame([[
        abs(rowA.C_value - rowB.C_value),
        max(rowA.C_value, rowB.C_value)/min(rowA.C_value, rowB.C_value),
        int(rowA.Family == rowB.Family),
        abs(rowA.tavg - rowB.tavg),
        rowA.perc_ag + rowB.perc_ag,
        rowA.perc_per + rowB.perc_per
    ]], columns=feature_names)

    X = preprocessor.transform(features)
    prob = model.predict_proba(X)[0][1] * 100
    return round(prob, 2)

import pandas as pd
import numpy as np
from utils.load_model import model, preprocessor
from .predict import df, feature_names

# Try to import SHAP with fallback
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")


def explain_prediction(genusA: str, genusB: str):
    """
    Explain prediction using SHAP or fallback method.
    """
    if genusA not in df['Genus'].values or genusB not in df['Genus'].values:
        raise ValueError("One or both genera not found in dataset.")

    rowA = df[df['Genus'] == genusA].iloc[0]
    rowB = df[df['Genus'] == genusB].iloc[0]

    # Create features
    features = pd.DataFrame([[
        abs(rowA.C_value - rowB.C_value),
        max(rowA.C_value, rowB.C_value) / min(rowA.C_value, rowB.C_value),
        int(rowA.Family == rowB.Family),
        abs(rowA.tavg - rowB.tavg),
        rowA.perc_ag + rowB.perc_ag,
        rowA.perc_per + rowB.perc_per
    ]], columns=feature_names)

    X = preprocessor.transform(features)

    # Get prediction probability
    prob = model.predict_proba(X)[0][1] * 100

    # Try to generate SHAP explanation
    explanation = generate_shap_explanation(X, model, preprocessor, features)

    # If SHAP fails, use feature importance or coefficients
    if explanation is None:
        explanation = generate_fallback_explanation(model, features, X)

    return {
        "genusA": genusA,
        "genusB": genusB,
        "probability": round(prob, 2),
        "features": {
            "names": feature_names,
            "values": features.values[0].tolist()
        },
        "explanation": explanation
    }


def generate_shap_explanation(X, model, preprocessor, features):
    """Generate SHAP explanation if available."""
    if not SHAP_AVAILABLE:
        return None

    try:
        # Check model type for appropriate explainer
        model_type = type(model).__name__

        if hasattr(model, 'estimator_type'):
            # Scikit-learn model
            if model.estimator_type == 'classifier':
                # For tree-based models
                if hasattr(model, 'feature_importances_'):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X)

                    # Handle different shap_values structures
                    if isinstance(shap_values, list) and len(shap_values) == 2:
                        # Binary classification: get positive class
                        shap_values = shap_values[1]
                    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                        # Multi-dimensional array
                        shap_values = shap_values[0, :, 1]  # First sample, positive class

                    # Flatten to 1D
                    if len(shap_values.shape) == 2:
                        shap_values = shap_values[0]

                    return {
                        "method": "shap_tree",
                        "values": shap_values.tolist(),
                        "feature_importance": model.feature_importances_.tolist()
                    }

                # For linear models
                elif hasattr(model, 'coef_'):
                    explainer = shap.LinearExplainer(model, X)
                    shap_values = explainer.shap_values(X)
                    return {
                        "method": "shap_linear",
                        "values": shap_values.tolist(),
                        "coefficients": model.coef_[0].tolist()
                    }

        # Generic Kernel SHAP as fallback
        explainer = shap.KernelExplainer(model.predict_proba, X)
        shap_values = explainer.shap_values(X, nsamples=50)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class

        return {
            "method": "shap_kernel",
            "values": shap_values.tolist()
        }

    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        return None


def generate_fallback_explanation(model, features, X):
    """Generate explanation without SHAP."""
    try:
        explanation = {
            "method": "fallback",
            "message": "SHAP explanation unavailable, using model coefficients/importance"
        }

        # Try to get feature importance
        if hasattr(model, 'feature_importances_'):
            explanation["feature_importance"] = model.feature_importances_.tolist()
            explanation["sorted_features"] = sorted(
                zip(features.columns, model.feature_importances_),
                key=lambda x: abs(x[1]),
                reverse=True
            )

        # Try to get coefficients
        elif hasattr(model, 'coef_'):
            explanation["coefficients"] = model.coef_[0].tolist()
            explanation["sorted_features"] = sorted(
                zip(features.columns, model.coef_[0]),
                key=lambda x: abs(x[1]),
                reverse=True
            )

        # Try to get prediction breakdown
        elif hasattr(model, 'predict_proba'):
            # Get base prediction
            base_prediction = model.predict_proba(np.zeros_like(X))[0][1]
            current_prediction = model.predict_proba(X)[0][1]

            explanation["base_prediction"] = float(base_prediction)
            explanation["current_prediction"] = float(current_prediction)
            explanation["difference"] = float(current_prediction - base_prediction)

        return explanation

    except Exception as e:
        print(f"Fallback explanation failed: {e}")
        return {
            "method": "error",
            "message": f"Could not generate explanation: {str(e)}"
        }
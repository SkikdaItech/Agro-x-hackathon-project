import joblib
import warnings

# Suppress version mismatch warnings when loading models
# The InconsistentVersionWarning is raised from sklearn.base
warnings.filterwarnings("ignore", category=UserWarning, message=".*Trying to unpickle.*version.*")

model = joblib.load("models/hybrid_model1.pkl")
preprocessor = joblib.load("models/imputer1.pkl")

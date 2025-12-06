from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from utils.analyse import create_analysis
from ml.predict import df, predict_pair_probability, numeric_cols
from ml.explain import explain_prediction
import itertools

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Genus Hybridization API")


# CORS settings
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "*",  # allow all (optional)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # domains allowed
    allow_credentials=True,
    allow_methods=["*"],              # GET, POST, PUT, DELETE...
    allow_headers=["*"],              # Authorization, Content-Type...
)

class GenusInput(BaseModel):
    genusA: str
    genusB: str

@app.post("/predict")
def predict(payload: GenusInput):
    try:
        prob = predict_pair_probability(payload.genusA, payload.genusB)
        return {"hybrid_percentage": prob}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
def explain(payload: GenusInput):
    try:
        explanation = explain_prediction(payload.genusA, payload.genusB)
        return explanation
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/top_pairs")
def top_pairs(n: int = 5):
    """
    Returns top `n` hybridization pairs with highest predicted probability.
    """
    genera = df['Genus'].head(50).unique()
    pairs = list(itertools.combinations(genera, 2))

    results = []
    for genusA, genusB in pairs:
        try:
            prob = predict_pair_probability(genusA, genusB)
            results.append({
                "Genus_A": genusA,
                "Genus_B": genusB,
                "Probability": prob
            })
        except:
            continue  # skip invalid pairs

    results = sorted(results, key=lambda x: x['Probability'], reverse=True)
    return results[:n]

@app.get("/analysis")
def analysis(premium: bool = False):
    try:
        return create_analysis(include_correlation=premium)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from utils.ranking import generate_ranking

@app.get("/ranking")
def ranking(top_n: int = 10):
    try:
        results = generate_ranking(top_n)
        return {"top_pairs": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

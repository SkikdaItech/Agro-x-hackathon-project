# schema/predict_schema.py
from pydantic import BaseModel

class GenusInput(BaseModel):
    genusA: str
    genusB: str

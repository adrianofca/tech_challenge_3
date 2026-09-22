from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"

app = FastAPI(title="Triagem de Laudos Médicos")
model = joblib.load(MODEL_PATH)


class LaudoRequest(BaseModel):
    texto: str


class LaudoResponse(BaseModel):
    classificacao: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=LaudoResponse)
def predict(laudo: LaudoRequest) -> LaudoResponse:
    classificacao = model.predict([laudo.texto])[0]
    return LaudoResponse(classificacao=classificacao)

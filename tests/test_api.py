"""Testes da API FastAPI (/health e /predict)."""

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

NIVEIS = {"normal", "atencao", "urgente"}


@pytest.fixture
def client(tmp_path: Path, modelo_sintetico: Pipeline) -> Iterator[TestClient]:
    """Sobe a API apontando MODEL_PATH para um modelo sintético."""
    caminho = tmp_path / "model.joblib"
    joblib.dump(modelo_sintetico, caminho)

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MODEL_PATH", str(caminho))
        # app.main carrega o modelo na importação; força recarregar com o novo path.
        sys.modules.pop("app.main", None)
        modulo = importlib.import_module("app.main")

    yield TestClient(modulo.app)
    sys.modules.pop("app.main", None)


def test_health_retorna_ok(client: TestClient) -> None:
    resposta = client.get("/health")

    assert resposta.status_code == 200
    assert resposta.json() == {"status": "ok"}


def test_predict_retorna_classificacao_e_score(client: TestClient) -> None:
    resposta = client.post(
        "/predict", json={"texto": "cardiac heart coronary infarction paciente"}
    )

    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["classificacao"] in NIVEIS
    assert 0.0 <= corpo["score"] <= 1.0


def test_predict_classifica_laudo_cardiaco_como_urgente(client: TestClient) -> None:
    resposta = client.post(
        "/predict", json={"texto": "cardiac heart coronary vascular infarction"}
    )

    assert resposta.json()["classificacao"] == "urgente"


@pytest.mark.parametrize("payload", [{}, {"texto": None}, {"texto": 123}])
def test_predict_rejeita_payload_invalido(
    client: TestClient, payload: dict[str, Any]
) -> None:
    resposta = client.post("/predict", json=payload)

    assert resposta.status_code == 422

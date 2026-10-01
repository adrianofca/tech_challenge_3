"""Testes da API FastAPI (/health e /predict)."""

import importlib
import sys

import joblib
import pytest
from fastapi.testclient import TestClient

NIVEIS = {"normal", "atencao", "urgente"}


@pytest.fixture
def client(tmp_path, modelo_sintetico):
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


def test_health_retorna_ok(client):
    resposta = client.get("/health")

    assert resposta.status_code == 200
    assert resposta.json() == {"status": "ok"}


def test_predict_retorna_classificacao_e_score(client):
    resposta = client.post(
        "/predict", json={"texto": "cardiac heart coronary infarction paciente"}
    )

    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["classificacao"] in NIVEIS
    assert 0.0 <= corpo["score"] <= 1.0


def test_predict_classifica_laudo_cardiaco_como_urgente(client):
    resposta = client.post(
        "/predict", json={"texto": "cardiac heart coronary vascular infarction"}
    )

    assert resposta.json()["classificacao"] == "urgente"


@pytest.mark.parametrize("payload", [{}, {"texto": None}, {"texto": 123}])
def test_predict_rejeita_payload_invalido(client, payload):
    resposta = client.post("/predict", json=payload)

    assert resposta.status_code == 422

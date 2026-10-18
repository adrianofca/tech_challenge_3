import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.onnx"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))

app = FastAPI(title="Triagem de Laudos Médicos")
session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name

# Registro próprio (em vez do padrão global): evita colisão de métricas
# duplicadas quando o módulo é recarregado mais de uma vez no mesmo processo
# (como acontece nos testes, que reimportam app.main por teste).
METRICS_REGISTRY = CollectorRegistry()

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total de requisicoes recebidas pela API",
    ["method", "endpoint", "status_code"],
    registry=METRICS_REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Tempo de resposta das requisicoes, em segundos",
    ["method", "endpoint"],
    registry=METRICS_REGISTRY,
)


@app.middleware("http")
async def registrar_metricas(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    # /metrics não entra na contagem para não poluir o painel com o scrape do Prometheus.
    if request.url.path == "/metrics":
        return await call_next(request)

    inicio = time.perf_counter()
    resposta = await call_next(request)
    duracao = time.perf_counter() - inicio

    endpoint = request.url.path
    REQUEST_LATENCY.labels(request.method, endpoint).observe(duracao)
    REQUEST_COUNT.labels(request.method, endpoint, resposta.status_code).inc()
    return resposta


class LaudoRequest(BaseModel):
    texto: str


class LaudoResponse(BaseModel):
    classificacao: str
    score: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(METRICS_REGISTRY), media_type=CONTENT_TYPE_LATEST)


def inferir(texto: str) -> tuple[str, float]:
    entrada = np.asarray([[texto]], dtype=object)
    nomes = [saida.name for saida in session.get_outputs()]
    valores = session.run(None, {INPUT_NAME: entrada})
    saidas = dict(zip(nomes, valores, strict=True))
    classificacao = str(saidas["label"][0])
    score = float(np.max(saidas["probabilities"][0]))
    return classificacao, score


@app.post("/predict", response_model=LaudoResponse)
def predict(laudo: LaudoRequest) -> LaudoResponse:
    classificacao, score = inferir(laudo.texto)
    return LaudoResponse(classificacao=classificacao, score=score)

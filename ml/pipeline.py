"""Etapas do pipeline de treino/retreino (Etapa 2).

A lógica fica aqui, separada da DAG, por dois motivos:

* o CI consegue testar cada etapa com ``pytest`` sem instalar o Airflow;
* a DAG (``dags/treino_laudos_dag.py``) fica só com a orquestração.

Fluxo: ``carregar_dados`` -> ``treinar_modelo`` -> ``salvar_modelo``.

O treino grava primeiro um modelo *candidato*. Só depois de passar pelo
critério mínimo de qualidade (``MIN_ACCURACY``) o candidato substitui o
``model.joblib`` em uso, de modo que um retreino ruim nunca derruba a API.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from ml.train import (
    DATA_DIR,
    LABEL_COLUMN,
    ROOT_DIR,
    TEXT_COLUMN,
    URGENCY_BY_LABEL,
    URGENCY_COLUMN,
    build_pipeline,
    save_model,
)

logger = logging.getLogger(__name__)

MODELS_DIR = ROOT_DIR / "models"

TRAIN_FILE = "medical_tc_train.csv"
TEST_FILE = "medical_tc_test.csv"

CANDIDATE_NAME = "model.candidate.joblib"
MODEL_NAME = "model.joblib"
METRICS_NAME = "metrics.json"

# O enunciado pede datasets com pelo menos 2.000 amostras.
MIN_SAMPLES = 2000
# Baseline atual: ~62% de acurácia. Abaixo disso o retreino é rejeitado.
MIN_ACCURACY = float(os.getenv("MIN_ACCURACY", "0.5"))


def _ler_csv(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    missing = {TEXT_COLUMN, LABEL_COLUMN} - set(dataframe.columns)
    if missing:
        raise ValueError(f"{path.name}: colunas ausentes: {sorted(missing)}")
    dataframe[URGENCY_COLUMN] = dataframe[LABEL_COLUMN].map(URGENCY_BY_LABEL)
    return dataframe


def _validar(dataframe: pd.DataFrame, nome: str, min_samples: int) -> None:
    if len(dataframe) < min_samples:
        raise ValueError(
            f"{nome}: {len(dataframe)} amostras (mínimo exigido: {min_samples})"
        )
    if dataframe[TEXT_COLUMN].isna().any():
        raise ValueError(f"{nome}: existem laudos com texto vazio")
    if dataframe[URGENCY_COLUMN].isna().any():
        raise ValueError(f"{nome}: existem rótulos fora do mapeamento de urgência")


def carregar_dados(
    data_dir: Path = DATA_DIR, min_samples: int = MIN_SAMPLES
) -> dict[str, Any]:
    """Lê os CSVs de treino/teste, valida e devolve um resumo (leve, p/ XCom)."""
    treino = _ler_csv(data_dir / TRAIN_FILE)
    teste = _ler_csv(data_dir / TEST_FILE)

    _validar(treino, "treino", min_samples)
    # O conjunto de teste só precisa ser válido e não estar vazio.
    _validar(teste, "teste", min_samples=1)

    resumo = {
        "train_rows": len(treino),
        "test_rows": len(teste),
        "class_distribution": {
            str(classe): int(qtd)
            for classe, qtd in treino[URGENCY_COLUMN].value_counts().items()
        },
    }
    logger.info("Dados carregados: %s", resumo)
    return resumo


def _calcular_metricas(teste: pd.DataFrame, previsoes: Any) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(teste[URGENCY_COLUMN], previsoes)),
        "f1_macro": float(f1_score(teste[URGENCY_COLUMN], previsoes, average="macro")),
    }


def treinar_modelo(
    data_dir: Path = DATA_DIR, models_dir: Path = MODELS_DIR
) -> dict[str, Any]:
    """Treina o pipeline TF-IDF + Regressão Logística e grava um *candidato*."""
    treino = _ler_csv(data_dir / TRAIN_FILE)
    teste = _ler_csv(data_dir / TEST_FILE)

    pipeline = build_pipeline()
    pipeline.fit(treino[TEXT_COLUMN], treino[URGENCY_COLUMN])
    previsoes = pipeline.predict(teste[TEXT_COLUMN])

    metricas = {
        **_calcular_metricas(teste, previsoes),
        "train_rows": len(treino),
        "test_rows": len(teste),
    }
    candidato = models_dir / CANDIDATE_NAME
    save_model(pipeline, candidato)
    logger.info("Candidato salvo em %s | métricas: %s", candidato, metricas)
    return {**metricas, "candidate_path": str(candidato)}


def _gravar_metricas(models_dir: Path, resultado: dict[str, Any]) -> None:
    metricas = {
        **{k: v for k, v in resultado.items() if k != "candidate_path"},
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (models_dir / METRICS_NAME).write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def salvar_modelo(
    resultado: dict[str, Any],
    models_dir: Path = MODELS_DIR,
    min_accuracy: float = MIN_ACCURACY,
) -> str:
    """Promove o candidato a ``model.joblib`` se atingir a qualidade mínima."""
    candidato = Path(resultado["candidate_path"])
    accuracy = resultado["accuracy"]

    if accuracy < min_accuracy:
        raise ValueError(
            f"Acurácia {accuracy:.3f} abaixo do mínimo {min_accuracy:.3f}: "
            f"modelo em produção mantido (candidato em {candidato})"
        )

    destino = models_dir / MODEL_NAME
    # os.replace é atômico: a API nunca vê um arquivo de modelo pela metade.
    os.replace(candidato, destino)
    _gravar_metricas(models_dir, resultado)

    logger.info("Modelo promovido para %s", destino)
    return str(destino)

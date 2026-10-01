"""DAG de treino/retreino do classificador de urgência de laudos médicos.

Fluxo (Etapa 2):

    carregar_dados  ->  treinar_modelo  ->  salvar_modelo

* ``carregar_dados``: lê os CSVs, valida e devolve um resumo.
* ``treinar_modelo``: treina TF-IDF + Regressão Logística e grava um candidato.
* ``salvar_modelo``: promove o candidato a ``models/model.joblib`` se a
  acurácia atingir o mínimo (``MIN_ACCURACY``).

A lógica de cada etapa está em ``ml/pipeline.py`` (testada no CI sem Airflow).

Execução local rápida, sem scheduler (precisa do Airflow instalado):

    python dags/treino_laudos_dag.py

Variáveis de ambiente opcionais:

* ``TC3_PROJECT_DIR``: raiz do projeto (padrão: pasta acima de ``dags/``).
* ``MIN_ACCURACY``: acurácia mínima para promover o modelo (padrão: 0.5).
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# A pasta ``dags/`` é carregada pelo Airflow isoladamente; garantimos que a raiz
# do projeto esteja no sys.path para que ``import ml`` funcione.
PROJECT_DIR = Path(os.getenv("TC3_PROJECT_DIR", Path(__file__).resolve().parents[1]))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:  # Airflow 3.x
    from airflow.sdk import dag, task
except ImportError:  # Airflow 2.x
    from airflow.decorators import dag, task

from ml import pipeline

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"


@dag(
    dag_id="treino_triagem_laudos",
    description="Retreino semanal do classificador de urgência de laudos médicos",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "tech-challenge-3",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["tech-challenge", "triagem", "treino"],
)
def treino_triagem_laudos():
    @task
    def carregar_dados() -> dict[str, Any]:
        return pipeline.carregar_dados(DATA_DIR)

    @task
    def treinar_modelo(resumo: dict[str, Any]) -> dict[str, Any]:
        logger.info("Treinando com %s amostras", resumo["train_rows"])
        return pipeline.treinar_modelo(DATA_DIR, MODELS_DIR)

    @task
    def salvar_modelo(resultado: dict[str, Any]) -> str:
        return pipeline.salvar_modelo(resultado, MODELS_DIR)

    salvar_modelo(treinar_modelo(carregar_dados()))


dag_object = treino_triagem_laudos()

if __name__ == "__main__":
    dag_object.test()
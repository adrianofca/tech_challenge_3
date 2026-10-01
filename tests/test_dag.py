"""Teste de integridade da DAG do Airflow.

O Airflow é pesado e não faz parte das dependências da API, então estes testes
são pulados (skip) quando ele não está instalado. No CI, o job ``dag`` instala o
Airflow em um ambiente separado e executa este arquivo.
"""

import os
from pathlib import Path

import pytest

DAGS_DIR = Path(__file__).resolve().parent.parent / "dags"
DAG_ID = "treino_triagem_laudos"


@pytest.fixture(scope="module")
def dagbag():
    pytest.importorskip("airflow")
    # Não carregar as DAGs de exemplo do Airflow (a config é lida na importação).
    os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "False")
    try:  # Airflow 3.x
        from airflow.dag_processing.dagbag import DagBag
    except ImportError:  # Airflow 2.x
        from airflow.models import DagBag

    return DagBag(dag_folder=str(DAGS_DIR))


def test_dag_importa_sem_erros(dagbag):
    assert dagbag.import_errors == {}
    assert DAG_ID in dagbag.dags


def test_dag_tem_as_tres_etapas_em_ordem(dagbag):
    dag = dagbag.dags[DAG_ID]

    assert set(dag.task_ids) == {"carregar_dados", "treinar_modelo", "salvar_modelo"}
    assert dag.get_task("carregar_dados").downstream_task_ids == {"treinar_modelo"}
    assert dag.get_task("treinar_modelo").downstream_task_ids == {"salvar_modelo"}
    assert dag.get_task("salvar_modelo").downstream_task_ids == set()


def test_dag_nao_faz_catchup(dagbag):
    assert dagbag.dags[DAG_ID].catchup is False

"""Testes do treino baseline (ml/train.py)."""

import joblib
from sklearn.pipeline import Pipeline

from ml import train

NIVEIS = {"normal", "atencao", "urgente"}


def test_mapeamento_cobre_todas_as_classes_do_dataset():
    assert set(train.URGENCY_BY_LABEL) == {1, 2, 3, 4, 5}
    assert set(train.URGENCY_BY_LABEL.values()) == NIVEIS


def test_load_dataset_adiciona_coluna_de_urgencia(dados_sinteticos, monkeypatch):
    monkeypatch.setattr(train, "DATA_DIR", dados_sinteticos)

    dataframe = train.load_dataset("medical_tc_train.csv")

    assert train.URGENCY_COLUMN in dataframe.columns
    assert dataframe[train.URGENCY_COLUMN].notna().all()
    assert set(dataframe[train.URGENCY_COLUMN]) == NIVEIS


def test_build_pipeline_tem_tfidf_e_classificador():
    pipeline = train.build_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert [nome for nome, _ in pipeline.steps] == ["tfidf", "clf"]


def test_save_model_grava_pipeline_reutilizavel(tmp_path, modelo_sintetico):
    destino = tmp_path / "subpasta" / "model.joblib"

    train.save_model(modelo_sintetico, destino)

    recarregado = joblib.load(destino)
    assert recarregado.predict(["cardiac heart coronary"])[0] == "urgente"

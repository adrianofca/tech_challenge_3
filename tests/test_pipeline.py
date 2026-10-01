"""Testes das etapas do pipeline de treino/retreino (ml/pipeline.py)."""

import json

import joblib
import pytest

from ml import pipeline


def test_carregar_dados_devolve_resumo(dados_sinteticos):
    resumo = pipeline.carregar_dados(dados_sinteticos, min_samples=10)

    assert resumo["train_rows"] == 60
    assert resumo["test_rows"] == 20
    assert set(resumo["class_distribution"]) == {"normal", "atencao", "urgente"}


def test_carregar_dados_rejeita_dataset_pequeno(dados_sinteticos):
    # O padrão do projeto exige 2.000 amostras (requisito do enunciado).
    with pytest.raises(ValueError, match="amostras"):
        pipeline.carregar_dados(dados_sinteticos)


def test_carregar_dados_rejeita_coluna_ausente(dados_sinteticos):
    (dados_sinteticos / "medical_tc_train.csv").write_text("a,b\n1,2\n")

    with pytest.raises(ValueError, match="colunas ausentes"):
        pipeline.carregar_dados(dados_sinteticos, min_samples=1)


def test_carregar_dados_rejeita_rotulo_desconhecido(dados_sinteticos):
    (dados_sinteticos / "medical_tc_train.csv").write_text(
        "condition_label,medical_abstract\n99,texto qualquer\n"
    )

    with pytest.raises(ValueError, match="rótulos"):
        pipeline.carregar_dados(dados_sinteticos, min_samples=1)


def test_treinar_modelo_gera_candidato_e_metricas(dados_sinteticos, tmp_path):
    models_dir = tmp_path / "models"

    resultado = pipeline.treinar_modelo(dados_sinteticos, models_dir)

    assert (models_dir / pipeline.CANDIDATE_NAME).exists()
    assert 0.0 <= resultado["accuracy"] <= 1.0
    assert 0.0 <= resultado["f1_macro"] <= 1.0
    # Treinar não pode tocar no modelo em produção.
    assert not (models_dir / pipeline.MODEL_NAME).exists()


def test_salvar_modelo_promove_candidato(dados_sinteticos, tmp_path):
    models_dir = tmp_path / "models"
    resultado = pipeline.treinar_modelo(dados_sinteticos, models_dir)

    destino = pipeline.salvar_modelo(resultado, models_dir, min_accuracy=0.5)

    assert destino == str(models_dir / pipeline.MODEL_NAME)
    assert not (models_dir / pipeline.CANDIDATE_NAME).exists()
    assert joblib.load(destino).predict(["cardiac heart coronary"])[0] == "urgente"

    metricas = json.loads((models_dir / pipeline.METRICS_NAME).read_text())
    assert metricas["accuracy"] == resultado["accuracy"]
    assert "trained_at" in metricas
    assert "candidate_path" not in metricas


def test_salvar_modelo_rejeita_acuracia_baixa_e_preserva_modelo_atual(
    dados_sinteticos, tmp_path
):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    atual = models_dir / pipeline.MODEL_NAME
    atual.write_bytes(b"modelo-em-producao")
    resultado = pipeline.treinar_modelo(dados_sinteticos, models_dir)
    resultado["accuracy"] = 0.1

    with pytest.raises(ValueError, match="abaixo do mínimo"):
        pipeline.salvar_modelo(resultado, models_dir, min_accuracy=0.5)

    assert atual.read_bytes() == b"modelo-em-producao"

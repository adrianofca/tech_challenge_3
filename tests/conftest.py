"""Fixtures compartilhadas: dados sintéticos pequenos e um modelo de teste.

Os testes NÃO dependem de ``models/model.joblib`` nem dos CSVs reais; assim o
CI roda em segundos e de forma determinística.
"""

from pathlib import Path

import pandas as pd
import pytest

from ml.train import (
    LABEL_COLUMN,
    TEXT_COLUMN,
    URGENCY_BY_LABEL,
    URGENCY_COLUMN,
    build_pipeline,
)

# Vocabulário distinto por classe original (1..5) do Medical Abstracts TC Corpus.
VOCABULARIO = {
    1: "tumor cancer carcinoma oncology neoplasm",
    2: "gastric liver intestinal digestive colon",
    3: "neuron brain cerebral nervous stroke",
    4: "cardiac heart coronary vascular infarction",
    5: "general infection pathology condition inflammation",
}


def gerar_dataframe(linhas_por_classe: int) -> pd.DataFrame:
    registros = [
        {LABEL_COLUMN: rotulo, TEXT_COLUMN: f"{palavras} paciente caso {i}"}
        for rotulo, palavras in VOCABULARIO.items()
        for i in range(linhas_por_classe)
    ]
    return pd.DataFrame(registros)


@pytest.fixture
def dados_sinteticos(tmp_path: Path) -> Path:
    """Cria um diretório ``data`` com CSVs de treino/teste sintéticos."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    gerar_dataframe(12).to_csv(data_dir / "medical_tc_train.csv", index=False)
    gerar_dataframe(4).to_csv(data_dir / "medical_tc_test.csv", index=False)
    return data_dir


@pytest.fixture(scope="session")
def modelo_sintetico():
    """Pipeline real (TF-IDF + LogReg) treinado com dados sintéticos."""
    dataframe = gerar_dataframe(12)
    dataframe[URGENCY_COLUMN] = dataframe[LABEL_COLUMN].map(URGENCY_BY_LABEL)
    pipeline = build_pipeline()
    pipeline.fit(dataframe[TEXT_COLUMN], dataframe[URGENCY_COLUMN])
    return pipeline

"""Treino baseline do classificador de urgência (Etapa 1/2).

Modelo: TF-IDF + Logistic Regression, escolhido por ser leve e ter
conversão limpa para ONNX na Etapa 4 (skl2onnx suporta bem modelos
lineares do sklearn).
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"

TEXT_COLUMN = "medical_abstract"
LABEL_COLUMN = "condition_label"
URGENCY_COLUMN = "urgencia"

# O dataset original classifica por sistema/órgão, não por urgência clínica.
# Mapeamos para 3 níveis pela criticidade típica de cada categoria:
# cardiovascular/nervoso (ex.: infarto, AVC) exigem resposta imediata;
# neoplasias/digestivas variam, mas tipicamente toleram acompanhamento;
# "condições patológicas gerais" é a categoria residual do corpus.
URGENCY_BY_LABEL: dict[int, str] = {
    4: "urgente",  # cardiovascular diseases
    3: "urgente",  # nervous system diseases
    1: "atencao",  # neoplasms
    2: "atencao",  # digestive system diseases
    5: "normal",  # general pathological conditions
}


def load_dataset(filename: str) -> pd.DataFrame:
    dataframe = pd.read_csv(DATA_DIR / filename)
    dataframe[URGENCY_COLUMN] = dataframe[LABEL_COLUMN].map(URGENCY_BY_LABEL)
    return dataframe


def build_pipeline() -> Pipeline:
    vectorizer = TfidfVectorizer(
        max_features=20000, stop_words="english", ngram_range=(1, 2)
    )
    classifier = LogisticRegression(max_iter=1000)
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def evaluate(pipeline: Pipeline, test_dataframe: pd.DataFrame) -> str:
    predictions = pipeline.predict(test_dataframe[TEXT_COLUMN])
    return classification_report(test_dataframe[URGENCY_COLUMN], predictions)


def save_model(pipeline: Pipeline, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, path)


def main() -> None:
    train_dataframe = load_dataset("medical_tc_train.csv")
    test_dataframe = load_dataset("medical_tc_test.csv")

    pipeline = build_pipeline()
    pipeline.fit(train_dataframe[TEXT_COLUMN], train_dataframe[URGENCY_COLUMN])

    print(evaluate(pipeline, test_dataframe))
    save_model(pipeline)
    print(f"Modelo salvo em {MODEL_PATH}")


if __name__ == "__main__":
    main()

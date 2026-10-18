"""Exporta o pipeline scikit-learn treinado para ONNX.

Uso:
    uv run python ml/export_onnx.py
    uv run python ml/export_onnx.py --model models/model.joblib --output models/model.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.pipeline import Pipeline

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT_DIR / "models" / "model.joblib"
DEFAULT_OUTPUT = ROOT_DIR / "models" / "model.onnx"


def _carregar_pipeline(model_path: Path) -> Pipeline:
    """Carrega e valida o pipeline scikit-learn treinado."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Modelo scikit-learn não encontrado: {model_path}. "
            "Execute primeiro: uv run python ml/train.py"
        )

    modelo = joblib.load(model_path)
    if not isinstance(modelo, Pipeline):
        raise TypeError(
            "O modelo esperado deve ser um sklearn.pipeline.Pipeline "
            "(TF-IDF + classificador)."
        )
    return modelo


def _converter_para_onnx(modelo: Pipeline) -> bytes:
    """Converte o pipeline para ONNX (entrada: coluna de strings, uma por laudo).

    zipmap=False mantém as probabilidades como matriz NumPy, facilitando
    inferência e benchmark, sem alterar as classes previstas.
    """
    classificador = modelo.steps[-1][1]
    modelo_onnx = convert_sklearn(
        modelo,
        initial_types=[("texto", StringTensorType([None, 1]))],
        options={id(classificador): {"zipmap": False}},
        target_opset=17,
    )
    return modelo_onnx.SerializeToString()


def exportar_modelo(model_path: Path, output_path: Path) -> Path:
    """Converte um Pipeline de texto sklearn para ONNX e retorna o caminho."""
    modelo = _carregar_pipeline(model_path)
    onnx_bytes = _converter_para_onnx(modelo)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(onnx_bytes)
    print(f"Modelo ONNX salvo em: {output_path}")
    print(f"Tamanho: {output_path.stat().st_size / 1024:.1f} KiB")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Converte o pipeline scikit-learn para ONNX."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Modelo de entrada .joblib (padrão: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Arquivo ONNX de saída (padrão: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    exportar_modelo(args.model, args.output)


if __name__ == "__main__":
    main()

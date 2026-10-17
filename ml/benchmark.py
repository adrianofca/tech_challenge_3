"""Compara latência de inferência scikit-learn versus ONNX Runtime.

Mede a inferência de um laudo por vez (batch=1), incluindo previsão e
probabilidades em ambos os runtimes. Os resultados são medidos em processo,
não incluem a latência HTTP/FastAPI/Docker.

Uso:
    uv run python ml/benchmark.py
    uv run python ml/benchmark.py --iterations 1000 --warmup 50
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SKLEARN_MODEL = ROOT_DIR / "models" / "model.joblib"
DEFAULT_ONNX_MODEL = ROOT_DIR / "models" / "model.onnx"

TEXTOS_BENCHMARK = [
    "patient with cardiac heart coronary infarction and chest pain",
    "patient presents with tumor cancer carcinoma oncology",
    "patient has gastric liver intestinal digestive symptoms",
    "neurological brain neuron cerebral nervous system disorder",
    "general infection pathology condition inflammation",
]


def _inferir_sklearn(modelo, texto: str) -> tuple[str, float]:
    """Executa as mesmas duas operações usadas pela API atual."""
    classe = str(modelo.predict([texto])[0])
    score = float(np.max(modelo.predict_proba([texto])[0]))
    return classe, score


def _inferir_onnx(sessao: ort.InferenceSession, texto: str) -> tuple[str, float]:
    entrada = sessao.get_inputs()[0].name
    dados = np.asarray([[texto]], dtype=object)
    nomes = [saida.name for saida in sessao.get_outputs()]
    valores = sessao.run(None, {entrada: dados})
    saidas = dict(zip(nomes, valores, strict=True))

    # skl2onnx gera normalmente "label" e "probabilities" com zipmap=False.
    rotulos = saidas.get("label", valores[0])
    probabilidades = saidas.get("probabilities")
    classe = str(rotulos[0])
    if probabilidades is None:
        raise RuntimeError(
            "O ONNX não expôs a saída 'probabilities'. "
            "Reexporte usando ml/export_onnx.py."
        )
    score = float(np.max(np.asarray(probabilidades)[0]))
    return classe, score


def _medir(funcao, textos: list[str], warmup: int, iterations: int) -> np.ndarray:
    for indice in range(warmup):
        funcao(textos[indice % len(textos)])

    duracoes = np.empty(iterations, dtype=np.float64)
    for indice in range(iterations):
        texto = textos[indice % len(textos)]
        inicio = time.perf_counter_ns()
        funcao(texto)
        duracoes[indice] = (time.perf_counter_ns() - inicio) / 1_000_000
    return duracoes


def _resumo(nome: str, duracoes: np.ndarray) -> dict[str, float | str]:
    return {
        "runtime": nome,
        "mean_ms": float(np.mean(duracoes)),
        "median_ms": float(np.percentile(duracoes, 50)),
        "p95_ms": float(np.percentile(duracoes, 95)),
        "min_ms": float(np.min(duracoes)),
        "max_ms": float(np.max(duracoes)),
    }


def benchmark(
    sklearn_path: Path = DEFAULT_SKLEARN_MODEL,
    onnx_path: Path = DEFAULT_ONNX_MODEL,
    iterations: int = 500,
    warmup: int = 30,
) -> dict[str, dict[str, float | str]]:
    if iterations < 1:
        raise ValueError("--iterations precisa ser maior que zero.")
    if warmup < 0:
        raise ValueError("--warmup não pode ser negativo.")
    if not sklearn_path.is_file():
        raise FileNotFoundError(
            f"Modelo sklearn não encontrado: {sklearn_path}. "
            "Execute uv run python ml/train.py."
        )
    if not onnx_path.is_file():
        raise FileNotFoundError(
            f"Modelo ONNX não encontrado: {onnx_path}. "
            "Execute uv run python ml/export_onnx.py."
        )

    modelo = joblib.load(sklearn_path)
    sessao = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )

    # Verifica equivalência funcional antes de comparar tempos.
    divergencias = []
    for texto in TEXTOS_BENCHMARK:
        classe_sk, score_sk = _inferir_sklearn(modelo, texto)
        classe_onnx, score_onnx = _inferir_onnx(sessao, texto)
        if classe_sk != classe_onnx or not np.isclose(
            score_sk, score_onnx, rtol=1e-4, atol=1e-5
        ):
            divergencias.append(
                (texto, classe_sk, classe_onnx, score_sk, score_onnx)
            )
    if divergencias:
        raise RuntimeError(
            "As saídas sklearn e ONNX divergem além da tolerância: "
            f"{divergencias[:2]}"
        )

    sklearn_ms = _medir(
        lambda texto: _inferir_sklearn(modelo, texto),
        TEXTOS_BENCHMARK, warmup, iterations,
    )
    onnx_ms = _medir(
        lambda texto: _inferir_onnx(sessao, texto),
        TEXTOS_BENCHMARK, warmup, iterations,
    )

    resultados = {
        "sklearn": _resumo("scikit-learn", sklearn_ms),
        "onnx": _resumo("ONNX Runtime", onnx_ms),
    }
    media_sk = float(np.mean(sklearn_ms))
    media_onnx = float(np.mean(onnx_ms))
    speedup = media_sk / media_onnx if media_onnx > 0 else float("inf")
    ganho = (1 - media_onnx / media_sk) * 100 if media_sk > 0 else 0.0

    print("\nComparação de latência — inferência local, batch=1")
    print(f"Iterações por runtime: {iterations} (aquecimento: {warmup})")
    print(f"{'Métrica':<18}{'scikit-learn (ms)':>20}{'ONNX Runtime (ms)':>20}")
    for chave, rotulo in [
        ("mean_ms", "Média"),
        ("median_ms", "Mediana / p50"),
        ("p95_ms", "p95"),
        ("min_ms", "Mínimo"),
        ("max_ms", "Máximo"),
    ]:
        print(
            f"{rotulo:<18}{resultados['sklearn'][chave]:>20.4f}"
            f"{resultados['onnx'][chave]:>20.4f}"
        )
    print(f"\nSpeedup (média): {speedup:.2f}x")
    print(f"Redução da latência média: {ganho:.1f}%")
    print("Validação: classes e scores equivalentes dentro da tolerância.")
    return resultados


def main() -> None:
    parser = argparse.ArgumentParser(   
        description="Compara latência de scikit-learn e ONNX Runtime."
    )
    parser.add_argument("--sklearn-model", type=Path, default=DEFAULT_SKLEARN_MODEL)
    parser.add_argument("--onnx-model", type=Path, default=DEFAULT_ONNX_MODEL)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=30)
    args = parser.parse_args()
    benchmark(args.sklearn_model, args.onnx_model, args.iterations, args.warmup)


if __name__ == "__main__":
    main()

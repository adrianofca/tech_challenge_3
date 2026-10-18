FROM python:3.11-slim

# O ONNX Runtime usa um normalizador de texto (StringNormalizer, parte do
# TfIdfVectorizer) que exige uma locale instalada no SO — a imagem slim não
# vem com nenhuma configurada, e sem isso a API falha ao carregar o modelo.
RUN apt-get update \
    && apt-get install -y --no-install-recommends locales \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen \
    && rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app ./app
COPY models ./models

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Tech Challenge Fase 3 — Triagem Automática de Laudos Médicos

Sistema de triagem automática de laudos médicos (texto) que classifica a urgência do caso em **normal**, **atenção** ou **urgente**, servido via API REST em container Docker, com pipeline de CI/CD, orquestração de retreino via Airflow, monitoramento com Prometheus + Grafana e otimização de latência via ONNX Runtime.

## Decisão de Arquitetura

### Batch vs. Real-time

Optamos por **inferência em tempo real (real-time)**, não batch.

O caso de uso é triagem de urgência clínica: um laudo médico precisa ser classificado no momento em que chega, porque o próprio objetivo do sistema é reduzir o tempo até a decisão de prioridade. Se um caso classificado como "urgente" ficasse esperando em uma janela de processamento em lote (ex.: de hora em hora), o atraso introduzido pelo próprio sistema de triagem contradiria o problema que ele deveria resolver. Batch processing faz sentido para decisões que toleram espera (ex.: relatórios agregados, retreino do modelo), não para uma resposta que dispara uma ação clínica imediata.

Essa escolha também é coerente com o restante dos requisitos do desafio: instrumentação de latência via `prometheus_client`, dashboards de tempo de resposta no Grafana e otimização de latência com ONNX só fazem sentido em um fluxo síncrono de baixa latência — não haveria motivo para medir e otimizar latência de milissegundos em um pipeline batch.

### Provedor de Nuvem: AWS

O grupo optou por **AWS**, com acesso disponibilizado pela FIAP via **AWS Academy Learner Lab**.

Isso traz duas restrições típicas desse ambiente, que influenciam a escolha dos serviços:

- **Não é possível criar novas IAM roles/policies** — apenas reutilizar a `LabRole`/`LabInstanceProfile` já provisionada na conta.
- **Catálogo de serviços restrito** — serviços básicos como EC2, S3, Lambda, RDS, DynamoDB e CloudWatch estão disponíveis; serviços gerenciados mais complexos (ex.: ECS Fargate) costumam esbarrar na restrição de IAM, pois exigem a criação de uma *task execution role* customizada.

**Serviço escolhido: EC2 (instância única) rodando o `docker-compose.yml` da stack completa** (API + Prometheus + Grafana), usando a `LabRole` existente. Essa abordagem funciona integralmente dentro das restrições do Learner Lab, sem exigir permissões de IAM que a conta não concede, e reaproveita exatamente o mesmo `docker-compose.yml` usado em desenvolvimento local — sem necessidade de reescrever a orquestração para um serviço gerenciado.

Alternativas descartadas:
- **ECS Fargate**: mais próximo de uma arquitetura de produção "ideal", mas inviável no Learner Lab pela restrição de criação de IAM role.
- **Batch (ex.: SageMaker Batch Transform)**: descartado pelo motivo já explicado — a natureza clínica do problema exige resposta imediata.

## Modelo

Baseline: **TF-IDF + Logistic Regression** (scikit-learn), escolhido por ser leve e ter conversão limpa para ONNX Runtime na Etapa 4.

O dataset ([Medical Abstracts TC Corpus](https://github.com/sebischair/Medical-Abstracts-TC-Corpus)) classifica por sistema/órgão, não por urgência clínica. Como não existe um dataset público de triagem de urgência pronto para uso, mapeamos as 5 classes originais para os 3 níveis de urgência do desafio, pela criticidade típica de cada categoria:

| Classe original | Nível de urgência | Motivo |
|---|---|---|
| Doenças cardiovasculares | **urgente** | Ex.: infarto — tempo é crítico |
| Doenças do sistema nervoso | **urgente** | Ex.: AVC, convulsões — tempo é crítico |
| Neoplasias | **atenção** | Requer acompanhamento próximo, raramente emergência imediata |
| Doenças digestivas | **atenção** | Amplo espectro de gravidade, tratado como "precisa de atenção" por padrão |
| Condições patológicas gerais | **normal** | Categoria residual/menos específica do dataset |

Acurácia do baseline no conjunto de teste: **62%** (ver `ml/train.py`, que imprime o `classification_report` completo ao treinar).

## Execução

### Pré-requisitos

- [uv](https://docs.astral.sh/uv/) (gerencia Python 3.11 e as dependências automaticamente, sem instalação manual)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) com WSL2 habilitado (Windows)

### Passo a passo

```bash
# 1. Instalar dependências (uv baixa o Python 3.11 se necessário)
uv sync

# 2. Treinar o modelo baseline (necessário: models/ não é versionado)
uv run python ml/train.py

# 3. Build da imagem Docker
docker build -t triagem-api:baseline .

# 4. Subir o container
docker run -d --name triagem-api -p 8000:8000 triagem-api:baseline
```

### Testando a API

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texto": "paciente apresenta dor toracica intensa e falta de ar subita"}'
```

Resposta esperada:
```json
{ "classificacao": "urgente", "score": 0.34 }
```

## Latência

Baseline medido localmente, com a API rodando em container Docker, 50 requisições ao `/predict` (após 1 chamada de aquecimento), medidas do lado do cliente:

| Métrica | Valor |
|---|---|
| Mínimo | 2,35 ms |
| Média | 2,90 ms |
| p50 | 2,59 ms |
| p95 | 3,36 ms |
| Máximo | 11,31 ms |

Esse número serve de referência para a comparação com o modelo otimizado via ONNX Runtime na Etapa 4.

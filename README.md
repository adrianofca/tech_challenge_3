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

## Execução

*(seção a ser preenchida conforme as próximas etapas)*

## Latência

*(baseline vs. modelo otimizado — Etapa 4)*

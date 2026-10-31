# Tech Challenge — Fase 2 · Projeto 1: otimização de modelos de diagnóstico

Continuação da Fase 1 (classificação de câncer de mama com o dataset Breast Cancer Wisconsin).
Nesta fase os modelos da Fase 1 — **Regressão Logística** e **Random Forest** — têm seus
hiperparâmetros otimizados por um **algoritmo genético implementado do zero**, e uma **LLM
pré-treinada** explica os resultados em linguagem natural para médicos.

> Projeto educacional de apoio à triagem. Não substitui exame clínico, exames de imagem,
> biópsia ou a decisão de profissionais de saúde.

**Integrantes:** Gabriel Wesley (RM376899) · Gustavo Leite (RM375733) · Luiz Fellipe (RM376854)

## Resultados (teste hold-out da Fase 1 — 114 pacientes, 42 malignas)

| Modelo | Versão | Recall | F1 | ROC-AUC | Falsos negativos | Falsos positivos | Fitness CV 5×3 |
|---|---|---|---|---|---|---|---|
| Regressão Logística | Original (Fase 1) | 95,2% | 0,964 | 0,995 | 2 | 1 | 0,9659 ± 0,024 |
| Regressão Logística | **AG — E1 (selecionado)** | **97,6%** | 0,965 | 0,997 | **1** | 2 | 0,9703 ± 0,018 |
| Random Forest | Original (Fase 1) | 92,9% | 0,963 | 0,997 | 3 | 0 | 0,9516 ± 0,030 |
| Random Forest | AG — E4 | 95,2% | 0,976 | 0,998 | 2 | 0 | 0,9613 ± 0,026 |

O AG melhorou os dois modelos, principalmente ajustando o limiar de decisão (0,26 na Regressão
Logística, em vez de 0,5) e a regularização. O ganho é consistente na direção, mas modesto: o
modelo original já tinha ROC-AUC ≈ 0,99 e cada caso maligno no teste vale 2,4 pontos de recall.
Detalhes, 5 experimentos por modelo e a comparação com busca aleatória em
[`reports/relatorio_tecnico_fase2.pdf`](reports/relatorio_tecnico_fase2.pdf).

![Convergência](reports/figures/convergencia_ag.png)

---

## Onde está cada requisito

| Requisito do enunciado | Onde |
|---|---|
| Codificação dos hiperparâmetros em genes | `src/ga/search_space.py`, `src/spaces.py` |
| Seleção, cruzamento e mutação | `src/ga/operators.py` (torneio, roleta, ranking · uniforme, um ponto, BLX-α · gaussiana, uniforme) |
| Função fitness com métricas (recall, F1, AUC) | `src/fitness.py` |
| Motor do AG (elitismo, cache, parada antecipada) | `src/ga/engine.py` |
| Comparação original × otimizado | `src/experiments.py` → `reports/results/experiments.json`, figuras em `reports/figures/` |
| ≥ 3 experimentos com configurações diferentes | `configs/experiments.yaml` (5 experimentos + busca aleatória) |
| Escalabilidade automática | `docker-compose.yml` (réplicas + nginx), `deploy/k8s/api.yaml` (HPA 2–10 réplicas), `deploy/k8s/ga-job.yaml` |
| Monitoramento e logging | `src/monitoring.py` (logs JSON, RunTracker/MLflow), `src/api/metrics.py` (Prometheus), Grafana |
| Arquitetura e decisões documentadas | `docs/arquitetura.md`, relatório técnico |
| LLM pré-treinada (LLaMA, GPT, Claude…) | `src/llm/providers.py` |
| Explicações em linguagem natural / insights para médicos | `src/llm/interpreter.py`, `src/explain.py`, rotas `/interpret`, `/report/performance`, `/ask` |
| Prompt engineering | `src/llm/prompts.py` (3 estratégias: básico, estruturado, few-shot) |
| Avaliação da qualidade das interpretações | `src/llm/evaluation.py`, `scripts/evaluate_llm.py` |
| Ambiente virtual | `pyproject.toml` (Poetry) ou `requirements.txt` (venv) |
| Testes automatizados | `tests/` (pytest) |
| Notebook/scripts de demonstração | `notebooks/02_demo_fase2.ipynb`, `scripts/demo.py` |
| Relatório técnico | `reports/relatorio_tecnico_fase2.pdf` (e `.md`) |
| Roteiro do vídeo | `reports/roteiro_video_fase2.md` |

## Estrutura

```text
tech-challenge-fase2/
├── configs/experiments.yaml     # configurações do AG e da fitness
├── data/data.csv                # mesmo dataset da Fase 1
├── deploy/                      # nginx, Prometheus, Grafana, Kubernetes (HPA + Job)
├── docs/arquitetura.md          # arquitetura e decisões
├── frontend/                    # aplicação web Angular (AG + LLM na tela)
├── models/                      # best_model.joblib + model_card.json (gerados)
├── notebooks/02_demo_fase2.ipynb
├── reports/
│   ├── figures/                 # gráficos do relatório
│   ├── results/                 # experiments.json, llm_evaluation_*.json
│   ├── runs/                    # rastreamento por execução do AG
│   ├── relatorio_tecnico_fase2.pdf
│   └── roteiro_video_fase2.md
├── scripts/                     # run_experiments, evaluate_llm, demo, make_figures, generate_report
├── src/
│   ├── ga/                      # algoritmo genético genérico
│   ├── llm/                     # provedores, prompts, interpretação, avaliação
│   ├── api/                     # FastAPI + métricas Prometheus
│   ├── data.py  models.py  spaces.py  fitness.py  experiments.py  explain.py  monitoring.py
├── tests/
├── Dockerfile  docker-compose.yml  pyproject.toml  requirements.txt  .env.example
```

## Como executar

```bash
cd tech-challenge-fase2
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # ou: poetry install
```

### 1. Algoritmo genético

```bash
python -m scripts.run_experiments                      # todos (≈ 40 min em 2 CPUs; o Random Forest é o mais lento)
python -m scripts.run_experiments --models logistic_regression --experiments E1_pop_pequena
python -m scripts.run_experiments --quick              # versão mínima para testar o fluxo
```

Gera `reports/results/experiments.json`, `reports/runs/<execução>/` e o melhor modelo em `models/`.

### 2. LLM

Copie `.env.example` para `.env` e escolha um provedor:

| Provedor | Modelo padrão | Como obter |
|---|---|---|
| `groq` (recomendado) | GPT-OSS 120B (`openai/gpt-oss-120b`) | chave gratuita em console.groq.com → `GROQ_API_KEY=...` |
| `ollama` | LLaMA 3.1 8B (local) | instale o Ollama e rode `ollama pull llama3.1:8b` |
| `openai` | GPT-4o mini | `OPENAI_API_KEY=...` |
| `anthropic` | Claude Haiku | `ANTHROPIC_API_KEY=...` |
| `offline` | template | sem chave — usado em testes e como fallback |

```bash
python -m scripts.demo --provider groq                       # demonstração ponta a ponta
python -m scripts.evaluate_llm --provider groq --judge groq  # avaliação das 3 estratégias de prompt
```

- O `.env` é lido automaticamente pelos scripts, pela API e pelo notebook. Variáveis já exportadas no terminal têm prioridade.
- Sem a chave do provedor escolhido, o erro aparece na hora (antes caía em silêncio no template).
- Limite de taxa (429) e erros temporários são repetidos até `LLM_MAX_RETRIES` vezes, respeitando o `Retry-After` da API — no plano gratuito do Groq a avaliação completa leva alguns minutos por causa do limite de tokens por minuto.
- Se mesmo assim alguma chamada falhar, `evaluate_llm` salva em `reports/results/incompleto/` e termina com erro: o relatório nunca mistura respostas reais com falhas de API.

### 3. Relatório

```bash
python -m scripts.make_figures && python -m scripts.generate_report
```

Todos os números do relatório vêm dos JSON de resultados — basta rodar de novo após novos experimentos.

### 4. API

```bash
uvicorn src.api.main:app --port 8000      # documentação em http://localhost:8000/docs
```

| Rota | Descrição |
|---|---|
| `POST /predict` | Probabilidade de malignidade e orientação (lote de amostras) |
| `POST /interpret` | Explicação em linguagem natural (`estrategia`: basico, estruturado, few_shot) |
| `POST /ask` | Pergunta livre sobre um caso |
| `GET /report/performance` | Insights "original × otimizado" para a equipe clínica |
| `GET /experiments` | Tabela de resultados do AG |
| `POST /optimize` · `GET /optimize/{id}` | Roda o AG em segundo plano e acompanha o progresso |
| `GET /model` · `/health` · `/ready` · `/metrics` | Modelo em produção, sondas e métricas Prometheus |
| `GET /samples` | Pacientes do teste com a previsão (erros e casos perto do limiar primeiro; filtro `erros`, `limiar`, `malignos`, `benignos`) |
| `POST /llm/inspect` · `POST /llm/compare` | Tudo o que entra e sai da LLM (prompts, contexto, resposta bruta, notas, juiz, tokens); as 3 estratégias em sequência |
| `GET /llm/evaluations` · `GET /llm/evaluations/{provider}` | Avaliações salvas da LLM |
| `GET /experiments/history` | Convergência por geração e curva da busca aleatória |

### 5. Aplicação web

Interface Angular em [`frontend/`](frontend/README.md), no mesmo estilo visual do frontend da Fase 1. Dois terminais:

```bash
uvicorn src.api.main:app --port 8000          # nesta pasta (tech-challenge-fase2/)

cd frontend
npm install --legacy-peer-deps                # só na primeira vez
npm start                                     # http://localhost:4201
```

| Tela | O que mostra |
|---|---|
| Visão geral | Modelo em produção e original × otimizado no teste |
| Algoritmo genético | Experimentos, convergência × busca aleatória, histórico por geração e o AG rodando ao vivo |
| Paciente + LLM | A jornada do médico: escolher a paciente, ver a previsão, pedir a explicação e fazer uma pergunta livre |
| Inspetor da LLM | Prompt de sistema e do usuário, contexto, resposta bruta, JSON, notas automáticas, juiz, latência e tokens; as 3 estratégias lado a lado |
| Avaliação da LLM | A avaliação salva em `reports/results/` (médias, gráficos, tabela por caso, exemplos, falhas) |
| Relatório clínico | Relatório "original × otimizado" para a equipe clínica |

- **Seletor de provedor** no topo: `offline` (padrão: template determinístico, sem chave nem cota) ou `groq`
  (LLM real). Para navegar ou gravar sem gastar a cota, deixe em `offline`.
- A LLM só é chamada quando alguém clica num botão. As respostas ficam em memória, então voltar a uma tela não gasta a
  cota (o Groq gratuito tem limite por minuto e 1.000 requisições por dia).
- Toda resposta mostra um selo com provedor e modelo. Se a LLM real falhar, aparece o alerta "template offline" com o
  motivo.
- A chave fica só no `.env` da API; o navegador nunca a recebe.

### 6. Escalabilidade e monitoramento

```bash
docker compose up --build --scale api=3          # API em :8080, Prometheus em :9090, Grafana em :3000
kubectl apply -f deploy/k8s/api.yaml             # Kubernetes com HPA (2–10 réplicas)
kubectl apply -f deploy/k8s/ga-job.yaml          # AG como Job em lote
```

### 7. Testes

```bash
pytest
```

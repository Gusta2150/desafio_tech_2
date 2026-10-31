# Frontend da Fase 2: algoritmo genético e LLM na tela (Angular 16)

Interface web que consome a API da Fase 2 (`src/api/main.py`). Mostra o algoritmo genético, a jornada do médico
com a LLM e tudo o que a LLM recebe e devolve, para conferência. Foi criado a partir de uma cópia do frontend da
Fase 1 e usa os mesmos tokens visuais (`styles.scss`), o mesmo tema dos gráficos (`core/chart-theme.ts`) e os
mesmos componentes de topo e de aviso clínico.

Angular 16, componentes **standalone**, gráficos com **ng2-charts/Chart.js**.

## Como rodar

Pré-requisito: a API rodando em `http://localhost:8000` (ver `src/environments/environment.ts`).

```bash
# na raiz do projeto
uvicorn src.api.main:app --port 8000

# em outro terminal, na pasta frontend/
npm install --legacy-peer-deps   # ng2-charts declara uma peer dep que conflita com Angular 16
npm start                        # porta 4201, para não conflitar com a Fase 1 (4200)
```

Abra `http://localhost:4201`.

## Telas

| Rota | O que mostra | Rotas da API |
|---|---|---|
| `/visao-geral` | Modelo em produção; original × otimizado no teste | `/model` |
| `/algoritmo-genetico` | Experimentos, convergência × busca aleatória, histórico por geração e o AG rodando ao vivo | `/experiments`, `/experiments/history`, `/optimize` |
| `/paciente` | Jornada do médico: paciente, previsão, explicação e pergunta livre | `/samples`, `/predict`, `/interpret`, `/ask` |
| `/inspetor` | Prompts, contexto, resposta bruta, JSON, notas, juiz, tokens; 3 estratégias lado a lado | `/llm/inspect`, `/llm/compare` |
| `/avaliacao` | Avaliação salva da LLM: médias, gráficos, tabela por caso e exemplos | `/llm/evaluations` |
| `/relatorio` | Relatório "original × otimizado" para a equipe clínica | `/report/performance` |

## Provedor da LLM e cota

- O seletor no topo escolhe `offline` (template determinístico, sem chave nem cota) ou `groq` (LLM real).
  O padrão é `offline`.
- A LLM só é chamada quando alguém clica num botão, nunca ao abrir uma tela.
- As respostas ficam em memória por paciente, estratégia e provedor. Voltar a uma tela pelo menu não gasta a cota.
  Recarregar a página (F5) limpa essa memória.
- A chave do Groq fica só no `.env` da API e nunca chega ao navegador.

## Estrutura

```
src/app/
├── core/       # api.service (HTTP, timeout de 120 s, cache), models, chart-theme
├── shared/     # top-bar, clinical-disclaimer, llm-badge, llm-loading, patient-picker
└── pages/      # overview, genetic, patient, inspector, evaluation, report
```

## Build de produção

```bash
npm run build
```

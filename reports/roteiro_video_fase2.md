# Roteiro do vídeo — Tech Challenge Fase 2 (até 15 minutos)

Sugestão de divisão entre os 3 integrantes. Antes de gravar: rode `python -m scripts.run_experiments`
(ou use os resultados já salvos), configure a LLM no `.env` e deixe a API aberta em `http://localhost:8000/docs`.
Suba também a aplicação web (`cd frontend && npm start`, em `http://localhost:4201`): várias partes ficam mais claras
nas telas do que no Swagger. As telas indicadas abaixo são **sugestões** e substituem trechos, sem aumentar o tempo.

| Tempo | Quem | O que mostrar | O que falar |
|---|---|---|---|
| 0:00–1:00 | Integrante 1 | README da Fase 2 | Apresentação do grupo; recapitular a Fase 1 (RL e RF, dataset Wisconsin, recall como métrica principal) e o objetivo da Fase 2 (Projeto 1). |
| 1:00–2:00 | Integrante 1 | `reports/figures/arquitetura.png` | Visão geral: otimização em lote (AG) → modelo salvo → API → LLM; monitoramento e autoscaling. |
| 2:00–4:30 | Integrante 2 | `src/ga/search_space.py`, `src/spaces.py`, `src/ga/operators.py` | Codificação real em [0,1] com genes tipados (log para C, inteiros, categóricos); limiar como gene e por quê; operadores de seleção (torneio, roleta, ranking), crossover (uniforme, um ponto, BLX-α) e mutação (gaussiana, uniforme); elitismo. |
| 4:30–5:30 | Integrante 2 | `src/fitness.py` | Fitness = 0,45·Recall + 0,35·F1 + 0,20·AUC em CV 5-fold só no treino; por que F1 evita "tudo maligno"; cache e reaproveitamento quando só o limiar muda. |
| 5:30–7:30 | Integrante 2 | **App → Algoritmo genético → "Rodar o AG ao vivo"** (Regressão Logística, população 12, 8 gerações: segundos). Alternativa: terminal com `python -m scripts.run_experiments --models logistic_regression --experiments E1_pop_pequena` | Mostrar a barra de progresso e o gráfico crescendo geração a geração (melhor, média, diversidade). Abrir `configs/experiments.yaml` e explicar as 5 configurações. |
| 7:30–9:30 | Integrante 3 | **App → Algoritmo genético** (tabela, convergência × busca aleatória, "Por geração") e **Visão geral** (original × otimizado). Figuras de apoio: `validacao_robusta.png`, `erros_teste.png` | Resultados: modelo original × otimizado (recall, falsos negativos, falsos positivos), diferenças entre experimentos, validação robusta e comparação com busca aleatória no mesmo orçamento. Ser honesto sobre o tamanho do ganho (teste pequeno). |
| 9:30–12:30 | Integrante 3 | **App → Paciente + LLM** (filtro "Erros do modelo", Explicar, Pergunta livre) e **Inspetor da LLM** (seções do prompt e do contexto; "Comparar as 3 estratégias") | A jornada do médico; grounding (contexto com probabilidade, limiar e fatores locais), regras do prompt, saída JSON e o selo de fallback. Mostrar o básico × few-shot lado a lado na mesma paciente. Gravar com `groq` só os cliques que forem ao ar; ensaiar com `offline`. |
| 12:30–13:30 | Integrante 1 | **App → Avaliação da LLM** (execução groq · openai/gpt-oss-120b) e **Relatório clínico** | Critérios de avaliação (formato, fidelidade numérica, cobertura, coerência, segurança, legibilidade, LLM juiz), resultados por estratégia e zero falhas. O relatório para a equipe clínica aparece já salvo, sem nova chamada. |
| 13:30–14:30 | Integrante 1 | `GET /metrics`, `docker-compose.yml`, `deploy/k8s/api.yaml`, `pytest` rodando | Monitoramento (Prometheus/Grafana, logs JSON com request_id), autoscaling (HPA 2–10 réplicas, Job do AG) e testes automatizados. |
| 14:30–15:00 | Todos | — | Limitações (dados públicos, validação clínica) e próximos passos para o Módulo 3. |

## Comandos úteis durante a gravação

```bash
python -m scripts.demo --provider groq          # demonstração ponta a ponta no terminal
uvicorn src.api.main:app --port 8000            # API
cd frontend && npm start                        # aplicação web em http://localhost:4201
pytest -q                                       # testes
```

Exemplo de corpo para o Swagger (`/interpret`): use `python -c "from src.data import load_split; import json; print(json.dumps(load_split().X_test.iloc[0].to_dict()))"` para gerar uma amostra.

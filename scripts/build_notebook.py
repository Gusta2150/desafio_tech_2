"""Cria notebooks/02_demo_fase2.ipynb — o roteiro da apresentação, passo a passo.

    python -m scripts.build_notebook
    jupyter nbconvert --to notebook --execute --inplace notebooks/02_demo_fase2.ipynb

Cada seção mostra a chamada e o retorno. A LLM usa o provedor da variável PROVIDER
(célula 0.1): ``groq`` se houver chave no .env, senão ``offline``. Com groq, o notebook
inteiro faz cerca de 7 chamadas à LLM.
"""

import nbformat as nbf

from src.data import PROJECT_DIR

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    # ------------------------------------------------------------------ 0
    md("# Tech Challenge · Fase 2 · Projeto 1\n"
       "## Otimização dos modelos de diagnóstico com algoritmo genético + interpretação com LLM\n\n"
       "Gabriel Wesley (RM376899) · Gustavo Leite (RM375733) · Luiz Fellipe (RM376854)\n\n"
       "Roteiro deste notebook:\n\n"
       "1. Dados e ponto de partida (modelos da Fase 1)\n"
       "2. Algoritmo genético: codificação, operadores, fitness, execução ao vivo\n"
       "3. Experimentos e comparação original × otimizado\n"
       "4. LLM: contexto, prompts, resposta, avaliação, fallback\n"
       "5. A API: chamadas e retornos\n"
       "6. Escalabilidade, monitoramento e testes\n\n"
       "> Projeto educacional de apoio à triagem. Não confirma nem descarta câncer e não substitui a avaliação "
       "médica."),
    md("### 0.1 Ambiente"),
    code("import os, sys, json, warnings\n"
         "from pathlib import Path\n"
         "# raiz do projeto: guardada na 1ª execução, para a célula poder rodar de novo sem mudar de pasta\n"
         "ROOT = globals().get('ROOT') or next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'src').is_dir())\n"
         "os.chdir(ROOT)\n"
         "if str(ROOT) not in sys.path:\n"
         "    sys.path.insert(0, str(ROOT))\n"
         "warnings.filterwarnings('ignore')\n"
         "import numpy as np\n"
         "import pandas as pd\n"
         "import matplotlib.pyplot as plt\n"
         "from IPython.display import Image, Markdown, display\n"
         "pd.set_option('display.width', 180)\n"
         "pd.set_option('display.max_colwidth', 120)\n\n"
         "from src.config import load_env\n"
         "load_env()  # lê o .env (chave do Groq); variáveis já exportadas têm prioridade\n\n"
         "# 'groq' = LLM real (gasta cota); 'offline' = template determinístico, sem chave\n"
         "PROVIDER = 'groq' if (os.getenv('GROQ_API_KEY') or os.getenv('LLM_API_KEY')) else 'offline'\n"
         "print('Provedor da LLM neste notebook:', PROVIDER)"),

    # ------------------------------------------------------------------ 1
    md("## 1. Dados e ponto de partida\n\n"
       "### 1.1 Mesmo dataset e mesmo split da Fase 1\n"
       "Mesmas regras de limpeza e o mesmo split estratificado 80/20 (`random_state=42`): as 114 pacientes de "
       "teste são exatamente as da Fase 1, o que torna justa a comparação original × otimizado. "
       "**O AG nunca vê o teste.**"),
    code("from src.data import load_split\n"
         "data = load_split()\n"
         "pd.DataFrame({\n"
         "    'amostras': [len(data.y_train), len(data.y_test)],\n"
         "    'malignos': [int(data.y_train.sum()), int(data.y_test.sum())],\n"
         "    'benignos': [int((data.y_train == 0).sum()), int((data.y_test == 0).sum())],\n"
         "}, index=['treino', 'teste'])"),
    md("### 1.2 Os modelos originais da Fase 1\n"
       "Regressão Logística e Random Forest com os hiperparâmetros fixos da Fase 1 e limiar 0,5. São a linha de "
       "base de todas as comparações."),
    code("from src.models import BASELINE_PARAMS\n"
         "for nome, params in BASELINE_PARAMS.items():\n"
         "    print(f'{nome}: {params}')"),
    code("res = json.load(open('reports/results/experiments.json', encoding='utf-8'))\n"
         "def metricas(t):\n"
         "    return {'recall': round(t['recall'], 3), 'precisão': round(t['precision'], 3), 'F1': round(t['f1'], 3),\n"
         "            'ROC-AUC': round(t['roc_auc'], 3), 'falsos negativos': t['false_negatives'],\n"
         "            'falsos positivos': t['false_positives']}\n"
         "pd.DataFrame.from_dict({r['display_name']: metricas(r['baseline']['test_metrics'])\n"
         "                        for r in res['models'].values()}, orient='index')"),

    # ------------------------------------------------------------------ 2
    md("## 2. Algoritmo genético (implementado do zero, em `src/ga/`)\n\n"
       "### 2.1 Codificação: cada gene é um hiperparâmetro\n"
       "O cromossomo é um vetor em [0, 1]ᵈ. Cada gene é decodificado conforme o seu tipo: real (com escala log "
       "para `C`), inteiro ou categórico. **O limiar de decisão também é um gene**: é ele que troca falsos "
       "negativos por falsos positivos."),
    code("from src.spaces import SPACES\n"
         "from src.ga.search_space import CategoricalGene, FloatGene, IntGene\n"
         "def descreve(g):\n"
         "    if isinstance(g, FloatGene):\n"
         "        return 'real' + (' (log)' if g.log else ''), f'{g.low} … {g.high}'\n"
         "    if isinstance(g, IntGene):\n"
         "        return 'inteiro', f'{g.low} … {g.high}'\n"
         "    return 'categórico', ', '.join(map(str, g.choices))\n"
         "linhas = [{'modelo': m, 'gene': g.name, 'tipo': descreve(g)[0], 'valores': descreve(g)[1]}\n"
         "          for m, space in SPACES.items() for g in space.genes]\n"
         "pd.DataFrame(linhas)"),
    code("space = SPACES['logistic_regression']\n"
         "cromossomo = np.random.default_rng(7).random(space.size)\n"
         "print('cromossomo:', cromossomo.round(3))\n"
         "print('decodificado:', space.decode(cromossomo))"),
    md("### 2.2 Operadores genéticos\n"
       "Seleção (torneio, roleta, ranking), cruzamento (uniforme, um ponto, BLX-α) e mutação (gaussiana, uniforme). "
       "Exemplo com dois pais:"),
    code("from src.ga.operators import blx_alpha_crossover, gaussian_mutation, tournament_selection\n"
         "rng = np.random.default_rng(1)\n"
         "pai, mae = rng.random(space.size), rng.random(space.size)\n"
         "filho1, filho2 = blx_alpha_crossover(pai, mae, rng, alpha=0.5)\n"
         "mutante = gaussian_mutation(filho1, rate=0.5, rng=rng, sigma=0.1)\n"
         "pd.DataFrame([pai, mae, filho1, mutante], columns=space.names,\n"
         "             index=['pai', 'mãe', 'filho (BLX-α)', 'filho após mutação']).round(3)"),
    code("fitness_pop = np.array([0.91, 0.95, 0.80, 0.97, 0.88, 0.93])\n"
         "escolhidos = tournament_selection(fitness_pop, n=6, rng=np.random.default_rng(3), k=3)\n"
         "print('fitness da população:', fitness_pop)\n"
         "print('índices escolhidos por torneio (k=3):', escolhidos, '→ os melhores aparecem mais')"),
    md("### 2.3 Função fitness\n"
       "Validação cruzada 5-fold **só no treino**:  \n"
       "`fitness = 0,45·Recall + 0,35·F1 + 0,20·ROC-AUC`  \n"
       "O recall pesa mais (não deixar passar câncer); o F1 impede a solução trivial \"tudo maligno\"."),
    code("from src.fitness import CVFitness\n"
         "fitness = CVFitness('logistic_regression', data.X_train, data.y_train)\n"
         "score, met = fitness(BASELINE_PARAMS['logistic_regression'])\n"
         "print('fitness do modelo original (Fase 1):', round(score, 4))\n"
         "pd.Series(met).round(4)"),
    md("### 2.4 O AG rodando ao vivo\n"
       "Regressão Logística, população 10, 6 gerações, BLX-α e torneio. Leva poucos segundos."),
    code("from src.ga import GAConfig, GeneticAlgorithm\n"
         "cfg = GAConfig(population_size=10, generations=6, crossover='blx', mutation_rate=0.15, seed=1)\n"
         "resultado = GeneticAlgorithm(space, fitness, cfg).run()\n"
         "hist = pd.DataFrame([{'geração': h.generation, 'melhor': round(h.best_fitness, 4),\n"
         "                      'média': round(h.mean_fitness, 4), 'diversidade': round(h.diversity, 3),\n"
         "                      'avaliações': h.evaluations} for h in resultado.history])\n"
         "hist"),
    code("fig, ax = plt.subplots(figsize=(8, 3.2))\n"
         "ax.plot(hist['geração'], hist['melhor'], marker='o', label='melhor')\n"
         "ax.plot(hist['geração'], hist['média'], marker='o', label='média da população')\n"
         "ax.axhline(score, ls='--', color='grey', label='original (Fase 1)')\n"
         "ax.set_xlabel('geração'); ax.set_ylabel('fitness (CV)'); ax.legend(); plt.show()\n"
         "print('melhor indivíduo:', resultado.best_params, '| fitness', round(resultado.best_fitness, 4))"),

    # ------------------------------------------------------------------ 3
    md("## 3. Experimentos e comparação\n\n"
       "### 3.1 As 5 configurações do AG\n"
       "Definidas em `configs/experiments.yaml` e rodadas por `python -m scripts.run_experiments` para os dois "
       "modelos. Os resultados abaixo foram salvos dessa execução."),
    code("import yaml\n"
         "cfgs = yaml.safe_load(open('configs/experiments.yaml', encoding='utf-8'))['experiments']\n"
         "cols = ['population_size', 'generations', 'mutation_rate', 'selection', 'crossover', 'mutation', 'description']\n"
         "pd.DataFrame(cfgs).T.reindex(columns=cols)"),
    md("### 3.2 Resultados de todos os experimentos\n"
       "`cv_fitness` é o que o AG otimiza (treino). As demais colunas são no **teste**. A busca aleatória usa o "
       "mesmo orçamento de avaliações do experimento de referência."),
    code("rows = []\n"
         "for model, r in res['models'].items():\n"
         "    todos = {'original (Fase 1)': r['baseline'], **r['experiments'], 'busca aleatória': r['random_search']}\n"
         "    for nome, rec in todos.items():\n"
         "        rows.append({'modelo': r['display_name'], 'configuração': nome, 'cv_fitness': round(rec['cv_fitness'], 4),\n"
         "                     'CV 5×3': round(rec['robust_cv']['fitness']['mean'], 4), **metricas(rec['test_metrics'])})\n"
         "pd.DataFrame(rows)"),
    code("for f in ['convergencia_ag.png', 'ag_vs_busca_aleatoria.png']:\n"
         "    display(Image(f'reports/figures/{f}'))"),
    md("### 3.3 Modelo escolhido e o ganho no teste\n"
       "A escolha é feita pela validação cruzada repetida 5×3 (mais robusta), **nunca pelo teste**."),
    code("sel = res['selected']\n"
         "base = res['models'][sel['model']]['baseline']['test_metrics']\n"
         "print('Selecionado:', res['models'][sel['model']]['display_name'], '/', sel['source'])\n"
         "print('Hiperparâmetros:', sel['params'])\n"
         "pd.DataFrame.from_dict({'original (Fase 1)': metricas(base), 'otimizado (AG)': metricas(sel['test_metrics'])},\n"
         "                       orient='index')"),
    code("for f in ['comparativo_teste.png', 'erros_teste.png', 'validacao_robusta.png']:\n"
         "    display(Image(f'reports/figures/{f}'))"),
    md("### 3.4 O modelo em produção prevendo uma paciente"),
    code("import joblib\n"
         "model = joblib.load('models/best_model.joblib')\n"
         "card = json.load(open('models/model_card.json', encoding='utf-8'))\n"
         "paciente = data.X_test[data.y_test == 1].iloc[[0]]\n"
         "p = model.predict_proba(paciente)[:, 1][0]\n"
         "print(f'Paciente #{paciente.index[0]} (real: maligno) → probabilidade {p:.1%}, limiar {model.threshold:.2f} →',\n"
         "      'ALERTA' if p >= model.threshold else 'sem alerta')"),

    # ------------------------------------------------------------------ 4
    md("## 4. Interpretação com LLM\n\n"
       "### 4.1 Provedor\n"
       "Interface única para APIs compatíveis com OpenAI (Groq, Ollama, OpenAI) e Anthropic. Padrão: Groq com "
       "`openai/gpt-oss-120b`. Respostas 429 (limite de taxa) são repetidas respeitando o `Retry-After`."),
    code("from src.llm import DiagnosisInterpreter, get_provider\n"
         "provider = get_provider(PROVIDER)\n"
         "interp = DiagnosisInterpreter(model, card, provider)\n"
         "print('provedor:', provider.name, '| modelo:', provider.model)"),
    md("### 4.2 Contexto numérico (grounding)\n"
       "A LLM **não recebe a paciente crua**: recebe a probabilidade, o limiar, a faixa de risco e os 5 fatores "
       "que mais pesaram para *esta* paciente (por oclusão), com as médias de benignos e malignos. São os únicos "
       "números que ela pode usar."),
    code("from src.explain import build_case_context\n"
         "contexto = build_case_context(model, card, paciente)\n"
         "print({k: v for k, v in contexto.items() if k not in ('fatores_principais', 'desempenho_modelo')})\n"
         "pd.DataFrame(contexto['fatores_principais'])"),
    md("### 4.3 Prompt engineering: as 3 estratégias\n"
       "`basico` (só a pergunta), `estruturado` (persona, regras e JSON) e `few_shot` (estruturado + exemplo)."),
    code("from src.llm.prompts import case_prompt\n"
         "for estrategia in ['basico', 'estruturado', 'few_shot']:\n"
         "    system, user, json_mode = case_prompt(contexto, estrategia)\n"
         "    print(f'{estrategia:<12} prompt de sistema: {len(system):>5} caracteres | usuário: {len(user):>5} | json_mode={json_mode}')\n"
         "system, user, _ = case_prompt(contexto, 'few_shot')\n"
         "print('\\n--- PROMPT DE SISTEMA (few_shot) ---\\n' + system)"),
    md("### 4.4 Chamada à LLM: resposta bruta → JSON → texto para o médico"),
    code("out = interp.explain_case(paciente, strategy='few_shot')\n"
         "print(f'provedor={out.provider} modelo={out.model} latência={out.latency_s}s fallback={out.used_fallback}')\n"
         "print('tokens:', {k: out.usage[k] for k in ('prompt_tokens', 'completion_tokens', 'total_tokens') if k in out.usage})\n"
         "print('\\n--- RESPOSTA BRUTA ---\\n' + out.raw[:900])"),
    code("print('JSON válido:', out.valid_json)\n"
         "print(out.text)"),
    md("### 4.5 Avaliação automática da resposta (e o LLM juiz)\n"
       "Formato, fidelidade numérica (números inventados), cobertura dos fatores, coerência com o modelo, segurança "
       "(aviso e sem diagnóstico definitivo) e legibilidade. O juiz é outra chamada à LLM que dá notas de 1 a 5."),
    code("from src.llm.evaluation import evaluate_interpretation, llm_judge\n"
         "notas = evaluate_interpretation(out)\n"
         "juiz = llm_judge(provider, out)  # None no provedor offline\n"
         "display(pd.Series({k: v for k, v in notas.items() if k not in ('violacoes',)}, name='notas'))\n"
         "print('LLM juiz:', juiz if juiz else '(indisponível no offline)')"),
    md("### 4.6 Por que o prompt importa: básico × few-shot na mesma paciente"),
    code("basico = interp.explain_case(paciente, strategy='basico')\n"
         "nb = evaluate_interpretation(basico)\n"
         "pd.DataFrame({'básico': {k: nb[k] for k in ('formato', 'fidelidade', 'seguranca', 'score_geral', 'palavras')},\n"
         "              'few-shot': {k: notas[k] for k in ('formato', 'fidelidade', 'seguranca', 'score_geral', 'palavras')}})"),
    md("### 4.7 Pergunta livre do médico"),
    code("resposta = interp.ask(paciente, 'Quais medidas mais contribuíram para esse resultado e o que devo '\n"
         "                     'considerar antes de pedir biópsia?')\n"
         "print(resposta.text)"),
    md("### 4.8 Fallback: o que acontece quando a LLM falha\n"
       "Simulamos uma falha de API (sem gastar cota). A resposta vem do template offline **e isso fica explícito**: "
       "`used_fallback=True`, com o motivo."),
    code("from src.llm import LLMAPIError\n"
         "class ProvedorQueFalha:\n"
         "    name, model = 'groq', 'simulado'\n"
         "    def generate(self, *a, **k):\n"
         "        raise LLMAPIError('HTTP 429 — limite de taxa esgotado (simulado)')\n"
         "falha = DiagnosisInterpreter(model, card, ProvedorQueFalha()).explain_case(paciente)\n"
         "print(f'provedor={falha.provider} fallback={falha.used_fallback} tipo={falha.error_kind}')\n"
         "print('motivo:', falha.error)"),
    md("### 4.9 Insights para a equipe clínica (original × otimizado)"),
    code("relatorio = interp.explain_performance(res)\n"
         "print(relatorio.text)"),
    md("### 4.10 Avaliação completa salva (10 pacientes × 3 estratégias, com juiz)\n"
       "Gerada por `python -m scripts.evaluate_llm --provider groq --judge groq`. É a execução citada no relatório."),
    code("ev = json.load(open('reports/results/llm_evaluation_groq.json', encoding='utf-8'))\n"
         "print(f\"provedor={ev['provider']} modelo={ev['model']} juiz={ev['judge_model']} casos={ev['cases']} falhas={ev['falhas']}\")\n"
         "pd.DataFrame(ev['summary']).T[['formato', 'fidelidade', 'cobertura', 'coerencia', 'seguranca',\n"
         "                               'legibilidade', 'score_geral', 'juiz_media', 'palavras', 'latencia_s']]"),
    code("display(Image('reports/figures/llm_avaliacao_groq.png'))"),

    # ------------------------------------------------------------------ 5
    md("## 5. A API: chamadas e retornos\n\n"
       "A mesma lógica, servida por FastAPI (`src/api/main.py`). Aqui a API roda dentro do notebook "
       "(`TestClient`), sem precisar subir o servidor. É ela que a aplicação web consome."),
    code("os.environ['LOG_LEVEL'] = 'WARNING'  # os logs JSON de cada requisição ficam para a seção 6\n"
         "from fastapi.testclient import TestClient\n"
         "from src.api.main import app\n"
         "api = TestClient(app).__enter__()\n"
         "def mostra(r, chaves=None):\n"
         "    corpo = r.json()\n"
         "    if chaves:\n"
         "        corpo = {k: corpo[k] for k in chaves}\n"
         "    print(r.request.method, r.request.url.path, '→ HTTP', r.status_code)\n"
         "    print(json.dumps(corpo, ensure_ascii=False, indent=2)[:1500])\n"
         "mostra(api.get('/health'))\n"
         "mostra(api.get('/model'), ['display_name', 'source', 'params'])"),
    code("amostra = paciente.iloc[0].to_dict()\n"
         "mostra(api.post('/predict', json={'amostras': [amostra]}))"),
    code("r = api.post('/interpret', json={'amostra': amostra, 'estrategia': 'estruturado', 'provider': PROVIDER})\n"
         "mostra(r, ['provider', 'model', 'latency_s', 'used_fallback', 'error', 'structured'])"),
    code("r = api.get('/samples', params={'filtro': 'erros'})\n"
         "print('GET /samples?filtro=erros → HTTP', r.status_code)\n"
         "pd.DataFrame(r.json()['amostras'])[['indice', 'diagnostico_real', 'probabilidade', 'alerta', 'acertou']]"),
    code("r = api.post('/optimize', json={'modelo': 'logistic_regression', 'population_size': 6, 'generations': 3})\n"
         "mostra(r)\n"
         "import time\n"
         "for _ in range(120):\n"
         "    job = api.get(r.json()['status_url']).json()\n"
         "    if job['status'] != 'executando':\n"
         "        break\n"
         "    time.sleep(0.5)\n"
         "print('status:', job['status'], '| melhor fitness:', round(job['resultado']['best_fitness'], 4),\n"
         "      '| gerações acompanhadas:', len(job['historico']))"),

    # ------------------------------------------------------------------ 6
    md("## 6. Escalabilidade, monitoramento e testes\n\n"
       "### 6.1 Métricas Prometheus (`/metrics`)\n"
       "Volume e latência por rota, previsões, chamadas/latência/fallback da LLM e jobs do AG."),
    code("linhas = api.get('/metrics').text.splitlines()\n"
         "print('\\n'.join(l for l in linhas if l.startswith(('http_requests_total', 'predictions_total', 'llm_requests_total',\n"
         "                                                     'ga_best_fitness', 'model_info')))[:2500])"),
    md("### 6.2 Logs JSON estruturados, com `request_id` em cada requisição"),
    code("import io, logging\n"
         "from src.monitoring import setup_logging\n"
         "setup_logging(level='INFO', fmt='json')\n"
         "buf = io.StringIO(); logging.getLogger().handlers[0].stream = buf\n"
         "api.get('/health', headers={'x-request-id': 'demo-123'})\n"
         "print(next(l for l in buf.getvalue().splitlines() if '\"msg\": \"request\"' in l))"),
    md("### 6.3 Escalabilidade automática\n"
       "`docker compose up --scale api=3` (nginx balanceando réplicas) e, no Kubernetes, HPA de 2 a 10 réplicas; "
       "o AG roda como Job em lote."),
    code("k8s = open('deploy/k8s/api.yaml', encoding='utf-8').read()\n"
         "print(k8s[k8s.index('apiVersion: autoscaling/v2'):][:820])"),
    md("### 6.4 Testes automatizados"),
    code("import subprocess\n"
         "print(subprocess.run([sys.executable, '-m', 'pytest', '-q', '--color=no', '-p', 'no:cacheprovider'],\n"
         "                     capture_output=True, text=True).stdout.strip().splitlines()[-1])"),
    md("---\n"
       "**Próximo passo da apresentação:** a aplicação web (`frontend/`, http://localhost:4201), que consome esta "
       "mesma API e mostra tudo integrado: o AG ao vivo, a jornada do médico e o inspetor da LLM."),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
path = PROJECT_DIR / "notebooks" / "02_demo_fase2.ipynb"
nbf.write(nb, path)
print(path)

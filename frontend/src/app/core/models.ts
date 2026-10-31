/** Tipos das respostas da API da Fase 2 (src/api/main.py). */

export type Provider = 'offline' | 'groq';
export type Strategy = 'basico' | 'estruturado' | 'few_shot';
export type SampleFilter = 'erros' | 'limiar' | 'malignos' | 'benignos';

export const STRATEGIES: { id: Strategy; label: string }[] = [
  { id: 'basico', label: 'Básico' },
  { id: 'estruturado', label: 'Estruturado' },
  { id: 'few_shot', label: 'Few-shot' },
];

export interface TestMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  false_negatives: number;
  false_positives: number;
}

export interface ModelInfo {
  model: string;
  display_name: string;
  source: string;
  params: Record<string, unknown>;
  test_metrics: TestMetrics;
  baseline_test_metrics: TestMetrics;
  fitness_weights: Record<string, number>;
  features: string[];
}

export interface ExperimentRow {
  modelo: string;
  experimento: string;
  cv_fitness: number;
  teste_accuracy: number;
  teste_precision: number;
  teste_recall: number;
  teste_f1: number;
  teste_roc_auc: number;
  teste_false_negatives: number;
  teste_false_positives: number;
}

export interface Selected {
  model: string;
  source: string;
  params: Record<string, unknown>;
  test_metrics: TestMetrics;
}

export interface ExperimentsResponse {
  selecionado: Selected;
  tabela: ExperimentRow[];
}

export interface GenerationPoint {
  geracao: number;
  melhor: number;
  media: number;
  diversidade: number;
  avaliacoes: number;
}

export interface ExperimentHistory {
  descricao: string;
  config: Record<string, unknown>;
  cv_fitness: number;
  avaliacoes: number;
  parada_antecipada: boolean;
  historico: GenerationPoint[];
}

export interface ModelHistory {
  display_name: string;
  baseline_cv_fitness: number;
  experimentos: Record<string, ExperimentHistory>;
  busca_aleatoria: {
    orcamento: number;
    experimento_referencia: string;
    cv_fitness: number;
    curva: { avaliacoes: number; melhor: number }[];
  } | null;
}

export interface HistoryResponse {
  selecionado: Selected;
  modelos: Record<string, ModelHistory>;
}

export interface OptimizeRequest {
  modelo: 'logistic_regression' | 'random_forest';
  population_size: number;
  generations: number;
  mutation_rate: number;
  crossover_rate: number;
  selection: 'tournament' | 'roulette' | 'rank';
  crossover: 'uniform' | 'one_point' | 'blx';
  mutation: 'gaussian' | 'uniform';
  seed: number;
}

export interface OptimizeJob {
  job_id: string;
  status: 'executando' | 'concluido' | 'erro';
  parametros: OptimizeRequest;
  progress?: { geracao: number; total: number; melhor_fitness: number };
  historico?: GenerationPoint[];
  erro?: string;
  resultado?: {
    best_params: Record<string, unknown>;
    best_fitness: number;
    baseline_fitness: number;
    cv_metrics: Record<string, number>;
    evaluations: number;
    elapsed_s: number;
  };
}

export interface SampleRow {
  indice: number;
  diagnostico_real: 'M' | 'B';
  probabilidade: number;
  limiar: number;
  alerta: boolean;
  acertou: boolean;
  distancia_limiar: number;
  medidas: Record<string, number>;
}

export interface SamplesResponse {
  limiar: number;
  perto_do_limiar: number;
  filtro: SampleFilter | null;
  total: number;
  amostras: SampleRow[];
}

export interface PredictResponse {
  modelo: string;
  origem: string;
  resultados: { probabilidade_malignidade: number; limiar: number; orientacao: string }[];
  aviso: string;
}

/** Campos comuns a qualquer resposta da LLM (para o selo de provedor/fallback). */
export interface LlmMeta {
  provider: string;
  model: string;
  used_fallback: boolean;
  error: string | null;
  error_kind: string | null;
}

export interface Interpretation extends LlmMeta {
  structured: Record<string, unknown> | null;
  text: string;
  context: Record<string, unknown>;
  strategy: string;
  prompt_version: string;
  latency_s: number;
  valid_json: boolean;
}

export interface Scores {
  formato: number;
  fidelidade: number;
  cobertura: number;
  coerencia: number;
  seguranca: number;
  legibilidade: number;
  score_geral: number;
  numeros_inventados: number[];
  violacoes: string[];
  latencia_s: number;
  fallback: boolean;
  palavras: number;
}

export interface JudgeScores {
  clareza: number;
  utilidade_clinica: number;
  fidelidade: number;
  seguranca: number;
  media: number;
  comentario: string;
}

export interface Inspection extends LlmMeta {
  estrategia: Strategy;
  provider_solicitado: string;
  prompt_version: string;
  prompt_system: string;
  prompt_user: string;
  json_mode: boolean;
  contexto: Record<string, unknown>;
  resposta_bruta: string;
  estruturado: Record<string, unknown> | null;
  texto: string;
  valid_json: boolean;
  notas: Scores;
  juiz: JudgeScores | null;
  juiz_status: 'nao_pedido' | 'indisponivel_offline' | 'ok' | 'falhou';
  latencia_s: number;
  usage: Record<string, unknown>;
}

export interface EvaluationSummary {
  provider: string;
  model: string;
  judge: string | null;
  judge_model: string | null;
  cases: number;
  estrategias: string[];
  falhas: Record<string, number>;
  total_falhas: number;
  arquivo: string;
}

export type StrategySummary = Record<string, number>;

export interface EvaluationRow extends Partial<Scores> {
  caso: number;
  diagnostico_real: number;
  estrategia: Strategy;
  juiz?: JudgeScores | null;
}

export interface EvaluationDetail {
  provider: string;
  model: string;
  judge: string | null;
  judge_model: string | null;
  cases: number;
  falhas: Record<string, number>;
  resumo: Record<string, StrategySummary>;
  linhas: EvaluationRow[];
  exemplos: { caso: number; estrategia: Strategy; texto: string; erro: string | null; scores: Scores }[];
  relatorio_desempenho: (Interpretation & { raw?: string }) | null;
}

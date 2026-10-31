import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, catchError, of, tap, throwError, timeout } from 'rxjs';

import { environment } from '../../environments/environment';
import {
  EvaluationDetail,
  EvaluationSummary,
  ExperimentsResponse,
  HistoryResponse,
  Inspection,
  Interpretation,
  ModelInfo,
  OptimizeJob,
  OptimizeRequest,
  PredictResponse,
  Provider,
  SampleFilter,
  SamplesResponse,
  Strategy,
} from './models';

/**
 * Chamadas à LLM levam 10–20 s e mais quando o Groq gratuito devolve 429 e a
 * API espera o Retry-After antes de tentar de novo.
 */
const LLM_TIMEOUT_MS = 120_000;

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly baseUrl = environment.apiBaseUrl;

  /** Provedor escolhido no seletor do topo — `offline` por padrão, para não gastar a cota do Groq. */
  readonly provider$ = new BehaviorSubject<Provider>('offline');

  /**
   * Respostas da LLM já obtidas, por (rota, paciente, estratégia, provedor...).
   * Ficam só em memória: voltar a uma tela não chama a LLM de novo.
   */
  private readonly llmCache = new Map<string, unknown>();

  constructor(private readonly http: HttpClient) {}

  get provider(): Provider {
    return this.provider$.value;
  }

  // ---- modelo e pacientes ----
  model(): Observable<ModelInfo> {
    return this.get<ModelInfo>('/model');
  }

  samples(filtro?: SampleFilter | null): Observable<SamplesResponse> {
    return this.get<SamplesResponse>('/samples' + (filtro ? `?filtro=${filtro}` : ''));
  }

  predict(amostra: Record<string, number>): Observable<PredictResponse> {
    return this.post<PredictResponse>('/predict', { amostras: [amostra] });
  }

  // ---- algoritmo genético ----
  experiments(): Observable<ExperimentsResponse> {
    return this.get<ExperimentsResponse>('/experiments');
  }

  history(): Observable<HistoryResponse> {
    return this.get<HistoryResponse>('/experiments/history');
  }

  optimize(req: OptimizeRequest): Observable<{ job_id: string }> {
    return this.post<{ job_id: string }>('/optimize', req);
  }

  optimizeStatus(jobId: string): Observable<OptimizeJob> {
    return this.get<OptimizeJob>(`/optimize/${jobId}`);
  }

  // ---- avaliações salvas (não chamam a LLM) ----
  evaluations(): Observable<EvaluationSummary[]> {
    return this.get<EvaluationSummary[]>('/llm/evaluations');
  }

  evaluation(provider: string): Observable<EvaluationDetail> {
    return this.get<EvaluationDetail>(`/llm/evaluations/${provider}`);
  }

  // ---- LLM: só chamadas a partir de um clique ----
  interpret(indice: number, amostra: Record<string, number>, estrategia: Strategy): Observable<Interpretation> {
    const provider = this.provider;
    return this.cached(`interpret|${indice}|${estrategia}|${provider}`, () =>
      this.post<Interpretation>('/interpret', { amostra, estrategia, provider }, LLM_TIMEOUT_MS)
    );
  }

  ask(indice: number, amostra: Record<string, number>, pergunta: string): Observable<Interpretation> {
    const provider = this.provider;
    return this.cached(`ask|${indice}|${provider}|${pergunta.trim()}`, () =>
      this.post<Interpretation>('/ask', { amostra, pergunta, provider }, LLM_TIMEOUT_MS)
    );
  }

  inspect(indice: number, amostra: Record<string, number>, estrategia: Strategy, juiz: boolean): Observable<Inspection> {
    const provider = this.provider;
    return this.cached(`inspect|${indice}|${estrategia}|${provider}|${juiz}`, () =>
      this.post<Inspection>('/llm/inspect', { amostra, estrategia, provider, juiz }, LLM_TIMEOUT_MS)
    );
  }

  compare(indice: number, amostra: Record<string, number>, juiz: boolean): Observable<Inspection[]> {
    const provider = this.provider;
    // 3 chamadas em sequência no servidor (6 com juiz): o timeout cresce junto.
    return this.cached(`compare|${indice}|${provider}|${juiz}`, () =>
      this.post<Inspection[]>('/llm/compare', { amostra, provider, juiz }, LLM_TIMEOUT_MS * 3)
    );
  }

  performanceReport(): Observable<Interpretation> {
    const provider = this.provider;
    return this.cached(`report|${provider}`, () =>
      this.get<Interpretation>(`/report/performance?provider=${provider}`, LLM_TIMEOUT_MS)
    );
  }

  /** Se a resposta já está em memória (para a tela mostrar sem novo clique). */
  peek<T>(key: string): T | undefined {
    return this.llmCache.get(key) as T | undefined;
  }

  // ---- helpers ----
  private cached<T>(key: string, call: () => Observable<T>): Observable<T> {
    if (this.llmCache.has(key)) {
      return of(this.llmCache.get(key) as T);
    }
    return call().pipe(tap((res) => this.llmCache.set(key, res)));
  }

  private get<T>(path: string, ms = 30_000): Observable<T> {
    return this.http.get<T>(this.baseUrl + path).pipe(timeout(ms), catchError(toMessage));
  }

  private post<T>(path: string, body: unknown, ms = 30_000): Observable<T> {
    return this.http.post<T>(this.baseUrl + path, body).pipe(timeout(ms), catchError(toMessage));
  }
}

/** Converte qualquer falha num Error com mensagem legível para a tela. */
function toMessage(err: unknown): Observable<never> {
  if (err instanceof HttpErrorResponse) {
    if (err.status === 0) {
      return throwError(() => new Error('A API não respondeu. Ela está rodando em ' + environment.apiBaseUrl + '?'));
    }
    const detail = typeof err.error?.detail === 'string' ? err.error.detail : JSON.stringify(err.error?.detail ?? err.message);
    return throwError(() => new Error(`HTTP ${err.status}: ${detail}`));
  }
  if (err instanceof Error && err.name === 'TimeoutError') {
    return throwError(() => new Error('Tempo esgotado esperando a resposta (a LLM pode estar no limite de taxa).'));
  }
  return throwError(() => (err instanceof Error ? err : new Error(String(err))));
}

import { CommonModule } from '@angular/common';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ChartConfiguration, ChartData } from 'chart.js';
import { NgChartsModule } from 'ng2-charts';
import { Subscription, interval, startWith, switchMap, takeWhile } from 'rxjs';

import { ApiService } from '../../core/api.service';
import { CHART_COLORS, REFERENCE_COLOR, SERIES_COLORS, darkLegend, darkScale } from '../../core/chart-theme';
import {
  ExperimentRow,
  GenerationPoint,
  HistoryResponse,
  ModelHistory,
  OptimizeJob,
  OptimizeRequest,
} from '../../core/models';

type XY = { x: number; y: number };

const AXIS_TITLE = '#8b96ad';

function lineOptions(xTitle: string, yTitle: string, withDiversity = false): ChartConfiguration<'line'>['options'] {
  return {
    responsive: true,
    maintainAspectRatio: false,
    parsing: false,
    animation: false,
    interaction: { mode: 'nearest', intersect: false },
    scales: {
      x: darkScale({ type: 'linear', title: { display: true, text: xTitle, color: AXIS_TITLE } }),
      y: darkScale({ title: { display: true, text: yTitle, color: AXIS_TITLE } }),
      ...(withDiversity
        ? {
            y1: darkScale({
              position: 'right',
              min: 0,
              title: { display: true, text: 'Diversidade', color: AXIS_TITLE },
              grid: { drawOnChartArea: false },
            }),
          }
        : {}),
    },
    plugins: { legend: darkLegend },
  };
}

@Component({
  selector: 'app-genetic',
  standalone: true,
  imports: [CommonModule, FormsModule, NgChartsModule],
  templateUrl: './genetic.component.html',
  styleUrls: ['./genetic.component.scss'],
})
export class GeneticComponent implements OnInit, OnDestroy {
  history: HistoryResponse | null = null;
  table: ExperimentRow[] = [];
  error = '';

  modelKey = 'logistic_regression';
  experimentKey = '';

  convergenceData: ChartData<'line', XY[]> = { datasets: [] };
  detailData: ChartData<'line', XY[]> = { datasets: [] };
  liveData: ChartData<'line', XY[]> = { datasets: [] };

  readonly convergenceOptions = lineOptions('Avaliações de fitness (modelos únicos treinados)', 'Melhor fitness (CV)');
  readonly detailOptions = lineOptions('Geração', 'Fitness (CV)', true);
  readonly liveOptions = lineOptions('Geração', 'Fitness (CV)', true);

  // ---- execução ao vivo ----
  form: OptimizeRequest = {
    modelo: 'logistic_regression',
    population_size: 12,
    generations: 8,
    mutation_rate: 0.15,
    crossover_rate: 0.9,
    selection: 'tournament',
    crossover: 'blx',
    mutation: 'gaussian',
    seed: 42,
  };
  job: OptimizeJob | null = null;
  running = false;
  jobError = '';
  private poll?: Subscription;

  constructor(private readonly api: ApiService) {}

  ngOnInit(): void {
    this.api.history().subscribe({
      next: (h) => {
        this.history = h;
        this.modelKey = h.selecionado.model;
        this.selectModel(this.modelKey);
      },
      error: (err: Error) => (this.error = err.message),
    });
    this.api.experiments().subscribe({
      next: (e) => (this.table = e.tabela),
      error: (err: Error) => (this.error = err.message),
    });
  }

  ngOnDestroy(): void {
    this.poll?.unsubscribe();
  }

  get models(): { key: string; label: string }[] {
    return Object.entries(this.history?.modelos ?? {}).map(([key, m]) => ({ key, label: m.display_name }));
  }

  get current(): ModelHistory | null {
    return this.history?.modelos[this.modelKey] ?? null;
  }

  get experimentNames(): string[] {
    return Object.keys(this.current?.experimentos ?? {});
  }

  get rows(): ExperimentRow[] {
    return this.table.filter((r) => r.modelo === this.modelKey);
  }

  get bestRowName(): string {
    const rows = this.rows.filter((r) => r.experimento !== 'baseline');
    return rows.reduce((best, r) => (r.cv_fitness > best.cv_fitness ? r : best), rows[0])?.experimento ?? '';
  }

  isSelected(row: ExperimentRow): boolean {
    const sel = this.history?.selecionado;
    return !!sel && sel.model === row.modelo && sel.source === row.experimento;
  }

  selectModel(key: string): void {
    this.modelKey = key;
    const m = this.current;
    if (!m) return;
    const names = Object.keys(m.experimentos);
    this.experimentKey = names.includes(this.experimentKey) ? this.experimentKey : names[0];

    const datasets: ChartData<'line', XY[]>['datasets'] = names.map((name, i) => ({
      label: name,
      data: m.experimentos[name].historico.map((h) => ({ x: h.avaliacoes, y: h.melhor })),
      borderColor: SERIES_COLORS[i % SERIES_COLORS.length],
      backgroundColor: SERIES_COLORS[i % SERIES_COLORS.length],
      borderWidth: 2,
      pointRadius: 2,
      stepped: 'before' as const,
    }));
    if (m.busca_aleatoria) {
      datasets.push({
        label: `Busca aleatória (orçamento de ${m.busca_aleatoria.experimento_referencia})`,
        data: m.busca_aleatoria.curva.map((p) => ({ x: p.avaliacoes, y: p.melhor })),
        borderColor: REFERENCE_COLOR,
        backgroundColor: REFERENCE_COLOR,
        borderDash: [6, 5],
        borderWidth: 1.5,
        pointRadius: 0,
        stepped: 'before' as const,
      });
    }
    datasets.push({
      label: 'Original (Fase 1)',
      data: [
        { x: 0, y: m.baseline_cv_fitness },
        { x: Math.max(...datasets.flatMap((d) => d.data.map((p) => p.x)), 1), y: m.baseline_cv_fitness },
      ],
      borderColor: '#4b5875',
      borderDash: [2, 4],
      borderWidth: 1.5,
      pointRadius: 0,
    });
    this.convergenceData = { datasets };
    this.selectExperiment(this.experimentKey);
  }

  selectExperiment(name: string): void {
    this.experimentKey = name;
    const exp = this.current?.experimentos[name];
    this.detailData = exp ? this.generationChart(exp.historico) : { datasets: [] };
  }

  // ---- execução ao vivo ----
  run(): void {
    this.poll?.unsubscribe();
    this.running = true;
    this.jobError = '';
    this.job = null;
    this.liveData = { datasets: [] };
    this.api.optimize(this.form).subscribe({
      next: ({ job_id }) => {
        this.poll = interval(1500)
          .pipe(
            startWith(0),
            switchMap(() => this.api.optimizeStatus(job_id)),
            takeWhile((job) => job.status === 'executando', true)
          )
          .subscribe({
            next: (job) => {
              this.job = job;
              this.liveData = this.generationChart(job.historico ?? []);
              if (job.status !== 'executando') {
                this.running = false;
                if (job.status === 'erro') this.jobError = job.erro ?? 'o job falhou';
              }
            },
            error: (err: Error) => {
              this.jobError = err.message;
              this.running = false;
            },
          });
      },
      error: (err: Error) => {
        this.jobError = err.message;
        this.running = false;
      },
    });
  }

  get progressPct(): number {
    const p = this.job?.progress;
    if (!p) return 0;
    // as gerações vão de 0 (população inicial) até `total`
    return Math.min(100, Math.round(((p.geracao + 1) / (p.total + 1)) * 100));
  }

  liveParams(): { key: string; value: unknown }[] {
    return Object.entries(this.job?.resultado?.best_params ?? {}).map(([key, value]) => ({ key, value }));
  }

  private generationChart(hist: GenerationPoint[]): ChartData<'line', XY[]> {
    return {
      datasets: [
        {
          label: 'Melhor',
          data: hist.map((h) => ({ x: h.geracao, y: h.melhor })),
          borderColor: CHART_COLORS.accent,
          backgroundColor: CHART_COLORS.accent,
          borderWidth: 2,
          pointRadius: 3,
        },
        {
          label: 'Média da população',
          data: hist.map((h) => ({ x: h.geracao, y: h.media })),
          borderColor: CHART_COLORS.violet,
          backgroundColor: CHART_COLORS.violet,
          borderWidth: 2,
          pointRadius: 2,
        },
        {
          label: 'Diversidade',
          data: hist.map((h) => ({ x: h.geracao, y: h.diversidade })),
          borderColor: REFERENCE_COLOR,
          backgroundColor: REFERENCE_COLOR,
          borderDash: [4, 4],
          borderWidth: 1.5,
          pointRadius: 0,
          yAxisID: 'y1',
        },
      ],
    };
  }
}

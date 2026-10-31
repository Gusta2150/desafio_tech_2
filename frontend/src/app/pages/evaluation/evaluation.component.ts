import { CommonModule } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { ChartConfiguration, ChartData } from 'chart.js';
import { NgChartsModule } from 'ng2-charts';

import { ApiService } from '../../core/api.service';
import { SERIES_COLORS, darkLegend, darkScale } from '../../core/chart-theme';
import { EvaluationDetail, EvaluationSummary, STRATEGIES } from '../../core/models';

const CRITERIA: [string, string][] = [
  ['formato', 'Formato'],
  ['fidelidade', 'Fidelidade'],
  ['cobertura', 'Cobertura'],
  ['coerencia', 'Coerência'],
  ['seguranca', 'Segurança'],
  ['legibilidade', 'Legibilidade'],
  ['score_geral', 'Score geral'],
];

function barOptions(max: number, yTitle: string): ChartConfiguration<'bar'>['options'] {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
      x: darkScale(),
      y: darkScale({ min: 0, max, title: { display: true, text: yTitle, color: '#8b96ad' } }),
    },
    plugins: { legend: darkLegend },
  };
}

@Component({
  selector: 'app-evaluation',
  standalone: true,
  imports: [CommonModule, NgChartsModule],
  templateUrl: './evaluation.component.html',
  styles: ['.chart-box { position: relative; height: 320px; }'],
})
export class EvaluationComponent implements OnInit {
  list: EvaluationSummary[] = [];
  detail: EvaluationDetail | null = null;
  selected = '';
  error = '';

  criteriaData: ChartData<'bar'> = { labels: [], datasets: [] };
  judgeData: ChartData<'bar'> = { labels: [], datasets: [] };
  readonly criteriaOptions = barOptions(1, 'Nota (0 a 1)');
  readonly judgeOptions = barOptions(5, 'Nota do juiz (1 a 5)');

  constructor(private readonly api: ApiService) {}

  ngOnInit(): void {
    this.api.evaluations().subscribe({
      next: (list) => {
        this.list = list;
        // a avaliação com LLM real é a que interessa; o offline é só referência
        const preferred = list.find((e) => e.provider !== 'offline') ?? list[0];
        if (preferred) this.select(preferred.provider);
      },
      error: (err: Error) => (this.error = err.message),
    });
  }

  select(provider: string): void {
    this.selected = provider;
    this.detail = null;
    this.api.evaluation(provider).subscribe({
      next: (d) => {
        this.detail = d;
        this.buildCharts(d);
      },
      error: (err: Error) => (this.error = err.message),
    });
  }

  get strategies(): string[] {
    return Object.keys(this.detail?.resumo ?? {});
  }

  get totalFailures(): number {
    return Object.values(this.detail?.falhas ?? {}).reduce((a, b) => a + b, 0);
  }

  label(id: string): string {
    return STRATEGIES.find((s) => s.id === id)?.label ?? id;
  }

  private buildCharts(d: EvaluationDetail): void {
    const strategies = Object.keys(d.resumo);
    const color = (i: number) => SERIES_COLORS[i % SERIES_COLORS.length];
    this.criteriaData = {
      labels: CRITERIA.map(([, l]) => l),
      datasets: strategies.map((s, i) => ({
        label: this.label(s),
        data: CRITERIA.map(([k]) => d.resumo[s][k] ?? 0),
        backgroundColor: color(i),
        borderRadius: 4,
      })),
    };
    const judged = strategies.filter((s) => d.resumo[s]['juiz_media'] !== undefined);
    this.judgeData = {
      labels: judged.map((s) => this.label(s)),
      datasets: [
        {
          label: 'Média do LLM juiz',
          data: judged.map((s) => d.resumo[s]['juiz_media']),
          backgroundColor: judged.map((s) => color(strategies.indexOf(s))),
          borderRadius: 4,
        },
      ],
    };
  }
}

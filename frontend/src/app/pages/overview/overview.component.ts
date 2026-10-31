import { CommonModule } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { RouterLink } from '@angular/router';

import { ApiService } from '../../core/api.service';
import { ModelInfo, TestMetrics } from '../../core/models';

interface MetricRow {
  label: string;
  key: keyof TestMetrics;
  percent: boolean;
  /** Para FN/FP, menor é melhor. */
  lowerIsBetter?: boolean;
}

@Component({
  selector: 'app-overview',
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: './overview.component.html',
})
export class OverviewComponent implements OnInit {
  model: ModelInfo | null = null;
  error = '';

  readonly metrics: MetricRow[] = [
    { label: 'Recall (sensibilidade)', key: 'recall', percent: true },
    { label: 'Precisão', key: 'precision', percent: true },
    { label: 'F1', key: 'f1', percent: false },
    { label: 'ROC-AUC', key: 'roc_auc', percent: false },
    { label: 'Acurácia', key: 'accuracy', percent: true },
    { label: 'Falsos negativos', key: 'false_negatives', percent: false, lowerIsBetter: true },
    { label: 'Falsos positivos', key: 'false_positives', percent: false, lowerIsBetter: true },
  ];

  constructor(private readonly api: ApiService) {}

  ngOnInit(): void {
    this.api.model().subscribe({
      next: (m) => (this.model = m),
      error: (err: Error) => (this.error = err.message),
    });
  }

  params(): { key: string; value: unknown }[] {
    return Object.entries(this.model?.params ?? {}).map(([key, value]) => ({ key, value }));
  }

  delta(row: MetricRow): number {
    if (!this.model) return 0;
    return this.model.test_metrics[row.key] - this.model.baseline_test_metrics[row.key];
  }

  improved(row: MetricRow): boolean | null {
    const d = this.delta(row);
    if (d === 0) return null;
    return row.lowerIsBetter ? d < 0 : d > 0;
  }
}

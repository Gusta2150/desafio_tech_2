import { CommonModule } from '@angular/common';
import { Component, OnInit } from '@angular/core';

import { ApiService } from '../../core/api.service';
import { Interpretation } from '../../core/models';
import { LlmBadgeComponent } from '../../shared/llm-badge/llm-badge.component';
import { LlmLoadingComponent } from '../../shared/llm-loading/llm-loading.component';

/**
 * Relatório "original × otimizado" para a equipe clínica (GET /report/performance).
 * Ao abrir, mostra o que já foi salvo na avaliação da LLM; gerar de novo só com clique.
 */
@Component({
  selector: 'app-report',
  standalone: true,
  imports: [CommonModule, LlmBadgeComponent, LlmLoadingComponent],
  templateUrl: './report.component.html',
})
export class ReportComponent implements OnInit {
  saved: Interpretation | null = null;
  savedFrom = '';
  generated: Interpretation | null = null;
  loading = false;
  error = '';

  constructor(readonly api: ApiService) {}

  ngOnInit(): void {
    this.generated = this.api.peek<Interpretation>(`report|${this.api.provider}`) ?? null;
    this.api.evaluations().subscribe({
      next: (list) => {
        const preferred = list.find((e) => e.provider !== 'offline') ?? list[0];
        if (!preferred) return;
        this.api.evaluation(preferred.provider).subscribe({
          next: (d) => {
            this.saved = d.relatorio_desempenho;
            this.savedFrom = `${d.provider} · ${d.model}`;
          },
        });
      },
    });
  }

  generate(): void {
    this.loading = true;
    this.error = '';
    this.api.performanceReport().subscribe({
      next: (r) => {
        this.generated = r;
        this.loading = false;
      },
      error: (err: Error) => {
        this.error = err.message;
        this.loading = false;
      },
    });
  }
}

import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { ApiService } from '../../core/api.service';
import { Inspection, STRATEGIES, SampleRow, Strategy } from '../../core/models';
import { LlmLoadingComponent } from '../../shared/llm-loading/llm-loading.component';
import { PatientPickerComponent } from '../../shared/patient-picker/patient-picker.component';
import { InspectionViewComponent } from './inspection-view.component';

@Component({
  selector: 'app-inspector',
  standalone: true,
  imports: [CommonModule, FormsModule, PatientPickerComponent, LlmLoadingComponent, InspectionViewComponent],
  templateUrl: './inspector.component.html',
})
export class InspectorComponent {
  readonly strategies = STRATEGIES;

  sample: SampleRow | null = null;
  strategy: Strategy = 'few_shot';
  judge = false;

  inspection: Inspection | null = null;
  comparison: Inspection[] | null = null;
  loading: 'inspect' | 'compare' | null = null;
  error = '';

  constructor(readonly api: ApiService) {}

  pick(row: SampleRow): void {
    this.sample = row;
    this.inspection = null;
    this.comparison = null;
    this.error = '';
  }

  inspect(): void {
    if (!this.sample) return;
    this.start('inspect');
    this.api.inspect(this.sample.indice, this.sample.medidas, this.strategy, this.judge).subscribe({
      next: (r) => {
        this.inspection = r;
        this.comparison = null;
        this.loading = null;
      },
      error: (err: Error) => this.fail(err),
    });
  }

  compare(): void {
    if (!this.sample) return;
    this.start('compare');
    this.api.compare(this.sample.indice, this.sample.medidas, this.judge).subscribe({
      next: (r) => {
        this.comparison = r;
        this.inspection = null;
        this.loading = null;
      },
      error: (err: Error) => this.fail(err),
    });
  }

  label(id: string): string {
    return this.strategies.find((s) => s.id === id)?.label ?? id;
  }

  /** Quantas chamadas à LLM o botão gasta (para a pessoa decidir antes de clicar). */
  calls(compare: boolean): number {
    return (compare ? 3 : 1) * (this.judge ? 2 : 1);
  }

  private start(kind: 'inspect' | 'compare'): void {
    this.loading = kind;
    this.error = '';
  }

  private fail(err: Error): void {
    this.error = err.message;
    this.loading = null;
  }
}

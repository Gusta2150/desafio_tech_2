import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { ApiService } from '../../core/api.service';
import { Interpretation, PredictResponse, STRATEGIES, SampleRow, Strategy } from '../../core/models';
import { LlmBadgeComponent } from '../../shared/llm-badge/llm-badge.component';
import { LlmLoadingComponent } from '../../shared/llm-loading/llm-loading.component';
import { PatientPickerComponent } from '../../shared/patient-picker/patient-picker.component';

/** A jornada do médico: escolher a paciente, ver a previsão, pedir a explicação e perguntar. */
@Component({
  selector: 'app-patient',
  standalone: true,
  imports: [CommonModule, FormsModule, PatientPickerComponent, LlmBadgeComponent, LlmLoadingComponent],
  templateUrl: './patient.component.html',
})
export class PatientComponent {
  readonly strategies = STRATEGIES;

  sample: SampleRow | null = null;
  prediction: PredictResponse | null = null;
  predictError = '';

  strategy: Strategy = 'few_shot';
  explanation: Interpretation | null = null;
  explaining = false;
  explainError = '';

  question = 'Quais medidas mais contribuíram para esse resultado e o que devo considerar antes de pedir biópsia?';
  answer: Interpretation | null = null;
  asking = false;
  askError = '';

  constructor(readonly api: ApiService) {}

  pick(row: SampleRow): void {
    this.sample = row;
    this.prediction = null;
    this.predictError = '';
    this.explanation = null;
    this.answer = null;
    this.explainError = this.askError = '';
    // /predict é o modelo (não a LLM): pode ser chamado ao escolher a paciente.
    this.api.predict(row.medidas).subscribe({
      next: (p) => (this.prediction = p),
      error: (err: Error) => (this.predictError = err.message),
    });
  }

  explain(): void {
    if (!this.sample) return;
    this.explaining = true;
    this.explainError = '';
    this.api.interpret(this.sample.indice, this.sample.medidas, this.strategy).subscribe({
      next: (r) => {
        this.explanation = r;
        this.explaining = false;
      },
      error: (err: Error) => {
        this.explainError = err.message;
        this.explaining = false;
      },
    });
  }

  ask(): void {
    if (!this.sample || this.question.trim().length < 3) return;
    this.asking = true;
    this.askError = '';
    this.api.ask(this.sample.indice, this.sample.medidas, this.question).subscribe({
      next: (r) => {
        this.answer = r;
        this.asking = false;
      },
      error: (err: Error) => {
        this.askError = err.message;
        this.asking = false;
      },
    });
  }

  get alert(): boolean {
    const r = this.prediction?.resultados[0];
    return !!r && r.probabilidade_malignidade >= r.limiar;
  }

  measures(): { key: string; value: number }[] {
    return Object.entries(this.sample?.medidas ?? {}).map(([key, value]) => ({ key, value }));
  }
}

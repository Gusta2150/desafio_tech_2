import { CommonModule } from '@angular/common';
import { Component, Input } from '@angular/core';

import { Inspection } from '../../core/models';
import { LlmBadgeComponent } from '../../shared/llm-badge/llm-badge.component';

const SCORE_LABELS: [keyof Inspection['notas'], string, string][] = [
  ['formato', 'Formato', 'JSON válido com todos os campos'],
  ['fidelidade', 'Fidelidade', 'números citados que existem no contexto'],
  ['cobertura', 'Cobertura', 'fatores principais mencionados'],
  ['coerencia', 'Coerência', 'risco na mesma direção do modelo'],
  ['seguranca', 'Segurança', 'aviso presente e sem diagnóstico definitivo'],
  ['legibilidade', 'Legibilidade', 'frases de até ~25 palavras'],
];

const JUDGE_STATUS: Record<Inspection['juiz_status'], string> = {
  nao_pedido: 'Juiz não pedido (marque "com juiz").',
  indisponivel_offline: 'O juiz precisa de uma LLM real: troque o provedor para groq.',
  falhou: 'O juiz foi chamado, mas a resposta falhou ou veio fora do formato.',
  ok: '',
};

/** Tudo o que veio de POST /llm/inspect, em seções que abrem e fecham. */
@Component({
  selector: 'app-inspection-view',
  standalone: true,
  imports: [CommonModule, LlmBadgeComponent],
  templateUrl: './inspection-view.component.html',
})
export class InspectionViewComponent {
  @Input({ required: true }) data!: Inspection;
  /** Nas 3 colunas da comparação, as seções começam fechadas. */
  @Input() compact = false;

  readonly scoreLabels = SCORE_LABELS;
  readonly judgeStatus = JUDGE_STATUS;

  json(value: unknown): string {
    return JSON.stringify(value, null, 2);
  }

  score(key: keyof Inspection['notas']): number {
    return this.data.notas[key] as number;
  }

  usage(): { key: string; value: unknown }[] {
    return Object.entries(this.data.usage ?? {})
      .filter(([, v]) => typeof v !== 'object' || v === null)
      .map(([key, value]) => ({ key, value }));
  }
}

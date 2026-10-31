import { CommonModule } from '@angular/common';
import { Component, Input } from '@angular/core';

import { LlmMeta } from '../../core/models';

/**
 * Selo de origem de toda resposta da LLM: provedor/modelo e, se a LLM real
 * falhou, um alerta com o motivo (a resposta veio do template offline ou não veio).
 */
@Component({
  selector: 'app-llm-badge',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="llm-badge">
      <span class="badge" [ngClass]="meta.used_fallback || meta.error ? 'badge--danger' : 'badge--best'">
        {{ meta.provider }} · {{ meta.model }}
      </span>
      <span class="badge badge--danger" *ngIf="meta.used_fallback">⚠ template offline</span>
      <span class="badge badge--danger" *ngIf="!meta.used_fallback && meta.error">⚠ a LLM falhou</span>
      <span class="badge badge--muted" *ngIf="latency !== undefined">{{ latency | number: '1.1-1' }} s</span>
    </div>
    <p class="llm-badge__reason error" *ngIf="meta.error">
      <strong>{{ meta.error_kind === 'formato' ? 'Resposta fora do formato' : 'Falha de API' }}:</strong>
      {{ meta.error }}
    </p>
  `,
  styles: [
    `
      .llm-badge {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-bottom: 0.5rem;
      }
      .llm-badge__reason {
        margin: 0 0 0.75rem;
        word-break: break-word;
      }
    `,
  ],
})
export class LlmBadgeComponent {
  @Input({ required: true }) meta!: LlmMeta;
  @Input() latency?: number;
}

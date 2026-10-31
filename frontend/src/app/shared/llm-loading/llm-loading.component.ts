import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-llm-loading',
  standalone: true,
  template: `
    <div class="llm-loading" role="status">
      <span class="spinner spinner--accent"></span>
      <span>{{ text }}</span>
    </div>
  `,
  styles: [
    `
      .llm-loading {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.8rem 1rem;
        margin: 0.75rem 0;
        border-radius: var(--radius-sm);
        background: var(--accent-soft);
        color: var(--text-secondary);
        font-size: 0.87rem;
      }
    `,
  ],
})
export class LlmLoadingComponent {
  @Input() text = 'Aguardando a LLM (pode esperar o limite do Groq)…';
}

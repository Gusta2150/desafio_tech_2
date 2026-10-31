/**
 * Paleta e configuração compartilhada dos gráficos (Chart.js) para o tema
 * escuro do dashboard. Mantém as cores em um único lugar em vez de repetir
 * hex mágicos em cada componente de aba.
 *
 * Cores derivadas da paleta categórica/sequencial validada para modo escuro
 * (contraste ≥ 3:1 sobre a superfície do card, pares adjacentes com ΔE ≥ 8
 * em simulação de daltonismo).
 */

export const CHART_COLORS = {
  /** Azul — série principal (correlações, importância por permutação, ROC). */
  accent: '#3987e5',
  accentSoft: 'rgba(57, 135, 229, 0.16)',
  /** Violeta — usado só no SHAP, pra diferenciar visualmente da permutation importance. */
  violet: '#9085e9',
  /** Verde/vermelho — reservados para o desfecho clínico (benigno/maligno), nunca usados como cor de série genérica. */
  good: '#34d399',
  goodSoft: 'rgba(52, 211, 153, 0.16)',
  critical: '#f87171',
  criticalSoft: 'rgba(248, 113, 113, 0.16)',
} as const;

const GRID_COLOR = '#232e47';
const TICK_COLOR = '#8b96ad';
const BORDER_COLOR = '#2f3b57';

/** Config de eixo pronta pra sobrescrever `scales.x`/`scales.y` em qualquer gráfico. */
export function darkScale(extra: Record<string, unknown> = {}) {
  return {
    ticks: { color: TICK_COLOR, font: { size: 11 } },
    grid: { color: GRID_COLOR },
    border: { color: BORDER_COLOR },
    ...extra,
  };
}

export const darkLegend = {
  labels: { color: TICK_COLOR, font: { size: 11 } },
};

/**
 * Séries categóricas (Fase 2: experimentos do AG, estratégias de prompt).
 * Sem verde/vermelho, que continuam reservados ao desfecho clínico.
 */
export const SERIES_COLORS = ['#3987e5', '#9085e9', '#2bb3b1', '#d9a23a', '#d46fa8'] as const;

/** Referência neutra (busca aleatória, linha de base). */
export const REFERENCE_COLOR = '#8b96ad';

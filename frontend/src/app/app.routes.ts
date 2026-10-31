import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', redirectTo: 'visao-geral', pathMatch: 'full' },
  {
    path: 'visao-geral',
    loadComponent: () => import('./pages/overview/overview.component').then((m) => m.OverviewComponent),
  },
  {
    path: 'algoritmo-genetico',
    loadComponent: () => import('./pages/genetic/genetic.component').then((m) => m.GeneticComponent),
  },
  {
    path: 'paciente',
    loadComponent: () => import('./pages/patient/patient.component').then((m) => m.PatientComponent),
  },
  {
    path: 'inspetor',
    loadComponent: () => import('./pages/inspector/inspector.component').then((m) => m.InspectorComponent),
  },
  {
    path: 'avaliacao',
    loadComponent: () => import('./pages/evaluation/evaluation.component').then((m) => m.EvaluationComponent),
  },
  {
    path: 'relatorio',
    loadComponent: () => import('./pages/report/report.component').then((m) => m.ReportComponent),
  },
  { path: '**', redirectTo: 'visao-geral' },
];

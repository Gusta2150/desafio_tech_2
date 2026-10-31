import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';

import { ApiService } from './core/api.service';
import { Provider } from './core/models';
import { ClinicalDisclaimerComponent } from './shared/clinical-disclaimer/clinical-disclaimer.component';
import { TopBarComponent } from './shared/top-bar/top-bar.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, RouterOutlet, RouterLink, RouterLinkActive, TopBarComponent, ClinicalDisclaimerComponent],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss'],
})
export class AppComponent {
  readonly links = [
    { path: '/visao-geral', label: 'Visão geral' },
    { path: '/algoritmo-genetico', label: 'Algoritmo genético' },
    { path: '/paciente', label: 'Paciente + LLM' },
    { path: '/inspetor', label: 'Inspetor da LLM' },
    { path: '/avaliacao', label: 'Avaliação da LLM' },
    { path: '/relatorio', label: 'Relatório clínico' },
  ];

  readonly providers: { id: Provider; label: string }[] = [
    { id: 'offline', label: 'offline (template, sem cota)' },
    { id: 'groq', label: 'groq (LLM real)' },
  ];

  constructor(readonly api: ApiService) {}

  setProvider(value: string): void {
    this.api.provider$.next(value as Provider);
  }
}

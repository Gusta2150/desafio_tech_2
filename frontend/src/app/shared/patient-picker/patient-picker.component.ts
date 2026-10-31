import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';

import { ApiService } from '../../core/api.service';
import { SampleFilter, SampleRow, SamplesResponse } from '../../core/models';

/** Lista as pacientes do conjunto de teste (GET /samples) com filtros; emite a escolhida. */
@Component({
  selector: 'app-patient-picker',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './patient-picker.component.html',
  styleUrls: ['./patient-picker.component.scss'],
})
export class PatientPickerComponent implements OnInit {
  @Input() selected: number | null = null;
  @Output() pick = new EventEmitter<SampleRow>();

  readonly filters: { id: SampleFilter | null; label: string }[] = [
    { id: null, label: 'Todas' },
    { id: 'erros', label: 'Erros do modelo' },
    { id: 'limiar', label: 'Perto do limiar' },
    { id: 'malignos', label: 'Malignos' },
    { id: 'benignos', label: 'Benignos' },
  ];

  filter: SampleFilter | null = null;
  data: SamplesResponse | null = null;
  loading = false;
  error = '';

  constructor(private readonly api: ApiService) {}

  ngOnInit(): void {
    this.load();
  }

  setFilter(filter: SampleFilter | null): void {
    this.filter = filter;
    this.load();
  }

  private load(): void {
    this.loading = true;
    this.error = '';
    this.api.samples(this.filter).subscribe({
      next: (data) => {
        this.data = data;
        this.loading = false;
      },
      error: (err: Error) => {
        this.error = err.message;
        this.loading = false;
      },
    });
  }
}

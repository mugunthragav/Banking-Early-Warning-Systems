import { Component, OnInit } from '@angular/core';
import { FormsModule, NgForm } from '@angular/forms'; // Import NgForm
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { catchError, throwError } from 'rxjs';

@Component({
  selector: 'app-risk-form',
  standalone: true,
  imports: [FormsModule, CommonModule],
  styles: [`
    .predictor-box {
      background: #fff;
      border-radius: 18px;
      box-shadow: 0 8px 32px rgba(44,62,80,0.18), 0 1.5px 6px rgba(44,62,80,0.10);
      padding: 36px 32px 28px 32px;
      max-width: 540px;
      width: 100%;
      border: 4px solid #368cbf;
      margin: 56px auto;
    }
    h2 {
      text-align: center;
      color: #368cbf;
      margin-bottom: 28px;
      letter-spacing: 1.5px;
      font-weight: bold;
      text-shadow: 0 2px 12px rgba(54,140,191,0.10);
    }
    form {
      display: grid;
      grid-template-columns: 1fr 1.2fr;
      gap: 14px 18px;
      align-items: center;
    }
    label {
      color: #222;
      font-size: 15px;
      font-weight: bold;
      justify-self: end;
      text-align: right;
      padding-right: 8px;
      letter-spacing: 0.2px;
    }
    input[type="text"], input[type="number"] {
      width: 100%;
      padding: 9px 12px;
      border: 2px solid #dbeafe;
      border-radius: 7px;
      font-size: 15px;
      background: #f5faff;
      color: #222;
      box-shadow: 0 1px 4px rgba(54,140,191,0.04);
      transition: box-shadow 0.2s, border 0.2s;
    }
    input:focus {
      outline: none;
      border: 2px solid #368cbf;
      box-shadow: 0 2px 8px rgba(54,140,191,0.10);
    }
    button {
      grid-column: 1 / span 2;
      margin-top: 18px;
      padding: 12px 0;
      background: linear-gradient(90deg, #368cbf 0%, #ffb6b9 100%);
      color: #fff;
      font-weight: 700;
      font-size: 17px;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      box-shadow: 0 2px 8px rgba(54,140,191,0.13);
      transition: background 0.2s, box-shadow 0.2s;
    }
    button:disabled {
      background: #e0e0e0;
      color: #aaa;
      cursor: not-allowed;
    }
    .result, .error {
      margin-top: 26px;
      padding: 15px 0;
      text-align: center;
      border-radius: 8px;
      font-size: 19px;
      font-weight: 600;
      box-shadow: 0 1px 8px rgba(54,140,191,0.07);
    }
    .result {
      background: #e3fcec;
      color: #1b5e20;
      border: 1.5px solid #b2dfdb;
    }
    .error {
      background: #ffebee;
      color: #b71c1c;
      border: 1.5px solid #ffcdd2;
    }
    .loading {
      background: #f5faff;
      color: #368cbf;
      border: 1.5px solid #dbeafe;
    }
    .csv-upload {
      grid-column: 1 / span 2;
      margin-bottom: 18px;
      text-align: center;
    }
    .csv-upload label {
      font-weight: normal;
      color: #368cbf;
      font-size: 16px;
      margin-right: 12px;
    }
    @media (max-width: 700px) {
      .predictor-box {
        padding: 18px 4px 12px 4px;
      }
      form {
        grid-template-columns: 1fr;
      }
      label {
        text-align: left;
        justify-self: start;
        padding-right: 0;
        margin-top: 2px;
      }
      button {
        font-size: 15px;
      }
    }
  `],
  template: `
    <div class="predictor-box">
      <h2>Liquidity Risk Predictor</h2>
      <form (ngSubmit)="onSubmit()" #riskForm="ngForm" autocomplete="off">
        <div class="csv-upload">
          <label for="csvInput">Or import a CSV file:</label>
          <input type="file" id="csvInput" accept=".csv" (change)="onCsvUpload($event)" />
        </div>
        <ng-container *ngFor="let feature of features">
          <label [for]="feature.id">{{feature.label}}</label>
          <input
            [type]="feature.type"
            [id]="feature.id"
            [(ngModel)]="inputData[feature.id]"
            [name]="feature.id"
            [step]="feature.type === 'number' ? 'any' : null"
            required
          />
        </ng-container>
        <button type="submit" [disabled]="loading || !riskForm.valid">Predict</button>
      </form>
      <div *ngIf="loading" class="result loading">
        Predicting...
      </div>
      <div *ngIf="error" class="error">{{error}}</div>
    </div>
  `
})
export class RiskFormComponent implements OnInit {
  features: { id: string; label: string; type: string }[] = [
    { id: '13_CASH', label: 'Cash', type: 'number' },
    { id: '22_TREASURY_BILLS', label: 'Treasury Bills', type: 'number' },
    { id: '23_OTHER_GOV_SECURITIES', label: 'Other Gov. Securities', type: 'number' },
    { id: '01_CURR_ACC', label: 'Current Accounts', type: 'number' },
    { id: '02_TIME_DEPOSIT', label: 'Time Deposit', type: 'number' },
    { id: '03_SAVINGS', label: 'Savings', type: 'number' },
    { id: '06_BORROWING_FROM_PUBLIC', label: 'Borrowing from Public', type: 'number' },
    { id: '07_INTERBANKS_LOAN_PAYABLE', label: 'Interbanks Loan Payable', type: 'number' },
    { id: '11_OFF_BALSHEET_COMMITMENTS', label: 'Off-Balance Sheet Commitments', type: 'number' },
    { id: 'EWAQ_Capital', label: 'EWAQ Capital', type: 'number' },
    { id: 'EWAQ_GrossLoans', label: 'EWAQ Gross Loans', type: 'number' },
    { id: '25_COMMERCIAL_BILLS', label: 'Commercial Bills', type: 'number' },
    { id: 'F125_LIAB_TOTAL', label: 'F125 Total Liabilities', type: 'number' },
    { id: 'Deposit_Growth_Rate', label: 'Deposit Growth Rate', type: 'number' },
    { id: 'Funding_Cost_Change_Proxy', label: 'Funding Cost Change Proxy', type: 'number' },
    { id: 'exposed_banks', label: 'Exposed Banks (comma-separated)', type: 'text' },
    { id: 'exposure_amounts', label: 'Exposure Amounts (comma-separated)', type: 'text' },
  ];

  inputData: Record<string, any> = {};
  loading = false;
  error: string | null = null;
  private readonly API_URL = 'http://localhost:8000/predict';

  constructor(private http: HttpClient, private router: Router) {}

  ngOnInit(): void {
    this.features.forEach((feature) => {
      this.inputData[feature.id] = '';
    });
  }

  onSubmit(): void {
    this.loading = true;
    this.error = null;

    const payload: Record<string, any> = {};
    this.features.forEach((feature) => {
      const value = this.inputData[feature.id];
      if (feature.type === 'number') {
        payload[feature.id] = Number(value);
      } else {
        payload[feature.id] = String(value);
      }
    });

    this.http.post<any>(this.API_URL, payload)
      .pipe(
        catchError((err: HttpErrorResponse) => {
          console.error('API Error:', err);
          let errorMessage = 'Prediction failed. Please check your input or try again.';
          if (err.error && err.error.detail) {
            errorMessage = `Error: ${err.error.detail}`;
          } else if (err.message) {
            errorMessage = `Network or server error: ${err.message}`;
          }
          this.error = errorMessage;
          this.loading = false;
          return throwError(() => new Error(errorMessage));
        })
      )
      .subscribe({
        next: (res) => {
          this.loading = false;
          this.router.navigate(['/risk-result'], { state: {
            risk_label: res.risk_label,
            explanation: res.shap_explanation,
            advice: res.advice,
            model_input_features: res.model_input_features
          }});
        }
      });
  }

  onCsvUpload(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) {
      this.error = "No file selected.";
      return;
    }
    const file = input.files[0];
    const reader = new FileReader();

    reader.onload = (e: ProgressEvent<FileReader>) => {
      const text = e.target?.result as string;
      this.parseCsv(text);
    };
    reader.onerror = () => {
      this.error = "Failed to read file.";
    };
    reader.readAsText(file);
  }

  /**
   * Parses a CSV string, handling headers and a single data row.
   * Includes robust parsing for fields that might contain commas,
   * assuming those fields are enclosed in double quotes.
   * Does not handle escaped quotes (e.g., "" inside a quoted field).
   */
  parseCsv(csv: string): void {
    this.error = null;
    const lines = csv.trim().split('\n');

    if (lines.length < 2) {
      this.error = "CSV must have a header and at least one row of data.";
      return;
    }

    const headers = this.parseCsvLine(lines[0]); // Use robust parser for headers too
    const values = this.parseCsvLine(lines[1]);   // Use robust parser for data row

    if (headers.length !== values.length) {
      this.error = `CSV header count (${headers.length}) does not match value count (${values.length}) in the first data row. Please check quoting.`;
      console.error('CSV Headers:', headers);
      console.error('CSV Values:', values);
      return;
    }

    const tempInputData: Record<string, any> = {};
    let allHeadersFound = true;

    this.features.forEach(feature => {
      const headerIndex = headers.indexOf(feature.id);
      if (headerIndex !== -1) {
        let value = values[headerIndex];
        // Remove surrounding quotes if present (only if the field was quoted)
        if (value.startsWith('"') && value.endsWith('"')) {
            value = value.substring(1, value.length - 1);
        }

        if (feature.type === 'number') {
          const numValue = parseFloat(value);
          if (isNaN(numValue)) {
            this.error = `Invalid number format for "${feature.label}" ("${value}") in CSV.`;
            allHeadersFound = false;
            return;
          }
          tempInputData[feature.id] = numValue;
        } else {
          tempInputData[feature.id] = value;
        }
      } else {
        this.error = `Missing required column "${feature.id}" in CSV.`;
        allHeadersFound = false;
      }
    });

    if (allHeadersFound && !this.error) {
      this.inputData = tempInputData;
      console.log('CSV data loaded:', this.inputData);
    }
  }

  /**
   * Helper function to parse a single CSV line, respecting double quotes.
   * Handles commas within quoted fields.
   * Does NOT handle escaped quotes (e.g., "" inside a quoted field).
   */
  private parseCsvLine(line: string): string[] {
    const result: string[] = [];
    let inQuote = false;
    let currentField = '';
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      if (char === '"') {
        inQuote = !inQuote;
      } else if (char === ',' && !inQuote) {
        result.push(currentField.trim());
        currentField = '';
      } else {
        currentField += char;
      }
    }
    result.push(currentField.trim()); // Add the last field
    return result;
  }
}

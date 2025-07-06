import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';

// Interface for individual result entry
interface Result {
  pd_ml_probability: number;
  pd_ml_prediction: number;
  probability_of_repayment: number;
  lgd_ml_ann: number;
  recovery_rate_ml: number;
  ead_ml_meta: number;
  expected_loss_ml: number;
}

// Main response structure
interface RiskResponse {
  cumulative_expected_loss: number;
  credit_risk_percentage: number;
  defaulters_percentage: number;
  aggregate_metrics_ai_summary: string;
  results: Result[];
  batch_id: any;
  processing_summary: {
    total_applications_in_batch: number;
    applications_attempted_processing: number;
    successfully_scored_applications: number;
    applications_dropped_preprocessing: number;
  };
}


@Component({
  selector: 'app-market-risk',
  imports: [CommonModule, FormsModule],
  templateUrl: './Liquidity-risk.component.html',
  styleUrls: ['./Liquidity-risk.component.css'],
})
export class MarketRiskComponent {
  selectedOption: string = 'import';
  showResult: boolean = false;
  historicalVarInrFormatted: string = '';
  Cumulative: string = '';


  response: RiskResponse | null = null;





  constructor(private http: HttpClient) {}

  selectedFile: File | null = null;
  csvData: string[][] = [];

  csvFiles: { file: File | null; weightage: number | null; csvData: string[][] }[] = [
    { file: null, weightage: null, csvData: [] },
  ];

  // Tab switch: reset result and data
  setSelectedOption(option: string): void {
    this.selectedOption = option;
    this.clearResult();
  }

  // Handle single stock file upload
  onFileChanges(event: any): void {
    const file = event.target.files[0];
    this.selectedFile = file;
    console.log('File selected:', this.selectedFile);
  }



  // CSV parser
  parseCSV(text: string): string[][] {
    return text
      .trim()
      .split('\n')
      .map((row) => row.split(',').map((cell) => cell.trim()));
  }



  // Remove a file row (except first)
  removeCsvFile(index: number): void {
    if (index > 0) {
      this.csvFiles.splice(index, 1);
    }
  }




  // Handle single stock result generation
  toggleResult(): void {
  if (!this.selectedFile) {
    alert('Please select a CSV file first.');
    return;
  }

  const formData = new FormData();
  formData.append('file', this.selectedFile);

  this.http.post<any>('http://localhost:8000/api/v1/predict/batch_csv', formData).subscribe({
    next: (res: any) => {
      console.log('Upload results:', res);
        this.response = res;
        if (this.response) {
          console.log('Response:', this.response['results']);
        }

   this.showResult = true;
    },
    error: (err: any) => {
      console.error('Upload error:', err);
      // alert('Upload failed.');
      this.showResult = true;
    },
  });
}

  // Clear result and reset data
  clearResult(): void {
    this.showResult = false;
    this.selectedFile = null;
    this.csvData = [];

    this.csvFiles = [
      { file: null, weightage: null, csvData: [] }
    ];
  }
}
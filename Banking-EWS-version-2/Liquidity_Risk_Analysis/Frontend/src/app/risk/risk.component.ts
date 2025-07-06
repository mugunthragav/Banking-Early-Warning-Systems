import { Component, OnInit, OnDestroy, ViewChild, ElementRef } from '@angular/core';
import { Router } from '@angular/router';
import { CommonModule, KeyValuePipe, DecimalPipe } from '@angular/common';
import Chart, { TooltipItem } from 'chart.js/auto'; // Import Chart and TooltipItem

// Define interfaces for better type safety and clarity
interface ShapExplanation {
  [feature: string]: number;
}

interface AdviceItem {
  feature: string;
  impact_value: number;
  impact_on_risk: string;
  advice: string;
  explanation_text: string;
}

interface PredictionState {
  message: string;
  explanation: ShapExplanation;
  advice: AdviceItem[];
  model_input_features: Record<string, number>;
}

@Component({
  selector: 'app-risk',
  standalone: true,
  imports: [CommonModule, KeyValuePipe, DecimalPipe],
  styles: [`
    .predictor-box {
      background: #fff;
      border-radius: 18px;
      box-shadow: 0 8px 32px rgba(44,62,80,0.18), 0 1.5px 6px rgba(44,62,80,0.10);
      padding: 36px 32px 28px 32px;
      max-width: 700px;
      width: 100%;
      border: 4px solid #d32f2f;
      margin: 56px auto;
      text-align: center;
    }
    h2 {
      color: #d32f2f;
      margin-bottom: 18px;
      font-weight: bold;
      letter-spacing: 1.2px;
      text-shadow: 0 2px 12px rgba(211,47,47,0.10);
    }
    .action-btns {
      margin-bottom: 30px;
    }
    button {
      background: linear-gradient(90deg, #d32f2f 0%, #ffb6b9 100%);
      color: #fff;
      font-weight: 700;
      font-size: 17px;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      margin: 0 10px 0 10px;
      padding: 10px 28px;
      box-shadow: 0 2px 8px rgba(211,47,47,0.13);
      transition: background 0.2s, box-shadow 0.2s;
    }
    button:focus {
      outline: 2px solid #d32f2f;
    }
    .output-box {
      background: #f8f9fa;
      border-radius: 10px;
      padding: 20px;
      margin: 18px auto 12px auto;
      text-align: left;
      font-size: 16px;
      color: #222;
      box-shadow: 0 1px 8px rgba(211,47,47,0.07);
      min-height: 80px;
      white-space: pre-wrap;
      display: block;
      width: 90%;
      max-width: 600px;
    }
    .output-box h3 {
      color: #d32f2f;
      margin-top: 0;
      margin-bottom: 10px;
      font-size: 1.2em;
    }
    .output-box p {
      margin-bottom: 5px;
    }
    .output-box ul {
      list-style: none;
      padding: 0;
      margin: 0;
    }
    .output-box li {
      margin-bottom: 8px;
      padding-left: 10px;
      border-left: 3px solid #d32f2f;
    }
    .chart-container {
      position: relative;
      width: 100%;
      height: 250px; /* Adjust height as needed */
      margin-top: 20px;
    }
    .back-btn {
      margin-top: 25px;
      background: #fff;
      color: #d32f2f;
      border: 2px solid #d32f2f;
      padding: 8px 22px;
      border-radius: 8px;
      font-weight: 600;
      font-size: 15px;
      cursor: pointer;
      transition: background 0.2s, color 0.2s;
    }
    .back-btn:hover {
      background: #d32f2f;
      color: #fff;
    }
    @media (max-width: 700px) {
      .predictor-box {
        padding: 18px 4px 12px 4px;
      }
      .output-box {
        width: 95%;
        padding: 15px;
      }
      button {
        font-size: 15px;
        padding: 8px 20px;
      }
    }
  `],
  template: `
    <div class="predictor-box">
      <h2>{{ message }}</h2>
      <div class="action-btns">
        <button (click)="showExplain()">Explain</button>
        <button (click)="showAdvice()">Advice</button>
      </div>

      <div *ngIf="showing === 'explain'" class="output-box">
        <h3>Top Feature Impacts:</h3>
        <div *ngIf="advice && advice.length > 0 && !(advice && advice[0].explanation_text === 'SHAP values format not recognized or positive class values not found.')">
          <ul>
            <li *ngFor="let item of advice">
              <strong>{{ item.feature }}:</strong> {{ item.explanation_text }}
            </li>
          </ul>
          <div class="chart-container">
            <canvas #impactChart></canvas>
          </div>
        </div>
        <!-- Fallback for Explain section if no valid advice is available -->
        <p *ngIf="!advice || advice.length === 0 || (advice && advice[0].explanation_text === 'SHAP values format not recognized or positive class values not found.')">
          No feature explanations available.
        </p>
      </div>

      <div *ngIf="showing === 'advice'" class="output-box">
        <h3>Actionable Advice:</h3>
        <ul *ngIf="advice && advice.length > 0 && !(advice && advice[0].advice === 'An error occurred during advice generation.')">
          <li *ngFor="let item of advice">
            <strong>{{ item.feature }}:</strong> {{ item.advice }}
          </li>
        </ul>
        <!-- Fallback for Advice section if no valid advice is available -->
        <p *ngIf="!advice || advice.length === 0 || (advice && advice[0].advice === 'An error occurred during advice generation.')">
          No specific advice generated for this prediction.
        </p>
      </div>

      <button class="back-btn" (click)="navigateBack()">Back to Predictor</button>
    </div>
  `
})
export class RiskComponent implements OnInit, OnDestroy {
  @ViewChild('impactChart') impactChartRef!: ElementRef<HTMLCanvasElement>;

  message = '';
  explanation: ShapExplanation | null = null;
  advice: AdviceItem[] | null = null;
  showing: 'explain' | 'advice' | '' = '';
  modelInputFeatures: Record<string, number> | null = null;

  private chartInstance: Chart | null = null;

  constructor(private router: Router) {
    const navigation = this.router.getCurrentNavigation();
    const state = navigation?.extras.state as PredictionState;

    if (state) {
      this.message = state.message;
      this.explanation = state.explanation;
      this.advice = state.advice;
      this.modelInputFeatures = state.model_input_features;
    } else {
      console.warn('Navigated to /risk without prediction state. Redirecting to form.');
      this.router.navigate(['/']);
    }
  }

  ngOnInit(): void {
    // No default 'showing' set here, so it starts hidden.
  }

  ngOnDestroy(): void {
    if (this.chartInstance) {
      this.chartInstance.destroy();
      this.chartInstance = null;
    }
  }

  showExplain(): void {
    this.showing = 'explain';
    setTimeout(() => {
      this.renderImpactChart();
    }, 50);
  }

  showAdvice(): void {
    this.showing = 'advice';
    if (this.chartInstance) {
      this.chartInstance.destroy();
      this.chartInstance = null;
    }
  }

  navigateBack(): void {
    this.router.navigate(['/']);
  }

  private renderImpactChart(): void {
    if (!this.impactChartRef || !this.advice || this.advice.length === 0) {
      return;
    }

    if (this.chartInstance) {
      this.chartInstance.destroy();
    }

    const ctx = this.impactChartRef.nativeElement.getContext('2d');
    if (!ctx) {
      console.error('Failed to get 2D rendering context for chart canvas.');
      return;
    }

    const labels = this.advice.map(item => item.feature);
    const dataValues = this.advice.map(item => item.impact_value);

    const backgroundColors = dataValues.map(value =>
      value > 0 ? 'rgba(211, 47, 47, 0.7)' : 'rgba(56, 142, 60, 0.7)'
    );
    const borderColors = dataValues.map(value =>
      value > 0 ? 'rgba(211, 47, 47, 1)' : 'rgba(56, 142, 60, 1)'
    );

    this.chartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Impact on Risk',
          data: dataValues,
          backgroundColor: backgroundColors,
          borderColor: borderColors,
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        scales: {
          x: {
            title: {
              display: true,
              text: 'SHAP Value (Impact)'
            },
            beginAtZero: true
          },
          y: {
            title: {
              display: true,
              text: 'Feature'
            }
          }
        },
        plugins: {
          legend: {
            display: false
          },
          tooltip: {
            callbacks: {
              label: function(context: TooltipItem<'bar'>) { // Explicitly type 'context' here
                let label = context.dataset.label || '';
                if (label) {
                  label += ': ';
                }
                const value = context.raw as number;
                const impactDirection = value > 0 ? 'increases risk' : 'decreases risk';
                return `${label} ${impactDirection} by ${Math.abs(value).toFixed(4)}`;
              }
            }
          }
        }
      }
    });
  }
}

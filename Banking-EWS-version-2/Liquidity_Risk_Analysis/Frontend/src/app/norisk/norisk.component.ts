import { Component } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-norisk',
  standalone: true,
  styles: [`
    .predictor-box {
      background: #fff;
      border-radius: 18px;
      box-shadow: 0 8px 32px rgba(44,62,80,0.18), 0 1.5px 6px rgba(44,62,80,0.10);
      padding: 36px 32px 28px 32px;
      max-width: 700px;
      width: 100%;
      border: 4px solid #388e3c;
      margin: 56px auto;
      text-align: center;
    }
    h2 {
      color: #388e3c;
      margin-bottom: 18px;
      font-weight: bold;
      letter-spacing: 1.2px;
      text-shadow: 0 2px 12px rgba(56,142,60,0.10);
    }
    .back-btn {
      margin-top: 25px;
      background: #fff;
      color: #388e3c;
      border: 2px solid #388e3c;
      padding: 8px 22px;
      border-radius: 8px;
      font-weight: 600;
      font-size: 15px;
      cursor: pointer;
      transition: background 0.2s, color 0.2s;
    }
    .back-btn:hover {
      background: #388e3c;
      color: #fff;
    }
  `],
  template: `
    <div class="predictor-box">
      <h2>No Risk Predicted.</h2>
      <button class="back-btn" (click)="navigateBack()">Back to Predictor</button>
    </div>
  `
})
export class NoRiskComponent {
  constructor(private router: Router) {}
  navigateBack() {
    this.router.navigate(['/']);
  }
}

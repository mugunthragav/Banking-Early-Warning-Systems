import { Routes } from '@angular/router';
import { RiskFormComponent } from './risk-form/risk-form.component';
import { RiskResultComponent } from './risk-result/risk-result.component'; // Import the new component

export const routes: Routes = [
  { path: '', component: RiskFormComponent },
  { path: 'risk-result', component: RiskResultComponent }, // New route for the combined result page
  { path: '**', redirectTo: '' } // Redirect any unknown paths to the form
];
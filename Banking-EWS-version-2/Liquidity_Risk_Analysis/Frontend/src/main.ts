import { bootstrapApplication } from '@angular/platform-browser';
import { provideRouter, Routes } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';

import { AppComponent } from './app/app.component';
import { RiskFormComponent } from './app/risk-form/risk-form.component';
// Import the new RiskResultComponent
import { RiskResultComponent } from './app/risk-result/risk-result.component';

// Define your application routes
const routes: Routes = [
  { path: '', component: RiskFormComponent },
  // Use the new RiskResultComponent for the prediction results
  { path: 'risk-result', component: RiskResultComponent },
  { path: '**', redirectTo: '' } // Redirect any unknown paths to the form
];

bootstrapApplication(AppComponent, {
  providers: [
    provideHttpClient(),
    provideRouter(routes) // Provide the corrected routes
  ]
});

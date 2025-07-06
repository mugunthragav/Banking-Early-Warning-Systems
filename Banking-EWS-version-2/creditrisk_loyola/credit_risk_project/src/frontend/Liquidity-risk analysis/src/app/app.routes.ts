import { Routes } from '@angular/router';
import { MarketRiskComponent } from './pages/Liquidity-risk/Liquidity-risk.component';

export const appRoutes: Routes = [
  {
    path: 'Market-risk',
    component: MarketRiskComponent // Change 'loadChildren' to 'component' and reference the module directly
},

  {
    path: '',
    redirectTo: '/market-risk',
    pathMatch: 'full'
  },
  {
    path: '**',
    redirectTo: '/mar'
  }
];

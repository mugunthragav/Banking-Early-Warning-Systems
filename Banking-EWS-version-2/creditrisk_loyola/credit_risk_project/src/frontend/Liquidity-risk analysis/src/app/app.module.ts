import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppComponent } from './app.component';
import { SidebarComponent } from './shared/components/sidebar/sidebar.component';
import { HeaderComponent } from "./shared/components/header/header.component";
import {MarketRiskComponent} from './pages/Liquidity-risk/Liquidity-risk.component';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { SharedModule } from './shared/shared.module';
import { appRoutes } from './app.routes';
import { HttpClientModule } from '@angular/common/http';


@NgModule({
  declarations: [
    AppComponent,

  ],
  imports: [
    BrowserModule,
    HeaderComponent,
    SidebarComponent,
    CommonModule,
    RouterModule,
MarketRiskComponent,
    SharedModule,
    HttpClientModule,
    RouterModule.forRoot(appRoutes)
],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }

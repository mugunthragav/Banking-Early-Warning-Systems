import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { SidebarComponent } from './components/sidebar/sidebar.component';
import { HighlightDirective } from './directives/highlight.directive';

// Ensure the correct path to HeaderComponent
import { HeaderComponent } from './components/header/header.component';

@NgModule({
  declarations: [
  ],
  imports: [
    CommonModule,
    HighlightDirective,
    SidebarComponent,
    HeaderComponent
  ],
  exports: [
    SidebarComponent,
    HeaderComponent,
    HighlightDirective,
    CommonModule
  ]
})
export class SharedModule { }

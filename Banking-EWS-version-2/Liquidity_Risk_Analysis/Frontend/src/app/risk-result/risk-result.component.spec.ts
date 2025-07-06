import { ComponentFixture, TestBed } from '@angular/core/testing';

import { RiskResultComponent } from './risk-result.component';

describe('RiskResultComponent', () => {
  let component: RiskResultComponent;
  let fixture: ComponentFixture<RiskResultComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [RiskResultComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(RiskResultComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

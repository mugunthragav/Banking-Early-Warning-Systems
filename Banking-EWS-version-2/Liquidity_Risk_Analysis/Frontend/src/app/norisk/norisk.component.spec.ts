import { ComponentFixture, TestBed } from '@angular/core/testing';

import { NoriskComponent } from './norisk.component';

describe('NoriskComponent', () => {
  let component: NoriskComponent;
  let fixture: ComponentFixture<NoriskComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [NoriskComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(NoriskComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

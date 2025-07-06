import { Directive, ElementRef, Renderer2, HostListener } from '@angular/core';

@Directive({
  selector: '[appHighlight]'  // This is the attribute you will use in your HTML
})
export class HighlightDirective {
  
  constructor(private el: ElementRef, private renderer: Renderer2) { }

  // Change background color when the mouse hovers over the element
  @HostListener('mouseenter') onMouseEnter() {
    this.highlight('yellow');
  }

  // Revert background color when the mouse leaves the element
  @HostListener('mouseleave') onMouseLeave() {
    this.highlight(null);
  }

  // Method to set the background color
  private highlight(color: string | null) {
    this.renderer.setStyle(this.el.nativeElement, 'backgroundColor', color);
  }
}

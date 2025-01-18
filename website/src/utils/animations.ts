// Utility functions for handling animations
import { animate, stagger } from "motion";

export function animateArticles() {
  const articles = document.querySelectorAll('.article-card');
  if (articles.length === 0) return;

  animate(
    articles,
    { 
      opacity: [0, 1],
      y: [20, 0]
    },
    { 
      duration: 0.8,
      delay: stagger(0.15),
      easing: [0.215, 0.610, 0.355, 1.000]
    }
  );
}
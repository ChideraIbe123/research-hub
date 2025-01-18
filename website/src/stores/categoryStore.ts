import { papers } from "../data/papers";

interface CategoryState {
  selectedCategory: string;
  results: typeof papers;
}

class CategoryStore {
  private state: CategoryState = {
    selectedCategory: "All",
    results: papers,
  };

  private listeners: Set<() => void> = new Set();

  selectCategory(category: string) {
    this.state.selectedCategory = category;

    if (category === "All") {
      this.state.results = papers;
    } else {
      this.state.results = papers.filter(
        (paper) =>
          paper.journal === category ||
          paper.title.includes(category) ||
          paper.abstract.includes(category)
      );
    }

    this.notify();
  }

  getResults() {
    return this.state.results;
  }

  getSelectedCategory() {
    return this.state.selectedCategory;
  }

  subscribe(listener: () => void) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notify() {
    this.listeners.forEach((listener) => listener());
  }
}

export const categoryStore = new CategoryStore();

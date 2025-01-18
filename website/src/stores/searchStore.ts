import { papers } from "../data/papers";

interface SearchState {
  query: string;
  results: typeof papers;
}

class SearchStore {
  private state: SearchState = {
    query: "",
    results: papers,
  };

  private listeners: Set<() => void> = new Set();

  search(query: string) {
    this.state.query = query;

    if (!query.trim()) {
      this.state.results = papers;
    } else {
      const searchTerms = query.toLowerCase().split(" ");
      this.state.results = papers.filter((paper) => {
        const searchableText = [
          paper.title,
          paper.abstract,
          paper.journal,
          ...paper.authors,
        ]
          .join(" ")
          .toLowerCase();

        return searchTerms.every((term) => searchableText.includes(term));
      });
    }

    this.notify();
  }

  getResults() {
    return this.state.results;
  }

  getQuery() {
    return this.state.query;
  }

  subscribe(listener: () => void) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notify() {
    this.listeners.forEach((listener) => listener());
  }
}

export const searchStore = new SearchStore();

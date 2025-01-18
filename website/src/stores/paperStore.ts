import { papers as initialPapers } from "../data/papers";

interface PaperState {
  papers: typeof initialPapers;
}

class PaperStore {
  private state: PaperState = {
    papers: [...initialPapers],
  };

  private listeners: Set<() => void> = new Set();

  getPapers() {
    return this.state.papers;
  }

  subscribe(listener: () => void) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notify() {
    this.listeners.forEach((listener) => listener());
  }
}

export const paperStore = new PaperStore();

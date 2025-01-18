interface FilterState {
  paperCount: number;
  daysRange: number;
}

class FilterStore {
  private state: FilterState = {
    paperCount: 10,
    daysRange: 30,
  };

  private listeners: Set<() => void> = new Set();

  updateFilters(filters: Partial<FilterState>) {
    this.state = { ...this.state, ...filters };
    this.notify();
  }

  getFilters() {
    return this.state;
  }

  subscribe(listener: () => void) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notify() {
    this.listeners.forEach((listener) => listener());
  }
}

export const filterStore = new FilterStore();

import "@testing-library/jest-dom/vitest";

// Node >= 24 ships an experimental built-in `localStorage` global that stays
// undefined unless the process is started with --localstorage-file. That
// global shadows jsdom's own localStorage in the vitest jsdom environment,
// so tests crash with "Cannot read properties of undefined (reading 'clear')".
// Install an in-memory fallback whenever no working localStorage is present.
if (typeof globalThis.localStorage === "undefined" || globalThis.localStorage === null) {
  const store = new Map<string, string>();
  const memoryStorage: Storage = {
    getItem: (key: string) => (store.has(key) ? store.get(key)! : null),
    setItem: (key: string, value: string) => {
      store.set(key, String(value));
    },
    removeItem: (key: string) => {
      store.delete(key);
    },
    clear: () => {
      store.clear();
    },
    key: (index: number) => [...store.keys()][index] ?? null,
    get length() {
      return store.size;
    },
  };
  Object.defineProperty(globalThis, "localStorage", {
    value: memoryStorage,
    configurable: true,
    writable: true,
  });
}

beforeEach(() => {
  localStorage.clear();
});

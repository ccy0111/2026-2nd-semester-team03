/** Select a conservative ORT WASM thread count without changing inference. */
export function cpuWasmThreadCount(scope = globalThis) {
  if (scope.crossOriginIsolated !== true || typeof scope.SharedArrayBuffer !== "function") return 1;
  const cpus = scope.navigator?.hardwareConcurrency;
  if (!Number.isInteger(cpus) || cpus < 4) return 1;
  if (cpus < 8) return 2;
  return 4;
}

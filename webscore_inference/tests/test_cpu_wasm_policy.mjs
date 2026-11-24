import assert from "node:assert/strict";
import test from "node:test";
import {cpuWasmThreadCount} from "../web/cpu-wasm-policy.mjs";

const scope = (hardwareConcurrency, crossOriginIsolated = true, shared = true) => ({
  crossOriginIsolated, SharedArrayBuffer: shared ? function SharedArrayBuffer() {} : undefined,
  navigator: {hardwareConcurrency},
});
test("non-isolated and no-shared-memory contexts stay on one WASM thread", () => {
  assert.equal(cpuWasmThreadCount(scope(32, false)), 1);
  assert.equal(cpuWasmThreadCount(scope(32, true, false)), 1);
  assert.equal(cpuWasmThreadCount({}), 1);
});
test("use no more than half of a reasonable logical core count and cap at four", () => {
  for (const [cpus, expected] of [[0,1],[1,1],[2,1],[3,1],[4,2],[7,2],[8,4],[32,4],[1024,4]]) {
    assert.equal(cpuWasmThreadCount(scope(cpus)), expected);
  }
});
test("missing or invalid capability values fail safe", () => {
  for (const n of [undefined, null, NaN, Infinity, "32", 6.5, -4]) assert.equal(cpuWasmThreadCount(scope(n)), 1);
});

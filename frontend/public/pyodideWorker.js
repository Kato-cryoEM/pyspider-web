/**
 * pyspider Pyodide worker.
 *
 * Runs Pyodide (WASM CPython + numpy/scipy/etc.) off the main thread so
 * the browser UI stays responsive while graphs execute. Communicates
 * with the main thread via `postMessage` using a flat RPC schema:
 *
 *   main → worker: { id, method, args }
 *   worker → main: { id, event?: string, done?: boolean, result?, error?: string }
 *
 * Streaming methods post one or more `event` messages, then a final
 * `done` message. Non-streaming methods post a single `result` (or
 * `error`) message.
 */

// eslint-disable-next-line no-undef
importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

let pyodide = null;

function stage(name, t0) {
  self.postMessage({ stage: name, elapsedMs: performance.now() - t0 });
}

async function init(bundleBytes) {
  const t0 = performance.now();
  stage("runtime", t0);
  // eslint-disable-next-line no-undef
  pyodide = await loadPyodide({
    indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/",
  });
  stage("packages", t0);
  await pyodide.loadPackage(["numpy", "scipy", "pillow", "scikit-image"]);
  stage("bundle", t0);
  pyodide.unpackArchive(bundleBytes, "zip");
  stage("plugins", t0);
  await pyodide.runPythonAsync(`
import sys
if "/" not in sys.path:
    sys.path.insert(0, "/")
import backend
from backend import plugin_loader
plugin_loader.load_plugins()
from backend.api_shim import (
    get_catalog_json, get_plugins_json, get_health_json,
    predict_shapes_json, list_tutorials_json,
    list_challenges_json, start_challenge_json, fixture_preview_json,
    judge_challenge_json, get_settings_json, update_settings_json,
    run_graph_json, run_graph_stream, request_abort,
)
`);
  stage("ready", t0);
}

async function callShim(fn, jsonArg) {
  // Stringified JSON argument in, stringified JSON out.
  const f = pyodide.globals.get(fn);
  const raw = jsonArg !== undefined ? await f(jsonArg) : await f();
  f.destroy?.();
  return raw;
}

async function runGraphStream(reqJson, onEvent) {
  // The Python shim is an async function that calls the JS callback for
  // each event. We expose `__on_event` + `__req_json` as globals, then
  // await the coroutine. `asyncio.sleep(0)` inside the shim yields back
  // here between events so we can post incrementally.
  //
  // Defensive: tolerate a sync `run_graph_stream` too — if an older
  // bundle.zip is in play, calling the sync version still works, we
  // just lose the per-event paint.
  pyodide.globals.set("__req_json", reqJson);
  pyodide.globals.set("__on_event", onEvent);
  await pyodide.runPythonAsync(`
import inspect as _inspect
_r = run_graph_stream(__req_json, __on_event)
if _inspect.iscoroutine(_r):
    await _r
`);
}

self.onmessage = async (ev) => {
  const { id, method, args } = ev.data;
  try {
    if (method === "init") {
      await init(args[0]);
      self.postMessage({ id, done: true });
      return;
    }
    if (!pyodide) throw new Error("worker not initialised");
    if (method === "abort") {
      // Fire the Python-side abort flag synchronously. The in-flight
      // runGraphStream reads it between events and returns early.
      try {
        const f = pyodide.globals.get("request_abort");
        f();
        f.destroy?.();
      } catch (e) { /* no in-flight stream yet — OK */ }
      self.postMessage({ id, result: '{"ok":true}' });
      return;
    }
    if (method === "runGraphStream") {
      const [reqJson] = args;
      const relay = (evtJson) => {
        self.postMessage({ id, event: evtJson });
      };
      await runGraphStream(reqJson, relay);
      self.postMessage({ id, done: true });
      return;
    }
    // Generic single-call shim methods — all return a JSON string.
    const result = await callShim(method, args[0]);
    self.postMessage({ id, result });
  } catch (e) {
    const detail = (e && typeof e === "object")
      ? (e.stack || e.message || String(e))
      : String(e);
    self.postMessage({ id, error: detail });
  }
};

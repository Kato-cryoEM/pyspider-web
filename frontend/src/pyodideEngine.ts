/**
 * Pyodide-powered API client (Web Worker edition).
 *
 * Runs the full Python backend (numpy / scipy / plugins / engine) inside
 * a Web Worker. The main (UI) thread stays fully interactive while
 * graphs execute, and streaming events arrive incrementally so per-node
 * previews repaint as each node completes.
 */

// Append a build-stamp query so a new deploy busts any stale browser cache
// of bundle.zip / pyodideWorker.js (both are plain static assets that
// upstream HTTP caches tend to hold onto).
const CACHE_BUST = "v=20261110g";
const WORKER_URL = `./pyodideWorker.js?${CACHE_BUST}`;
const BUNDLE_URL = `./bundle.zip?${CACHE_BUST}`;

let worker: Worker | null = null;
let bootPromise: Promise<void> | null = null;
let nextCallId = 1;

type PendingEntry = {
  resolve: (v: unknown) => void;
  reject: (e: Error) => void;
  onEvent?: (json: string) => void;
};
const pending = new Map<number, PendingEntry>();

export type BootProgress = (stage: string, elapsedMs: number) => void;
let progressCb: BootProgress | null = null;
export function onBootProgress(cb: BootProgress) { progressCb = cb; }

function handleMessage(ev: MessageEvent) {
  const d = ev.data as {
    id?: number; stage?: string; elapsedMs?: number;
    event?: string; done?: boolean; result?: string; error?: string;
  };
  if (d.stage && progressCb) {
    progressCb(d.stage, d.elapsedMs ?? 0);
    return;
  }
  if (!d.id) return;
  const p = pending.get(d.id);
  if (!p) return;
  if (d.event !== undefined) {
    p.onEvent?.(d.event);
    return;
  }
  if (d.error !== undefined) {
    pending.delete(d.id);
    p.reject(new Error(d.error));
    return;
  }
  if (d.done || d.result !== undefined) {
    pending.delete(d.id);
    p.resolve(d.result);
  }
}

async function doBoot(): Promise<void> {
  // Resolve relative to the page so the worker loads correctly whether
  // served from the site root (`/`) or a GitHub Pages subpath.
  worker = new Worker(new URL(WORKER_URL, window.location.href));
  worker.onmessage = handleMessage;
  const resp = await fetch(BUNDLE_URL);
  if (!resp.ok) throw new Error(`bundle.zip HTTP ${resp.status}`);
  const buf = new Uint8Array(await resp.arrayBuffer());
  if (progressCb) progressCb("bundle-fetched", 0);
  await rpc("init", [buf], undefined, [buf.buffer]);
}

export function boot(): Promise<void> {
  if (!bootPromise) bootPromise = doBoot();
  return bootPromise;
}

function rpc(
  method: string, args: unknown[],
  onEvent?: (json: string) => void,
  transfer?: Transferable[],
): Promise<string | undefined> {
  if (!worker) throw new Error("worker not ready");
  const id = nextCallId++;
  return new Promise((resolve, reject) => {
    pending.set(id, {
      resolve: resolve as (v: unknown) => void,
      reject,
      onEvent,
    });
    worker!.postMessage({ id, method, args }, transfer ?? []);
  });
}

async function callJson<T>(method: string, arg?: unknown): Promise<T> {
  await boot();
  const raw = await rpc(method, arg === undefined ? [] : [JSON.stringify(arg)]);
  if (raw === undefined) throw new Error(`${method}: no result`);
  return JSON.parse(raw) as T;
}

// -- Public API ------------------------------------------------------------

export interface NodeSpecJSON {
  id: string; label: string; category: string;
  inputs: { name: string; kind: string; optional?: boolean }[];
  outputs: { name: string; kind: string }[];
  params: unknown[];
  description?: string;
}

export async function getCatalog(): Promise<NodeSpecJSON[]> {
  const j = await callJson<{ nodes: NodeSpecJSON[] }>("get_catalog_json");
  return j.nodes;
}

export async function getHealth(): Promise<{ backend: string; cuda_available: boolean }> {
  return callJson("get_health_json");
}

export async function getSettings(): Promise<{ default_save_dir?: string }> {
  return callJson("get_settings_json");
}

export async function updateSettings(patch: Record<string, unknown>) {
  return callJson<Record<string, unknown>>("update_settings_json", patch);
}

export async function predictShapes(req: { nodes: unknown[]; edges: unknown[] }) {
  return callJson<{ shapes: Record<string, unknown> }>("predict_shapes_json", req);
}

export async function listTutorials() {
  return callJson<{ lessons: unknown[]; errors: unknown[] }>("list_tutorials_json");
}

export async function listChallenges() {
  return callJson<{ challenges: unknown[]; errors: unknown[] }>("list_challenges_json");
}

export async function startChallenge(challengeId: string) {
  return callJson<Record<string, unknown>>("start_challenge_json", { challenge_id: challengeId });
}

export async function fixturePreview(sessionId: string, name: string) {
  return callJson<Record<string, unknown>>("fixture_preview_json", { session_id: sessionId, name });
}

export async function judgeChallenge(req: {
  challenge_id: string; session_id: string; answer_node_id: string;
  nodes: unknown[]; edges: unknown[];
}) {
  return callJson<Record<string, unknown>>("judge_challenge_json", req);
}

export async function runGraph(req: { nodes: unknown[]; edges: unknown[] }) {
  const j = await callJson<{ error?: string; results?: unknown }>("run_graph_json", req);
  if (j.error) throw new Error(j.error);
  return j as { results: Record<string, { ok: boolean; error: string | null; outputs: Record<string, unknown> }> };
}

export type StreamEvent =
  | { type: "node_start"; id: string }
  | { type: "node_progress"; id: string; progress?: number; log?: string }
  | { type: "node_done"; id: string; ok: boolean; error: string | null;
      outputs: Record<string, unknown>; saved?: Record<string, string> }
  | { type: "done" }
  | { type: "error"; message: string };

export async function runGraphStream(
  req: { nodes: unknown[]; edges: unknown[] },
  onEvent: (evt: StreamEvent) => void,
) {
  await boot();
  await rpc(
    "runGraphStream",
    [JSON.stringify(req)],
    (json: string) => {
      try { onEvent(JSON.parse(json) as StreamEvent); }
      catch (e) { console.error("stream parse error", e, json); }
    },
  );
}

/** Ask the worker to abort the current graph run. The in-flight stream
 *  returns an `error: "aborted"` event at the next safe point. */
export async function abortGraph(): Promise<void> {
  if (!worker) return;
  try {
    await rpc("abort", []);
  } catch { /* nothing in flight to abort — fine */ }
}

import type { NodeSpec, NodeResult, AppSettings } from "./types";
import * as py from "./pyodideEngine";

export async function fetchNodes(): Promise<NodeSpec[]> {
  return (await py.getCatalog()) as unknown as NodeSpec[];
}

export async function fetchHealth(): Promise<{ backend: string; cuda_available: boolean }> {
  return py.getHealth();
}

export async function switchBackend(_name: "cpu" | "cuda"): Promise<string> {
  // CUDA is never available in the browser — this is a stub for UI compat.
  return "cpu";
}

export interface RunReq {
  nodes: {
    id: string;
    type: string;
    params: Record<string, unknown>;
    parent?: string | null;
    save?: Record<string, { dir?: string }>;
  }[];
  edges: { src: string; srcPort: string; dst: string; dstPort: string }[];
}

export async function fetchSettings(): Promise<AppSettings> {
  return (await py.getSettings()) as unknown as AppSettings;
}

export async function updateSettings(patch: Partial<AppSettings>): Promise<AppSettings> {
  return (await py.updateSettings(patch)) as unknown as AppSettings;
}

export async function runGraph(req: RunReq): Promise<Record<string, NodeResult>> {
  const j = await py.runGraph(req);
  return j.results as unknown as Record<string, NodeResult>;
}

export type StreamEvent = py.StreamEvent;

export async function fetchTutorials(): Promise<{ lessons: unknown[]; errors: unknown[] }> {
  return py.listTutorials();
}

export async function runGraphStream(
  req: RunReq,
  onEvent: (evt: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  // Wire the AbortController the caller passed to the Pyodide worker: on
  // abort, send an abort message to the worker so it bails out at the
  // next inter-node yield point.
  const onAbort = () => { void py.abortGraph(); };
  signal?.addEventListener("abort", onAbort);
  try {
    await py.runGraphStream(req, onEvent);
  } finally {
    signal?.removeEventListener("abort", onAbort);
  }
}

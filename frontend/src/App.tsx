import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactFlow, {
  addEdge, applyNodeChanges, applyEdgeChanges,
  Background, Controls, MiniMap,
  Connection, Edge, Node,
  NodeChange, EdgeChange,
  ReactFlowProvider, useReactFlow,
} from "reactflow";

import SpiderNode, { SpiderNodeData } from "./SpiderNode";
import DoNode from "./DoNode";
import ConnectionLineLive from "./ConnectionLineLive";
import Inspector from "./Inspector";
import Palette from "./Palette";
import FileBrowser from "./FileBrowser";
import TutorialOverlay from "./tutorial/TutorialOverlay";
import TutorialComplete from "./tutorial/TutorialComplete";
import { useTutorial } from "./tutorial/useTutorial";
import ChallengePanel from "./challenge/ChallengePanel";
import { useChallenge } from "./challenge/useChallenge";
import { fetchNodes, fetchHealth, switchBackend, runGraphStream,
         fetchSettings, updateSettings } from "./api";
import * as py from "./pyodideEngine";
import type { NodeSpec, NodeResult, AppSettings, EdgeInfo, EdgeStatus } from "./types";


/**
 * Classify a wire from a source port (kind) to a destination port (kind).
 *
 * Policy (kept permissive to minimise UX friction — see design note in the
 * plugin guide):
 *
 *   - Same kind → OK.
 *   - Numeric param slots (int/float) accept a scalar source.
 *   - 3D kinds (stack / volume) are wire-level interchangeable — the
 *     distinction is only a *presentation hint*, not a hard type. Wiring
 *     between them is silently OK.
 *   - 3D → 2D image is OK; the engine passes the middle slice (equivalent
 *     to an implicit `stack.index` at the middle). Info is still available
 *     when the user clicks the edge.
 *   - 2D image → 3D (stack/volume) is a warning; usually the user meant to
 *     insert `stack.build` or similar.
 *   - Fourier ↔ real is an error (FT / IFT node needed).
 *   - Everything else that doesn't type-check is an error.
 */
function classifyEdge(srcKind: string, dstKind: string): EdgeInfo {
  const normDst = (dstKind === "int" || dstKind === "float") ? "scalar" : dstKind;
  // `fourier3d` is a complex 3D array — same geometric shape as a volume,
  // so wiring it into a volume-typed port (and vice versa) is fine. Nodes
  // that only care about ndim can accept the broader family.
  const is3D = (k: string) => k === "stack" || k === "volume" || k === "fourier3d";

  if (srcKind === normDst) return { status: "ok", srcKind, dstKind };
  if (srcKind === "scalar" && normDst === "scalar") return { status: "ok", srcKind, dstKind };
  if (srcKind === "string" && normDst === "string") return { status: "ok", srcKind, dstKind };

  // A scalar can flow into an image port when the receiving node is
  // broadcast-friendly (Calculator, arithmetic, filters that treat
  // scalars as constants). NumPy handles this natively so the engine
  // Just Works — the wire status stays OK to avoid a false red edge.
  if (srcKind === "scalar" && normDst === "image") {
    return {
      status: "ok", srcKind, dstKind,
      message: "scalar → image: broadcasts. NumPy treats the scalar as a "
             + "constant of the receiving array's shape.",
    };
  }

  // An image-kind port that carries a 0-d array (Calculator's result
  // when the formula reduces to a scalar, FS's dimensions, etc.) can
  // feed a scalar consumer — the value at runtime is a single number.
  // Marked OK so wiring Calculator → RT SQ's angle_deg stays green.
  if (srcKind === "image" && normDst === "scalar") {
    return {
      status: "ok", srcKind, dstKind,
      message: "image → scalar: OK when the runtime value is a 0-d array "
             + "(single number) — Calculator emits this for scalar formulas.",
    };
  }

  // 3D-to-3D: stack ↔ volume, same underlying array shape.
  if (is3D(srcKind) && is3D(normDst)) {
    return {
      status: "ok",
      srcKind, dstKind,
      message: `${srcKind} → ${normDst}: same underlying 3D shape — the `
             + `distinction is a presentation hint (${srcKind} = filmstrip, `
             + `${normDst} = 3D viewer).`,
    };
  }

  // 3D → 2D image:
  //   volume → image  : engine auto-passes the middle slice (density-map peek)
  //   stack  → image  : preserved intact; downstream broadcasts per frame
  //                     (e.g. PJ 3Q batch → MU × CTF → IFT works on every frame).
  //                     Use `stack.index` if you want a specific frame.
  if (is3D(srcKind) && normDst === "image") {
    const isStack = srcKind === "stack";
    return {
      status: "ok",
      srcKind, dstKind,
      message: isStack
        ? `stack → image: stack is passed through intact — arithmetic / FT / `
        + `Apply CTF broadcast per frame. Insert stack.index for a single frame.`
        : `${srcKind} → image: middle frame is passed through. `
        + `Insert Stack pick (stack.index) to select a specific frame.`,
    };
  }

  // 2D → 3D: not automatic; usually the user meant Stack build.
  if (srcKind === "image" && is3D(normDst)) {
    return {
      status: "warn",
      srcKind, dstKind,
      message: `image → ${normDst}: a single 2D image is not a ${normDst}. `
             + `Insert Stack build (2–4 images) to make one explicitly, `
             + `or wrap in a length-1 stack node.`,
    };
  }

  // Fourier ↔ image (both 2D, same shape because FT uses full fft2 + fftshift):
  // arithmetic nodes (MU, AD, SU, ...) promote to complex64 automatically, so
  // multiplying an FT'd image by a real-valued mask, or feeding the product
  // into IFT, works out of the box. Allow it as OK.
  if ((srcKind === "fourier" && normDst === "image") ||
      (srcKind === "image" && normDst === "fourier")) {
    return {
      status: "ok",
      srcKind, dstKind,
      message: `${srcKind} → ${normDst}: same 2D shape; arithmetic nodes `
             + `(MU / AD / SU) promote real ↔ complex transparently, so a `
             + `real mask × Fourier data (or product → IFT) is fine.`,
    };
  }
  // Fourier mixed with any other kind still needs an explicit bridge.
  if (srcKind === "fourier" || normDst === "fourier") {
    return {
      status: "error",
      srcKind, dstKind,
      message: `${srcKind} → ${normDst}: Fourier domain needs an explicit `
             + `FT / IFT conversion node — direct wiring won't work.`,
    };
  }

  return {
    status: "error",
    srcKind, dstKind,
    message: `${srcKind} → ${normDst}: incompatible types. The engine will `
           + `pass the value through, but the receiving node will likely fail.`,
  };
}

const EDGE_COLOR: Record<EdgeStatus, string> = {
  ok:      "#7bb0ff",   // accent blue
  warn:    "#ffa64d",   // amber
  error:   "#ff4a4a",   // red
  unknown: "#888",
};

const nodeTypes = { spider: SpiderNode, do_loop: DoNode };

const DO_NODE_ID = "flow.do";
const DO_DEFAULT_WIDTH = 420;
const DO_DEFAULT_HEIGHT = 300;


function Canvas({ specs }: { specs: NodeSpec[] }) {
  const [nodes, setNodes] = useState<Node<SpiderNodeData>[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [health, setHealth] = useState<{ backend: string; cuda_available: boolean }>({
    backend: "cpu", cuda_available: false,
  });
  const [running, setRunning] = useState(false);
  const [settings, setSettings] = useState<AppSettings>({ default_save_dir: "" });
  const [showSettings, setShowSettings] = useState(false);
  // Copyable error dialog replaces browser `alert()` so users can select /
  // copy the full stack trace into a bug report.
  const [errorPopup, setErrorPopup] = useState<string | null>(null);
  const showError = useCallback((msg: unknown) => {
    const text = (msg instanceof Error)
      ? (msg.stack || msg.message || String(msg))
      : String(msg);
    console.error(text);
    setErrorPopup(text);
  }, []);
  // Right-click context menu on a node. Only shown when the target node has
  // no required inputs (source-like), so the run-standalone action is well-defined.
  const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number; nodeId: string } | null>(null);
  // Live AbortController for the current graph run, so the "Stop" button
  // (the running-state variant of "Run") can cancel the SSE stream.
  const runAbort = useRef<AbortController | null>(null);
  const { screenToFlowPosition } = useReactFlow();

  // Declared high so memoised node-data computations downstream can read
  // `challenge.session` to toggle challenge-only UI (e.g. hiding dangling
  // output handles on answer nodes).
  const challenge = useChallenge();

  useEffect(() => { fetchHealth().then(setHealth); }, []);
  useEffect(() => { fetchSettings().then(setSettings).catch(() => {}); }, []);

  const specById = useMemo(
    () => new Map(specs.map((s) => [s.id, s])),
    [specs],
  );

  const onNodesChange = useCallback(
    (chs: NodeChange[]) => setNodes((ns) => applyNodeChanges(chs, ns) as Node<SpiderNodeData>[]),
    [],
  );
  const onEdgesChange = useCallback(
    (chs: EdgeChange[]) => setEdges((es) => applyEdgeChanges(chs, es)),
    [],
  );
  const onConnect = useCallback(
    (c: Connection) => setEdges((es) => addEdge({ ...c, animated: true }, es)),
    [],
  );

  // Backstop for React Flow's drop-detection: on connect-end, if RF didn't
  // call onConnect we probe the element under the real cursor and create
  // the edge ourselves. Needed because RF 11.11.4 uses the (sometimes
  // off-by-a-fixed-amount) pane bbox to project pointer events onto flow
  // coords, so a drop that LOOKS on-target can miss the handle entirely.
  const connectStartRef = useRef<{
    nodeId: string; handleId: string | null; handleType: "source" | "target";
  } | null>(null);
  const onConnectStart = useCallback(
    (_: unknown, p: { nodeId: string | null; handleId: string | null;
                      handleType: "source" | "target" | null }) => {
      if (!p.nodeId || !p.handleType) return;
      connectStartRef.current = {
        nodeId: p.nodeId, handleId: p.handleId, handleType: p.handleType,
      };
    },
    [],
  );
  const onConnectEnd = useCallback((evt: MouseEvent | TouchEvent) => {
    const start = connectStartRef.current;
    connectStartRef.current = null;
    if (!start) return;
    const pt = ("clientX" in evt)
      ? { x: evt.clientX, y: evt.clientY }
      : { x: evt.changedTouches?.[0]?.clientX ?? 0,
          y: evt.changedTouches?.[0]?.clientY ?? 0 };

    // Two-stage pickup:
    //   1. If the cursor is literally on top of a handle, use it.
    //   2. Otherwise, scan all handles in the DOM for the nearest VALID
    //      candidate within a generous radius — same magnetic feel that
    //      ConnectionLineLive uses for the preview, so a drop at the
    //      snap-highlighted position never silently misses.
    const wantTargetHandle = start.handleType === "source";
    const wantClass = wantTargetHandle ? "target" : "source";
    const DROP_SNAP_PX = 40;

    const direct = document.elementFromPoint(pt.x, pt.y)
      ?.closest(".react-flow__handle") as HTMLElement | null;

    let handleEl: HTMLElement | null = null;
    if (direct && direct.classList.contains(wantClass)
        && direct.getAttribute("data-nodeid") !== start.nodeId) {
      handleEl = direct;
    } else {
      let bestD2 = DROP_SNAP_PX * DROP_SNAP_PX;
      const candidates = document.querySelectorAll(
        `.react-flow__handle.${wantClass}`,
      ) as NodeListOf<HTMLElement>;
      for (const el of Array.from(candidates)) {
        if (el.getAttribute("data-nodeid") === start.nodeId) continue;
        const r = el.getBoundingClientRect();
        const cx = r.left + r.width / 2;
        const cy = r.top + r.height / 2;
        const dx = cx - pt.x;
        const dy = cy - pt.y;
        const d2 = dx * dx + dy * dy;
        if (d2 <= bestD2) {
          bestD2 = d2;
          handleEl = el;
        }
      }
    }
    if (!handleEl) return;

    const targetNodeId = handleEl.getAttribute("data-nodeid");
    const targetHandleId = handleEl.getAttribute("data-handleid");
    if (!targetNodeId) return;
    const connection: Connection = wantTargetHandle
      ? {
          source: start.nodeId,
          sourceHandle: start.handleId,
          target: targetNodeId,
          targetHandle: targetHandleId,
        }
      : {
          source: targetNodeId,
          sourceHandle: targetHandleId,
          target: start.nodeId,
          targetHandle: start.handleId,
        };
    setEdges((es) => {
      // Guard against double-add: RF may still fire its own onConnect for
      // handles that happened to land within its (buggy) radius.
      const already = es.some(
        (e) => e.source === connection.source
            && e.sourceHandle === connection.sourceHandle
            && e.target === connection.target
            && e.targetHandle === connection.targetHandle,
      );
      if (already) return es;
      return addEdge({ ...connection, animated: true }, es);
    });
  }, []);

  const onDragStart = (evt: React.DragEvent, specId: string) => {
    evt.dataTransfer.setData("application/spider-node", specId);
    evt.dataTransfer.effectAllowed = "move";
  };

  const onDrop = useCallback(
    (evt: React.DragEvent) => {
      evt.preventDefault();
      const id = evt.dataTransfer.getData("application/spider-node");
      const spec = specById.get(id);
      if (!spec) return;
      const dropAbs = screenToFlowPosition({ x: evt.clientX, y: evt.clientY });
      const isDo = spec.id === DO_NODE_ID;

      // Find a DO container whose bounds enclose the drop point. If several
      // DOs enclose (nested case), pick the SMALLEST — the innermost — so
      // dropping into an inner DO parents to the inner one, not the outer.
      // DOs themselves can now nest too; we allow a DO to be a child of
      // another DO.
      const enclosing = nodes.filter((n) => {
        if (n.data.spec.id !== DO_NODE_ID) return false;
        // Compute the DO's ABSOLUTE position: sum ancestor positions if it
        // is itself nested. n.position is relative to n.parentNode.
        let ax = n.position.x, ay = n.position.y;
        let cur = n.parentNode ? nodes.find((m) => m.id === n.parentNode) : undefined;
        while (cur) {
          ax += cur.position.x;
          ay += cur.position.y;
          cur = cur.parentNode ? nodes.find((m) => m.id === cur!.parentNode) : undefined;
        }
        const w = Number((n.style as { width?: number } | undefined)?.width ?? DO_DEFAULT_WIDTH);
        const h = Number((n.style as { height?: number } | undefined)?.height ?? DO_DEFAULT_HEIGHT);
        return dropAbs.x >= ax && dropAbs.x <= ax + w &&
               dropAbs.y >= ay && dropAbs.y <= ay + h;
      });
      const parentDo = enclosing.length === 0
        ? undefined
        : enclosing.reduce((smallest, cand) => {
            const smallestArea =
              Number((smallest.style as { width?: number } | undefined)?.width ?? DO_DEFAULT_WIDTH) *
              Number((smallest.style as { height?: number } | undefined)?.height ?? DO_DEFAULT_HEIGHT);
            const candArea =
              Number((cand.style as { width?: number } | undefined)?.width ?? DO_DEFAULT_WIDTH) *
              Number((cand.style as { height?: number } | undefined)?.height ?? DO_DEFAULT_HEIGHT);
            return candArea < smallestArea ? cand : smallest;
          });

      const nid = `n${Date.now()}${Math.floor(Math.random() * 1e4)}`;
      const params: Record<string, unknown> = {};
      for (const p of spec.params) {
        // Any param literally named `seed` auto-initialises to the drop-
        // time timestamp (unix seconds, clamped into int32) so freshly-
        // dragged random nodes get independent draws without the user
        // touching the field. Users can still overwrite to lock a seed.
        if (p.name === "seed" && p.kind === "int") {
          params[p.name] = Math.floor(Date.now() / 1000) & 0x7FFFFFFF;
        } else {
          params[p.name] = p.default;
        }
      }

      // Position for a nested child must be relative to its DIRECT parent —
      // but we know parentDo's ABSOLUTE position by walking its own ancestors.
      let position = dropAbs;
      if (parentDo) {
        let pax = parentDo.position.x, pay = parentDo.position.y;
        let cur = parentDo.parentNode ? nodes.find((m) => m.id === parentDo.parentNode) : undefined;
        while (cur) {
          pax += cur.position.x;
          pay += cur.position.y;
          cur = cur.parentNode ? nodes.find((m) => m.id === cur!.parentNode) : undefined;
        }
        position = { x: dropAbs.x - pax, y: dropAbs.y - pay };
      }

      const newNode: Node<SpiderNodeData> = {
        id: nid,
        type: isDo ? "do_loop" : "spider",
        position,
        data: { spec, params },
        ...(isDo && {
          style: { width: DO_DEFAULT_WIDTH, height: DO_DEFAULT_HEIGHT },
          dragHandle: ".do-header",
        }),
        ...(parentDo && {
          parentNode: parentDo.id,
          extent: "parent" as const,
        }),
      };
      // React Flow requires the parent to appear before its children in the
      // nodes array; since we always push to the end, and DO containers were
      // added earlier, this is naturally satisfied.
      setNodes((ns) => ns.concat(newNode));
    },
    [nodes, specById, screenToFlowPosition],
  );

  const onDragOver = (evt: React.DragEvent) => {
    evt.preventDefault();
    evt.dataTransfer.dropEffect = "move";
  };

  const onParamChange = useCallback(
    (nodeId: string, name: string, value: unknown) => {
      setNodes((ns) =>
        ns.map((n) =>
          n.id === nodeId
            ? { ...n, data: { ...n.data, params: { ...n.data.params, [name]: value } } }
            : n,
        ),
      );
    },
    [],
  );

  // Shared runner: `subset` (if given) restricts execution to those node IDs
  // and any edges between them — used by "Run this node" from the context menu.
  // `override` lets callers pass fresh nodes/edges instead of reading the
  // state closure — essential right after a setNodes(...) call, before React
  // has committed the update (e.g. "auto-run after dropping challenge
  // fixtures").
  const runSubset = useCallback(async (
    subset?: Set<string>,
    override?: {
      nodes: Node<SpiderNodeData>[],
      edges: Edge[],
    },
  ) => {
    setRunning(true);
    const inSubset = (id: string) => subset ? subset.has(id) : true;
    const useNodes = override?.nodes ?? nodes;
    const useEdges = override?.edges ?? edges;
    // Reset status only for nodes we're about to run — leave others intact
    // so the user's last full-graph coloring survives a partial re-run.
    setNodes((ns) =>
      ns.map((n) => inSubset(n.id)
        ? { ...n, data: { ...n.data, status: "idle" as const, logs: [] } }
        : n),
    );
    try {
      const req = {
        nodes: useNodes.filter((n) => inSubset(n.id)).map((n) => ({
          id: n.id,
          type: n.data.spec.id,
          params: n.data.params,
          parent: n.parentNode && inSubset(n.parentNode) ? n.parentNode : null,
          save: n.data.saveConfig && Object.keys(n.data.saveConfig).length
                  ? n.data.saveConfig : undefined,
        })),
        edges: useEdges
          .filter((e) => inSubset(e.source) && inSubset(e.target))
          .map((e) => ({
            src: e.source,
            srcPort: e.sourceHandle ?? "",
            dst: e.target,
            dstPort: e.targetHandle ?? "",
          })),
      };
      const ctrl = new AbortController();
      runAbort.current = ctrl;
      await runGraphStream(req, (evt) => {
        if (evt.type === "node_start") {
          setNodes((ns) => ns.map((n) => n.id === evt.id
            ? { ...n, data: { ...n.data, status: "running" as const,
                               logs: [] } }
            : n));
        } else if (evt.type === "node_progress") {
          // Progress bar was removed by user request (bar was misleading for
          // uninstrumented nodes that don't emit progress updates). We still
          // capture log lines so the Inspector shows a live tail per node.
          if (!evt.log) return;
          setNodes((ns) => ns.map((n) => {
            if (n.id !== evt.id) return n;
            const prevLogs = n.data.logs ?? [];
            return {
              ...n,
              data: { ...n.data, logs: [...prevLogs, evt.log!].slice(-200) },
            };
          }));
        } else if (evt.type === "done") {
          // Logs stay visible after the run completes so the user can click
          // any node and inspect what happened; they get wiped at the next
          // run's start (see `runSubset` above, which resets logs on the
          // set of nodes about to execute).
        } else if (evt.type === "node_done") {
          setNodes((ns) => ns.map((n) => n.id === evt.id
            ? {
                ...n,
                data: {
                  ...n.data,
                  status: evt.ok ? ("success" as const) : ("error" as const),
                  result: {
                    ok: evt.ok,
                    error: evt.error,
                    outputs: evt.outputs,
                    saved: evt.saved,
                  } as NodeResult,
                },
              }
            : n));
        } else if (evt.type === "error") {
          showError(`engine error: ${evt.message}`);
        }
      }, ctrl.signal);
    } catch (e) {
      showError(e);
    } finally {
      runAbort.current = null;
      setRunning(false);
      // If the user hit Stop while a node was executing, clear its running
      // status so the UI doesn't leave a red halo behind.
      setNodes((ns) => ns.map((n) => n.data.status === "running"
        ? { ...n, data: { ...n.data, status: "idle" as const } }
        : n));
    }
  }, [nodes, edges]);

  const run = useCallback(() => runSubset(undefined), [runSubset]);

  // A node is "runnable standalone" when it has zero required (non-optional)
  // input ports. Nodes with required inputs need upstream to feed them and so
  // don't make sense as a one-node run — the context menu omits the entry.
  const canRunStandalone = useCallback((n: Node<SpiderNodeData>): boolean => {
    const reqInputs = n.data.spec.inputs.filter((p) => !p.optional);
    return reqInputs.length === 0;
  }, []);

  const runOne = useCallback((nodeId: string) => {
    setCtxMenu(null);
    runSubset(new Set([nodeId]));
  }, [runSubset]);

  const toggleBackend = async () => {
    const next = health.backend === "cuda" ? "cpu" : "cuda";
    try {
      const b = await switchBackend(next);
      setHealth((h) => ({ ...h, backend: b }));
    } catch (e) {
      showError(e);
    }
  };

  // Compute per-node "which params are supplied by a wire" set from edges.
  // A wire targets a param when its targetHandle matches a param name on that node.
  const wiredParamsByNode = useMemo(() => {
    const m = new Map<string, Set<string>>();
    for (const e of edges) {
      const dst = nodes.find((n) => n.id === e.target);
      if (!dst) continue;
      const paramNames = new Set(dst.data.spec.params.map((p) => p.name));
      if (e.targetHandle && paramNames.has(e.targetHandle)) {
        let s = m.get(e.target);
        if (!s) { s = new Set(); m.set(e.target, s); }
        s.add(e.targetHandle);
      }
    }
    return m;
  }, [edges, nodes]);

  // Track which nodes have any edge (in or out).
  const nodesWithEdges = useMemo(() => {
    const s = new Set<string>();
    for (const e of edges) { s.add(e.source); s.add(e.target); }
    return s;
  }, [edges]);

  // Predicted output shapes per node (from /api/predict_shapes). Used by
  // SpiderNode to size the pre-Run preview slot to the CORRECT aspect
  // (e.g., clock 256² → PO 128×~804) instead of a placeholder square.
  // Recomputed on debounced graph changes so it stays in sync as wires
  // are drawn or params tuned.
  const [predictedShapes, setPredictedShapes] = useState<Record<string, number[]>>({});
  useEffect(() => {
    const req = {
      nodes: nodes.map((n) => ({
        id: n.id, type: n.data.spec.id, params: n.data.params ?? {},
      })),
      edges: edges.map((e) => ({
        src: e.source, srcPort: e.sourceHandle ?? "",
        dst: e.target, dstPort: e.targetHandle ?? "",
      })),
    };
    const t = window.setTimeout(async () => {
      try {
        const j = await py.predictShapes(req);
        setPredictedShapes((j.shapes as Record<string, number[]>) ?? {});
      } catch { /* Pyodide not booted yet — leave last-known shapes */ }
    }, 200);
    return () => window.clearTimeout(t);
  }, [nodes, edges]);

  // Per-fixture `hide_output` flag (from the challenge JSON) — pure-answer
  // fixtures (e.g. `particle` in angle_assignment, `projection` in
  // projection_theorem) suppress their output handle so the student
  // doesn't accidentally wire the target into the pipeline. Fixtures
  // meant as problem inputs (reference / shifted / rotated / volume /
  // micrograph) are absent from this set and keep their handles.
  const hideOutputFixtureNames = useMemo(() => {
    const s = new Set<string>();
    for (const f of challenge.session?.fixtures ?? []) {
      if (f.hide_output) s.add(f.name);
    }
    return s;
  }, [challenge.session]);

  const nodesWithWireInfo = useMemo(
    () => nodes.map((n) => {
      const spec = n.data.spec;
      const isHiddenFixture = hideOutputFixtureNames.size > 0
        && (spec.id === "data.challenge_input"
            || spec.id === "data.challenge_input_volume")
        && typeof n.data.params?.name === "string"
        && hideOutputFixtureNames.has(n.data.params.name as string);
      return {
        ...n,
        data: {
          ...n.data,
          wiredParams: wiredParamsByNode.get(n.id) ?? new Set<string>(),
          hasWires: nodesWithEdges.has(n.id),
          predictedShape: predictedShapes[n.id],
          hideAllOutputs: isHiddenFixture,
        },
      };
    }),
    [nodes, wiredParamsByNode, nodesWithEdges, predictedShapes,
     hideOutputFixtureNames],
  );

  // Attach a type-compatibility check to each edge and colour it.
  const edgesTyped = useMemo(() => {
    return edges.map((e): Edge => {
      const src = nodes.find((n) => n.id === e.source);
      const dst = nodes.find((n) => n.id === e.target);
      let info: EdgeInfo = { status: "unknown" };
      if (src && dst && e.sourceHandle && e.targetHandle) {
        const srcPort = src.data.spec.outputs.find((p) => p.name === e.sourceHandle);
        const dstIn = dst.data.spec.inputs.find((p) => p.name === e.targetHandle);
        const dstParam = dst.data.spec.params.find((p) => p.name === e.targetHandle);
        const dstKind = dstIn?.kind ?? dstParam?.kind;
        if (srcPort && dstKind) info = classifyEdge(srcPort.kind, dstKind);
      }
      const stroke = EDGE_COLOR[info.status];
      return {
        ...e,
        data: { ...(e.data ?? {}), info },
        style: {
          ...(e.style ?? {}),
          stroke,
          strokeWidth: info.status === "error" ? 2.5
                     : info.status === "warn" ? 2 : 1.6,
          strokeDasharray: info.status === "error" ? "6 3"
                         : info.status === "warn" ? "3 3" : undefined,
        },
        selected: e.id === selectedEdgeId,
      };
    });
  }, [edges, nodes, selectedEdgeId]);

  const selected = nodesWithWireInfo.find((n) => n.id === selectedId) ?? null;
  const selectedWired = selected
    ? (wiredParamsByNode.get(selected.id) ?? new Set<string>())
    : new Set<string>();
  const selectedEdge = edgesTyped.find((e) => e.id === selectedEdgeId) ?? null;

  const tutorial = useTutorial(nodesWithWireInfo, edges, selectedId);
  const [showLessons, setShowLessons] = useState(false);
  // Two-step picker: null = show category buttons; string = show lessons in
  // that category with a back arrow. Reset on close so each open starts at
  // the category screen.
  const [pickedCategory, setPickedCategory] = useState<string | null>(null);

  const [showChallenges, setShowChallenges] = useState(false);

  // Drop two fixture nodes on the canvas for the chosen challenge, then
  // call the hook so the panel opens with prompt + previews. Overrides
  // the previous graph entirely — challenge mode is a fresh workspace.
  const startChallenge = useCallback(async (challengeId: string) => {
    const picked = challenge.available.find((c) => c.id === challengeId);
    if (!picked) return;
    setShowChallenges(false);
    const sess = await challenge.begin(challengeId);
    if (!sess) return;
    const imageSpec = specById.get("data.challenge_input");
    const volumeSpec = specById.get("data.challenge_input_volume");
    if (!imageSpec) {
      // Plugin unavailable (fresh server without the challenge plugin);
      // bail out so we don't leave an empty canvas.
      showError("data.challenge_input node is not registered on the server.");
      return;
    }
    const now = Date.now();
    // judge_only fixtures are the hidden ground-truth: visible only as a
    // preview in the challenge panel, never dropped on the canvas, so a
    // student cannot wire the answer directly as a graph input.
    const canvasFixtures = sess.fixtures.filter((f) => !f.judge_only);
    const fixtureNodes: Node<SpiderNodeData>[] = canvasFixtures.map((f, i) => {
      const spec = f.kind === "volume" ? (volumeSpec ?? imageSpec) : imageSpec;
      return {
        id: `fx_${f.name}_${now}${i}`,
        type: "spider",
        position: { x: 80, y: 80 + i * 240 },
        data: {
          spec,
          params: { session_id: sess.session_id, name: f.name },
        },
      };
    });
    setNodes(fixtureNodes);
    setEdges([]);
    setSelectedId(null);
    setSelectedEdgeId(null);
    // Auto-run Run Graph once so the student sees the fixtures' previews
    // right away without manually pressing the button. Pass the fresh
    // fixtureNodes / [] explicitly — reading from the nodes/edges closure
    // would be stale (React hasn't committed the setNodes above yet).
    void runSubset(undefined, { nodes: fixtureNodes, edges: [] });
  }, [challenge, specById, runSubset]);

  const submitChallenge = useCallback(async () => {
    if (!selectedId) return;
    // Infer parent by geometric containment at submit time: a node is a
    // child of the smallest DO whose bounding box encloses its absolute
    // position. This matches what the user visually sees — being "inside"
    // the DO rectangle — rather than requiring they drop every node
    // directly inside at placement time. React Flow doesn't auto-reparent
    // when a node is dragged into a container after creation, so without
    // this inference the engine sees a flat graph and the `value_in`
    // feedback edge is never filtered, creating a top-level cycle.
    const absPos = (n: Node<SpiderNodeData>): { x: number; y: number } => {
      let x = n.position.x, y = n.position.y;
      let cur = n.parentNode ? nodes.find((m) => m.id === n.parentNode) : undefined;
      while (cur) {
        x += cur.position.x;
        y += cur.position.y;
        cur = cur.parentNode ? nodes.find((m) => m.id === cur!.parentNode) : undefined;
      }
      return { x, y };
    };
    const doBoxes = nodes
      .filter((n) => n.data.spec.id === DO_NODE_ID)
      .map((n) => {
        const p = absPos(n);
        const w = Number((n.style as { width?: number } | undefined)?.width ?? DO_DEFAULT_WIDTH);
        const h = Number((n.style as { height?: number } | undefined)?.height ?? DO_DEFAULT_HEIGHT);
        return { id: n.id, x: p.x, y: p.y, w, h };
      });
    const inferParent = (n: Node<SpiderNodeData>): string | null => {
      if (n.data.spec.id === DO_NODE_ID) {
        // A nested DO keeps whatever React Flow has — don't geometrically
        // re-nest DOs here, they're already handled by the drop path.
        return n.parentNode ?? null;
      }
      const p = absPos(n);
      let best: { id: string; area: number } | null = null;
      for (const b of doBoxes) {
        if (p.x >= b.x && p.x <= b.x + b.w &&
            p.y >= b.y && p.y <= b.y + b.h) {
          const area = b.w * b.h;
          if (best === null || area < best.area) best = { id: b.id, area };
        }
      }
      return best ? best.id : null;
    };
    const engineNodes = nodes.map((n) => ({
      id: n.id, type: n.data.spec.id,
      params: n.data.params ?? {},
      parent: inferParent(n),
    }));
    const engineEdges = edges.map((e) => ({
      src: e.source, srcPort: e.sourceHandle ?? "",
      dst: e.target, dstPort: e.targetHandle ?? "",
    }));
    await challenge.submit(selectedId, engineNodes, engineEdges);
  }, [challenge, nodes, edges, selectedId]);

  const endChallenge = useCallback(() => {
    challenge.close();
    setNodes([]);
    setEdges([]);
    setSelectedId(null);
    setSelectedEdgeId(null);
  }, [challenge]);

  return (
    <div className="app">
      <div className="topbar">
        <strong style={{ marginRight: 12 }}>pyspider</strong>
        <button
          className={running ? "run-btn running" : "run-btn"}
          data-tutor-id="run-button"
          onClick={() => {
            if (running) {
              runAbort.current?.abort();
            } else {
              run();
            }
          }}
          title={running ? "Click to stop the current run" : "Run the full graph"}
        >
          {running ? "■ Stop" : "▶ Run graph"}
        </button>
        <button onClick={toggleBackend} disabled={!health.cuda_available && health.backend === "cpu"}>
          Toggle backend
        </button>
        <span className={`backend-tag ${health.backend === "cuda" ? "cuda" : ""}`}>
          {health.backend.toUpperCase()}
          {!health.cuda_available && " (no CUDA)"}
        </span>
        <button onClick={() => setShowSettings(true)} title="Settings">⚙</button>
        <button onClick={() => setShowLessons(true)}
                title="Open interactive tutorial">
          📖 Tutorial
        </button>
        <button onClick={() => setShowChallenges(true)}
                title="Start a palette-limited puzzle challenge">
          🎯 Challenge
        </button>
        <span style={{ marginLeft: "auto", color: "var(--muted)", fontSize: 11 }}>
          {nodes.length} nodes · {edges.length} edges
        </span>
        <button
          style={{ marginLeft: 16 }}
          onClick={() => {
            if (nodes.length === 0 && edges.length === 0) return;
            const msg = `Clear the canvas? This removes ${nodes.length} node(s) `
                      + `and ${edges.length} edge(s). This cannot be undone.`;
            if (!window.confirm(msg)) return;
            setNodes([]);
            setEdges([]);
            setSelectedId(null);
            setSelectedEdgeId(null);
            setCtxMenu(null);
          }}
          disabled={nodes.length === 0 && edges.length === 0}
          title="Remove all nodes and edges from the canvas">
          🗑 Clear
        </button>
      </div>

      {showSettings && (
        <SettingsDialog
          settings={settings}
          onSave={async (patch) => {
            try {
              const next = await updateSettings(patch);
              setSettings(next);
              setShowSettings(false);
            } catch (e) { showError(e); }
          }}
          onClose={() => setShowSettings(false)}
        />
      )}

      <Palette specs={specs} onDragStart={onDragStart}
               forceExpandCategory={tutorial.expandCategory}
               allowList={challenge.session?.allowed_nodes ?? null} />

      <div className="canvas" onDrop={onDrop} onDragOver={onDragOver}>
        <ReactFlow
          nodes={nodesWithWireInfo}
          edges={edgesTyped}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onConnectStart={onConnectStart}
          onConnectEnd={onConnectEnd}
          onNodeClick={(_e, n) => { setSelectedId(n.id); setSelectedEdgeId(null); }}
          onEdgeClick={(_e, edge) => { setSelectedEdgeId(edge.id); setSelectedId(null); }}
          onPaneClick={() => { setSelectedId(null); setSelectedEdgeId(null); setCtxMenu(null); }}
          onNodeContextMenu={(e, n) => {
            e.preventDefault();
            // Only offer the menu when the target has no required inputs —
            // otherwise a standalone run is guaranteed to fail (missing input).
            if (!canRunStandalone(n as Node<SpiderNodeData>)) {
              setCtxMenu(null);
              return;
            }
            setCtxMenu({ x: e.clientX, y: e.clientY, nodeId: n.id });
          }}
          fitView
          fitViewOptions={{ padding: 0.15, minZoom: 0.05, maxZoom: 2 }}
          minZoom={0.05}
          maxZoom={3}
          connectionLineComponent={ConnectionLineLive}
        >
          <Background gap={16} color="#333" />
          <MiniMap pannable zoomable maskColor="rgba(0,0,0,0.6)" />
          <Controls />
        </ReactFlow>
      </div>

      <Inspector
        node={selected}
        edge={selectedEdge}
        onParamChange={onParamChange}
        wiredParams={selectedWired}
        onDeleteEdge={() => {
          if (!selectedEdgeId) return;
          setEdges((es) => es.filter((e) => e.id !== selectedEdgeId));
          setSelectedEdgeId(null);
        }}
      />

      {ctxMenu && (
        <NodeContextMenu
          x={ctxMenu.x}
          y={ctxMenu.y}
          onRun={() => runOne(ctxMenu.nodeId)}
          onClose={() => setCtxMenu(null)}
          disabled={running}
        />
      )}

      {showLessons && (
        <div className="tutor-picker-backdrop"
             onClick={() => { setShowLessons(false); setPickedCategory(null); }}>
          <div className="tutor-picker" onClick={(e) => e.stopPropagation()}>
            {/* Group lessons by category in first-seen order. */}
            {(() => {
              const groups = new Map<string, typeof tutorial.availableLessons>();
              for (const l of tutorial.availableLessons) {
                const cat = l.category ?? "Uncategorised";
                const g = groups.get(cat) ?? [];
                g.push(l);
                groups.set(cat, g);
              }
              if (tutorial.availableLessons.length === 0) {
                return (
                  <>
                    <div className="tutor-picker-title">Choose a tutorial</div>
                    <div style={{ color: "var(--muted)", fontSize: 12,
                                    padding: "8px 0" }}>Loading tutorials…</div>
                  </>
                );
              }
              const current = pickedCategory !== null
                ? groups.get(pickedCategory) ?? null
                : null;

              if (current) {
                // Step 2 — lessons inside the chosen category.
                return (
                  <>
                    <div className="tutor-picker-title">
                      <button type="button" className="tutor-picker-back"
                              onClick={() => setPickedCategory(null)}
                              aria-label="back to categories">◀</button>
                      {pickedCategory}
                    </div>
                    <div className="tutor-picker-list">
                      {current.map((l) => (
                        <button key={l.id} className="tutor-picker-item"
                                onClick={() => {
                                  tutorial.start(l.id);
                                  setShowLessons(false);
                                  setPickedCategory(null);
                                }}>
                          <div className="tutor-picker-item-title">{l.title}</div>
                          <div className="tutor-picker-item-summary">{l.summary}</div>
                        </button>
                      ))}
                    </div>
                  </>
                );
              }

              // Step 1 — category buttons.
              return (
                <>
                  <div className="tutor-picker-title">Choose a tutorial</div>
                  <div className="tutor-picker-cats">
                    {[...groups.entries()].map(([cat, lessons]) => (
                      <button key={cat} className="tutor-picker-cat"
                              onClick={() => setPickedCategory(cat)}>
                        <span className="tutor-picker-cat-name">{cat}</span>
                        <span className="tutor-picker-cat-count">
                          {lessons.length} lessons
                        </span>
                      </button>
                    ))}
                  </div>
                </>
              );
            })()}
            <button className="tutor-picker-close"
                    onClick={() => { setShowLessons(false); setPickedCategory(null); }}>
              Cancel
            </button>
          </div>
        </div>
      )}

      {errorPopup !== null && (
        <div className="settings-overlay" onClick={() => setErrorPopup(null)}>
          <div className="settings-dialog"
               style={{ maxWidth: 680, minWidth: 420 }}
               onClick={(e) => e.stopPropagation()}>
            <h3 style={{ color: "#ff6b6b", marginTop: 0 }}>Error</h3>
            <textarea
              readOnly
              value={errorPopup}
              style={{
                width: "100%", height: 200, fontFamily: "monospace",
                fontSize: 11, background: "#0b0c10", color: "#eee",
                border: "1px solid #333", padding: 8, borderRadius: 4,
                resize: "vertical", whiteSpace: "pre",
              }}
              onFocus={(e) => e.currentTarget.select()}
            />
            <div style={{ display: "flex", justifyContent: "flex-end",
                          gap: 8, marginTop: 12 }}>
              <button onClick={() => {
                navigator.clipboard?.writeText(errorPopup).catch(() => {});
              }}>Copy</button>
              <button onClick={() => setErrorPopup(null)}>Close</button>
            </div>
          </div>
        </div>
      )}

      <TutorialOverlay state={tutorial} nodes={nodesWithWireInfo} edges={edges} />
      {tutorial.completed && (
        <TutorialComplete
          title={tutorial.completed.title}
          onClose={tutorial.clearCompleted}
        />
      )}

      <ChallengePanel
        state={challenge}
        pickerOpen={showChallenges}
        onClosePicker={() => setShowChallenges(false)}
        onPickChallenge={startChallenge}
        onSubmit={submitChallenge}
        onEnd={endChallenge}
        selectedNodeId={selectedId}
        selectedNodeLabel={selected?.data.spec.label ?? null}
      />

    </div>
  );
}


function NodeContextMenu({ x, y, onRun, onClose, disabled }: {
  x: number; y: number;
  onRun: () => void; onClose: () => void;
  disabled?: boolean;
}) {
  // Close on any outside click or Escape — mount a one-shot listener while
  // the menu is visible.
  useEffect(() => {
    const closeOnKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    const closeOnClick = () => onClose();
    window.addEventListener("keydown", closeOnKey);
    window.addEventListener("mousedown", closeOnClick);
    return () => {
      window.removeEventListener("keydown", closeOnKey);
      window.removeEventListener("mousedown", closeOnClick);
    };
  }, [onClose]);

  return (
    <div
      className="node-context-menu"
      style={{ left: x, top: y }}
      onMouseDown={(e) => e.stopPropagation()}
    >
      <button
        className="ctx-item"
        disabled={disabled}
        onMouseDown={(e) => e.stopPropagation()}
        onClick={() => { onRun(); onClose(); }}
        title="Execute this node alone using its current parameter values"
      >
        ▶ Run this node
      </button>
    </div>
  );
}


function SettingsDialog({ settings, onSave, onClose }: {
  settings: AppSettings;
  onSave: (patch: Partial<AppSettings>) => void;
  onClose: () => void;
}) {
  const [dir, setDir] = useState(settings.default_save_dir);
  const [browsing, setBrowsing] = useState(false);
  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-dialog" onClick={(e) => e.stopPropagation()}>
        <h3>Settings</h3>
        <label>
          <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 4 }}>
            Default save directory
          </div>
          <div style={{ display: "flex", gap: 4 }}>
            <input
              type="text"
              value={dir}
              onChange={(e) => setDir(e.target.value)}
              style={{ flex: 1 }}
            />
            <button onClick={() => setBrowsing(true)} title="Browse...">📁</button>
          </div>
        </label>
        <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 6 }}>
          Used when a node's "save" checkbox is on but no per-output directory is set.
          Files are written as MRC (images/volumes) or STAR (docs, scalars, strings).
        </div>
        <div style={{ display: "flex", justifyContent: "flex-end",
                      gap: 8, marginTop: 16 }}>
          <button onClick={onClose}>Cancel</button>
          <button onClick={() => onSave({ default_save_dir: dir })}>Save</button>
        </div>
        {browsing && (
          <FileBrowser
            initialPath={dir}
            onSelect={(path) => { setDir(path); setBrowsing(false); }}
            onClose={() => setBrowsing(false)}
          />
        )}
      </div>
    </div>
  );
}


export default function App() {
  const [specs, setSpecs] = useState<NodeSpec[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [bootStage, setBootStage] = useState<string>("starting");

  useEffect(() => {
    py.onBootProgress((stage) => setBootStage(stage));
    fetchNodes()
      .then(setSpecs)
      .catch((e) => setError(String(e)));
  }, []);

  if (error) {
    return (
      <div style={{ padding: 20 }}>
        <h2>Boot failed</h2>
        <pre>{error}</pre>
      </div>
    );
  }
  if (!specs) {
    const labels: Record<string, string> = {
      starting: "starting",
      script:   "downloading Pyodide runtime…",
      runtime:  "initialising Python…",
      packages: "loading numpy / scipy / Pillow…",
      bundle:   "unpacking backend…",
      plugins:  "registering nodes…",
      ready:    "finalising…",
    };
    return (
      <div style={{ padding: 20, fontFamily: "sans-serif" }}>
        <h3 style={{ margin: 0 }}>pyspider</h3>
        <p style={{ color: "#888", fontSize: 13 }}>
          {labels[bootStage] ?? bootStage}
        </p>
      </div>
    );
  }

  return (
    <ReactFlowProvider>
      <Canvas specs={specs} />
    </ReactFlowProvider>
  );
}

import { Fragment, useCallback, useEffect } from "react";
import { Handle, Position, NodeProps, useUpdateNodeInternals } from "reactflow";
import type { NodeSpec, NodeResult, Param, SaveConfig } from "./types";

export type NodeRunStatus = "idle" | "running" | "success" | "error";

export interface SpiderNodeData {
  spec: NodeSpec;
  params: Record<string, unknown>;
  result?: NodeResult;
  wiredParams?: Set<string>;
  status?: NodeRunStatus;
  /** Per-output save configuration; missing key = save disabled. */
  saveConfig?: SaveConfig;
  /** 0..1 progress from the latest node_progress event (running only). */
  progress?: number;
  /** Live log lines from the latest run; wiped when a fresh run starts and
   *  again when the run completes (per user's design decision). */
  logs?: string[];
  /** True when the node has at least one edge (in or out). Used to reserve
   *  preview space so nothing suddenly grows after Run and pushes other
   *  nodes around — the layout the user sees at wire-time matches the
   *  post-run layout. */
  hasWires?: boolean;
  /** Predicted output shape from /api/predict_shapes — populated by App on
   *  every graph change. Used to size the pre-Run preview slot to the
   *  correct aspect (clock 256² → PO 128×~804) instead of an arbitrary
   *  square. May be undefined for nodes the predictor doesn't know. */
  predictedShape?: number[];
  /** Names of output ports on THIS node that currently feed at least one
   *  edge. Paired with `hideUnconnectedOutputs` to render a cleaner
   *  "answer node" during challenges (no dangling output handles). */
  connectedOutputs?: Set<string>;
  hideUnconnectedOutputs?: boolean;
}

const ROW_HEIGHT = 16;
const HEADER_HEIGHT = 40;    // top area reserved for category + title
const BOTTOM_PAD = 6;

export default function SpiderNode({ id, data, selected }: NodeProps<SpiderNodeData>) {
  const { spec, params, result, wiredParams } = data;
  const wired = wiredParams ?? new Set<string>();

  // React Flow caches node dimensions until it sees a resize. When a preview
  // image loads AFTER first render, the node's DOM grows but React Flow's
  // internal size stays stale — handles land in wrong positions and other
  // nodes' auto-layout doesn't shove aside. `updateNodeInternals(id)`
  // forces a re-measure on load. Fires per <img> so a stack-thumbnail
  // filmstrip or a subsequent re-run also triggers it.
  const updateNodeInternals = useUpdateNodeInternals();
  const onImgLoad = useCallback(
    () => updateNodeInternals(id),
    [updateNodeInternals, id],
  );

  // Predicted shape (from /api/predict_shapes) changes size → node's
  // handles need to re-measure so the drag-preview line stays anchored
  // to the CENTRE of the moved handle rather than a stale old position
  // (fixes: "wire preview line doesn't follow the cursor"). We also
  // ping updateNodeInternals when the wire count changes because that
  // toggles the preview slot on/off, which shifts handle positions.
  const predictedShapeKey = (data.predictedShape ?? []).join(",");
  // Any param that gates a port/param via `depends_on` can change which
  // handles are visible. We stringify the values of such gate params so
  // React Flow re-measures handle positions whenever one flips.
  const gateKeys = Array.from(new Set([
    ...spec.inputs.map((p) => p.depends_on?.[0]).filter(Boolean) as string[],
    ...spec.outputs.map((p) => p.depends_on?.[0]).filter(Boolean) as string[],
    ...spec.params.map((p) => p.depends_on?.[0]).filter(Boolean) as string[],
  ]));
  const gateSnapshot = gateKeys.map((k) => `${k}=${String(params?.[k])}`).join("|");
  useEffect(() => {
    updateNodeInternals(id);
  }, [updateNodeInternals, id, predictedShapeKey, data.hasWires, gateSnapshot]);

  const outs = result?.outputs ?? {};
  // Prefer a port named `preview` if it exists — some nodes (mask.circle,
  // mask.gaussian, PK …) publish a purpose-built visualisation there while
  // keeping the raw scientific output on other ports for downstream use.
  const pngEntry = Object.entries(outs).find(([name, v]) => name === "preview" && v.png)
                ?? Object.entries(outs).find(([, v]) => v.png);
  const scalarEntry = Object.entries(outs).find(([, v]) => v.kind === "scalar");

  const status = data.status ?? "idle";
  const cls = ["spider-node", `status-${status}`];
  if (selected) cls.push("selected");
  if (status !== "running" && result && !result.ok) cls.push("error");

  // Generic depends_on check — used for both Params and Ports. A port whose
  // gate is unmet is hidden from the node face (its wire handle disappears);
  // params fall back to "keep visible when a wire is already connected" so
  // an edge isn't orphaned mid-render when the user toggles a selector.
  const dependsOnSatisfied = (item: { depends_on?: [string, unknown] | null }): boolean => {
    if (!item.depends_on) return true;
    const [other, want] = item.depends_on;
    const otherVal = params?.[other] ?? spec.params.find((q) => q.name === other)?.default;
    return Array.isArray(want) ? want.includes(otherVal) : otherVal === want;
  };
  const visibleInputs = spec.inputs.filter(dependsOnSatisfied);
  const connectedOuts = data.connectedOutputs ?? new Set<string>();
  const visibleOutputs = spec.outputs
    .filter(dependsOnSatisfied)
    .filter((p) => !data.hideUnconnectedOutputs || connectedOuts.has(p.name));
  const wireableParams: Param[] = spec.params.filter(
    (p) => p.wireable && (dependsOnSatisfied(p) || wired.has(p.name)),
  );
  const leftRowCount = visibleInputs.length + wireableParams.length;
  const rightRowCount = visibleOutputs.length;
  const portsHeight = Math.max(leftRowCount, rightRowCount) * ROW_HEIGHT;

  // Nodes with many wireable params + outputs (e.g. align.ml_angle) can have
  // long left + right labels colliding at 150px min-width. Estimate the widest
  // (left + right) label pair and bump min-width so no row overlaps. Char
  // widths are approximations tuned to the two font sizes used (10 px for
  // I/O ports, 9 px monospace for params).
  const chW = 5.5;    // avg glyph width at the smaller font size
  const sidePad = 24; // 10px + 10px + a bit of safety
  const leftTexts = [
    ...visibleInputs.map((p) => `◂ ${p.name}`),
    ...wireableParams.map((p) => `◇ ${p.name}`),
  ];
  const rightTexts = visibleOutputs.map((p) => `${p.name} ▸`);
  const nRows = Math.max(leftTexts.length, rightTexts.length);
  let dynamicMinWidth = 150;
  for (let i = 0; i < nRows; i++) {
    const l = leftTexts[i]?.length ?? 0;
    const r = rightTexts[i]?.length ?? 0;
    const w = Math.ceil((l + r) * chW) + sidePad;
    if (w > dynamicMinWidth) dynamicMinWidth = w;
  }

  // Helper: compute the *vertical centre* of row `i` relative to the node top.
  // Both the Handle and its label are positioned at exactly this y, so they
  // are guaranteed to line up — no font-metric guesswork.
  const leftCenter  = (i: number) => HEADER_HEIGHT + i * ROW_HEIGHT + ROW_HEIGHT / 2;
  const rightCenter = (i: number) => HEADER_HEIGHT + i * ROW_HEIGHT + ROW_HEIGHT / 2;

  return (
    <div className={cls.join(" ")} data-tutor-id={`node:${spec.id}`}
         style={{ minWidth: `${dynamicMinWidth}px` }}>
      <div className="header">
        <div className="category">{spec.category}</div>
        <div className="title">{spec.label}</div>
      </div>

      {/* Reserve vertical space for all port rows; handles + labels are
          absolutely positioned inside `.spider-node` (which is
          position: relative), sharing the same `top` per row so they align. */}
      <div className="ports-space" style={{ height: portsHeight + BOTTOM_PAD }} />

      {visibleInputs.map((p, i) => {
        const top = leftCenter(i);
        // `data-tutor-handle` lets the tutorial spotlight blink the right
        // connection point on wire steps. Format: "dst:<spec.id>:<port>".
        return (
          <Fragment key={`in-${p.name}`}>
            <Handle type="target" position={Position.Left}
                    id={p.name} style={{ top }}
                    data-tutor-handle={`dst:${spec.id}:${p.name}`}
                    data-tutor-nid={id} />
            <span className="port-label port-label-left port-io"
                  style={{ top }}>
              ◂ {p.name}
            </span>
          </Fragment>
        );
      })}

      {wireableParams.map((p, i) => {
        const top = leftCenter(visibleInputs.length + i);
        return (
          <Fragment key={`pp-${p.name}`}>
            <Handle type="target" position={Position.Left} id={p.name}
                    className="param-handle" style={{ top }}
                    data-tutor-handle={`dst:${spec.id}:${p.name}`}
                    data-tutor-nid={id} />
            <span className={"port-label port-label-left port-param" +
                             (wired.has(p.name) ? " wired" : "")}
                  style={{ top }}>
              ◇ {p.name}
            </span>
          </Fragment>
        );
      })}

      {visibleOutputs.map((p, i) => {
        const top = rightCenter(i);
        return (
          <Fragment key={`out-${p.name}`}>
            <span className="port-label port-label-right port-io"
                  style={{ top }}>
              {p.name} ▸
            </span>
            <Handle type="source" position={Position.Right}
                    id={p.name} style={{ top }}
                    data-tutor-handle={`src:${spec.id}:${p.name}`}
                    data-tutor-nid={id} />
          </Fragment>
        );
      })}

      {pngEntry ? (
        (() => {
          // Explicit display dimensions from the output's ACTUAL pixel
          // shape — the browser doesn't guess. Cap at MAX_W × MAX_H while
          // preserving aspect. 256² fits at natural size (256×256 < 480×360);
          // extreme aspects like polar 128×804 hit the width cap and render
          // proportionally shorter.
          const MAX_W = 480, MAX_H = 360;
          const shape = (pngEntry[1] as { shape?: number[] }).shape ?? [];
          // Shape may be [H, W] (grayscale 2D), [H, W, 3] (RGB), or
          // [N, H, W] (stack — middle slice shown, so use last two dims).
          let iw = 0, ih = 0;
          if (shape.length >= 2) {
            if (shape.length === 3 && shape[2] === 3) {          // (H, W, 3)
              ih = shape[0]; iw = shape[1];
            } else if (shape.length === 3) {                     // (N, H, W)
              ih = shape[1]; iw = shape[2];
            } else {                                              // (H, W)
              ih = shape[0]; iw = shape[1];
            }
          }
          const style: React.CSSProperties = {};
          if (iw > 0 && ih > 0) {
            const scale = Math.min(MAX_W / iw, MAX_H / ih, 1);
            style.width = Math.round(iw * scale);
            style.height = Math.round(ih * scale);
          }
          return (
            <img alt={pngEntry[0]} onLoad={onImgLoad} style={style}
                 src={`data:image/png;base64,${pngEntry[1].png}`} />
          );
        })()
      ) : (
        // Pre-Run preview slot sized by the predictor. For nodes the
        // predictor knows (clock face, PO, CC, PK, most passthroughs) we
        // set explicit width/height so the layout at wire-time already
        // matches the post-Run layout — no surprise growth on Run.
        // Unknown-shape nodes fall back to a thin stub so they at least
        // show that a preview will appear.
        data.hasWires
        && spec.outputs.some((p) =>
             p.kind === "image" || p.kind === "stack" || p.kind === "volume")
        && (() => {
          const MAX_W = 480, MAX_H = 360;
          const s = data.predictedShape;
          let iw = 0, ih = 0;
          if (s && s.length >= 2) {
            if (s.length === 3 && s[2] === 3)       { ih = s[0]; iw = s[1]; }
            else if (s.length === 3)                { ih = s[1]; iw = s[2]; }
            else                                     { ih = s[0]; iw = s[1]; }
          }
          if (iw > 0 && ih > 0) {
            const scale = Math.min(MAX_W / iw, MAX_H / ih, 1);
            return <div className="preview-slot" style={{
              width: Math.round(iw * scale),
              height: Math.round(ih * scale),
            }} />;
          }
          // Calculator with all-scalar inputs → no shape → no stub either
          // (the node is purely arithmetic, a preview would just be noise).
          if (spec.id === "arith.calc") return null;
          return <div className="preview-stub">▷ preview appears after Run</div>;
        })()
      )}
      {!pngEntry && scalarEntry && (
        <div className="scalar-out" title={String(scalarEntry[1].value ?? "")}>
          {scalarEntry[0]} = {formatScalar(scalarEntry[1].value)}
        </div>
      )}
      {result && !result.ok && <div className="err">{result.error}</div>}
    </div>
  );
}


function formatScalar(v: unknown, max = 40): string {
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(4);
  const s = String(v ?? "");
  const firstNewline = s.indexOf("\n");
  const hasMore = firstNewline !== -1;
  const firstLine = hasMore ? s.slice(0, firstNewline) : s;
  if (firstLine.length > max) return firstLine.slice(0, max) + "…";
  return hasMore ? firstLine + " …" : firstLine;
}



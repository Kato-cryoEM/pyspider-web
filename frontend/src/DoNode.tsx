import { Handle, NodeProps, NodeResizer, Position } from "reactflow";
import type { SpiderNodeData } from "./SpiderNode";

/**
 * Container node for the `flow.do` control-flow primitive.
 *
 * Visual: a resizable red-bordered rectangle. Only the header is a drag
 * handle (`.do-header`), so children can be interacted with normally
 * without accidentally dragging the container.
 *
 * The DO node exposes a scalar `i` output on the right edge of its header.
 * Users wire that into any child's parameter to make it depend on the
 * current loop iteration. Backend engine iterates the child subgraph
 * per SPIDER DO semantics: i = start, start+step, ..., ≤ end.
 */
export default function DoNode({ data, selected }: NodeProps<SpiderNodeData>) {
  const { spec, params } = data;
  const start = Number(params.start ?? 1);
  const end = Number(params.end ?? 10);
  const step = Number(params.step ?? 1);
  const count = step === 0
    ? 0
    : Math.max(0, Math.floor((end - start) / step) + 1);

  const status = data.status ?? "idle";
  // Parse the latest `iter k / N` log line — the engine emits one per
  // iteration via node_progress, so the DO node face can read the live
  // iteration count without any new plumbing.
  const latestLog = (data.logs ?? []).slice(-1)[0] ?? "";
  const iterMatch = latestLog.match(/iter\s+(\d+)\s*\/\s*(\d+)/);
  const liveIter = iterMatch
    ? { now: Number(iterMatch[1]), total: Number(iterMatch[2]) }
    : null;
  return (
    <div className={`do-container status-${status} ${selected ? "selected" : ""}`}>
      <NodeResizer
        minWidth={280}
        minHeight={180}
        isVisible={selected}
        color="#e57373"
        handleStyle={{ width: 8, height: 8, background: "#e57373",
                       border: "1px solid #ffdcdc" }}
      />

      <div className="do-header">
        <div className="do-title">
          {spec.label}
          {liveIter && status === "running" && (
            <span className="do-iter-badge">
              iter {liveIter.now} / {liveIter.total}
            </span>
          )}
        </div>
        <div className="do-info">
          i = {start} .. {end} step {step}  ({count} passes)
        </div>
      </div>

      {/* Loop-counter chip lives INSIDE the container body, top-left,
          with the handle protruding to the right — so children dropped
          into the DO can wire from `i` naturally without a wire that has
          to leave and re-enter the container. */}
      <div className="do-i-chip">
        <span>i&nbsp;▸</span>
        <Handle
          type="source"
          position={Position.Right}
          id="i"
          className="do-i-handle"
        />
      </div>

      <div className="do-hint">
        drop nodes here · wire <code>i</code> into any child param
      </div>
    </div>
  );
}

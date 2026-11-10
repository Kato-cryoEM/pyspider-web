import { useCallback, useEffect, useRef, useState } from "react";
import type { Edge, Node } from "reactflow";
import type { SpiderNodeData } from "./SpiderNode";
import type { OutputPreview, EdgeInfo } from "./types";
import Volume3D from "./Volume3D";
import FilterRing from "./FilterRing";
import MaskProfile from "./MaskProfile";
import SliceDiagram from "./SliceDiagram";
import SumAxisDiagram from "./SumAxisDiagram";
import RotateVolumeDiagram from "./RotateVolumeDiagram";
import VoeaSphereDiagram from "./VoeaSphereDiagram";

interface Props {
  node: Node<SpiderNodeData> | null;
  edge?: Edge | null;
  onParamChange: (nodeId: string, name: string, value: unknown) => void;
  onDeleteEdge?: () => void;
  /** Which params on the selected node are currently driven by a wire. */
  wiredParams?: Set<string>;
}

/**
 * One parameter row. Numeric inputs use a local "draft" buffer so partial
 * entries like "0." or "0.0" survive React's controlled-input round-trip
 * (which would otherwise collapse them to the parsed number and strip the
 * trailing dot, making decimals impossible to type). We commit to the
 * parent only when the draft parses to a finite number, keep the draft
 * visible otherwise, and resync whenever an external change (wire, undo,
 * another node's edit) lands.
 */
interface ParamSpec {
  name: string;
  kind: string;
  default: unknown;
  min?: number | null;
  max?: number | null;
  step?: number | null;
  wireable?: boolean | null;
  placeholder?: string | null;
}

interface ParamInputProps {
  param: ParamSpec;
  value: unknown;
  isWired: boolean;
  onChange: (v: unknown) => void;
}

function ParamInput({ param: p, value, isWired, onChange }: ParamInputProps) {
  const isNumeric = p.kind === "int" || p.kind === "float";
  const inputType = p.kind === "string" ? "text" : "number";
  const externalStr = value === undefined || value === null
    ? "" : String(value);
  // Local draft only for numeric inputs — strings commit as-is.
  const [draft, setDraft] = useState<string>(externalStr);
  // Resync when the upstream value changes to a value that doesn't match
  // the current parsed draft (external edit, wire activation, etc.).
  useEffect(() => {
    if (!isNumeric) {
      setDraft(externalStr);
      return;
    }
    const parsedDraft = p.kind === "int" ? parseInt(draft, 10) : parseFloat(draft);
    const externalNum = typeof value === "number" ? value : NaN;
    if (!Number.isFinite(parsedDraft) || parsedDraft !== externalNum) {
      setDraft(externalStr);
    }
    // Only resync when the EXTERNAL value changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [externalStr, isNumeric]);

  const isEmpty = p.kind === "string" && externalStr.length === 0;
  return (
    <div className="param" data-tutor-id={`param:${p.name}`}>
      <label>
        {p.name}
        {p.wireable && !isWired && <span className="wire-hint">◇</span>}
        {isWired && <span className="wired-tag">← wired</span>}
      </label>
      <input
        type={inputType}
        value={isWired ? "" : draft}
        placeholder={isWired
          ? "(supplied by wire)"
          : (p.placeholder ?? "")}
        step={p.step ?? (p.kind === "float" ? 0.01 : 1)}
        min={p.min ?? undefined}
        max={p.max ?? undefined}
        disabled={isWired}
        className={(isEmpty || isWired) ? "is-empty" : ""}
        onChange={(e) => {
          const raw = e.target.value;
          setDraft(raw);
          if (!isNumeric) {
            onChange(raw);
            return;
          }
          const parsed = p.kind === "int" ? parseInt(raw, 10) : parseFloat(raw);
          // Only commit when the draft is a well-formed number — this
          // lets "0." / "-" / "" stay in the field without round-tripping
          // through the parent and getting normalised back to 0.
          if (Number.isFinite(parsed)) onChange(parsed);
        }}
        onBlur={() => {
          // On blur, if the field was left in a half-typed state, snap
          // back to the committed value so a stray "0." never ships.
          if (!isNumeric) return;
          const parsed = p.kind === "int" ? parseInt(draft, 10) : parseFloat(draft);
          if (!Number.isFinite(parsed)) setDraft(externalStr);
        }}
      />
    </div>
  );
}

export default function Inspector({
  node, edge, onParamChange, onDeleteEdge, wiredParams,
}: Props) {
  const [docOpen, setDocOpen] = useState(true);

  if (edge) {
    const info = (edge.data?.info ?? { status: "unknown" }) as EdgeInfo;
    const statusLabel = { ok: "OK", warn: "WARNING", error: "ERROR",
                          unknown: "?" }[info.status];
    const statusColor = { ok: "var(--ok)", warn: "#ffa64d",
                          error: "var(--err)", unknown: "var(--muted)"
                        }[info.status];
    return (
      <div className="inspector">
        <div className="ins-header">
          <div className="ins-cat">CONNECTION</div>
          <div className="ins-title">
            <span style={{ color: statusColor }}>[{statusLabel}]</span>
            &nbsp;wire
          </div>
          <div className="ins-ports">
            {edge.source} → {edge.target}
          </div>
        </div>
        <div className="ins-section-title">Endpoints</div>
        <div style={{ fontSize: 12, fontFamily: "ui-monospace, monospace" }}>
          <div><b>from</b> {edge.source} <span style={{ color: "var(--muted)" }}>·</span> {edge.sourceHandle} <span style={{ color: statusColor }}>({info.srcKind ?? "?"})</span></div>
          <div style={{ marginTop: 4 }}><b>to&nbsp;&nbsp;</b> {edge.target} <span style={{ color: "var(--muted)" }}>·</span> {edge.targetHandle} <span style={{ color: statusColor }}>({info.dstKind ?? "?"})</span></div>
        </div>
        {info.message && (
          <>
            <div className="ins-section-title">Diagnostics</div>
            <div style={{ fontSize: 12, lineHeight: 1.55, color: "var(--text)",
                          background: info.status === "error" ? "rgba(255,74,74,0.10)"
                                     : info.status === "warn"  ? "rgba(255,166,77,0.10)"
                                     : "var(--bg)",
                          border: `1px solid ${statusColor}`,
                          borderRadius: 4, padding: "8px 10px" }}>
              {info.message}
            </div>
          </>
        )}
        <div style={{ marginTop: 16 }}>
          <button onClick={onDeleteEdge}
                  style={{ background: "var(--panel2)", color: "var(--err)",
                           border: "1px solid var(--err)", padding: "4px 10px",
                           borderRadius: 3, cursor: "pointer" }}>
            Delete this connection
          </button>
        </div>
      </div>
    );
  }

  if (!node) {
    return (
      <div className="inspector">
        <h3>Inspector</h3>
        <div style={{ fontSize: 12, color: "var(--muted)" }}>
          Click a node or an edge to inspect it.
        </div>
      </div>
    );
  }
  const { spec, params, result } = node.data;
  const wired = wiredParams ?? new Set<string>();
  const logs = node.data.logs;
  const portsSummary =
    (spec.inputs.length
      ? `IN: ${spec.inputs.map((p) => `${p.name}:${p.kind}`).join(", ")}`
      : "IN: (source)") +
    "   " +
    (spec.outputs.length
      ? `OUT: ${spec.outputs.map((p) => `${p.name}:${p.kind}`).join(", ")}`
      : "OUT: (sink)");

  return (
    <div className="inspector">
      <div className="ins-header">
        <div className="ins-cat">{spec.category}</div>
        <div className="ins-title">{spec.label}</div>
        <div className="ins-ports">{portsSummary}</div>
      </div>

      {logs && logs.length > 0 && (
        <div className="ins-log-panel">
          <pre className="ins-log" ref={(el) => {
            if (el) el.scrollTop = el.scrollHeight;
          }}>
            {logs.join("\n")}
          </pre>
        </div>
      )}

      {spec.doc && (
        <div className={`doc-block ${docOpen ? "open" : "collapsed"}`}>
          <div
            className="doc-heading"
            onClick={() => setDocOpen((s) => !s)}
            title="Click to toggle"
          >
            <span className="doc-caret">{docOpen ? "▾" : "▸"}</span>
            <span>About this node</span>
          </div>
          {docOpen && <div className="doc">{spec.doc}</div>}
        </div>
      )}

      {spec.params.length > 0 && (
        <div className="ins-section-title">Parameters</div>
      )}

      {spec.params.map((p) => {
        // Hide params whose depends_on gate doesn't match the current value
        // of the referenced param. Backend still uses the schema default.
        // `want` can be a single value or an array of accepted values.
        if (p.depends_on) {
          const [other, want] = p.depends_on;
          const otherVal = params[other] ?? spec.params.find((q) => q.name === other)?.default;
          const match = Array.isArray(want) ? want.includes(otherVal) : otherVal === want;
          if (!match) return null;
        }

        const value = params[p.name] ?? p.default;
        const isWired = wired.has(p.name);

        if (p.kind === "bool") {
          return (
            <div className="param" key={p.name} data-tutor-id={`param:${p.name}`}>
              <label style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
                <input
                  type="checkbox"
                  checked={Boolean(value)}
                  onChange={(e) => onParamChange(node.id, p.name, e.target.checked)}
                  disabled={isWired}
                />
                <span>{p.name}</span>
                {isWired && <span className="wired-tag">← wired</span>}
              </label>
            </div>
          );
        }

        if (p.kind === "select" && p.choices) {
          return (
            <div className="param" key={p.name} data-tutor-id={`param:${p.name}`}>
              <label>
                {p.name}
                {p.wireable && !isWired && <span className="wire-hint">◇</span>}
                {isWired && <span className="wired-tag">← wired</span>}
              </label>
              <select
                value={String(value)}
                onChange={(e) => onParamChange(node.id, p.name, e.target.value)}
                disabled={isWired}
              >
                {p.choices.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
          );
        }

        if (p.kind === "text") {
          const raw = String(value ?? "");
          const numLines = raw.split("\n").length;
          const isEmpty = raw.length === 0;
          return (
            <div className="param" key={p.name} data-tutor-id={`param:${p.name}`}>
              <label>
                {p.name}
                {p.wireable && !isWired && <span className="wire-hint">◇</span>}
                {isWired && <span className="wired-tag">← wired</span>}
                <span className="wire-hint" style={{ marginLeft: "auto" }}>
                  {isEmpty ? "empty" : `${numLines} lines`}
                </span>
              </label>
              <textarea
                value={raw}
                rows={Math.min(14, Math.max(4, numLines + 1))}
                onChange={(e) => onParamChange(node.id, p.name, e.target.value)}
                disabled={isWired}
                placeholder={p.placeholder ?? ""}
                spellCheck={false}
                className={isEmpty ? "is-empty" : ""}
                style={{
                  fontFamily: "ui-monospace, monospace",
                  fontSize: 11,
                  resize: "vertical",
                }}
              />
            </div>
          );
        }

        return (
          <ParamInput
            key={p.name}
            param={p}
            value={value}
            isWired={isWired}
            onChange={(v) => onParamChange(node.id, p.name, v)}
          />
        );
      })}

      {spec.id.startsWith("filter.") && (
        <FilterRing nodeType={spec.id} params={params} />
      )}

      {(spec.id === "mask.circle" || spec.id === "mask.gaussian") && (
        <MaskProfile nodeType={spec.id} params={params} />
      )}

      {spec.id === "recon.slice3d" && (
        <SliceDiagram params={params} />
      )}

      {spec.id === "recon.sum_axis" && (
        <SumAxisDiagram params={params} />
      )}

      {spec.id === "geom.rotate3d" && (
        <RotateVolumeDiagram params={params} />
      )}

      {spec.id === "recon.voea" && (
        <VoeaSphereDiagram params={params} />
      )}

      {result && (
        <>
          <div className="ins-section-title">Preview</div>
          {!result.ok && <div className="err">{result.error}</div>}
          {Object.entries(result.outputs).map(([name, out]) => (
            <OutputBlock key={name} name={name} out={out}
                         savedPath={result.saved?.[name]} />
          ))}
        </>
      )}
    </div>
  );
}


function OutputBlock({ name, out, savedPath }: {
  name: string;
  out: OutputPreview;
  savedPath?: string;
}) {
  // For stack outputs with per-frame thumbnails, remember which frame the
  // user picked so the "big preview" tracks their filmstrip clicks.
  const [selected, setSelected] = useState<number | null>(null);
  const frames = out.frames ?? [];
  const hasFrames = frames.length > 0;
  // Position within the sampled `frames[]` array (not within the full stack
  // when it's larger than max_frames and sparsely sampled). This keeps
  // prev/next jumps aligned with the filmstrip thumbnails.
  const defaultPos = hasFrames ? Math.floor(frames.length / 2) : 0;
  const pos = (() => {
    if (!hasFrames || selected == null) return defaultPos;
    const i = frames.findIndex((f) => f.index === selected);
    return i >= 0 ? i : defaultPos;
  })();
  const chosen = hasFrames && selected != null
    ? frames.find((f) => f.index === selected) ?? null
    : null;
  const bigPng = chosen ? chosen.png : out.png;

  const step = useCallback((delta: number) => {
    if (!hasFrames) return;
    const next = Math.max(0, Math.min(frames.length - 1, pos + delta));
    setSelected(frames[next].index);
  }, [hasFrames, frames, pos]);

  // Auto-scroll the active filmstrip thumbnail into view.
  const stripRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!hasFrames || selected == null || !stripRef.current) return;
    const el = stripRef.current.querySelector<HTMLButtonElement>(
      `button[data-fidx="${selected}"]`);
    el?.scrollIntoView({ behavior: "smooth", block: "nearest",
                         inline: "nearest" });
  }, [selected, hasFrames]);

  // ←/→ frame navigation is scoped to this preview via keyboard focus:
  // clicking the preview arms the keys, clicking a canvas node moves focus
  // away so React Flow's own arrow-key node-nudging keeps working.
  const previewRef = useRef<HTMLDivElement | null>(null);

  return (
    <div
      ref={previewRef}
      className="preview"
      tabIndex={hasFrames ? 0 : undefined}
      onMouseDown={hasFrames ? () => {
        // Grab focus on click so arrow keys route here instead of the
        // canvas. Done on mousedown (not click) so the first keystroke
        // after a pointerup is already captured.
        previewRef.current?.focus();
      } : undefined}
      onKeyDown={hasFrames ? (e) => {
        if (e.key === "ArrowLeft")  { e.preventDefault(); step(-1); }
        if (e.key === "ArrowRight") { e.preventDefault(); step(+1); }
      } : undefined}
    >
      <div style={{ fontSize: 11, color: "var(--muted)",
                    display: "flex", alignItems: "center", gap: 6 }}>
        <span>{name}</span>
        {hasFrames && (
          <>
            <button
              type="button"
              className="frame-nav"
              onClick={() => step(-1)}
              disabled={pos <= 0}
              title="Previous frame (←)"
              aria-label="previous frame"
            >◀</button>
            <span>
              frame {chosen ? chosen.index : frames[defaultPos].index}
              {" / "}{out.n_frames}
            </span>
            <button
              type="button"
              className="frame-nav"
              onClick={() => step(+1)}
              disabled={pos >= frames.length - 1}
              title="Next frame (→)"
              aria-label="next frame"
            >▶</button>
          </>
        )}
      </div>

      {out.volume_u8 && out.volume_shape && out.volume_range ? (
        <Volume3D
          volumeB64={out.volume_u8}
          shape={out.volume_shape as [number, number, number]}
          range={out.volume_range as [number, number]}
        />
      ) : bigPng ? (
        <img alt={name} src={`data:image/png;base64,${bigPng}`} />
      ) : null}

      {hasFrames && (
        <div className="filmstrip" ref={stripRef}>
          {frames.map((f) => (
            <button
              key={f.index}
              data-fidx={f.index}
              className={"filmstrip-item" +
                (selected === f.index ? " active" : "")}
              onClick={() => setSelected(f.index === selected ? null : f.index)}
              title={`frame ${f.index}`}
            >
              <img src={`data:image/png;base64,${f.png}`} alt={`f${f.index}`} />
              <div className="filmstrip-idx">{f.index}</div>
            </button>
          ))}
        </div>
      )}

      {out.kind === "scalar" && (
        <div className="scalar">{String(out.value)}</div>
      )}

      {out.kind === "doc" && (
        <div className="doc-preview">
          <div style={{ fontSize: 10, color: "var(--muted)" }}>
            {out.n_rows} rows (showing first {out.preview?.length})
          </div>
          <table>
            {out.columns && out.columns.length > 0 && (
              <thead>
                <tr>
                  <th className="row-idx">#</th>
                  {out.columns.map((c, j) => <th key={j}>{c}</th>)}
                </tr>
              </thead>
            )}
            <tbody>
              {(out.preview ?? []).map((row, i) => (
                <tr key={i}>
                  <td className="row-idx">{i}</td>
                  {row.map((v, j) => <td key={j}>{
                    Number.isInteger(v) ? String(v) : v.toFixed(4)
                  }</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {out.shape && (
        <div style={{ fontSize: 10, color: "var(--muted)" }}>
          {out.dtype} · {out.shape.join(" × ")}
          {out.min != null && ` · [${out.min.toFixed(2)}, ${out.max?.toFixed(2)}]`}
        </div>
      )}

      {savedPath && (
        <div className={savedPath.startsWith("ERROR") ? "err" : "saved-hint"}>
          {savedPath.startsWith("ERROR") ? "" : "💾 "}{savedPath}
        </div>
      )}
    </div>
  );
}

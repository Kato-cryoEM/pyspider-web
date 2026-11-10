/**
 * Minimal 1D line-plot widget used by every node whose output is a 1D
 * numpy array (e.g. `align.ap.ref_index`, histograms, loss curves).
 *
 * The backend downsamples to ≤ 512 points before shipping the data over,
 * so we can plot without worrying about SVG node counts.
 */

interface Props {
  x: number[];
  y: number[];
  width?: number;
  height?: number;
  /** Shown as small text overlay — "N samples" (full pre-downsample count). */
  nTotal?: number;
  /** Axis-label min/max from the original full array (before downsample). */
  yRange?: [number, number] | null;
}

export default function LinePlot({
  x, y, width = 220, height = 120, nTotal, yRange,
}: Props) {
  if (!y || y.length === 0) {
    return (
      <div style={{ width, height, display: "grid", placeItems: "center",
                    background: "#0b0c10", border: "1px solid #2c313c",
                    borderRadius: 4, color: "#666", fontSize: 10 }}>
        (empty)
      </div>
    );
  }
  const n = y.length;
  const yMin = yRange ? yRange[0] : Math.min(...y);
  const yMax = yRange ? yRange[1] : Math.max(...y);
  const yPad = (yMax - yMin) * 0.05 || 1;
  const lo = yMin - yPad, hi = yMax + yPad;
  const xMin = x.length ? x[0] : 0;
  const xMax = x.length ? x[x.length - 1] : (n - 1);
  const xSpan = xMax - xMin || 1;
  const pad = { l: 28, r: 6, t: 6, b: 16 };
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;
  const mapX = (xv: number) =>
    pad.l + ((xv - xMin) / xSpan) * innerW;
  const mapY = (yv: number) =>
    pad.t + (1 - (yv - lo) / (hi - lo)) * innerH;
  const points = y.map((v, i) => `${mapX(x[i] ?? i)},${mapY(v)}`).join(" ");
  // Zero-line (only if it falls inside the view range).
  const zeroY = lo <= 0 && hi >= 0 ? mapY(0) : null;

  const fmt = (v: number) => {
    const abs = Math.abs(v);
    if (abs !== 0 && (abs < 0.01 || abs >= 10000)) return v.toExponential(1);
    return v.toFixed(abs < 1 ? 3 : (abs < 10 ? 2 : (abs < 100 ? 1 : 0)));
  };

  return (
    <svg width={width} height={height}
         style={{ background: "#0b0c10", borderRadius: 4,
                  border: "1px solid #2c313c", display: "block" }}>
      {/* Y-axis labels */}
      <text x={pad.l - 4} y={pad.t + 6} fontSize={9} textAnchor="end"
            fill="#8fa0b4">{fmt(hi)}</text>
      <text x={pad.l - 4} y={pad.t + innerH + 2} fontSize={9} textAnchor="end"
            fill="#8fa0b4">{fmt(lo)}</text>
      {/* X-axis labels */}
      <text x={pad.l} y={height - 3} fontSize={9} textAnchor="start"
            fill="#8fa0b4">0</text>
      <text x={width - pad.r} y={height - 3} fontSize={9} textAnchor="end"
            fill="#8fa0b4">
        {nTotal != null && nTotal !== n ? nTotal : n}
      </text>
      {/* Axes */}
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={pad.t + innerH}
            stroke="#333" strokeWidth={1} />
      <line x1={pad.l} y1={pad.t + innerH} x2={pad.l + innerW}
            y2={pad.t + innerH} stroke="#333" strokeWidth={1} />
      {zeroY !== null && (
        <line x1={pad.l} y1={zeroY} x2={pad.l + innerW} y2={zeroY}
              stroke="#333" strokeWidth={0.5} strokeDasharray="2 2" />
      )}
      <polyline points={points} fill="none" stroke="#2c6cf6" strokeWidth={1.2} />
    </svg>
  );
}

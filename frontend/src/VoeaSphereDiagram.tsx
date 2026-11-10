import { useMemo } from "react";

/**
 * Live preview for recon.voea. Pure SVG. Shows:
 *   - a wireframe unit sphere (silhouette + meridians + latitudes)
 *   - one radial line + dot per VO EA direction (phi, theta) generated
 *     by the same quasi-even algorithm the backend uses, so the student
 *     can watch the directions redistribute as they tweak delta_theta /
 *     theta_min..max / phi_min..max / symmetry.
 *   - a small axis triad (red x̂ / green ŷ / blue ẑ) for orientation
 *
 * Each viewing direction is encoded as the rotated z-axis under ZYZ
 * Euler (psi=0 in VO EA):  v = (sin θ cos φ, sin θ sin φ, cos θ).
 */

interface Props {
  params: Record<string, unknown>;
  size?: number;
}

type Vec3 = [number, number, number];

const AZIM = (-55 * Math.PI) / 180;
const ELEV = (20 * Math.PI) / 180;

const CAM_DIR: Vec3 = [
  Math.cos(ELEV) * Math.cos(AZIM),
  Math.cos(ELEV) * Math.sin(AZIM),
  Math.sin(ELEV),
];

function project([x, y, z]: Vec3): [number, number] {
  const rx = -Math.sin(AZIM), ry = Math.cos(AZIM);
  const ux = -Math.sin(ELEV) * Math.cos(AZIM);
  const uy = -Math.sin(ELEV) * Math.sin(AZIM);
  const uz = Math.cos(ELEV);
  return [x * rx + y * ry, -(x * ux + y * uy + z * uz)];
}

function depth(p: Vec3): number {
  return p[0] * CAM_DIR[0] + p[1] * CAM_DIR[1] + p[2] * CAM_DIR[2];
}

function num(params: Record<string, unknown>, k: string, def: number): number {
  const v = params[k];
  const n = typeof v === "number" ? v : parseFloat(String(v ?? def));
  return Number.isFinite(n) ? n : def;
}

function str(params: Record<string, unknown>, k: string, def: string): string {
  const v = params[k];
  return typeof v === "string" ? v : def;
}

/**
 * Replicates backend/nodes.py::_fn_voea so the diagram stays in sync
 * without a backend round-trip on every param tweak.
 */
function computeVoeaAngles(params: Record<string, unknown>): Array<[number, number]> {
  let p_lo = num(params, "phi_min", 0);
  let p_hi = num(params, "phi_max", 359.9);
  let t_lo = num(params, "theta_min", 0);
  let t_hi = num(params, "theta_max", 90);
  const dt = Math.max(1e-4, num(params, "delta_theta", 15));
  const sym = str(params, "symmetry", "c1").trim().toLowerCase();

  if (sym.startsWith("c") && /^\d+$/.test(sym.slice(1))) {
    const n = parseInt(sym.slice(1), 10);
    if (n > 1) p_hi = Math.min(p_hi, 360 / n - 1e-6);
  } else if (sym.startsWith("d") && /^\d+$/.test(sym.slice(1))) {
    const n = parseInt(sym.slice(1), 10);
    if (n >= 1) {
      p_hi = Math.min(p_hi, 360 / n - 1e-6);
      t_hi = Math.min(t_hi, 90);
    }
  }

  const out: Array<[number, number]> = [];
  // Hard cap so pathological delta_theta (e.g. 0.5°) doesn't lock the UI.
  const CAP = 2000;
  let theta = t_lo;
  while (theta <= t_hi + 1e-6 && out.length < CAP) {
    const s = Math.max(1e-6, Math.sin((theta * Math.PI) / 180));
    const n_phi = Math.max(1, Math.round(((p_hi - p_lo) * s) / dt));
    if (theta === 0 || theta === 180) {
      out.push([0, theta]);
    } else {
      for (let j = 0; j < n_phi && out.length < CAP; j++) {
        const phi = p_lo + ((p_hi - p_lo) * j) / n_phi;
        out.push([phi, theta]);
      }
    }
    theta += dt;
  }
  return out;
}

function dirVec(phiDeg: number, thetaDeg: number): Vec3 {
  const phi = (phiDeg * Math.PI) / 180;
  const theta = (thetaDeg * Math.PI) / 180;
  const st = Math.sin(theta), ct = Math.cos(theta);
  return [st * Math.cos(phi), st * Math.sin(phi), ct];
}

/** Sampled great-circle / small-circle arc for wireframe drawing. */
function arc(fn: (t: number) => Vec3, t0: number, t1: number, steps: number): Vec3[] {
  const pts: Vec3[] = [];
  for (let i = 0; i <= steps; i++) {
    pts.push(fn(t0 + ((t1 - t0) * i) / steps));
  }
  return pts;
}

export default function VoeaSphereDiagram({ params, size = 220 }: Props) {
  const angles = useMemo(() => computeVoeaAngles(params), [params]);

  const dt = num(params, "delta_theta", 15);
  const sym = str(params, "symmetry", "c1");

  // Fit sphere in the viewport with a bit of margin for axis arrows.
  const r = size * 0.40;
  const cx = size / 2, cy = size / 2;
  const toSvg = ([u, v]: [number, number]): [number, number] => [cx + u * r, cy + v * r];

  // Wireframe: silhouette circle + latitudes at θ = 30/60/90 + meridians
  // every 30° of φ.
  const latitudes = [30, 60, 90].map((thetaDeg) => {
    const pts = arc(
      (phi) => dirVec(phi, thetaDeg),
      0, 360, 48,
    );
    return pts;
  });
  const meridians: Vec3[][] = [];
  for (let phiDeg = 0; phiDeg < 360; phiDeg += 30) {
    meridians.push(arc((t) => dirVec(phiDeg, t), 0, 180, 24));
  }

  // Split a polyline on back/front according to depth so the back half
  // can be drawn dimmer.
  function splitFrontBack(pts: Vec3[]): { front: Vec3[][]; back: Vec3[][] } {
    const front: Vec3[][] = [];
    const back: Vec3[][] = [];
    let cur: Vec3[] = [];
    let curFront: boolean | null = null;
    for (const p of pts) {
      const f = depth(p) >= 0;
      if (curFront === null) { curFront = f; cur = [p]; continue; }
      if (f === curFront) { cur.push(p); continue; }
      // transition: finish current segment
      (curFront ? front : back).push(cur);
      curFront = f;
      cur = [p];
    }
    if (cur.length) (curFront ? front : back).push(cur);
    return { front, back };
  }

  const toPath = (pts: Vec3[]) =>
    pts.map((p, i) => {
      const [sx, sy] = toSvg(project(p));
      return `${i === 0 ? "M" : "L"}${sx.toFixed(1)},${sy.toFixed(1)}`;
    }).join(" ");

  const latPaths = latitudes.map(splitFrontBack);
  const merPaths = meridians.map(splitFrontBack);

  // Axis triad at r*1.15.
  const axisLen = 1.15;
  const axEnd: Vec3[] = [
    [axisLen, 0, 0],
    [0, axisLen, 0],
    [0, 0, axisLen],
  ];
  const axNames = ["x", "y", "z"];
  const axColors = ["#c66", "#6c6", "#88c"];
  const [ox, oy] = toSvg(project([0, 0, 0]));

  // Precompute projected angle endpoints with depth for ordering
  // (draw back lines first, then front).
  const samples = angles.map(([phi, theta]) => {
    const v = dirVec(phi, theta);
    const d = depth(v);
    const [sx, sy] = toSvg(project(v));
    return { phi, theta, sx, sy, d };
  });
  samples.sort((a, b) => a.d - b.d);  // back → front

  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 4 }}>
        VO EA ·
        <b style={{ color: "#e4a94b", margin: "0 4px" }}>{samples.length}</b>
        directions  (Δθ =
        <b style={{ color: "#e4a94b", margin: "0 4px" }}>{dt.toFixed(1)}°</b>
        , sym =
        <b style={{ color: "#e4a94b", margin: "0 4px" }}>{sym}</b>
        )
      </div>
      <svg width={size} height={size}
           style={{ background: "#0f0f14", borderRadius: 6,
                    border: "1px solid #2a2a35", display: "block" }}>
        <defs>
          <marker id="voea-arrow-x" viewBox="0 0 10 10" refX="8" refY="5"
                  markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="#c66" />
          </marker>
          <marker id="voea-arrow-y" viewBox="0 0 10 10" refX="8" refY="5"
                  markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="#6c6" />
          </marker>
          <marker id="voea-arrow-z" viewBox="0 0 10 10" refX="8" refY="5"
                  markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="#88c" />
          </marker>
        </defs>

        {/* Silhouette outline */}
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="#3a3a48"
                strokeWidth={1} />

        {/* Back-facing latitudes + meridians (dim dashed) */}
        {latPaths.map((sp, i) => sp.back.map((seg, j) => (
          <path key={`lb${i}-${j}`} d={toPath(seg)} fill="none"
                stroke="#2b2b36" strokeWidth={0.8} strokeDasharray="2 3" />
        )))}
        {merPaths.map((sp, i) => sp.back.map((seg, j) => (
          <path key={`mb${i}-${j}`} d={toPath(seg)} fill="none"
                stroke="#2b2b36" strokeWidth={0.8} strokeDasharray="2 3" />
        )))}

        {/* Front-facing latitudes + meridians */}
        {latPaths.map((sp, i) => sp.front.map((seg, j) => (
          <path key={`lf${i}-${j}`} d={toPath(seg)} fill="none"
                stroke="#444555" strokeWidth={1} />
        )))}
        {merPaths.map((sp, i) => sp.front.map((seg, j) => (
          <path key={`mf${i}-${j}`} d={toPath(seg)} fill="none"
                stroke="#444555" strokeWidth={1} />
        )))}

        {/* Direction lines (origin → unit sphere). Depth-dependent
            color so back half is darker and reads as "behind". */}
        {samples.map((s, i) => {
          const back = s.d < 0;
          const color = back ? "#6b4420" : "#f0a63b";
          const dotColor = back ? "#9a6a2e" : "#ffd27a";
          const opacity = back ? 0.55 : 1;
          return (
            <g key={`a${i}`} opacity={opacity}>
              <line x1={ox} y1={oy} x2={s.sx} y2={s.sy}
                    stroke={color} strokeWidth={back ? 0.7 : 1} />
              <circle cx={s.sx} cy={s.sy} r={back ? 1.6 : 2.2}
                      fill={dotColor} />
            </g>
          );
        })}

        {/* Axis triad */}
        {axEnd.map((end, i) => {
          const [ex, ey] = toSvg(project(end));
          return (
            <g key={`ax${i}`}>
              <line x1={ox} y1={oy} x2={ex} y2={ey}
                    stroke={axColors[i]} strokeWidth={1.6}
                    markerEnd={`url(#voea-arrow-${axNames[i]})`} />
              <text x={ex} y={ey} fontSize={10} fill={axColors[i]}
                    textAnchor="middle" dx={6} dy={-2}>{axNames[i]}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

import { useMemo } from "react";

/**
 * Live preview for geom.rotate3d. Pure SVG. Shows:
 *   - faded reference cube (identity, dashed)
 *   - solid rotated cube (phi, theta, psi applied, SPIDER ZYZ)
 *   - a small axis triad (red x̂ / green ŷ / blue ẑ) rotated the same way
 *     so the student can see which direction the content is pointing after
 *     rotation.
 */

interface Props {
  params: Record<string, unknown>;
  size?: number;
}

type Vec3 = [number, number, number];
type Mat3 = [Vec3, Vec3, Vec3];

const AZIM = (-55 * Math.PI) / 180;
const ELEV = (20 * Math.PI) / 180;

function project([x, y, z]: Vec3): [number, number] {
  const rx = -Math.sin(AZIM), ry = Math.cos(AZIM), rz = 0;
  const ux = -Math.sin(ELEV) * Math.cos(AZIM);
  const uy = -Math.sin(ELEV) * Math.sin(AZIM);
  const uz = Math.cos(ELEV);
  return [x * rx + y * ry + z * rz, -(x * ux + y * uy + z * uz)];
}

function num(params: Record<string, unknown>, k: string, def: number): number {
  const v = params[k];
  const n = typeof v === "number" ? v : parseFloat(String(v ?? def));
  return Number.isFinite(n) ? n : def;
}

function matmul(A: Mat3, B: Mat3): Mat3 {
  const out: Mat3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      out[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
  return out;
}

function apply(R: Mat3, v: Vec3): Vec3 {
  return [
    R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
    R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
    R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
  ];
}

// SPIDER convention: R = Rz(psi) · Ry(theta) · Rz(phi), degrees.
function eulerZYZ(phiDeg: number, thetaDeg: number, psiDeg: number): Mat3 {
  const d2r = Math.PI / 180;
  const phi = phiDeg * d2r, theta = thetaDeg * d2r, psi = psiDeg * d2r;
  const cp = Math.cos(phi), sp = Math.sin(phi);
  const ct = Math.cos(theta), st = Math.sin(theta);
  const cs = Math.cos(psi), ss = Math.sin(psi);
  const Rz_phi: Mat3   = [[cp, -sp, 0], [sp, cp, 0], [0, 0, 1]];
  const Ry_theta: Mat3 = [[ct, 0, st], [0, 1, 0], [-st, 0, ct]];
  const Rz_psi: Mat3   = [[cs, -ss, 0], [ss, cs, 0], [0, 0, 1]];
  return matmul(matmul(Rz_psi, Ry_theta), Rz_phi);
}


export default function RotateVolumeDiagram({ params, size = 220 }: Props) {
  const phi = num(params, "phi", 0);
  const theta = num(params, "theta", 0);
  const psi = num(params, "psi", 0);
  const R = useMemo(() => eulerZYZ(phi, theta, psi), [phi, theta, psi]);

  const refCorners: Vec3[] = [
    [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
    [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
  ];
  const rotCorners = refCorners.map((c) => apply(R, c));
  const triad: Vec3[] = [
    apply(R, [1.1, 0, 0]),
    apply(R, [0, 1.1, 0]),
    apply(R, [0, 0, 1.1]),
  ];

  // Project everything to find viewport bounds.
  const projRef = refCorners.map(project);
  const projRot = rotCorners.map(project);
  const projTriad = triad.map(project);
  const projOrigin = project([0, 0, 0]);
  const all = [...projRef, ...projRot, ...projTriad, projOrigin];
  const xs = all.map(([x]) => x), ys = all.map(([, y]) => y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const span = Math.max(maxX - minX, maxY - minY) * 1.08;
  const cx0 = (minX + maxX) / 2, cy0 = (minY + maxY) / 2;
  const toSvg = ([x, y]: [number, number]): [number, number] => [
    (size / 2) + ((x - cx0) / span) * size,
    (size / 2) + ((y - cy0) / span) * size,
  ];

  const edgeIdx: [number, number][] = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];
  const refLines = edgeIdx.map(([a, b]) => {
    const [x1, y1] = toSvg(projRef[a]);
    const [x2, y2] = toSvg(projRef[b]);
    return { x1, y1, x2, y2 };
  });
  const rotLines = edgeIdx.map(([a, b]) => {
    const [x1, y1] = toSvg(projRot[a]);
    const [x2, y2] = toSvg(projRot[b]);
    return { x1, y1, x2, y2 };
  });
  const [ox, oy] = toSvg(projOrigin);
  const [xtx, xty] = toSvg(projTriad[0]);
  const [ytx, yty] = toSvg(projTriad[1]);
  const [ztx, zty] = toSvg(projTriad[2]);

  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 4 }}>
        ZYZ Euler — phi =
        <b style={{ color: "#e4a94b", margin: "0 4px" }}>{phi.toFixed(1)}°</b>
        theta =
        <b style={{ color: "#e4a94b", margin: "0 4px" }}>{theta.toFixed(1)}°</b>
        psi =
        <b style={{ color: "#e4a94b", margin: "0 4px" }}>{psi.toFixed(1)}°</b>
      </div>
      <svg width={size} height={size}
           style={{ background: "#0f0f14", borderRadius: 6,
                    border: "1px solid #2a2a35", display: "block" }}>
        <defs>
          <marker id="rot-arrow-x" viewBox="0 0 10 10" refX="8" refY="5"
                  markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="#c66" />
          </marker>
          <marker id="rot-arrow-y" viewBox="0 0 10 10" refX="8" refY="5"
                  markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="#6c6" />
          </marker>
          <marker id="rot-arrow-z" viewBox="0 0 10 10" refX="8" refY="5"
                  markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="#88c" />
          </marker>
        </defs>

        {/* Reference cube (identity, dashed gray) */}
        {refLines.map((e, i) => (
          <line key={`r${i}`} x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2}
                stroke="#444" strokeWidth={1} strokeDasharray="3 3" />
        ))}

        {/* Rotated cube (bright blue solid) */}
        {rotLines.map((e, i) => (
          <line key={`t${i}`} x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2}
                stroke="#4ea3ff" strokeWidth={1.4} />
        ))}

        {/* Rotated axis triad from origin */}
        <line x1={ox} y1={oy} x2={xtx} y2={xty}
              stroke="#c66" strokeWidth={1.8} markerEnd="url(#rot-arrow-x)" />
        <line x1={ox} y1={oy} x2={ytx} y2={yty}
              stroke="#6c6" strokeWidth={1.8} markerEnd="url(#rot-arrow-y)" />
        <line x1={ox} y1={oy} x2={ztx} y2={zty}
              stroke="#88c" strokeWidth={1.8} markerEnd="url(#rot-arrow-z)" />
        <text x={xtx} y={xty} fontSize={10} fill="#c66"
              textAnchor="middle" dx={6} dy={-2}>x</text>
        <text x={ytx} y={yty} fontSize={10} fill="#6c6"
              textAnchor="middle" dx={6} dy={-2}>y</text>
        <text x={ztx} y={zty} fontSize={10} fill="#88c"
              textAnchor="middle" dx={6} dy={-2}>z</text>
      </svg>
    </div>
  );
}

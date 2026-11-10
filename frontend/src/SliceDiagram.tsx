import { useMemo } from "react";

/**
 * Live 3D-cube-with-slice preview for recon.slice3d. Pure SVG — no backend
 * round-trip, no WebGL — so it ticks instantly as the user drags axis /
 * index / tilt sliders. The geometry mirrors _fn_slice3d exactly, so the
 * diagram and the sampled plane are the same object.
 */

interface Props {
  params: Record<string, unknown>;
  size?: number;
}

type Vec3 = [number, number, number];

function num(params: Record<string, unknown>, k: string, def: number): number {
  const v = params[k];
  const n = typeof v === "number" ? v : parseFloat(String(v ?? def));
  return Number.isFinite(n) ? n : def;
}

function str(params: Record<string, unknown>, k: string, def: string): string {
  const v = params[k];
  return typeof v === "string" ? v : def;
}

// Fixed camera (azim=-55°, elev=20°) — matches the earlier matplotlib shot.
const AZIM = (-55 * Math.PI) / 180;
const ELEV = (20 * Math.PI) / 180;

function project([x, y, z]: Vec3): [number, number] {
  // Camera forward = (cos(E)cos(A), cos(E)sin(A), sin(E))
  // Right        = (-sin(A), cos(A), 0)
  // Up           = (-sin(E)cos(A), -sin(E)sin(A), cos(E))
  const rx = -Math.sin(AZIM), ry = Math.cos(AZIM), rz = 0;
  const ux = -Math.sin(ELEV) * Math.cos(AZIM);
  const uy = -Math.sin(ELEV) * Math.sin(AZIM);
  const uz = Math.cos(ELEV);
  // Screen Y points up in math but down in SVG — negate.
  return [x * rx + y * ry + z * rz, -(x * ux + y * uy + z * uz)];
}

function sliceGeometry(axis: string, tiltRad: number, offset: number): {
  uRow: Vec3; uCol: Vec3; normal: Vec3; centre: Vec3;
} {
  const c = Math.cos(tiltRad), s = Math.sin(tiltRad);
  // (x, y, z) order below (matches SVG world coords). The backend fn uses
  // (dz, dy, dx) ordering for array indexing; converting here.
  let uRow: Vec3, uCol: Vec3, normal: Vec3;
  if (axis === "z") {
    // Base: plane spans (x, y), normal = +z. Tilt rotates around x.
    normal = [s, 0, c];
    uRow = [0, c, -s];
    uCol = [1, 0, 0];
  } else if (axis === "y") {
    // Base: plane spans (x, z), normal = +y. Tilt rotates around x.
    normal = [0, c, s];
    uRow = [0, -s, c];
    uCol = [1, 0, 0];
  } else { // "x"
    // Base: plane spans (y, z), normal = +x. Tilt rotates around y.
    normal = [c, 0, s];
    uRow = [-s, 0, c];
    uCol = [0, 1, 0];
  }
  const centre: Vec3 = [
    offset * normal[0],
    offset * normal[1],
    offset * normal[2],
  ];
  return { uRow, uCol, normal, centre };
}


export default function SliceDiagram({ params, size = 220 }: Props) {
  const geom = useMemo(() => {
    const axis = str(params, "axis", "z");
    const tilt = num(params, "tilt_deg", 0);
    const index = Math.max(-1, Math.trunc(num(params, "index", -1)));
    // index -1 → centre (offset = 0). Otherwise map 0..N-1 to [-1..+1] with
    // N = 64 as a stand-in — we don't know the actual volume size here, so
    // normalise under the assumption the user is near the centre. The
    // visual intent (middle = plane through centre) stays correct.
    const offset = index < 0 ? 0 : (index - 32) / 32;
    const tiltRad = (tilt * Math.PI) / 180;
    const { uRow, uCol, normal, centre } = sliceGeometry(axis, tiltRad, offset);
    const corners: Vec3[] = [
      [centre[0] + uRow[0] + uCol[0], centre[1] + uRow[1] + uCol[1], centre[2] + uRow[2] + uCol[2]],
      [centre[0] + uRow[0] - uCol[0], centre[1] + uRow[1] - uCol[1], centre[2] + uRow[2] - uCol[2]],
      [centre[0] - uRow[0] - uCol[0], centre[1] - uRow[1] - uCol[1], centre[2] - uRow[2] - uCol[2]],
      [centre[0] - uRow[0] + uCol[0], centre[1] - uRow[1] + uCol[1], centre[2] - uRow[2] + uCol[2]],
    ];
    return { axis, tilt, offset, normal, centre, corners };
  }, [params]);

  // Project world → viewport. All coords live in [-1, +1]³ cube; pick a
  // 2D viewport so the projected cube fills ~90% of the SVG.
  const projected = {
    cubeCorners: ([
      [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
      [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
    ] as Vec3[]).map(project),
    sliceCorners: geom.corners.map(project),
    normalEnd: project([
      geom.centre[0] + 0.6 * geom.normal[0],
      geom.centre[1] + 0.6 * geom.normal[1],
      geom.centre[2] + 0.6 * geom.normal[2],
    ]),
    centre: project(geom.centre),
    axisTips: {
      x: project([1.3, 0, 0]),
      y: project([0, 1.3, 0]),
      z: project([0, 0, 1.3]),
    },
  };
  // Fit all projected points into the SVG viewport.
  const all = [
    ...projected.cubeCorners,
    ...projected.sliceCorners,
    projected.normalEnd,
    projected.axisTips.x, projected.axisTips.y, projected.axisTips.z,
  ];
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
  const cubeLines = edgeIdx.map(([a, b]) => {
    const [x1, y1] = toSvg(projected.cubeCorners[a]);
    const [x2, y2] = toSvg(projected.cubeCorners[b]);
    return { x1, y1, x2, y2 };
  });
  const slicePts = projected.sliceCorners.map(toSvg)
    .map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" ");
  const [cx, cy] = toSvg(projected.centre);
  const [nx, ny] = toSvg(projected.normalEnd);
  const [xtx, xty] = toSvg(projected.axisTips.x);
  const [ytx, yty] = toSvg(projected.axisTips.y);
  const [ztx, zty] = toSvg(projected.axisTips.z);

  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 4 }}>
        slice preview — axis = <b style={{ color: "#e4a94b" }}>{geom.axis}</b>
        {", tilt = "}
        <b style={{ color: "#e4a94b" }}>{geom.tilt.toFixed(1)}°</b>
      </div>
      <svg width={size} height={size}
           style={{ background: "#0f0f14", borderRadius: 6,
                    border: "1px solid #2a2a35", display: "block" }}>
        <defs>
          <marker id="slice-arrow" viewBox="0 0 10 10" refX="8" refY="5"
                  markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="#e4a94b" />
          </marker>
        </defs>
        {/* Cube wireframe */}
        {cubeLines.map((e, i) => (
          <line key={i} x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2}
                stroke="#555" strokeWidth={1} />
        ))}
        {/* Slice plane */}
        <polygon points={slicePts}
                 fill="rgba(78,163,255,0.35)"
                 stroke="#4ea3ff" strokeWidth={1.4} />
        {/* Normal arrow */}
        <line x1={cx} y1={cy} x2={nx} y2={ny}
              stroke="#e4a94b" strokeWidth={1.6}
              markerEnd="url(#slice-arrow)" />
        {/* Axis labels */}
        <text x={xtx} y={xty} fontSize={10} fill="#c66" textAnchor="middle">x</text>
        <text x={ytx} y={yty} fontSize={10} fill="#6c6" textAnchor="middle">y</text>
        <text x={ztx} y={zty} fontSize={10} fill="#88c" textAnchor="middle">z</text>
      </svg>
    </div>
  );
}

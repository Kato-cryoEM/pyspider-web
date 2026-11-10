import { useMemo } from "react";

/**
 * Live cube + sum-direction preview for recon.sum_axis. Pure SVG. Shows:
 *   - the unit cube
 *   - a bold yellow arrow spanning the cube along the chosen axis (the
 *     direction being integrated over)
 *   - a translucent blue rectangle on the output face (the 2D image plane
 *     that remains after summation).
 */

interface Props {
  params: Record<string, unknown>;
  size?: number;
}

type Vec3 = [number, number, number];

const AZIM = (-55 * Math.PI) / 180;
const ELEV = (20 * Math.PI) / 180;

function project([x, y, z]: Vec3): [number, number] {
  const rx = -Math.sin(AZIM), ry = Math.cos(AZIM), rz = 0;
  const ux = -Math.sin(ELEV) * Math.cos(AZIM);
  const uy = -Math.sin(ELEV) * Math.sin(AZIM);
  const uz = Math.cos(ELEV);
  return [x * rx + y * ry + z * rz, -(x * ux + y * uy + z * uz)];
}

function str(params: Record<string, unknown>, k: string, def: string): string {
  const v = params[k];
  return typeof v === "string" ? v : def;
}


export default function SumAxisDiagram({ params, size = 220 }: Props) {
  const axis = str(params, "axis", "z");

  const geom = useMemo(() => {
    // Direction of summation. Arrow is drawn through the cube centre.
    let dir: Vec3;
    // Four corners of the output-image plane (perpendicular to `dir`, drawn
    // at the "far face" of the cube so the arrow visibly pierces it).
    let plane: Vec3[];
    if (axis === "z") {
      dir = [0, 0, 1];
      plane = [[-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1]];
    } else if (axis === "y") {
      dir = [0, 1, 0];
      plane = [[-1, +1, -1], [+1, +1, -1], [+1, +1, +1], [-1, +1, +1]];
    } else {
      dir = [1, 0, 0];
      plane = [[+1, -1, -1], [+1, +1, -1], [+1, +1, +1], [+1, -1, +1]];
    }
    const tail: Vec3 = [-dir[0], -dir[1], -dir[2]];
    const head: Vec3 = [dir[0], dir[1], dir[2]];
    return { axis, dir, tail, head, plane };
  }, [axis]);

  // Project everything.
  const cubeCorners: Vec3[] = [
    [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
    [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
  ];
  const projected = {
    cube: cubeCorners.map(project),
    plane: geom.plane.map(project),
    tail: project(geom.tail),
    head: project(geom.head),
    axisTips: {
      x: project([1.3, 0, 0]),
      y: project([0, 1.3, 0]),
      z: project([0, 0, 1.3]),
    },
  };
  const all = [
    ...projected.cube, ...projected.plane,
    projected.tail, projected.head,
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
    const [x1, y1] = toSvg(projected.cube[a]);
    const [x2, y2] = toSvg(projected.cube[b]);
    return { x1, y1, x2, y2 };
  });
  const planePts = projected.plane.map(toSvg)
    .map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" ");
  const [tx, ty] = toSvg(projected.tail);
  const [hx, hy] = toSvg(projected.head);
  const [xtx, xty] = toSvg(projected.axisTips.x);
  const [ytx, yty] = toSvg(projected.axisTips.y);
  const [ztx, zty] = toSvg(projected.axisTips.z);

  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 4 }}>
        sum direction — axis =
        <b style={{ color: "#e4a94b", marginLeft: 4 }}>{geom.axis}</b>
      </div>
      <svg width={size} height={size}
           style={{ background: "#0f0f14", borderRadius: 6,
                    border: "1px solid #2a2a35", display: "block" }}>
        <defs>
          <marker id="sumax-arrow" viewBox="0 0 10 10" refX="8" refY="5"
                  markerWidth="7" markerHeight="7" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="#e4a94b" />
          </marker>
        </defs>
        {/* Cube wireframe */}
        {cubeLines.map((e, i) => (
          <line key={i} x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2}
                stroke="#555" strokeWidth={1} />
        ))}
        {/* Output 2D plane (the face that survives summation) */}
        <polygon points={planePts}
                 fill="rgba(78,163,255,0.28)"
                 stroke="#4ea3ff" strokeWidth={1.2} />
        {/* Sum-direction arrow (from -axis to +axis through the centre) */}
        <line x1={tx} y1={ty} x2={hx} y2={hy}
              stroke="#e4a94b" strokeWidth={2.2}
              markerEnd="url(#sumax-arrow)" />
        {/* Axis labels */}
        <text x={xtx} y={xty} fontSize={10} fill="#c66" textAnchor="middle">x</text>
        <text x={ytx} y={yty} fontSize={10} fill="#6c6" textAnchor="middle">y</text>
        <text x={ztx} y={zty} fontSize={10} fill="#88c" textAnchor="middle">z</text>
      </svg>
    </div>
  );
}

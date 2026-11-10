import {
  ConnectionLineComponentProps, Position, useReactFlow, useStore,
} from "reactflow";
import { useEffect, useRef, useState } from "react";

/**
 * Custom connection line that tracks the pointer directly against the
 * `.react-flow__viewport` element's current bounding rect. React Flow 11's
 * built-in connection line computes the drag endpoint from
 * `connectionPosition = clientY - domNode.getBoundingClientRect().top`,
 * and in this app that measurement comes out offset upward from the
 * actual cursor by a fixed amount. Measuring directly against the
 * viewport (which carries the pan + zoom transform in its own
 * `getBoundingClientRect`) bypasses that bug entirely:
 *
 *     flowX = (clientX - viewport.left) / zoom
 *     flowY = (clientY - viewport.top ) / zoom
 *
 * On top of that we re-implement React Flow's "snap to nearest valid
 * handle" behaviour: on every pointermove we walk every `.react-flow__handle`
 * in the DOM, project its screen centre to flow coords (same viewport
 * bbox / zoom used for the cursor), and pick the closest handle within
 * `SNAP_RADIUS_PX` SCREEN pixels — screen-space so the magnet feel stays
 * constant regardless of zoom. We filter to the opposite handle type
 * (source ↔ target) and skip handles on the source node itself.
 */
const SNAP_RADIUS_PX = 24;

export default function ConnectionLineLive({
  fromNode, fromX, fromY, fromPosition,
}: ConnectionLineComponentProps) {
  const rf = useReactFlow();
  // React Flow tracks the handle type that initiated the drag.
  // We mirror it to decide which handles are valid snap targets.
  const fromHandleType = useStore(
    (s) => (s as unknown as { connectionHandleType: "source" | "target" | null })
      .connectionHandleType,
  );
  const [to, setTo] = useState<
    { x: number; y: number; snapped: boolean } | null
  >(null);
  const latestClient = useRef<{ x: number; y: number } | null>(null);
  const raf = useRef<number | null>(null);

  useEffect(() => {
    const schedule = () => {
      if (raf.current != null) return;
      raf.current = requestAnimationFrame(() => {
        raf.current = null;
        const c = latestClient.current;
        if (!c) return;
        const viewport = document.querySelector(
          ".react-flow__viewport",
        ) as HTMLElement | null;
        if (!viewport) return;
        const bbox = viewport.getBoundingClientRect();
        const { zoom } = rf.getViewport();
        const zClamped = Math.max(1e-6, zoom);
        // Raw cursor in flow coords.
        const cursorFlow = {
          x: (c.x - bbox.left) / zClamped,
          y: (c.y - bbox.top) / zClamped,
        };
        // Snap: walk DOM handles and find the nearest valid one (opposite
        // type, different node) within SNAP_RADIUS_PX screen pixels of
        // the cursor's screen position.
        const oppositeType = fromHandleType === "source" ? "target" : "source";
        const nodes = document.querySelectorAll(
          `.react-flow__handle.${oppositeType}`,
        );
        let best: { x: number; y: number; d2: number } | null = null;
        const r2 = SNAP_RADIUS_PX * SNAP_RADIUS_PX;
        for (const el of Array.from(nodes) as HTMLElement[]) {
          if (el.getAttribute("data-nodeid") === fromNode?.id) continue;
          const hbbox = el.getBoundingClientRect();
          const hx = hbbox.left + hbbox.width / 2;
          const hy = hbbox.top + hbbox.height / 2;
          const dx = hx - c.x;
          const dy = hy - c.y;
          const d2 = dx * dx + dy * dy;
          if (d2 <= r2 && (!best || d2 < best.d2)) {
            best = {
              x: (hx - bbox.left) / zClamped,
              y: (hy - bbox.top) / zClamped,
              d2,
            };
          }
        }
        if (best) {
          setTo({ x: best.x, y: best.y, snapped: true });
        } else {
          setTo({ x: cursorFlow.x, y: cursorFlow.y, snapped: false });
        }
      });
    };
    const onMove = (e: PointerEvent) => {
      latestClient.current = { x: e.clientX, y: e.clientY };
      schedule();
    };
    window.addEventListener("pointermove", onMove);
    return () => {
      window.removeEventListener("pointermove", onMove);
      if (raf.current != null) cancelAnimationFrame(raf.current);
    };
  }, [rf, fromNode?.id, fromHandleType]);

  if (!to) return null;

  // Simple bezier modelled after React Flow's default. For a cleaner
  // S-curve we offset the control points based on the source Position.
  const dx = to.x - fromX;
  const dy = to.y - fromY;
  const horizontal = fromPosition === Position.Left || fromPosition === Position.Right;
  const offset = horizontal
    ? Math.max(50, Math.abs(dx) * 0.4)
    : Math.max(50, Math.abs(dy) * 0.4);
  const sign = (fromPosition === Position.Right || fromPosition === Position.Bottom) ? 1 : -1;
  const cx1 = horizontal ? fromX + sign * offset : fromX;
  const cy1 = horizontal ? fromY : fromY + sign * offset;
  const cx2 = horizontal ? to.x - sign * offset : to.x;
  const cy2 = horizontal ? to.y : to.y - sign * offset;

  const stroke = to.snapped ? "#55dd88" : "#b1b1b7";

  return (
    <g className="rf-conn-live">
      <path
        d={`M ${fromX} ${fromY} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${to.x} ${to.y}`}
        fill="none"
        stroke={stroke}
        strokeWidth={2}
      />
      <circle cx={to.x} cy={to.y} r={to.snapped ? 5 : 3} fill={stroke} />
    </g>
  );
}

/**
 * Renders the tutorial spotlight + info card on top of the app.
 *
 * The spotlight is a full-viewport translucent backdrop with a
 * rectangular hole punched around the current step's target element
 * (via `data-tutor-id`). The info card is placed adjacent to the target
 * on whichever side has room; buttons let the user press Next (for info
 * steps) or Close the tutorial.
 *
 * We poll `getBoundingClientRect()` on every animation frame so the
 * spotlight follows moving targets (React Flow nodes being dragged,
 * scrolling palette, etc.) without needing observer hooks per element.
 */

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { Edge, Node } from "reactflow";
import type { SpiderNodeData } from "../SpiderNode";
import type { TutorialState } from "./useTutorial";

interface Props {
  state: TutorialState;
  nodes: Node<SpiderNodeData>[];
  edges: Edge[];
}

interface TargetRect {
  x: number; y: number; w: number; h: number;
}

/** Find the current target element and return its viewport rect (or null).
 *  If `preferNid` is given, and the tutorId matches multiple elements (e.g.
 *  `node:geom.po` when two PO nodes exist), prefer the one whose enclosing
 *  React-Flow node has that id. Falls back to the last matching element so
 *  the "latest-added" wins — matches the tutorial's natural ordering. */
function measure(tutorId: string | undefined,
                 preferNid?: string | null): TargetRect | null {
  if (!tutorId) return null;
  const els = document.querySelectorAll<HTMLElement>(
    `[data-tutor-id="${CSS.escape(tutorId)}"]`);
  if (els.length === 0) return null;
  let el: HTMLElement | null = null;
  if (preferNid) {
    for (const e of els) {
      const wrap = e.closest("[data-id]") as HTMLElement | null;
      if (wrap?.getAttribute("data-id") === preferNid) { el = e; break; }
    }
  }
  // Fallback: last match (in DOM order === node insertion order for React
  // Flow nodes, so this picks the most-recently-added one).
  if (!el) el = els[els.length - 1];
  const r = el.getBoundingClientRect();
  return { x: r.left, y: r.top, w: r.width, h: r.height };
}

export default function TutorialOverlay({ state, nodes, edges }: Props) {
  const [rect, setRect] = useState<TargetRect | null>(null);
  // Actual rendered card height — measured after mount so the positioning
  // logic below never assumes a wrong CARD_H (previously hard-coded to 200,
  // which pushed long-body cards off the bottom of the viewport).
  const [cardH, setCardH] = useState(240);
  const cardRef = useRef<HTMLDivElement | null>(null);

  // For steps whose tutorId can match multiple instances (e.g.
  // `node:geom.po` when two PO nodes exist), compute the SPECIFIC instance
  // the step really refers to so `measure()` picks the right one:
  //   - wire_check: prefer the free dst node (the just-dropped destination)
  //   - add_node / select_node: prefer the most-recently-added match
  //   - set_param: prefer the currently-selected node
  const preferNid: string | null = (() => {
    const a = state.step?.action;
    if (!a) return null;
    const latestOfType = (typeId: string) => {
      for (let i = nodes.length - 1; i >= 0; i--) {
        if (nodes[i].data?.spec?.id === typeId) return nodes[i].id;
      }
      return null;
    };
    if (a.kind === "wire_check") {
      // Prefer the destination whose (id, dstPort) has no incoming edge.
      const usedDst = new Set<string>();
      for (const e of edges) usedDst.add(`${e.target}:${e.targetHandle ?? ""}`);
      const matches = nodes.filter((n) => n.data?.spec?.id === a.dstNodeType);
      const free = matches.filter((n) => !usedDst.has(`${n.id}:${a.dstPort}`));
      return (free.length ? free[free.length - 1]
                          : matches[matches.length - 1])?.id ?? null;
    }
    if (a.kind === "add_node")    return latestOfType(a.nodeType);
    if (a.kind === "select_node") return latestOfType(a.nodeType);
    return null;
  })();

  // Poll the target's position each frame while the tutorial is active.
  useEffect(() => {
    if (!state.active) return;
    let raf = 0;
    const loop = () => {
      setRect(measure(state.step?.target?.tutorId, preferNid));
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [state.active, state.step, preferNid]);

  // Reset rect immediately on step change so the last step's rect doesn't
  // flash before the next measurement lands.
  useLayoutEffect(() => {
    setRect(measure(state.step?.target?.tutorId, preferNid));
  }, [state.step, preferNid]);

  // Scroll the target into view once when the step first mounts. Otherwise
  // a Palette item deep in a scrolled category (or an off-screen node)
  // could yield a rect outside the viewport, sending the popup card off
  // the visible area with it.
  useEffect(() => {
    const tid = state.step?.target?.tutorId;
    if (!tid) return;
    // Give the DOM one tick — the target category may have JUST been
    // uncollapsed by Palette's forceExpandCategory effect.
    const t = window.setTimeout(() => {
      const el = document.querySelector(
        `[data-tutor-id="${CSS.escape(tid)}"]`,
      ) as HTMLElement | null;
      if (!el) return;
      const r = el.getBoundingClientRect();
      const above = r.top < 60;
      const below = r.bottom > window.innerHeight - 60;
      if (above || below) {
        el.scrollIntoView({ block: "center", behavior: "smooth" });
      }
    }, 80);
    return () => window.clearTimeout(t);
  }, [state.step]);

  // Measure the actual card height after every render (step change → new
  // text → potentially different height). ResizeObserver keeps it in sync
  // when the viewport is resized (max-height clip changes).
  useLayoutEffect(() => {
    const el = cardRef.current;
    if (!el) return;
    const update = () => setCardH(el.offsetHeight);
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    window.addEventListener("resize", update);
    return () => { ro.disconnect(); window.removeEventListener("resize", update); };
  }, [state.step]);

  // Blink the specific handles (▸ src / ◂ dst) the current wire step is
  // waiting for. If several nodes of the target type exist (common when a
  // tutorial drops two PO or two AS S), we prefer the ones whose specific
  // (nodeId, port) has NO existing wire — those are the ones the user is
  // almost certainly meant to connect. If NONE are free, fall back to
  // blinking all matching handles.
  useEffect(() => {
    const a = state.step?.action;
    if (!a || a.kind !== "wire_check") return;

    // Which specific node IDs are "free" on the src port / dst port?
    // src is free if no outgoing edge exists from (nodeId, srcPort).
    // dst is free if no incoming edge exists into (nodeId, dstPort).
    const usedSrc = new Set<string>();
    const usedDst = new Set<string>();
    for (const e of edges) {
      usedSrc.add(`${e.source}:${e.sourceHandle ?? ""}`);
      usedDst.add(`${e.target}:${e.targetHandle ?? ""}`);
    }
    const freeIds = (typeId: string, port: string, side: "src" | "dst") => {
      const matches = nodes.filter((n) => n.data?.spec?.id === typeId);
      if (matches.length === 0) return new Set<string>();
      const isFree = (nid: string) => side === "src"
        ? !usedSrc.has(`${nid}:${port}`)
        : !usedDst.has(`${nid}:${port}`);
      const free = matches.filter((n) => isFree(n.id));
      // Pick exactly ONE handle to blink: the LATEST-added candidate that's
      // still unwired on this port. `nodes` is in insertion order, so the
      // most recent drop sits at the end. If nothing is free (the user
      // re-drew wires), fall back to the most recent match overall so at
      // least ONE handle blinks. This matches natural tutorial ordering:
      // "drop this node, THEN wire it" → the freshly-dropped node is what
      // the current step is about.
      const pick = free.length
        ? free[free.length - 1]
        : matches[matches.length - 1];
      return new Set([pick.id]);
    };
    const srcFree = freeIds(a.srcNodeType, a.srcPort, "src");
    const dstFree = freeIds(a.dstNodeType, a.dstPort, "dst");

    const marks: HTMLElement[] = [];
    const mark = (kind: "src" | "dst", nodeType: string, port: string,
                  allowed: Set<string>) => {
      const sel = `[data-tutor-handle="${kind}:${nodeType}:${port}"]`;
      document.querySelectorAll(sel).forEach((el) => {
        // Handle's own `data-tutor-nid` (set by SpiderNode from its React
        // Flow `id` prop) tells us the node-instance id directly — no need
        // to walk ancestors.
        const nid = (el as HTMLElement).getAttribute("data-tutor-nid") ?? "";
        if (nid && !allowed.has(nid)) return;
        (el as HTMLElement).classList.add("tutor-handle-blink");
        marks.push(el as HTMLElement);
      });
    };
    const paint = () => {
      mark("src", a.srcNodeType, a.srcPort, srcFree);
      mark("dst", a.dstNodeType, a.dstPort, dstFree);
    };
    paint();
    // Poll: nodes can be added/removed at any time, so re-mark every 250ms.
    const t = window.setInterval(paint, 250);
    return () => {
      window.clearInterval(t);
      marks.forEach((el) => el.classList.remove("tutor-handle-blink"));
      document.querySelectorAll(".tutor-handle-blink")
        .forEach((el) => el.classList.remove("tutor-handle-blink"));
    };
  }, [state.step, nodes, edges]);

  if (!state.active || !state.step) return null;

  const { step, lesson, stepIndex } = state;
  const total = lesson?.steps.length ?? 0;
  // Pre-compute flags needed by the auto-side resolver below.
  const isFinalInfo =
    step.action.kind === "info" && stepIndex >= total - 1;

  // Card side: default to "auto" — place opposite the target relative to
  // the viewport centre so the card doesn't overlap the target.
  let cardX = 24, cardY = 24;
  const CARD_W = 320, MARGIN = 16;
  // Guaranteed-vertical-fit clamp: the maximum cardY such that a card of
  // the currently-measured height still fits above the viewport bottom.
  const maxY = Math.max(MARGIN, window.innerHeight - cardH - MARGIN);
  const clampY = (y: number) => Math.max(MARGIN, Math.min(y, maxY));
  if (rect) {
    // Resolve "auto": the final 🎉 info card places sideways so the result
    // node's preview stays fully visible (the step is read-only, no ports
    // need to be unoccluded). All other steps prefer top/bottom to avoid
    // covering input/output handles; left/right placement is reserved for
    // tutor-side-requested overrides.
    let side = step.target?.cardSide ?? "auto";
    if (side === "auto") {
      if (isFinalInfo) {
        const centreX = window.innerWidth / 2;
        side = rect.x + rect.w / 2 > centreX ? "left" : "right";
      } else {
        const spaceAbove = rect.y;
        const spaceBelow = window.innerHeight - (rect.y + rect.h);
        side = spaceBelow >= spaceAbove ? "bottom" : "top";
      }
    }
    if (side === "left") {
      cardX = Math.max(MARGIN, rect.x - CARD_W - MARGIN);
      cardY = clampY(rect.y);
    } else if (side === "right") {
      cardX = Math.min(window.innerWidth - CARD_W - MARGIN, rect.x + rect.w + MARGIN);
      cardY = clampY(rect.y);
    } else if (side === "top") {
      cardX = Math.max(MARGIN, Math.min(rect.x, window.innerWidth - CARD_W - MARGIN));
      // If there's no room above the target, fall back to placing below —
      // otherwise clampY would land the card ON TOP of the target and block
      // any drag that starts from there.
      const above = rect.y - cardH - MARGIN;
      cardY = above >= MARGIN
        ? above
        : Math.min(maxY, rect.y + rect.h + MARGIN);
    } else if (side === "bottom") {
      cardX = Math.max(MARGIN, Math.min(rect.x, window.innerWidth - CARD_W - MARGIN));
      // Symmetric: if there's no room below the target, place above instead
      // so the user can still drag from the highlighted target unobstructed.
      const below = rect.y + rect.h + MARGIN;
      cardY = below + cardH + MARGIN <= window.innerHeight
        ? clampY(below)
        : clampY(rect.y - cardH - MARGIN);
    }
  } else {
    // No target — centre the card horizontally, snap to the bottom-third
    // vertically (but clamped by maxY so it always fits).
    cardX = (window.innerWidth - CARD_W) / 2;
    cardY = clampY(window.innerHeight - cardH - 40);
  }

  const isInfo = step.action.kind === "info";
  // For wire_check steps: no Next button and no Skip — the user must
  // actually draw the wire (auto-detected by useTutorial) to advance.
  const isWire = step.action.kind === "wire_check";
  // Last step of the lesson: show a single celebratory "Got it!" button
  // that just closes the tutorial (no End / Skip clutter).
  const isFinal = stepIndex >= total - 1 && isInfo;

  return (
    <div className="tutor-root">
      {/* Backdrop with a rectangular hole around the target. Uses SVG for
          the punch-out so pointer events still reach the highlighted
          element (the hole area doesn't intercept clicks). */}
      <svg className="tutor-backdrop"
           viewBox={`0 0 ${window.innerWidth} ${window.innerHeight}`}
           preserveAspectRatio="none">
        <defs>
          <mask id="tutor-hole">
            <rect width="100%" height="100%" fill="white" />
            {rect && (
              <rect x={rect.x - 6} y={rect.y - 6}
                    width={rect.w + 12} height={rect.h + 12}
                    rx={6} ry={6} fill="black" />
            )}
          </mask>
        </defs>
        {/* The backdrop dims non-target UI to draw attention. It's kept
            light enough (25% black) that a node's rendered preview inside
            the canvas is still clearly visible while the ring blinks. */}
        <rect width="100%" height="100%"
              fill="rgba(0, 0, 0, 0.25)" mask="url(#tutor-hole)" />
        {rect && (
          <rect x={rect.x - 6} y={rect.y - 6}
                width={rect.w + 12} height={rect.h + 12}
                rx={6} ry={6}
                fill="none" stroke="#ffd54a" strokeWidth={3}
                className="tutor-ring" />
        )}
      </svg>

      <div ref={cardRef}
           className={"tutor-card" + (isFinal ? " tutor-card-final" : "")}
           style={{ left: cardX, top: cardY }}>
        <div className="tutor-progress">
          {lesson?.title} · step {stepIndex + 1} / {total}
        </div>
        <div className="tutor-title">{step.title}</div>
        <div className="tutor-body">
          {step.body.split("\n\n").map((p, i) => <p key={i}>{p}</p>)}
        </div>
        <div className="tutor-buttons">
          {isFinal ? (
            // Final celebratory step: single centered "Got it!" button, no
            // End / Skip clutter — the lesson is complete.
            <button className="tutor-btn got-it" onClick={state.close}>
              🎉  Got it!
            </button>
          ) : (
            <>
              <button className="tutor-btn skip" onClick={state.close}>
                End tutorial
              </button>
              {isInfo && (
                <button className="tutor-btn next" onClick={state.next}>
                  Next →
                </button>
              )}
              {!isInfo && !isWire && (
                <button className="tutor-btn skip-step" onClick={state.skip}
                        title="Skip this step">
                  Skip step
                </button>
              )}
              {isWire && (
                <span className="tutor-wait">
                  waiting for the wire…
                </span>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * Tutorial runner state hook.
 *
 * Owns which lesson + step is active, watches the app's node graph for the
 * change the current step is waiting on, and auto-advances when it sees it.
 * Info-only steps and "wire two nodes" steps require the user to press Next.
 *
 * `pyspider_tutorial_v1` in localStorage remembers which lesson + step the
 * user was on across page reloads (so a mid-lesson refresh doesn't lose
 * progress).
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Edge, Node } from "reactflow";
import type { SpiderNodeData } from "../SpiderNode";
import { getLessons, loadLessons } from "./lessons";
import type { Lesson, Step } from "./types";

const STORAGE_KEY = "pyspider_tutorial_v1";

interface Persisted {
  lessonId: string;
  stepIndex: number;
}

function loadPersisted(): Persisted | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as Persisted;
  } catch { return null; }
}

function savePersisted(p: Persisted | null) {
  try {
    if (p === null) localStorage.removeItem(STORAGE_KEY);
    else localStorage.setItem(STORAGE_KEY, JSON.stringify(p));
  } catch { /* ignore quota / private-mode failures */ }
}

export interface TutorialState {
  active: boolean;
  lesson: Lesson | null;
  stepIndex: number;
  step: Step | null;
  /** All lessons currently available (fetched from /api/tutorials).
   *  Empty array until the initial fetch resolves. */
  availableLessons: Lesson[];
  /** Palette category the current step needs opened (or null). Consumed by
   *  App → Palette's `forceExpandCategory` prop. */
  expandCategory: string | null;
  /** Set to the just-completed lesson's title for one render-cycle burst so
   *  the UI can show a "Congratulations!" popup. Cleared when the user
   *  dismisses it via `clearCompleted()`. */
  completed: { title: string } | null;
  clearCompleted: () => void;
  start: (lessonId: string) => void;
  next: () => void;
  skip: () => void;
  close: () => void;
}

/**
 * @param nodes React Flow nodes with pyspider `data` payloads — the runner
 *   needs them to detect add_node / set_param / run_node conditions.
 * @param edges React Flow edges — needed for `wire_check` steps.
 * @param selectedId Currently-selected node id (to check the right node's
 *   params on set_param steps).
 */
export function useTutorial(
  nodes: Node<SpiderNodeData>[],
  edges: Edge[],
  selectedId: string | null,
): TutorialState {
  const [state, setState] = useState<Persisted | null>(() => loadPersisted());
  // Force-refresh trigger — flips when `loadLessons()` finishes so the memos
  // pick up the freshly-fetched catalogue.
  const [lessonsVersion, setLessonsVersion] = useState(0);
  // One-shot celebration state: set the moment a lesson advances past its
  // last step, cleared by the UI once the Congratulations popup is closed.
  const [completed, setCompleted] = useState<{ title: string } | null>(null);

  // Fetch the lesson catalogue on mount and mark ourselves ready.
  useEffect(() => {
    let alive = true;
    loadLessons().then(() => { if (alive) setLessonsVersion((v) => v + 1); });
    return () => { alive = false; };
  }, []);

  const lesson = useMemo(() => {
    if (!state) return null;
    return getLessons().find((l) => l.id === state.lessonId) ?? null;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state, lessonsVersion]);

  const step: Step | null = useMemo(() => {
    if (!lesson) return null;
    return lesson.steps[state!.stepIndex] ?? null;
  }, [lesson, state]);

  // Persist state whenever it changes.
  useEffect(() => { savePersisted(state); }, [state]);

  // Was any non-info step bypassed via Skip during this run? If yes, we
  // suppress the end-of-lesson "Congratulations" popup — the user never
  // actually did the work, so celebrating is misleading. Reset each time
  // a lesson starts.
  const skippedRun = useRef(false);

  const advance = useCallback(() => {
    setState((s) => {
      if (!s) return s;
      const l = getLessons().find((ll) => ll.id === s.lessonId);
      if (!l) return null;
      const next = s.stepIndex + 1;
      if (next >= l.steps.length) {
        // Lesson finished — raise the one-shot completion signal before
        // tearing down active state so the UI can celebrate. Skip the
        // celebration if the user bypassed any real step with Skip.
        if (!skippedRun.current) {
          setCompleted({ title: l.title });
        }
        return null;
      }
      return { ...s, stepIndex: next };
    });
  }, []);

  const clearCompleted = useCallback(() => setCompleted(null), []);

  // Auto-advance: only when the *graph state changes* from what it was at
  // step entry. This prevents chain-firing through steps whose conditions
  // are already satisfied by a leftover graph from a previous tutorial run
  // (e.g. an existing stack.ass node making `drop-ass` fire immediately).
  //
  // For each step we snapshot the relevant metric on entry (count of
  // matching nodes, presence of matching edge, current param value, …).
  // The step then requires a POSITIVE transition from that snapshot.
  //
  // `select_node` is intentionally lenient: if a matching node happens to
  // already be selected on entry, we advance immediately — clicking to
  // select is a trivial action to force otherwise.
  const lastAdvancedForStep = useRef<string | null>(null);
  const stepEntrySnapshot = useRef<{ key: string; snap: unknown }>({ key: "", snap: null });

  useEffect(() => {
    if (!step || !state) return;
    const stepKey = `${state.lessonId}::${state.stepIndex}`;
    if (lastAdvancedForStep.current === stepKey) return;

    const a = step.action;

    // Compute the current metric for this step.
    const currentMetric = (): unknown => {
      if (a.kind === "add_node") {
        return nodes.filter((n) => n.data?.spec?.id === a.nodeType).length;
      }
      if (a.kind === "set_param") {
        // Count (not existence) of nodes whose param equals the target —
        // so a tutorial that sets the SAME value on two instances of the
        // same node type (e.g. angle_deg=15 on Rotate#1 then Rotate#2)
        // can still advance by growing the count. The matcher below
        // requires a positive delta from entry.
        //
        // For string params (formulas, paths, expressions) normalise
        // whitespace so `-P1 * 360 / P2` matches `-P1*360/P2` and users
        // don't have to hit the tutorial's exact spacing.
        const norm = (s: unknown) =>
          typeof s === "string" ? s.replace(/\s+/g, "") : String(s);
        const target = norm(a.value);
        return nodes.filter((n) => {
          const v = n.data?.params?.[a.nodeName];
          return v !== undefined && norm(v) === target;
        }).length;
      }
      if (a.kind === "wire_check") {
        // Count (not existence) so a second instance of the same wire spec
        // — e.g. Checker → Rotate#2 after Checker → Rotate#1 — can still
        // advance by growing the count. The matcher below requires a
        // positive delta from entry.
        const typeOf = new Map(nodes.map((n) => [n.id, n.data?.spec?.id]));
        return edges.filter((e) =>
          typeOf.get(e.source) === a.srcNodeType
          && e.sourceHandle === a.srcPort
          && typeOf.get(e.target) === a.dstNodeType
          && e.targetHandle === a.dstPort).length;
      }
      if (a.kind === "run_node") {
        return nodes.some((n) =>
          n.data?.spec?.id === a.nodeType && n.data?.status === "success");
      }
      return null;
    };

    // On first entry to a step, capture its baseline metric and do NOT
    // auto-advance yet — even if the condition is already met.
    if (stepEntrySnapshot.current.key !== stepKey) {
      stepEntrySnapshot.current = { key: stepKey, snap: currentMetric() };
      // `select_node` is the one lenient case: allow immediate advance when
      // the matching node is already selected (the user just clicked it in
      // a prior step and doesn't need to re-select).
      if (a.kind === "select_node") {
        const sel = nodes.find((n) => n.id === selectedId);
        if (sel?.data?.spec?.id === a.nodeType) {
          lastAdvancedForStep.current = stepKey;
          advance();
        }
      }
      return;
    }

    const before = stepEntrySnapshot.current.snap;
    let matched = false;
    if (a.kind === "add_node") {
      // Advance when the count grows — i.e. a NEW node of the type appears.
      matched = (currentMetric() as number) > (before as number);
    } else if (a.kind === "set_param") {
      // Snapshot was the count of matching nodes at step entry; advance
      // when the count grows (someone just set another node to the target).
      matched = (currentMetric() as number) > (before as number);
    } else if (a.kind === "wire_check") {
      matched = (currentMetric() as number) > (before as number);
    } else if (a.kind === "run_node") {
      matched = (currentMetric() as boolean) && !(before as boolean);
    } else if (a.kind === "select_node") {
      const sel = nodes.find((n) => n.id === selectedId);
      matched = sel?.data?.spec?.id === a.nodeType;
    }
    if (matched) {
      lastAdvancedForStep.current = stepKey;
      advance();
    }
  }, [step, state, nodes, edges, selectedId, advance]);

  const start = useCallback((lessonId: string) => {
    lastAdvancedForStep.current = null;
    stepEntrySnapshot.current = { key: "", snap: null };
    skippedRun.current = false;
    setState({ lessonId, stepIndex: 0 });
  }, []);

  const close = useCallback(() => {
    lastAdvancedForStep.current = null;
    stepEntrySnapshot.current = { key: "", snap: null };
    skippedRun.current = false;
    setState(null);
  }, []);

  const skip = useCallback(() => {
    lastAdvancedForStep.current = null;
    // Reset the snapshot too so the NEW step's baseline is captured fresh
    // — otherwise it might carry a stale snapshot from a previous session.
    stepEntrySnapshot.current = { key: "", snap: null };
    // Only `info` steps are "free to skip" without disqualifying the
    // completion popup — they're prose, not work. Any other action kind
    // (add_node, wire_check, set_param, select_node, run_node) implies
    // the user is bypassing real work, so suppress the celebration.
    if (step && step.action?.kind !== "info") {
      skippedRun.current = true;
    }
    advance();
  }, [advance, step]);

  return {
    active: !!step,
    lesson,
    stepIndex: state?.stepIndex ?? 0,
    step,
    // Read the memoised list; re-computed when lessonsVersion bumps.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    availableLessons: useMemo(getLessons, [lessonsVersion]),
    expandCategory: step?.target?.expandPaletteCategory ?? null,
    completed,
    clearCompleted,
    start,
    next: skip,
    skip,
    close,
  };
}

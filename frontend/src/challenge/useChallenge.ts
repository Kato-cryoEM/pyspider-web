/**
 * Challenge-mode state container.
 *
 * Thin wrapper around the backend's /api/challenges/* endpoints. Owns the
 * current session (if any), the latest judge verdict, and the available-
 * challenge catalogue. Imperative canvas work (clearing nodes, dropping the
 * fixture nodes, picking an answer node) lives in App.tsx — this hook stays
 * pure so it's reusable in tests.
 */

import { useCallback, useEffect, useState } from "react";

import { listChallenges, startChallenge, judgeChallenge, fetchFixturePreview } from "./api";
import type { ChallengeSummary, ChallengeSession, JudgeResult } from "./types";

export interface ChallengeState {
  available: ChallengeSummary[];
  /** IDs of challenges the user has solved (verdict.passed === true).
   *  Persisted in localStorage so progress survives reloads. */
  cleared: Set<string>;
  session: ChallengeSession | null;
  /** Base64 data-URL previews of each fixture, filled asynchronously after
   *  `begin()` resolves — the picker panel shows these so students see what
   *  they're dealing with before touching the canvas. */
  previews: Record<string, string>;
  result: JudgeResult | null;
  busy: boolean;
  /** Last error from start/judge (networking or server-side); non-null means
   *  the UI should surface it instead of a stale verdict. */
  error: string | null;
  /** True once the catalogue fetch finishes (success or empty). Guards the
   *  picker's "no challenges" message from flashing on first paint. */
  loaded: boolean;

  begin: (challengeId: string) => Promise<ChallengeSession | null>;
  submit: (
    answerNodeId: string,
    engineNodes: { id: string; type: string; params: Record<string, unknown> }[],
    engineEdges: { src: string; srcPort: string; dst: string; dstPort: string }[],
  ) => Promise<JudgeResult | null>;
  close: () => void;
  clearResult: () => void;
}


const CLEARED_KEY = "pyspider_cleared_challenges";

function _loadCleared(): Set<string> {
  try {
    const raw = localStorage.getItem(CLEARED_KEY);
    if (!raw) return new Set();
    const arr = JSON.parse(raw);
    return new Set(Array.isArray(arr) ? arr.filter((x) => typeof x === "string") : []);
  } catch {
    return new Set();
  }
}

function _saveCleared(s: Set<string>) {
  try { localStorage.setItem(CLEARED_KEY, JSON.stringify([...s])); } catch { /* ignore quota */ }
}

export function useChallenge(): ChallengeState {
  const [available, setAvailable] = useState<ChallengeSummary[]>([]);
  const [cleared, setCleared] = useState<Set<string>>(() => _loadCleared());
  const [loaded, setLoaded] = useState(false);
  const [session, setSession] = useState<ChallengeSession | null>(null);
  const [previews, setPreviews] = useState<Record<string, string>>({});
  const [result, setResult] = useState<JudgeResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    listChallenges()
      .then((xs) => {
        if (!alive) return;
        // Sort by `order` so the sequence-lock UI shows challenges in the
        // author-intended progression. Items missing `order` fall to the end.
        const sorted = [...xs].sort(
          (a, b) => (a.order ?? 1e9) - (b.order ?? 1e9),
        );
        setAvailable(sorted);
        setLoaded(true);
      })
      .catch(() => { if (alive) setLoaded(true); });
    return () => { alive = false; };
  }, []);

  const begin = useCallback(async (challengeId: string) => {
    setBusy(true);
    setError(null);
    setResult(null);
    setPreviews({});
    try {
      const sess = await startChallenge(challengeId);
      setSession(sess);
      // Load previews in parallel. Failures are non-fatal — the student can
      // still see the fixtures via node previews after they Run.
      Promise.all(
        sess.fixture_names.map((name) =>
          fetchFixturePreview(sess.session_id, name).then((p) => [name, p] as const)
        )
      ).then((entries) => {
        setPreviews(Object.fromEntries(
          entries.filter(([, p]) => p !== null) as [string, string][]
        ));
      });
      return sess;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return null;
    } finally {
      setBusy(false);
    }
  }, []);

  const submit = useCallback(async (
    answerNodeId: string,
    engineNodes: { id: string; type: string; params: Record<string, unknown> }[],
    engineEdges: { src: string; srcPort: string; dst: string; dstPort: string }[],
  ) => {
    if (!session) {
      setError("no active challenge session");
      return null;
    }
    setBusy(true);
    setError(null);
    try {
      const verdict = await judgeChallenge({
        challenge_id: session.challenge_id,
        session_id: session.session_id,
        answer_node_id: answerNodeId,
        nodes: engineNodes,
        edges: engineEdges,
      });
      setResult(verdict);
      // Persist clear state when the judge passes — unlocks the next item.
      if (verdict && verdict.pass && session) {
        setCleared((prev) => {
          if (prev.has(session.challenge_id)) return prev;
          const next = new Set(prev);
          next.add(session.challenge_id);
          _saveCleared(next);
          return next;
        });
      }
      return verdict;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return null;
    } finally {
      setBusy(false);
    }
  }, [session]);

  const close = useCallback(() => {
    setSession(null);
    setPreviews({});
    setResult(null);
    setError(null);
  }, []);

  const clearResult = useCallback(() => setResult(null), []);

  return {
    available, cleared, session, previews, result, busy, error, loaded,
    begin, submit, close, clearResult,
  };
}

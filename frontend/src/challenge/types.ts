/** Public shape of a challenge spec as returned by GET /api/challenges.
 *  Server-side judging fields (compare_to / metric / pass_score) are stripped
 *  from this listing — they live only on the server. */
export interface ChallengeSummary {
  id: string;
  title: string;
  summary?: string;
  prompt: string;
  fixtures: { name: string; recipe: Record<string, unknown> }[];
  allowed_nodes: string[];
  answer_port?: string;
  hint?: string;
  /** Sequence index (1-based). Challenges with lower order unlock first.
   *  Items without an order are placed at the end. */
  order?: number;
}

/** One fixture descriptor in the start response. */
export interface FixtureMeta {
  name: string;
  /** "image" (2D float) or "volume" (3D float). Picks the fixture plugin. */
  kind: "image" | "volume";
  /** When true, the fixture is only shown as a preview in the challenge
   *  panel — no canvas node is dropped — so the student cannot wire it
   *  into their graph (= hidden ground-truth / judge reference). */
  judge_only?: boolean;
  /** When true, the fixture is dropped on the canvas but its output
   *  handle is suppressed, preventing the student from wiring it into
   *  their pipeline. Used for pure-answer fixtures. */
  hide_output?: boolean;
}

/** Payload returned by POST /api/challenges/start. */
export interface ChallengeSession {
  session_id: string;
  challenge_id: string;
  title: string;
  prompt: string;
  fixtures: FixtureMeta[];
  /** Deprecated — kept for one release while the UI migrates to `fixtures`. */
  fixture_names: string[];
  allowed_nodes: string[];
  hint: string;
}

/** Payload returned by POST /api/challenges/judge. */
export interface JudgeResult {
  ok: boolean;
  pass: boolean;
  score: number | null;
  metric?: string;
  threshold?: number;
  compared_to?: string;
  error?: string;
}

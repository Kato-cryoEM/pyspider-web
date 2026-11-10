import type { ChallengeSummary, ChallengeSession, JudgeResult } from "./types";
import * as py from "../pyodideEngine";

export async function listChallenges(): Promise<ChallengeSummary[]> {
  const j = (await py.listChallenges()) as { challenges?: ChallengeSummary[] };
  return j.challenges ?? [];
}

export async function startChallenge(challengeId: string): Promise<ChallengeSession> {
  const j = (await py.startChallenge(challengeId)) as unknown as ChallengeSession & { error?: string };
  if (j.error) throw new Error(j.error);
  return j;
}

export async function fetchFixturePreview(sessionId: string, name: string): Promise<string | null> {
  const j = (await py.fixturePreview(sessionId, name)) as
    { error?: string; png?: { png?: string | null } };
  if (j.error) return null;
  const b64 = j.png?.png;
  return b64 ? `data:image/png;base64,${b64}` : null;
}

export interface JudgeReq {
  challenge_id: string;
  session_id: string;
  answer_node_id: string;
  nodes: { id: string; type: string; params: Record<string, unknown> }[];
  edges: { src: string; srcPort: string; dst: string; dstPort: string }[];
}

export async function judgeChallenge(req: JudgeReq): Promise<JudgeResult> {
  const j = (await py.judgeChallenge(req)) as unknown as JudgeResult & { error?: string };
  if (j.error && typeof (j as JudgeResult).pass !== "boolean") {
    throw new Error(j.error);
  }
  return j;
}

import { useState } from "react";

import type { ChallengeState } from "./useChallenge";
import "./challenge.css";

interface Props {
  state: ChallengeState;
  /** True when the picker modal should be shown. */
  pickerOpen: boolean;
  onClosePicker: () => void;
  /** Called when the user picks a challenge; the parent clears the canvas,
   *  drops the fixture nodes, then kicks off `state.begin()`. */
  onPickChallenge: (challengeId: string) => void;
  /** Hands the parent the "submit" trigger — the parent builds the engine
   *  graph from React Flow state and designates the answer node id. */
  onSubmit: () => void;
  onEnd: () => void;
  /** Currently-selected node id, surfaced so the submit button can tell the
   *  student which node will be judged. */
  selectedNodeId: string | null;
  selectedNodeLabel: string | null;
}


export default function ChallengePanel({
  state, pickerOpen, onClosePicker, onPickChallenge,
  onSubmit, onEnd, selectedNodeId, selectedNodeLabel,
}: Props) {
  const [showHint, setShowHint] = useState(false);

  // Fixture generation can take a few seconds (3D volume projection,
  // noisy averaging). Show a blocking overlay so the user sees something
  // is happening instead of staring at an empty canvas.
  const loading = state.busy && !state.session;

  return (
    <>
      {loading && (
        <div className="challenge-loading-backdrop">
          <div className="challenge-loading-card">
            <div className="challenge-loading-spinner" />
            <div className="challenge-loading-text">Preparing the puzzle…</div>
            <div className="challenge-loading-sub">
              Building fixtures (stack + ground truth)
            </div>
          </div>
        </div>
      )}
      {pickerOpen && (
        <div className="challenge-picker-backdrop" onClick={onClosePicker}>
          <div className="challenge-picker" onClick={(e) => e.stopPropagation()}>
            <div className="challenge-picker-title">🎯 Choose a challenge</div>
            {!state.loaded && (
              <div style={{ color: "var(--muted)", fontSize: 12 }}>Loading…</div>
            )}
            {state.loaded && state.available.length === 0 && (
              <div style={{ color: "var(--muted)", fontSize: 12 }}>
                No challenges found. Add JSON files under
                <code>&nbsp;backend/challenges/</code> or
                <code>&nbsp;~/.pyspider/challenges/</code>.
              </div>
            )}
            {state.available.map((c) => {
              const isCleared = state.cleared.has(c.id);
              return (
                <button key={c.id}
                        className={"challenge-picker-item"
                          + (isCleared ? " is-cleared" : "")}
                        onClick={() => onPickChallenge(c.id)}>
                  <div className="challenge-picker-item-title">
                    {isCleared && (
                      <span className="challenge-cleared-badge">
                        ✅ Cleared
                      </span>
                    )}
                    {c.title}
                  </div>
                  <div className="challenge-picker-item-summary">
                    {c.summary ?? c.prompt.split("\n")[0]}
                  </div>
                </button>
              );
            })}
            <button className="challenge-picker-close" onClick={onClosePicker}>
              Cancel
            </button>
          </div>
        </div>
      )}

      {state.session && (
        <div className="challenge-panel">
          <h2>🎯 {state.session.title}</h2>

          <div className="challenge-prompt">{state.session.prompt}</div>

          <div className="challenge-fixtures">
            {state.session.fixtures.map((f) => (
              <figure key={f.name}>
                {state.previews[f.name]
                  ? <img src={state.previews[f.name]} alt={f.name} />
                  : <div style={{
                      width: "100%", aspectRatio: "1 / 1",
                      background: "#000", border: "1px solid #2a2a35",
                      borderRadius: 4, display: "grid", placeItems: "center",
                      color: "#555", fontSize: 11,
                    }}>…</div>}
                <figcaption>
                  {f.judge_only && (
                    <span title="Judge-only — cannot be used as a graph input"
                          style={{ marginRight: 4 }}>🔒</span>
                  )}
                  {f.name}
                </figcaption>
              </figure>
            ))}
          </div>

          <div className="challenge-allowed-nodes">
            Available nodes:{" "}
            {state.session.allowed_nodes.map((id) => (
              <code key={id}>{id}</code>
            ))}
          </div>

          <div className="challenge-actions">
            <button
              className="primary"
              disabled={state.busy || !selectedNodeId}
              onClick={onSubmit}
              title={selectedNodeId
                ? `Judge "${selectedNodeLabel ?? selectedNodeId}"`
                : "Select the node whose output should be judged"}
            >
              {state.busy ? "Judging…" : "Submit"}
            </button>
            <button onClick={() => setShowHint((v) => !v)}>
              {showHint ? "Hint ▴" : "Hint ▾"}
            </button>
            <button onClick={onEnd}>End</button>
          </div>

          {!selectedNodeId && (
            <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 6 }}>
              Select the node to be judged, then press Submit.
            </div>
          )}

          {showHint && state.session.hint && (
            <div className="challenge-hint">{state.session.hint}</div>
          )}

          {state.result && (
            <div className={`challenge-result ${state.result.pass ? "pass" : "fail"}`}>
              {state.result.pass ? "✓ Passed" : "✗ Not yet"}
              {state.result.score != null && (
                <div className="challenge-result-score">
                  {state.result.metric ?? "score"} = {state.result.score.toFixed(4)}
                  {state.result.threshold != null && (
                    <> &nbsp;(pass threshold: {state.result.threshold.toFixed(2)})</>
                  )}
                </div>
              )}
              {state.result.error && (
                <div className="challenge-result-score">
                  {state.result.error}
                </div>
              )}
            </div>
          )}

          {state.error && (
            <div className="challenge-error">Error: {state.error}</div>
          )}
        </div>
      )}
    </>
  );
}

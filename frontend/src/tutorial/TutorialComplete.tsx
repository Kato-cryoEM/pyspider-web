/**
 * One-shot celebration popup shown when a tutorial advances past its last
 * step. The parent (App) renders this whenever `tutorial.completed` is
 * non-null; dismissing calls `tutorial.clearCompleted()`.
 *
 * Visual: full-screen soft backdrop + centered card with the tutorial
 * title, a big 🎉, and a confetti burst (pure CSS, no deps).
 */

import { useEffect } from "react";

interface Props {
  title: string;
  onClose: () => void;
}

export default function TutorialComplete({ title, onClose }: Props) {
  // Auto-close after a few seconds so the user can still click OK sooner.
  useEffect(() => {
    const t = window.setTimeout(onClose, 6000);
    return () => window.clearTimeout(t);
  }, [onClose]);

  // Generate confetti pieces with randomised offsets / colours / delays.
  const confetti = Array.from({ length: 60 }, (_, i) => {
    const left = Math.random() * 100;
    const delay = Math.random() * 0.6;
    const dur = 2.2 + Math.random() * 1.6;
    const rot = Math.random() * 360;
    const colours = ["#ff5f6d", "#ffc371", "#4a7eff", "#8fe3a7",
                     "#f1a0a0", "#c48aff", "#6ad7ff", "#ffdd57"];
    const bg = colours[i % colours.length];
    const size = 6 + Math.round(Math.random() * 6);
    return (
      <span
        key={i}
        className="tutor-confetti-piece"
        style={{
          left: `${left}%`,
          background: bg,
          width: size,
          height: size,
          animationDelay: `${delay}s`,
          animationDuration: `${dur}s`,
          transform: `rotate(${rot}deg)`,
        }}
      />
    );
  });

  return (
    <div className="tutor-complete-backdrop" onClick={onClose}>
      <div className="tutor-confetti-layer">{confetti}</div>
      <div className="tutor-complete-card" onClick={(e) => e.stopPropagation()}>
        <div className="tutor-complete-emoji">🎉</div>
        <div className="tutor-complete-title">Congratulations!</div>
        <div className="tutor-complete-lesson">{title}</div>
        <div className="tutor-complete-sub">Tutorial complete!</div>
        <button className="tutor-complete-ok" onClick={onClose}>OK</button>
      </div>
    </div>
  );
}

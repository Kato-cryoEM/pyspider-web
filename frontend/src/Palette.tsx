import { useMemo, useState, useEffect } from "react";
import type { NodeSpec } from "./types";

interface Props {
  specs: NodeSpec[];
  onDragStart: (evt: React.DragEvent, specId: string) => void;
  /** External bump: when this token changes, uncollapse the named category.
   *  Used by the tutorial to open the right section before highlighting an
   *  item inside it. Ignored on first render (falsy value). */
  forceExpandCategory?: string | null;
  /** If set, show ONLY nodes whose `id` is in this list. Used by Challenge
   *  mode to limit the palette to the puzzle's allowed node set. */
  allowList?: string[] | null;
}

const STORAGE_KEY = "pyspider.palette.collapsed";

// The visual order for categories. Anything not listed drops to the end.
const CAT_ORDER = [
  "I/O", "Data", "Arithmetic", "Statistics", "Geometry", "Masking",
  "Filter", "Fourier", "Correlation", "Alignment",
  "CTF", "Reconstruction", "Classification", "Stack",
];


// Node ids that are internal plumbing for the Challenge mode — they get
// placed on the canvas automatically from a challenge spec's `fixtures`
// and should never be drag-and-dropped by the user. Hidden from the
// Palette unless the active challenge's allow-list explicitly lists one.
const HIDDEN_NODE_IDS = new Set<string>([
  "data.challenge_input",
  "data.challenge_input_volume",
]);

export default function Palette({ specs, onDragStart, forceExpandCategory, allowList }: Props) {
  // Apply the allow-list (if any) before the user's search query so the
  // visible count in the category headers reflects what the user can drag.
  // When no allow-list is active, drop plumbing-only nodes from the base
  // list so they never show up in the user-facing palette.
  const allowed = useMemo(() => {
    if (!allowList || allowList.length === 0) {
      return specs.filter((s) => !HIDDEN_NODE_IDS.has(s.id));
    }
    const set = new Set(allowList);
    return specs.filter((s) => set.has(s.id));
  }, [specs, allowList]);
  const [query, setQuery] = useState("");
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : {};
    } catch {
      return {};
    }
  });

  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(collapsed)); } catch {}
  }, [collapsed]);

  // Tutorial-driven expansion: if the parent bumps a category name in,
  // uncollapse it so the target inside the section becomes visible before
  // the spotlight moves onto it. Also clear the search box — otherwise a
  // filter can hide the palette item the tutorial is trying to spotlight.
  useEffect(() => {
    if (!forceExpandCategory) return;
    setCollapsed((s) => (s[forceExpandCategory] ? { ...s, [forceExpandCategory]: false } : s));
    setQuery("");
  }, [forceExpandCategory]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q
      ? allowed.filter(
          (s) =>
            s.label.toLowerCase().includes(q) ||
            s.id.toLowerCase().includes(q) ||
            s.doc.toLowerCase().includes(q),
        )
      : allowed;
  }, [allowed, query]);

  const byCategory = useMemo(() => {
    const m = new Map<string, NodeSpec[]>();
    for (const s of filtered) {
      const arr = m.get(s.category) ?? [];
      arr.push(s);
      m.set(s.category, arr);
    }
    return [...m.entries()].sort((a, b) => {
      const ai = CAT_ORDER.indexOf(a[0]);
      const bi = CAT_ORDER.indexOf(b[0]);
      return (ai < 0 ? 999 : ai) - (bi < 0 ? 999 : bi);
    });
  }, [filtered]);

  const toggle = (cat: string) =>
    setCollapsed((s) => ({ ...s, [cat]: !s[cat] }));

  const allCollapsed = byCategory.every(([c]) => collapsed[c]);
  const expandAll = () => setCollapsed({});
  const collapseAll = () =>
    setCollapsed(Object.fromEntries(byCategory.map(([c]) => [c, true])));

  return (
    <div className="palette">
      <div className="palette-controls">
        <input
          type="search"
          placeholder="Search nodes…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button
          className="link"
          onClick={allCollapsed ? expandAll : collapseAll}
          title={allCollapsed ? "Expand all" : "Collapse all"}
        >
          {allCollapsed ? "＋" : "−"}
        </button>
      </div>
      {byCategory.map(([cat, items]) => {
        const isCollapsed = !!collapsed[cat] && !query;
        return (
          <div key={cat} className="pal-cat">
            <div className="pal-header" onClick={() => toggle(cat)}>
              <span className="pal-caret">{isCollapsed ? "▸" : "▾"}</span>
              <span className="pal-catname">{cat}</span>
              <span className="pal-count">{items.length}</span>
            </div>
            {!isCollapsed &&
              items.map((s) => (
                <div
                  key={s.id}
                  className="pal-item"
                  data-tutor-id={`palette:${s.id}`}
                  draggable
                  onDragStart={(e) => onDragStart(e, s.id)}
                  title={s.doc}
                >
                  {s.label}
                </div>
              ))}
          </div>
        );
      })}
      {byCategory.length === 0 && (
        <div className="pal-empty">No matches</div>
      )}
    </div>
  );
}

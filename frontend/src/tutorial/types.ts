/**
 * Tutorial data model.
 *
 * A `Lesson` is a linear sequence of `Step`s. Each step tells the runner
 *   1. what to explain (title + body)
 *   2. which UI element to spotlight (via a `[data-tutor-id="..."]` selector)
 *   3. what the user must do to advance (drag a node, edit a param, click
 *      Run, or just click "Next" for an info-only step)
 *
 * The runner watches the app state and auto-advances when the expected
 * change is observed — so the user never has to hunt for the "Next" button
 * on action steps.
 */

/** What the tutorial is expecting the user to do on this step. */
export type StepAction =
  /** Info card only. User clicks "Next" to advance. */
  | { kind: "info" }
  /** Wait until a node of the given `nodeType` appears on the canvas. */
  | { kind: "add_node"; nodeType: string }
  /** Wait until the currently-selected node's param `name` equals `value`
   *  (loose equality via String()). */
  | { kind: "set_param"; nodeName: string; value: string | number | boolean }
  /** Wait until a node of the given `nodeType` is selected on the canvas
   *  (so its params are visible in the Inspector). */
  | { kind: "select_node"; nodeType: string }
  /** Wait until an edge exists from any `srcNodeType`.`srcPort` to any
   *  `dstNodeType`.`dstPort`. The step's Next button stays hidden until
   *  the wire is drawn — matches user's "cannot skip until connected"
   *  requirement. */
  | { kind: "wire_check"; srcNodeType: string; srcPort: string;
      dstNodeType: string; dstPort: string }
  /** Wait until any node of `nodeType` has finished running (status=success). */
  | { kind: "run_node"; nodeType: string }
  /** Wait until the user clicks the target itself (e.g. the Run button). */
  | { kind: "click_target" };

/** Where to draw the spotlight ring. */
export interface StepTarget {
  /** CSS selector against `data-tutor-id`, e.g. "palette:data.mo".  */
  tutorId: string;
  /** Position hint for the description card — "auto" places it opposite
   *  the target relative to the viewport centre. */
  cardSide?: "left" | "right" | "top" | "bottom" | "auto";
  /** If set, the tutorial asks the Palette to expand this category before
   *  measuring the target — otherwise a collapsed category hides the
   *  palette item and the spotlight has nothing to point at. */
  expandPaletteCategory?: string;
}

export interface Step {
  /** Stable id within a lesson (used for jump-to-step). */
  id: string;
  /** Bold title — one short line. */
  title: string;
  /** Description body — a couple of sentences. Rendered as plain text with
   *  paragraph breaks on `\n\n`. */
  body: string;
  /** Element to highlight; omit for global info steps. */
  target?: StepTarget;
  /** How the user advances this step. */
  action: StepAction;
}

export interface Lesson {
  id: string;
  /** Palette-order title, e.g. "1 · Masking with a Gaussian". */
  title: string;
  /** One-line summary shown in the lesson picker. */
  summary: string;
  /** Grouping in the picker — "Basic" / "Application" / ... — free-form. */
  category?: string;
  steps: Step[];
}

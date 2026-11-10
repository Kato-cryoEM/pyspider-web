export type PortKind =
  | "image" | "stack" | "volume" | "fourier" | "phase"
  | "peaks" | "labels" | "scalar" | "string" | "doc";

export interface Port {
  name: string;
  kind: PortKind;
  optional?: boolean;
  /** [otherParamName, requiredValue] — port is hidden in the node body
      when the other param currently has a different value. `requiredValue`
      can be a scalar or a list of accepted values. Runtime is unaffected
      (a hidden port is still a valid wire target) — this just tidies the
      node face when a param toggles between shapes/formats. */
  depends_on?: [string, unknown] | null;
}

export interface Param {
  name: string;
  kind: "float" | "int" | "string" | "text" | "select" | "bool";
  default: number | string | boolean;
  choices?: string[] | null;
  min?: number | null;
  max?: number | null;
  wireable: boolean;
  /** [otherParamName, requiredValue] — hidden when the other param
      currently has a different value. If `requiredValue` is an array, the
      param is shown when the other param's value equals any element. Used
      to build toggle-driven UIs. */
  depends_on?: [string, unknown] | null;
  /** Greyed hint shown when the field is empty (HTML placeholder attr). */
  placeholder?: string | null;
  /** Up/down spinner increment. When nullish, the Inspector falls back to
   *  1 for int params and 0.01 for float params. */
  step?: number | null;
}

export interface NodeSpec {
  id: string;
  label: string;
  category: string;
  inputs: Port[];
  outputs: Port[];
  params: Param[];
  doc: string;
}

export interface StackFrame {
  index: number;
  png: string;
}

export interface OutputPreview {
  kind: "image" | "array" | "scalar" | "doc" | "other";
  shape?: number[];
  dtype?: string;
  png?: string | null;
  min?: number | null;
  max?: number | null;
  value?: number | string | boolean;
  repr?: string;
  /** Downsampled 3D volume (raw uint8 bytes, base64). */
  volume_u8?: string;
  /** Shape of the downsampled volume: [nz, ny, nx]. */
  volume_shape?: number[];
  /** Original value range [min, max] before uint8 normalisation. */
  volume_range?: number[];
  /** Per-frame thumbnails for stack outputs (up to 32 frames). */
  frames?: StackFrame[];
  /** Total number of frames in the stack (may be > frames.length). */
  n_frames?: number;
  /** Doc preview: number of rows and a sample of the first rows. */
  n_rows?: number;
  preview?: number[][];
  /** Optional column headers for the doc-preview table (e.g. eulers_csv
   *  → ["phi","theta","psi"]). null / undefined → unnamed columns. */
  columns?: string[] | null;
  /** For FQ-family filter nodes: base64 PNG of the 2D filter frequency
   *  response with the cutoff drawn as a red ring. */
  filter_ring?: string;
}

export interface NodeResult {
  ok: boolean;
  error: string | null;
  outputs: Record<string, OutputPreview>;
  /** {portName: absolutePath} for outputs that were auto-saved. */
  saved?: Record<string, string>;
}

/** Per-output save configuration stored on each node. Missing entry
 *  means saving is disabled for that port. */
export interface SaveConfig {
  [portName: string]: { dir?: string };
}

export interface AppSettings {
  default_save_dir: string;
}

/** Compatibility class of a wire (edge) between two ports. */
export type EdgeStatus = "ok" | "warn" | "error" | "unknown";

/** Explanation attached to each edge so the Inspector can show why the
 *  connection is coloured / rejected. */
export interface EdgeInfo {
  status: EdgeStatus;
  srcKind?: string;
  dstKind?: string;
  message?: string;
}

import { useEffect, useRef } from "react";

/**
 * Real-time 1D radial cross-section of an FQ-family filter, drawn on a
 * canvas.  Much clearer than the 2D ring for reading off shape / cutoff /
 * transition width. Updates live as params change — no backend call.
 *
 * X axis = spatial frequency (0 .. 0.5 = Nyquist, cyc/pixel).
 * Y axis = filter gain (0 .. 1).
 */
interface Props {
  nodeType: string;
  params: Record<string, unknown>;
  width?: number;
  height?: number;
}

const parseFilterType = (
  nodeType: string, params: Record<string, unknown>,
) => {
  if (!nodeType.startsWith("filter.")) return null;
  const rest = nodeType.slice("filter.".length);
  // Consolidated nodes: filter.lp / filter.hp carry the family in `kind`.
  if (rest === "lp" || rest === "hp") {
    const kind = String(params.kind ?? "gauss");
    return { fam: kind, pass: rest as "lp" | "hp" };
  }
  // Legacy per-family nodes: filter.<fam>_<lp|hp>.
  if (rest.endsWith("_lp")) return { fam: rest.slice(0, -3), pass: "lp" as const };
  if (rest.endsWith("_hp")) return { fam: rest.slice(0, -3), pass: "hp" as const };
  return null;
};

interface FilterInfo {
  H: (f: number) => number;
  markers: { f: number; label: string; color: string }[];
}

function makeFilter(
  fam: string, pass: "lp" | "hp",
  params: Record<string, unknown>,
): FilterInfo {
  const num = (k: string, def: number) => {
    const v = params[k];
    return typeof v === "number" ? v : parseFloat(String(v ?? def));
  };
  let base: (f: number) => number = () => 0;
  const markers: FilterInfo["markers"] = [];

  if (fam === "tophat") {
    const fc = num("fc", 0.1);
    base = (f) => (f <= fc ? 1 : 0);
    markers.push({ f: fc, label: "fc", color: "#ff5a5a" });
  } else if (fam === "gauss") {
    const fc = Math.max(1e-6, num("fc", 0.1));
    base = (f) => Math.exp(-0.5 * (f / fc) ** 2);
    markers.push({ f: fc, label: "fc (σ)", color: "#ff5a5a" });
  } else if (fam === "fermi") {
    const fc = num("fc", 0.1);
    const temp = Math.max(1e-4, num("temp", 0.02));
    base = (f) => 1.0 / (1.0 + Math.exp((f - fc) / temp));
    markers.push({ f: fc, label: "fc", color: "#ff5a5a" });
  } else if (fam === "butter") {
    const fp = num("fp", 0.10);
    const fs = num("fs", 0.15);
    const fc = 0.5 * (fp + fs);
    base = (f) => 1.0 / (1.0 + (f / Math.max(1e-6, fc)) ** 8);
    markers.push({ f: fp, label: "fp", color: "#7bd88f" });
    markers.push({ f: fs, label: "fs", color: "#ff5a5a" });
  } else if (fam === "raised_cos") {
    const fp = num("fp", 0.10);
    const fs = num("fs", 0.15);
    const w = Math.max(1e-6, fs - fp);
    base = (f) => {
      if (f >= fs) return 0;
      if (f <= fp) return 1;
      return 0.5 * (1.0 + Math.cos(Math.PI * (f - fp) / w));
    };
    markers.push({ f: fp, label: "fp", color: "#7bd88f" });
    markers.push({ f: fs, label: "fs", color: "#ff5a5a" });
  }

  const H = pass === "hp"
    ? (f: number) => Math.min(1, Math.max(0, 1 - base(f)))
    : (f: number) => Math.min(1, Math.max(0, base(f)));
  return { H, markers };
}

export default function FilterRing({
  nodeType, params, width = 260, height = 140,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const parsed = parseFilterType(nodeType, params);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !parsed) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    // Layout: margins for axes labels
    const PL = 34, PR = 8, PT = 10, PB = 22;
    const plotW = width - PL - PR;
    const plotH = height - PT - PB;
    const fMax = 0.5;

    const x = (f: number) => PL + (f / fMax) * plotW;
    const y = (g: number) => PT + (1 - g) * plotH;

    // Panel background
    ctx.fillStyle = "#0f0f14";
    ctx.fillRect(PL, PT, plotW, plotH);

    // Gridlines (every 0.1 in f, every 0.25 in gain)
    ctx.strokeStyle = "#2a2a35";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let f = 0.1; f < 0.5; f += 0.1) {
      const xx = x(f);
      ctx.moveTo(xx, PT); ctx.lineTo(xx, PT + plotH);
    }
    for (let g = 0.25; g < 1; g += 0.25) {
      const yy = y(g);
      ctx.moveTo(PL, yy); ctx.lineTo(PL + plotW, yy);
    }
    ctx.stroke();

    // Axes
    ctx.strokeStyle = "#666";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PL, PT); ctx.lineTo(PL, PT + plotH);
    ctx.lineTo(PL + plotW, PT + plotH);
    ctx.stroke();

    // Filter curve
    const { H, markers } = makeFilter(parsed.fam, parsed.pass, params);
    ctx.strokeStyle = "#7bb0ff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    const N = 300;
    for (let i = 0; i <= N; i++) {
      const f = (i / N) * fMax;
      const px = x(f), py = y(H(f));
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Marker vertical lines
    ctx.font = "10px ui-monospace, monospace";
    for (const m of markers) {
      const xx = x(m.f);
      ctx.strokeStyle = m.color;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(xx, PT); ctx.lineTo(xx, PT + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = m.color;
      ctx.fillText(`${m.label}=${m.f.toFixed(3)}`, xx + 3, PT + 12);
    }

    // Tick labels — X
    ctx.fillStyle = "#9a9aa2";
    ctx.font = "10px ui-monospace, monospace";
    ctx.textAlign = "center";
    for (const f of [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]) {
      ctx.fillText(f.toFixed(1), x(f), PT + plotH + 12);
    }
    ctx.textAlign = "right";
    for (const g of [0.0, 0.25, 0.5, 0.75, 1.0]) {
      ctx.fillText(g.toFixed(2), PL - 4, y(g) + 3);
    }
    ctx.textAlign = "center";
    ctx.fillStyle = "#666";
    ctx.fillText("spatial frequency (cyc/pixel)  · 0.5 = Nyquist",
                 PL + plotW / 2, height - 2);
  }, [nodeType, params, width, height, parsed]);

  if (!parsed) return null;
  return (
    <div className="filter-ring-live">
      <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 6 }}>
        filter response · 1D radial cross-section · live-updates with params
      </div>
      <canvas ref={canvasRef}
              style={{ display: "block", marginTop: 3,
                       borderRadius: 3, background: "#0a0a0e" }} />
    </div>
  );
}

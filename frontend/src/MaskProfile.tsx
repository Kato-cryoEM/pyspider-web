import { useEffect, useRef } from "react";

/**
 * Real-time 1D radial cross-section of a mask node (mask.circle or
 * mask.gaussian). Updates live as outer_radius / inner_radius / edge /
 * edge_width / sigma change — no backend call. Mirrors the FilterRing
 * component used for filter.* nodes.
 *
 * X axis = radial distance from the centre of the image (pixels).
 * Y axis = mask gain (0 .. 1).
 */
interface Props {
  nodeType: string;
  params: Record<string, unknown>;
  width?: number;
  height?: number;
  /** Max radius plotted on the X axis. Defaults to half the image size
   *  if we can guess it; otherwise 64 (= half of a 128² test image). */
  rMax?: number;
}

interface MaskInfo {
  H: (r: number) => number;
  markers: { r: number; label: string; color: string }[];
  bands: { r0: number; r1: number; color: string }[];
}

function makeMask(nodeType: string,
                  params: Record<string, unknown>,
                  rMax: number): MaskInfo | null {
  const num = (k: string, def: number) => {
    const v = params[k];
    return typeof v === "number" ? v : parseFloat(String(v ?? def));
  };
  const markers: MaskInfo["markers"] = [];
  const bands: MaskInfo["bands"] = [];
  let H: (r: number) => number = () => 0;

  if (nodeType === "mask.gaussian") {
    // Gaussian with sigma = `sigma` × half-width. We approximate half-width
    // as rMax (matches the backend's `min(plane) / 2` scaling).
    const sigmaFrac = Math.max(1e-4, num("sigma", 0.25));
    const sigma = sigmaFrac * rMax;
    H = (r) => Math.exp(-0.5 * (r / sigma) ** 2);
    markers.push({ r: sigma, label: `σ=${sigma.toFixed(1)}`, color: "#ff5a5a" });
    return { H, markers, bands };
  }

  if (nodeType === "mask.circle") {
    const ro = Math.max(0, num("outer_radius", 30));
    const ri = Math.max(0, num("inner_radius", 0));
    const w  = Math.max(1e-4, num("edge_width", 3));
    const edge = (params["edge"] as string) ?? "disk";

    if (edge === "true_gauss") {
      // Pure Gaussian; outer_radius plays the role of sigma.
      H = (r) => Math.exp(-0.5 * (r / Math.max(1e-4, ro)) ** 2);
      markers.push({ r: ro, label: `σ=${ro.toFixed(1)}`, color: "#ff5a5a" });
    } else {
      // Radial pass band [ri, ro] with soft outer edge of width `w`.
      const outer = (r: number): number => {
        if (r <= ro) return 1.0;
        if (edge === "cosine") {
          const t = Math.min(1, (r - ro) / w);
          return 0.5 * (Math.cos(Math.PI * t) + 1.0);
        }
        if (edge === "gauss_edge") {
          return Math.exp(-0.5 * ((r - ro) / w) ** 2);
        }
        return 0; // disk
      };
      const inner = (r: number): number => (ri > 0 ? (r >= ri ? 1 : 0) : 1);
      H = (r) => outer(r) * inner(r);
      markers.push({ r: ro, label: `ro=${ro.toFixed(1)}`, color: "#ff5a5a" });
      if (ri > 0) {
        markers.push({ r: ri, label: `ri=${ri.toFixed(1)}`, color: "#8fe3a7" });
      }
      if (edge === "cosine" || edge === "gauss_edge") {
        bands.push({ r0: ro, r1: Math.min(rMax, ro + w),
                     color: "rgba(255, 221, 87, 0.14)" });
      }
    }
    return { H, markers, bands };
  }

  return null;
}

export default function MaskProfile({
  nodeType, params, width = 220, height = 100, rMax,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  // Guess a reasonable X-axis range. For most teaching images (128²) half
  // the image = 64 is right; cap at 2× outer_radius so the mask stays
  // readable even when ro is small.
  const roGuess = (() => {
    const v = params["outer_radius"];
    return typeof v === "number" ? v : parseFloat(String(v ?? 30));
  })();
  const effRMax = rMax ?? Math.max(32, roGuess * 1.8, 64);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const PL = 34, PR = 8, PT = 10, PB = 22;
    const plotW = width - PL - PR;
    const plotH = height - PT - PB;
    const x = (r: number) => PL + (r / effRMax) * plotW;
    const y = (g: number) => PT + (1 - g) * plotH;

    // Panel background
    ctx.fillStyle = "#0f0f14";
    ctx.fillRect(PL, PT, plotW, plotH);

    const info = makeMask(nodeType, params, effRMax);
    if (!info) return;

    // Shaded transition bands (drawn under the curve)
    for (const b of info.bands) {
      ctx.fillStyle = b.color;
      ctx.fillRect(x(b.r0), PT, x(b.r1) - x(b.r0), plotH);
    }

    // Gridlines
    ctx.strokeStyle = "#2a2a35";
    ctx.lineWidth = 1;
    ctx.beginPath();
    const rStep = effRMax <= 50 ? 10 : effRMax <= 100 ? 20 : 50;
    for (let r = rStep; r < effRMax; r += rStep) {
      const xx = x(r);
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

    // Mask curve
    ctx.strokeStyle = "#7bb0ff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    const N = 300;
    for (let i = 0; i <= N; i++) {
      const r = (i / N) * effRMax;
      const px = x(r), py = y(info.H(r));
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Marker vertical lines + labels
    ctx.font = "10px ui-monospace, monospace";
    for (const m of info.markers) {
      const xx = x(m.r);
      ctx.strokeStyle = m.color;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(xx, PT); ctx.lineTo(xx, PT + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = m.color;
      ctx.fillText(m.label, xx + 3, PT + 12);
    }

    // Tick labels — X
    ctx.fillStyle = "#9a9aa2";
    ctx.font = "10px ui-monospace, monospace";
    ctx.textAlign = "center";
    const tickVals: number[] = [0];
    for (let r = rStep; r <= effRMax; r += rStep) tickVals.push(r);
    for (const r of tickVals) {
      ctx.fillText(String(r), x(r), PT + plotH + 12);
    }
    // Tick labels — Y
    ctx.textAlign = "right";
    for (const g of [0.0, 0.25, 0.5, 0.75, 1.0]) {
      ctx.fillText(g.toFixed(2), PL - 4, y(g) + 3);
    }
    ctx.textAlign = "center";
    ctx.fillStyle = "#666";
    ctx.fillText("radial distance (pixels)",
                 PL + plotW / 2, height - 2);
  }, [nodeType, params, width, height, effRMax]);

  return (
    <div className="mask-profile-live">
      <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 6 }}>
        mask response · 1D radial cross-section · live-updates with params
      </div>
      <canvas ref={canvasRef}
              style={{ display: "block", marginTop: 3,
                       borderRadius: 3, background: "#0a0a0e" }} />
    </div>
  );
}

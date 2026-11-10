"""Convert array outputs into a browser-friendly preview.

Images become PNG (base64-encoded), scalars pass through as JSON.
Kept minimal — grayscale, auto-contrast, no colormap.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from . import xp


def _hsv_to_rgb_arrays(hue: np.ndarray, value: np.ndarray,
                       saturation: float = 1.0) -> np.ndarray:
    """Vectorised HSV → RGB (uint8, shape H×W×3). Saturation is a scalar."""
    h6 = hue * 6.0
    i = np.floor(h6).astype(np.int32)
    f = h6 - i
    v = value
    s = float(saturation)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    r = np.zeros_like(v); g = np.zeros_like(v); b = np.zeros_like(v)
    idx0 = i == 0; r[idx0]=v[idx0]; g[idx0]=t[idx0]; b[idx0]=p[idx0]
    idx1 = i == 1; r[idx1]=q[idx1]; g[idx1]=v[idx1]; b[idx1]=p[idx1]
    idx2 = i == 2; r[idx2]=p[idx2]; g[idx2]=v[idx2]; b[idx2]=t[idx2]
    idx3 = i == 3; r[idx3]=p[idx3]; g[idx3]=q[idx3]; b[idx3]=v[idx3]
    idx4 = i == 4; r[idx4]=t[idx4]; g[idx4]=p[idx4]; b[idx4]=v[idx4]
    idx5 = i == 5; r[idx5]=v[idx5]; g[idx5]=p[idx5]; b[idx5]=q[idx5]
    return np.stack([
        (r * 255).clip(0, 255).astype(np.uint8),
        (g * 255).clip(0, 255).astype(np.uint8),
        (b * 255).clip(0, 255).astype(np.uint8),
    ], axis=-1)


def _phase_to_png_b64(phase: np.ndarray) -> str:
    """Colour-code a 2D phase array (radians) — hue = phase, S = V = 1."""
    a = np.asarray(phase, dtype=np.float32)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    hue = ((a + np.pi) / (2.0 * np.pi)) % 1.0
    rgb = _hsv_to_rgb_arrays(hue, np.ones_like(hue), 1.0)
    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _complex_to_png_b64(z: np.ndarray) -> str:
    """Domain-colour a 2D complex spectrum:
        hue        = arg(z) wrapped to [0, 1]  → phase
        value      = log(1 + |z|) normalised to [0, 1]  → amplitude (log)
        saturation = 1

    The convention in this app is that Fourier data from `ft.forward` is
    already DC-centred (fftshifted), so no extra shift is applied here.
    """
    hue = ((np.angle(z) + np.pi) / (2.0 * np.pi)) % 1.0
    magn = np.log1p(np.abs(z)).astype(np.float32)
    mmax = float(magn.max()) if magn.size else 1.0
    value = (magn / mmax) if mmax > 0 else np.zeros_like(magn)
    rgb = _hsv_to_rgb_arrays(hue.astype(np.float32), value, 1.0)
    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _image_to_png_b64(a: np.ndarray) -> str:
    if np.iscomplexobj(a) and a.ndim == 2:
        # Domain-coloured Fourier preview: hue = phase, brightness = |F|.
        return _complex_to_png_b64(a)
    if np.iscomplexobj(a):
        a = np.log1p(np.abs(a))
    a = np.asarray(a, dtype=np.float32)
    # RGB annotated preview — (H, W, 3) uint8 or float. Callers (e.g. PK's
    # peak preview) build the image themselves with drawn overlays. Detected
    # BEFORE the "stack middle slice" branch so an RGB image isn't sliced
    # channel-wise into a nonsense 2D map.
    if a.ndim == 3 and a.shape[-1] == 3 and a.shape[0] != 3:
        if a.dtype != np.uint8:
            lo, hi = float(a.min()), float(a.max())
            if hi > lo:
                a = ((a - lo) / (hi - lo) * 255.0).clip(0, 255)
            else:
                a = np.zeros_like(a)
        u8 = a.astype(np.uint8)
        img = Image.fromarray(u8, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    if a.ndim == 3 and a.shape[0] > 1:
        # Show middle slice of a volume / stack.
        a = a[a.shape[0] // 2]
    if a.ndim != 2:
        a = a.squeeze()
    if a.ndim != 2:
        a = np.atleast_2d(a)
    lo, hi = float(a.min()), float(a.max())
    if hi > lo:
        norm = (a - lo) / (hi - lo)
    else:
        norm = np.zeros_like(a)
    u8 = (norm * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(u8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _downsample_volume_to_uint8(arr: np.ndarray, max_dim: int = 64
                                ) -> tuple[bytes, list[int], float, float]:
    """Stride-downsample a 3D array to <= max_dim per axis and normalise to uint8.

    Returns (raw bytes, [nz, ny, nx], min, max). Nearest-neighbour is fine for
    preview quality and keeps the payload small (64^3 = 256 KB → ~350 KB b64).
    """
    a = np.asarray(arr, dtype=np.float32)
    if np.iscomplexobj(arr):
        a = np.log1p(np.abs(arr)).astype(np.float32)
    strides = tuple(max(1, (s + max_dim - 1) // max_dim) for s in a.shape)
    a_ds = a[::strides[0], ::strides[1], ::strides[2]]
    a_ds = np.ascontiguousarray(a_ds, dtype=np.float32)
    lo = float(a_ds.min()) if a_ds.size else 0.0
    hi = float(a_ds.max()) if a_ds.size else 0.0
    if hi > lo:
        norm = (a_ds - lo) / (hi - lo)
    else:
        norm = np.zeros_like(a_ds)
    u8 = (norm * 255.0).clip(0, 255).astype(np.uint8)
    return u8.tobytes(), list(u8.shape), lo, hi


def _frame_to_png_b64(a: np.ndarray, target_max: int = 128) -> str:
    """Render one stack frame as a small grayscale PNG (max side = target_max).
    Complex frames (e.g. Fourier stacks after FT) are shown as log-magnitude."""
    if np.iscomplexobj(a):
        a = np.log1p(np.abs(a))
    a = np.asarray(a, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    norm = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
    u8 = (norm * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(u8, mode="L")
    if max(img.size) > target_max:
        scale = target_max / max(img.size)
        img = img.resize((max(1, int(img.width * scale)),
                          max(1, int(img.height * scale))),
                         Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _stack_frames_preview(arr: np.ndarray, max_frames: int = 32
                          ) -> list[dict]:
    """Sample up to `max_frames` frames evenly from a 3D stack and encode
    each as a thumbnail PNG. Returns a list of {index, png} entries."""
    n = arr.shape[0]
    if n <= 0:
        return []
    if n <= max_frames:
        indices = list(range(n))
    else:
        step = n / max_frames
        indices = [int(round(i * step)) for i in range(max_frames)]
        # de-dupe while preserving order (edge case at small step)
        seen: set[int] = set()
        dedup: list[int] = []
        for i in indices:
            if i not in seen and 0 <= i < n:
                seen.add(i)
                dedup.append(i)
        indices = dedup
    return [{"index": int(i), "png": _frame_to_png_b64(arr[i])} for i in indices]


def render_filter_ring_png(node_type: str, params: dict,
                           canvas: int = 192) -> str | None:
    """For an FQ-family filter node, render a small 2D image showing the
    filter's frequency response with the cutoff drawn as a red ring.

    The image is centred on DC. Radius in the plot is spatial frequency
    (0 at centre → 0.5 = Nyquist at the edge). White = pass, black = cut.
    """
    if not node_type.startswith("filter."):
        return None
    rest = node_type[len("filter."):]
    if rest.endswith("_lp"):
        pass_dir, fam = "lp", rest[:-3]
    elif rest.endswith("_hp"):
        pass_dir, fam = "hp", rest[:-3]
    else:
        return None

    N = int(canvas)
    ax_ = np.linspace(-0.5, 0.5, N)
    u, v = np.meshgrid(ax_, ax_)
    freq = np.sqrt(u * u + v * v).astype(np.float32)  # 0 .. ~0.707

    marker = None
    if fam == "tophat":
        fc = float(params.get("fc", 0.1))
        H = (freq <= fc).astype(np.float32)
        marker = fc
    elif fam == "gauss":
        fc = max(1e-6, float(params.get("fc", 0.1)))
        H = np.exp(-0.5 * (freq / fc) ** 2).astype(np.float32)
        marker = fc
    elif fam == "fermi":
        fc = float(params.get("fc", 0.1))
        temp = max(1e-4, float(params.get("temp", 0.02)))
        H = 1.0 / (1.0 + np.exp((freq - fc) / temp))
        marker = fc
    elif fam == "butter":
        fp = float(params.get("fp", 0.10))
        fs = float(params.get("fs", 0.15))
        fc = 0.5 * (fp + fs)
        # 8th-order-like Butterworth silhouette
        H = 1.0 / (1.0 + (freq / max(1e-6, fc)) ** 8)
        marker = fc
    elif fam == "raised_cos":
        fp = float(params.get("fp", 0.10))
        fs = float(params.get("fs", 0.15))
        H = np.ones_like(freq)
        H[freq >= fs] = 0.0
        tr = (freq >= fp) & (freq < fs)
        w = max(1e-6, fs - fp)
        H[tr] = 0.5 * (1.0 + np.cos(np.pi * (freq[tr] - fp) / w))
        marker = 0.5 * (fp + fs)
    else:
        return None

    if pass_dir == "hp":
        H = 1.0 - H

    # Convert to 8-bit greyscale, colour-convert to RGB so we can draw a ring.
    u8 = (H * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(u8, mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)

    # Cutoff frequency → pixel radius. The plot spans -0.5..+0.5 across N px,
    # so 1 cyc/pixel corresponds to N pixels; radius = marker * N pixels.
    if marker is not None and marker > 0:
        r = float(marker) * N
        cx = cy = N / 2.0
        # Ring
        draw.ellipse((cx - r, cy - r, cx + r, cy + r),
                     outline=(255, 90, 90), width=2)
        # Crosshair at DC
        draw.line((cx - 6, cy, cx + 6, cy), fill=(90, 200, 255), width=1)
        draw.line((cx, cy - 6, cx, cy + 6), fill=(90, 200, 255), width=1)
        # Tiny label showing fc value in the corner
        try:
            draw.text((4, 4), f"fc={marker:.3f}", fill=(255, 200, 200))
        except Exception:
            pass  # PIL font may be missing on some setups

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _csv_string_to_doc(text: str, port_name: str = "") -> dict | None:
    """If ``text`` looks like a multi-row numeric CSV, encode it as a
    doc-file preview (`kind: "doc"`) so the Inspector renders a table
    instead of squashing the string into a single line via `.scalar`.

    Column names are inferred from the port name for the common cases:
      eulers*    → phi, theta, psi
      assignments_csv → ref, angle, dy, dx, mirror, peak
      scree_csv  → index, value, fraction
    Otherwise the columns are unnamed (Inspector shows just the values).

    Returns ``None`` when the string doesn't parse as a consistent-width
    numeric table (falls back to plain scalar preview).
    """
    if "\n" not in text or not text.strip():
        return None
    lines = [ln for ln in text.splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    if len(lines) < 2:
        return None
    rows: list[list[float]] = []
    n_cols: int | None = None
    for ln in lines:
        clean = ln.split("#", 1)[0].strip()
        parts = [p for p in clean.replace(";", ",").split(",") if p.strip()]
        try:
            row = [float(p) for p in parts]
        except ValueError:
            return None
        if n_cols is None:
            n_cols = len(row)
        elif len(row) != n_cols:
            return None
        rows.append(row)
    if n_cols is None or n_cols < 1:
        return None

    p = port_name.lower()
    columns: list[str] | None = None
    if p.startswith("eulers") and n_cols == 3:
        columns = ["phi", "theta", "psi"]
    elif p == "assignments_csv":
        # AP MR family emits 6 cols: ref, angle, dy, dx, mirror, peak.
        # ML angle emits 8 cols:    map_idx, phi, theta, psi, dy, dx, log_lik, entropy.
        if n_cols == 8:
            columns = ["map_idx", "phi", "theta", "psi",
                       "dy", "dx", "log_lik", "entropy"]
        else:
            columns = ["ref", "angle", "dy", "dx", "mirror", "peak"][:n_cols]
    elif p == "scree_csv" and n_cols == 3:
        columns = ["index", "value", "fraction"]
    elif p == "loss_history" and n_cols == 1:
        columns = ["loss"]

    return {
        "kind": "doc",
        "n_rows": len(rows),
        "preview": [list(map(float, r)) for r in rows[:32]],
        "columns": columns,
    }


def _plot_payload(arr: np.ndarray, max_points: int = 512) -> dict:
    """Downsample a 1D array to a flat JSON-safe list for the frontend's
    line-plot widget. Keep complex arrays magnitude-only (plot the modulus).
    """
    a = np.asarray(arr).ravel()
    if np.iscomplexobj(a):
        a = np.abs(a)
    a = np.asarray(a, dtype=np.float64)
    n = int(a.size)
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(np.int64)
        y = a[idx]
        x = idx.astype(np.float64)
    else:
        y = a
        x = np.arange(n, dtype=np.float64)
    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "n_total": n,
    }


def encode_output(value: Any, kind: str | None = None,
                  port_name: str = "") -> dict:
    """Return a JSON-friendly preview of a single port value.

    **Ndim-based dispatch** — the port spec's ``kind`` no longer selects
    the preview style. Instead:

      - 0-D (scalar array)      → scalar text
      - 1-D array               → line plot
      - 2-D array (real/complex)→ PNG image (HSV colour for complex)
      - 3-D array               → rotatable Three.js volume  (+ centre-slice PNG)
      - 4+D array               → shape summary only

    ``port_name`` is still used to guess column headers for CSV strings.
    """
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        arr = xp.to_host(value)
        real = np.log1p(np.abs(arr)) if np.iscomplexobj(arr) else arr
        if arr.ndim == 0:
            try:
                val = float(arr)
            except (TypeError, ValueError):
                val = str(arr)
            return {"kind": "scalar", "value": val,
                    "shape": [], "dtype": str(arr.dtype)}
        if arr.ndim == 1:
            return {
                "kind": "plot",
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "plot": _plot_payload(arr),
                "min": float(real.min()) if arr.size else None,
                "max": float(real.max()) if arr.size else None,
            }
        png: str | None = None
        if arr.ndim == 2:
            png = (_phase_to_png_b64(arr)
                   if kind == "phase" and not np.iscomplexobj(arr)
                   else _image_to_png_b64(arr))
        elif arr.ndim == 3:
            # Centre-slice thumbnail so a 2D fallback is always available
            # (e.g. inline SpiderNode preview before Three.js initialises).
            png = _image_to_png_b64(arr)
        result: dict = {
            "kind": "image" if arr.ndim == 2 else (
                "volume" if arr.ndim == 3 else "array"),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "png": png,
            "min": float(real.min()) if arr.size else None,
            "max": float(real.max()) if arr.size else None,
        }
        if arr.ndim == 3 and arr.size > 0:
            raw, vshape, vmin, vmax = _downsample_volume_to_uint8(arr)
            result["volume_u8"] = base64.b64encode(raw).decode("ascii")
            result["volume_shape"] = vshape
            result["volume_range"] = [vmin, vmax]
        return result
    if isinstance(value, str):
        # Multi-row numeric CSV → render as a table (with column headers when
        # the port name is a well-known csv output).
        doc = _csv_string_to_doc(value, port_name)
        if doc is not None:
            return doc
        return {"kind": "scalar", "value": value}
    if isinstance(value, (int, float, bool)):
        return {"kind": "scalar", "value": value}
    if isinstance(value, list):
        # in-memory doc table (list of rows)
        rows = value
        preview = rows[:8]
        return {
            "kind": "doc",
            "n_rows": len(rows),
            "preview": [list(map(float, r)) for r in preview],
        }
    return {"kind": "other", "repr": repr(value)[:200]}

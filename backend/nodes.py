"""Node catalog.

Each entry wraps a pyspider function or a compact numpy/scipy/skimage
snippet, exposing it as a typed node the frontend can drag and wire.

The heavy lifting comes from pyspider where possible (SPIDER-accurate,
binary-validated); everything else stays a one-liner leveraging the
scientific-Python stack — that's how we keep the source small while
covering most of SPIDER's operator vocabulary.

Port kinds:  "image" (2D), "stack" (3D N,ny,nx), "volume" (3D nz,ny,nx),
             "fourier" (2D complex rfft2), "peaks", "labels",
             "scalar", "string", "doc" (in-memory list-of-rows table).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import ndimage as ndi
from skimage.transform import resize as sk_resize

import pyspider
from pyspider import (
    read_image, write_image, read_stack,
    filter_image,
    cross_correlate, translational_shift, rotational_angle,
    find_peaks,
)
from pyspider import ctf as _ctf
from pyspider import reconstruct as _recon
from pyspider import classify as _classify
from pyspider import testdata as _td

from . import formats, xp
from .man import MANUAL


# ---------------------------------------------------------------------------
# Schema.

@dataclass
class Port:
    name: str
    kind: str
    optional: bool = False
    # (other_param_name, value) — this port is hidden in the frontend when
    # the current value of `other_param_name` doesn't match. `value` can be
    # a scalar or a list of accepted values (same convention as Param).
    # Runtime behavior is unaffected — a hidden port is still valid on the
    # wire and still populated by the node fn.
    depends_on: tuple[str, Any] | None = None


@dataclass
class Param:
    name: str
    kind: str
    default: Any
    choices: list[str] | None = None
    min: float | None = None
    max: float | None = None
    wireable: bool | None = None  # None = default (int/float wireable; string/select not)
    # (other_param_name, value) — this param is only shown in the Inspector
    # when the current value of `other_param_name` equals `value`. The engine
    # still passes the schema default when hidden, so unused fields stay quiet.
    depends_on: tuple[str, Any] | None = None
    # Grey hint text shown when the field is empty (rendered as the browser's
    # placeholder attribute — no visual weight when the user has typed something).
    placeholder: str | None = None
    # Up/down spinner increment for the Inspector's numeric input. When None,
    # the frontend falls back to 1 for int / 0.01 for float. Set to 1.0 for
    # angle params that read naturally in whole degrees.
    step: float | None = None

    @property
    def is_wireable(self) -> bool:
        if self.wireable is not None:
            return self.wireable
        return self.kind in ("float", "int")


@dataclass
class NodeSpec:
    id: str
    label: str
    category: str
    inputs: list[Port]
    outputs: list[Port]
    params: list[Param]
    doc: str = ""
    manual: str | None = None  # SPIDER command name for man-page lookup
    fn: Callable[..., Any] | None = field(default=None, repr=False)

    def to_public(self) -> dict:
        d = asdict(self)
        d.pop("fn", None)
        # Surface each param's effective wireable flag so the frontend can
        # decide whether to render a handle for it.
        for pubp, p in zip(d["params"], self.params):
            pubp["wireable"] = p.is_wireable
        return d


# Helpers to keep node declarations tight.

def _ports(spec) -> list[Port]:
    """Parse a port spec. Accepts three forms:
      * ``""`` / ``None``  → no ports
      * ``"a:image,b:image,mask:image?"`` — string, ``?`` marks optional
      * ``[Port(...), Port(...)]`` — list of pre-built Port objects
        (needed when a port has ``depends_on`` set — the string form
        can't carry that).
    """
    if not spec:
        return []
    if isinstance(spec, (list, tuple)):
        return [p if isinstance(p, Port) else Port(*p) for p in spec]
    out: list[Port] = []
    for part in spec.split(","):
        name, kind = part.split(":")
        optional = kind.endswith("?")
        if optional:
            kind = kind[:-1]
        out.append(Port(name=name, kind=kind, optional=optional))
    return out


def _params(items) -> list[Param]:
    out: list[Param] = []
    for it in items:
        if isinstance(it, Param):
            out.append(it)
        else:
            out.append(Param(*it))
    return out


def N(id, label, category, ins, outs, params, doc, fn, *,
      manual: str | None = None) -> NodeSpec:
    return NodeSpec(id=id, label=label, category=category,
                    inputs=_ports(ins), outputs=_ports(outs),
                    params=_params(params), doc=doc, fn=fn,
                    manual=manual)


def _fold_manual(spec: NodeSpec) -> None:
    """Attach the SPIDER manual cross-reference as a SUPPLEMENT to the
    node's own doc — no longer replaces it or leads with the SPIDER
    command name. The algorithms (polar transform, cross-correlate,
    back-project, …) are universal cryo-EM primitives; the SPIDER
    command code is useful for veterans but shouldn't be the first
    thing a new user reads.

    Rule:
      - Node's own `doc` (set at register time) leads. Kept as-is.
      - If there IS a SPIDER manual match: append a small
        `\n\nSPIDER equivalent: <CODE>. See also: <REF>, <REF>` block.
      - If there's NO node doc at all (unusual), fall back to the
        old behaviour so at least the manual's PURPOSE shows up.
    """
    if not spec.manual:
        return
    entry = MANUAL.get(spec.manual)
    if entry is None:
        return

    tag_lines = [f"SPIDER equivalent: {spec.manual}"]
    if entry.see_also:
        refs = ", ".join(c for c, _ in entry.see_also)
        tag_lines.append(f"See also: {refs}")
    tail = "  ·  ".join(tag_lines)

    if spec.doc and spec.doc.strip():
        spec.doc = f"{spec.doc.rstrip()}\n\n{tail}"
    else:
        # No node doc — fall back to SPIDER's own PURPOSE so something shows.
        spec.doc = f"{entry.purpose.strip()}\n\n{tail}"


# host/device shortcuts.
h = xp.to_host
d = xp.to_device


# =========================================================================
# I/O + Data
# =========================================================================

def _fn_load(path: str):
    """Load a 2D image or a 3D array from disk via the format registry.

    Extension picks the reader (.spi / .mrc / .map / .ccp4 / ...). Plugins
    can register new formats — see backend/plugin_api.py.
    Any 3D file is returned as-is; use `stack.index` downstream to pick a
    frame if you actually wanted a single image.
    """
    arr = formats.read_array(path)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]  # unwrap trivial (1, ny, nx)
    return {"image": d(arr.astype(np.float32))}

def _fn_save(image, path: str, title: str = "", index: int = -1):
    """Save an image to disk via the format registry (extension → writer).

    If `index >= 0` and `{i}` appears in `path`, it is replaced with the
    integer — standard DO-loop pattern (wire DO's `i` into `index`).
    `title` is ignored by non-SPIDER writers.
    """
    if index >= 0 and "{i}" in path:
        path = path.replace("{i}", str(int(index)))
    formats.write_array(path, h(image).astype(np.float32))
    return {}

def _fn_load_stack(path: str, n_max: int = 32):
    """Load a stack of 2D frames.

    * SPIDER (.spi / .dat / .spider) with the ISTACK header is read via
      pyspider.read_stack up to `n_max` frames.
    * Any other extension is routed through the format registry — a
      3D array becomes an N-frame stack (N = shape[0]), and a plain
      2D image becomes a stack of length 1.
    """
    from pathlib import Path as _P
    ext = _P(path).suffix.lower()
    if ext in (".spi", ".dat", ".spider"):
        items = list(read_stack(path))[: int(n_max)]
        if not items:
            return {"stack": d(np.zeros((0, 1, 1), np.float32))}
        arr = np.stack([im.astype(np.float32) for _, im in items], axis=0)
        return {"stack": d(arr)}
    arr = formats.read_array(path).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"cannot interpret shape {arr.shape} as a stack")
    return {"stack": d(arr[: int(n_max)])}


# ---- Save Image (PNG / JPEG / TIFF / BMP via Pillow) ----------------------

def _fn_save_image(image, path: str, format: str = "auto",
                   normalize: str = "auto",
                   min_value: float = 0.0, max_value: float = 1.0,
                   bit_depth: str = "8", invert: bool = False,
                   index: int = -1):
    """Save an image as PNG / JPEG / TIFF / BMP via Pillow.

    Distinct from `io.save` (MRC — lossless float32). This node produces a
    display-ready file: the float array is normalised to [0, 1] and quantised
    to 8-bit (or 16-bit for PNG / TIFF), so it opens directly in any image
    viewer.

    - format=auto picks the writer from the file extension. Explicit values
      (png / jpeg / tiff / bmp) override / append the extension.
    - normalize=auto uses [min, max] of the data; fixed uses (min_value,
      max_value); none assumes the data is already in [0, 1] and just clips.
    - JPEG is 8-bit only; a 16-bit request is silently downgraded.
    - path supports the DO-loop `{i}` template (wire DO's `i` into `index`).
    """
    from PIL import Image as _PILImage
    from pathlib import Path as _P

    a = h(image)
    if np.iscomplexobj(a):
        # Show log-magnitude for complex data (Fourier arrays etc.).
        a = np.log1p(np.abs(a))
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3 and a.shape[0] > 1:
        a = a[a.shape[0] // 2]
    if a.ndim != 2:
        a = np.atleast_2d(a.squeeze())

    if index >= 0 and "{i}" in path:
        path = path.replace("{i}", str(int(index)))

    p = _P(path).expanduser()
    ext_map = {"png": ".png", "jpeg": ".jpg", "jpg": ".jpg",
               "tiff": ".tif", "tif": ".tif", "bmp": ".bmp"}
    fmt = format.lower().strip()
    if fmt != "auto":
        want = ext_map.get(fmt)
        if want and p.suffix.lower() not in (want, want.replace(".tif", ".tiff"),
                                             want.replace(".jpg", ".jpeg")):
            p = p.with_suffix(want)

    if normalize == "auto":
        lo, hi = float(a.min()), float(a.max())
    elif normalize == "fixed":
        lo, hi = float(min_value), float(max_value)
    else:  # "none"
        lo, hi = 0.0, 1.0
    norm = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
    norm = np.clip(norm, 0.0, 1.0)
    if invert:
        norm = 1.0 - norm

    ext = p.suffix.lower()
    want_16 = str(bit_depth) == "16" and ext not in (".jpg", ".jpeg", ".bmp")
    if want_16:
        u = (norm * 65535.0).astype(np.uint16)
        im = _PILImage.fromarray(u, mode="I;16")
    else:
        u = (norm * 255.0).astype(np.uint8)
        im = _PILImage.fromarray(u, mode="L")

    p.parent.mkdir(parents=True, exist_ok=True)
    im.save(str(p))
    return {}


# ---- Plot (matplotlib line / scatter / bar) -------------------------------

def _fn_plot(data: str = "", plot_type: str = "line",
             title: str = "", xlabel: str = "", ylabel: str = "",
             x_column: int = -1, y_column: int = 0,
             width: int = 480, height: int = 360,
             grid: bool = True,
             logscale_x: bool = False, logscale_y: bool = False,
             marker: str = "auto"):
    """Render numeric text data as a 2D plot image.

    `data` is parsed as one row per line, splitting on comma or whitespace.
    Non-numeric tokens and blank / '#'-prefixed lines are ignored, so this
    accepts CSV, whitespace-separated values, and even the mixed
    'key = value' style commonly emitted by loggers.

    x_column = -1 means: use the row index as X (useful for time-series
    text like SGD loss_history). Otherwise pick that column of the parsed
    table. y_column selects the value column.

    Returns a grayscale image (H × W float32) that plugs into any node
    accepting `image` — including Save Image and Save MRC.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[list[float]] = []
    for line in (data or "").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # Normalise separators: comma / semicolon / '=' → space.
        tok = s.replace(",", " ").replace(";", " ").replace("=", " ").split()
        row: list[float] = []
        for t in tok:
            try:
                row.append(float(t))
            except ValueError:
                pass
        if row:
            rows.append(row)

    W, H = max(64, int(width)), max(64, int(height))
    if not rows:
        # Empty-data placeholder.
        fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "no numeric data", ha="center", va="center",
                fontsize=12, color="#888")
        ax.set_axis_off()
    else:
        max_cols = max(len(r) for r in rows)
        for r in rows:
            r.extend([np.nan] * (max_cols - len(r)))
        M = np.asarray(rows, dtype=np.float64)

        yc = max(0, min(int(y_column), M.shape[1] - 1))
        y = M[:, yc]
        if int(x_column) < 0:
            x = np.arange(len(y), dtype=np.float64)
        else:
            xc = max(0, min(int(x_column), M.shape[1] - 1))
            x = M[:, xc]

        fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100)
        ax = fig.add_subplot(111)
        default_marker = {"line": None, "scatter": "o",
                          "bar": None, "stem": None}.get(plot_type, None)
        m = None if marker == "none" else (
            default_marker if marker == "auto" else marker)

        if plot_type == "scatter":
            ax.scatter(x, y, s=20)
        elif plot_type == "bar":
            w = float(x[1] - x[0]) * 0.8 if len(x) > 1 else 0.8
            ax.bar(x, y, width=w)
        elif plot_type == "stem":
            ax.stem(x, y)
        else:
            ax.plot(x, y, marker=m)

        if title:  ax.set_title(title)
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        if grid:   ax.grid(True, alpha=0.3)
        if logscale_x: ax.set_xscale("log")
        if logscale_y: ax.set_yscale("log")

    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    plt.close(fig)
    gray = (0.299 * buf[..., 0] + 0.587 * buf[..., 1]
            + 0.114 * buf[..., 2]).astype(np.float32)
    return {"image": d(gray)}


# ---- EMDB (Electron Microscopy Data Bank) fetch ---------------------------

import gzip as _gzip
import os as _os
import urllib.request as _urlreq

_EMDB_MIRRORS = (
    "https://ftp.wwpdb.org/pub/emdb/structures/EMD-{id}/map/emd_{id}.map.gz",
    "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{id}/map/emd_{id}.map.gz",
    "https://ftp.pdbj.org/pub/emdb/structures/EMD-{id}/map/emd_{id}.map.gz",
)


def _fourier_crop_3d(vol: np.ndarray, target: tuple[int, int, int]
                     ) -> np.ndarray:
    """Downscale a 3D volume by keeping only the central (tz, ty, tx) block of
    its fftshift'd DFT and inverse-transforming. This is an ideal low-pass
    followed by nearest-integer sub-sampling — no aliasing, no interpolation
    blur, unlike stride decimation."""
    tz, ty, tx = target
    F = np.fft.fftshift(np.fft.fftn(vol.astype(np.float32)))
    nz, ny, nx = F.shape
    z0 = (nz - tz) // 2
    y0 = (ny - ty) // 2
    x0 = (nx - tx) // 2
    F_c = F[z0:z0 + tz, y0:y0 + ty, x0:x0 + tx]
    out = np.fft.ifftn(np.fft.ifftshift(F_c)).real
    # Amplitude preservation: numpy ifftn normalises by prod(target); we want
    # constants to survive unchanged, so scale by prod(target) / prod(source).
    scale = (float(tz) * ty * tx) / (float(nz) * ny * nx)
    return (out * scale).astype(np.float32)


def _cubic_zoom_3d(vol: np.ndarray, target: tuple[int, int, int]
                   ) -> np.ndarray:
    """Upscale a 3D volume with cubic B-spline (order=3) interpolation —
    the 3D analog of bicubic."""
    factors = (target[0] / vol.shape[0],
               target[1] / vol.shape[1],
               target[2] / vol.shape[2])
    return ndi.zoom(vol.astype(np.float32), factors, order=3,
                    mode="constant", cval=0.0).astype(np.float32)


def _resize_volume(vol: np.ndarray, box_size: int) -> np.ndarray:
    """Resize a 3D volume to a cubic (box_size)³ shape.
    Fourier-crop if downscaling any axis, cubic-zoom if upscaling."""
    if box_size <= 0:
        return vol
    target = (int(box_size), int(box_size), int(box_size))
    if target == vol.shape:
        return vol.astype(np.float32)
    # If any axis needs downscaling, Fourier crop that axis (well-defined for
    # every axis simultaneously); if every axis needs upscaling, cubic zoom.
    if all(t <= s for t, s in zip(target, vol.shape)):
        return _fourier_crop_3d(vol, target)
    if all(t >= s for t, s in zip(target, vol.shape)):
        return _cubic_zoom_3d(vol, target)
    # Mixed: pass through cubic zoom (Fourier crop can't upscale, and mixing
    # the two per-axis would need padding logic — cubic is close enough).
    return _cubic_zoom_3d(vol, target)


def _fn_fetch_emdb(emdb_id: str = "1010", box_size: int = 0,
                   cache_dir: str = ""):
    """Download an EMDB map (MRC .map.gz), cache it, and return the volume.

    Cache is keyed by EMDB id, so repeated runs reuse the local file.
    `box_size` sets the returned cubic dimension (0 = keep original shape):
      - smaller than the source → Fourier crop (ideal LP + subsample)
      - larger  than the source → cubic B-spline interpolation (order=3)
    """
    raw_id = str(emdb_id).strip()
    for prefix in ("EMD-", "emd-", "EMD_", "emd_"):
        if raw_id.startswith(prefix):
            raw_id = raw_id[len(prefix):]
    if not raw_id.isdigit():
        raise ValueError(f"emdb_id must be a number (or EMD-XXXX), got {emdb_id!r}")

    cache_root = _os.path.expanduser(cache_dir or "~/.pyspider_emdb_cache")
    _os.makedirs(cache_root, exist_ok=True)
    cache_file = _os.path.join(cache_root, f"emd_{raw_id}.map")

    if not _os.path.exists(cache_file) or _os.path.getsize(cache_file) == 0:
        last_err: Exception | None = None
        for template in _EMDB_MIRRORS:
            url = template.format(id=raw_id)
            try:
                req = _urlreq.Request(url, headers={"User-Agent": "pyspider-web/0.1"})
                tmp = cache_file + ".partial"
                with _urlreq.urlopen(req, timeout=60) as resp, \
                     _gzip.GzipFile(fileobj=resp) as gz, \
                     open(tmp, "wb") as out:
                    while True:
                        chunk = gz.read(1 << 20)  # 1 MiB
                        if not chunk:
                            break
                        out.write(chunk)
                _os.replace(tmp, cache_file)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                if _os.path.exists(cache_file + ".partial"):
                    _os.remove(cache_file + ".partial")
        else:
            raise RuntimeError(
                f"failed to fetch EMD-{raw_id} from all mirrors: {last_err}"
            )

    vol = formats.read_array(cache_file).astype(np.float32)
    if int(box_size) > 0:
        vol = _resize_volume(vol, int(box_size))
    return {"volume": d(vol)}


# ---- In-memory doc-file (SPIDER-style table) --------------------------------

def _parse_doc_text(text: str) -> list[list[float]]:
    """Parse a multi-line CSV/whitespace-separated table into rows of floats.

    Lines starting with ``#`` and blank lines are ignored. Values may be
    separated by commas OR whitespace. Non-numeric cells become 0.0 so a
    malformed row still fits the 4-column schema.
    """
    rows: list[list[float]] = []
    for raw in text.splitlines():
        s = raw.split("#", 1)[0].strip()
        if not s:
            continue
        parts = [p for p in re.split(r"[,\s]+", s) if p]
        vals: list[float] = []
        for p in parts:
            try:
                vals.append(float(p))
            except ValueError:
                vals.append(0.0)
        rows.append(vals)
    return rows


def _fn_doc_text(text: str = ""):
    rows = _parse_doc_text(text)
    n = len(rows)
    # `last_row` = n - 1 is the natural target for DO.end when using
    # start=0, step=1 (SPIDER DO is inclusive of end).
    return {"doc": rows, "n_rows": n, "last_row": max(0, n - 1)}


def _fn_doc_row(doc, row: int = 0):
    """Return columns 1..4 of a doc row as scalars. Missing cells → 0.0.

    Row index is zero-based. If out of range, all outputs are 0.0 (the
    graph does not fail; downstream code can guard with n_rows).
    """
    rows = doc if isinstance(doc, list) else []
    idx = int(row)
    if 0 <= idx < len(rows):
        r = rows[idx]
    else:
        r = []
    def _col(i: int) -> float:
        return float(r[i]) if i < len(r) else 0.0
    return {"v1": _col(0), "v2": _col(1), "v3": _col(2), "v4": _col(3)}


# =========================================================================
# Fourier (real-image <-> complex-fourier port kind).
# =========================================================================

def _fn_ft(image, dims: str = "auto"):
    """Forward FFT — picks per-plane 2D or whole-volume 3D via `dims`.

    Both modes pre-shift so DC sits at the geometric centre of the output,
    which means a user-drawn centred filter (MO disk, etc.) multiplies the
    FT output directly without any manual fftshift. `ft.inverse` undoes the
    shift before inverse-transforming so FT → IFT is identity.

    `dims=auto` (default): 2D input → 2D FFT; 3D input → full 3D FFT. Use
    this when wiring a volume straight through without thinking about the
    dimensionality. The only time you need to override is when a 3D array
    is a *stack of 2D images* and you want per-frame FFT (= `dims=2d`).

    `dims=2d`: numpy fft2 restricted to the last two axes. 2D image → one
    2D FT; 3D stack `(N, ny, nx)` → per-frame 2D FT (PJ 3Q → FT → MU → IFT
    just works). Port kind on the `fourier` output.

    `dims=3d`: numpy fftn over all three axes. Requires a 3D volume input.
    Port kind on the `fourier3d` output (a different kind so downstream
    nodes that only consume 2D FTs don't accidentally type-check-pass a
    volume-FT and produce nonsense).
    """
    a = h(image)
    mode = dims
    if mode == "auto":
        mode = "3d" if a.ndim == 3 else "2d"
    if mode == "3d":
        if a.ndim != 3:
            raise ValueError(
                f"ft.forward dims=3d needs a 3D volume, got shape {a.shape}")
        return {"fourier3d": np.fft.fftshift(np.fft.fftn(a))}
    axes = (-2, -1)
    return {"fourier": np.fft.fftshift(np.fft.fft2(a, axes=axes), axes=axes)}

def _fn_ift(fourier=None, fourier3d=None, dims: str = "auto",
            ny: int = 0, nx: int = 0):
    """Inverse FFT — matching `dims` to forward.

    `dims=auto` (default): emit a volume when the `fourier3d` input is
    wired, otherwise emit a 2D image from the `fourier` input.

    `dims=2d`: consumes `fourier` (2D FT or per-frame stack), returns an
    image. `ny`/`nx` can force the output shape if the FT was cropped /
    padded (0 = auto-infer from input).

    `dims=3d`: consumes `fourier3d` (volume FT), returns a volume.
    """
    mode = dims
    if mode == "auto":
        mode = "3d" if fourier3d is not None else "2d"
    if mode == "3d":
        F = fourier3d if fourier3d is not None else fourier
        if F is None:
            raise ValueError("ft.inverse dims=3d: wire a fourier3d input")
        return {"volume": d(np.fft.ifftn(np.fft.ifftshift(F)).real.astype(np.float32))}
    F = fourier if fourier is not None else fourier3d
    if F is None:
        raise ValueError("ft.inverse dims=2d: wire a fourier input")
    axes = (-2, -1)
    s = (int(ny), int(nx)) if ny and nx else None
    unshift = np.fft.ifftshift(F, axes=axes)
    return {"image": d(np.fft.ifft2(unshift, s=s, axes=axes).real.astype(np.float32))}


def _fn_amplitude(fourier):
    # Fourier data is already centred (fft.forward pre-shifts) — no shift here.
    return {"image": d(np.abs(fourier).astype(np.float32))}

def _fn_log_amplitude(fourier):
    return {"image": d(np.log1p(np.abs(fourier)).astype(np.float32))}

def _fn_phase_spec(fourier):
    return {"image": d(np.angle(fourier).astype(np.float32))}

def _fn_power_spectrum(image):
    a = h(image)
    axes = (-2, -1)
    F = np.fft.fftshift(np.fft.fft2(a, axes=axes), axes=axes)
    return {"image": d(np.log1p(np.abs(F) ** 2).astype(np.float32))}


# =========================================================================
# Arithmetic (SPIDER AD/SU/MU/DI/SQ/LN/SP/NEG et al).
# =========================================================================

def _promote(*xs) -> str:
    """Pick the smallest numpy dtype that fits all inputs; keep complex if
    any input is complex so arith ops don't silently drop the imaginary
    component (needed for user-authored Fourier filters like `F * H`)."""
    if any(np.iscomplexobj(x) for x in xs):
        return "complex64"
    return "float32"


def _bin(op):
    def fn(a, b):
        ah, bh = h(a), h(b)
        dt = _promote(ah, bh)
        return {"out": d(op(np.asarray(ah, dtype=dt), np.asarray(bh, dtype=dt)))}
    return fn

def _un(op):
    def fn(image):
        ih = h(image)
        dt = _promote(ih)
        return {"out": d(op(np.asarray(ih, dtype=dt)))}
    return fn

def _scalar(op):
    def fn(image, value: float = 0.0):
        ih = h(image)
        dt = _promote(ih)
        return {"out": d(op(np.asarray(ih, dtype=dt), float(value)))}
    return fn


# --- AR · Expression (SPIDER AR: per-pixel formula evaluator) --------------

import ast as _ast

_AR_FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "atan2": np.arctan2,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "log10": np.log10, "log2": np.log2,
    "sqrt": np.sqrt, "abs": np.abs, "sign": np.sign,
    "floor": np.floor, "ceil": np.ceil, "round": np.round,
    "min": np.minimum, "max": np.maximum,
    "clip": lambda a, lo, hi: np.clip(a, lo, hi),
    "where": np.where,
}
_AR_CONSTS = {"pi": np.pi, "e": np.e}

# Node types allowed in AST expressions. Everything else is rejected — no
# attribute access, no comprehensions, no imports, no arbitrary calls.
_AR_ALLOWED_NODES = (
    _ast.Expression, _ast.BinOp, _ast.UnaryOp, _ast.BoolOp, _ast.Compare,
    _ast.IfExp,  # ternary a if cond else b
    _ast.Call, _ast.Name, _ast.Load,
    _ast.Num, _ast.Constant, _ast.Tuple, _ast.List,
    _ast.Add, _ast.Sub, _ast.Mult, _ast.Div, _ast.FloorDiv, _ast.Mod, _ast.Pow,
    _ast.USub, _ast.UAdd,
    _ast.And, _ast.Or, _ast.Not,
    _ast.Eq, _ast.NotEq, _ast.Lt, _ast.LtE, _ast.Gt, _ast.GtE,
)


def _ar_translate(formula: str) -> str:
    """Rewrite SPIDER AR-style syntax into a Python-parseable expression.

    - `[x]` / `[y]` (SPIDER coordinate refs) → `x`, `y`
    - `^` (Fortran power) → `**`
    - Function names are case-insensitive: `SIN(P1)` → `sin(P1)`
    - `P1..P4` stay as-is (bound in eval namespace).
    """
    import re as _re
    s = formula.replace("[x]", "x").replace("[X]", "x")
    s = s.replace("[y]", "y").replace("[Y]", "y")
    s = s.replace("^", "**")
    # P1..P4 references: fold lowercase p1..p4 to the canonical uppercase form.
    for i in range(1, 5):
        s = _re.sub(rf"\bp{i}\b", f"P{i}", s)
    for fn in _AR_FUNCS:
        s = _re.sub(r"\b" + fn + r"\b", fn, s, flags=_re.IGNORECASE)
    for const in _AR_CONSTS:
        s = _re.sub(r"\b" + const + r"\b", const, s, flags=_re.IGNORECASE)
    return s


def _ar_validate(tree: _ast.AST) -> None:
    for node in _ast.walk(tree):
        if not isinstance(node, _AR_ALLOWED_NODES):
            raise ValueError(
                f"AR: expression element {type(node).__name__} is not allowed"
            )
        if isinstance(node, _ast.Call) and not isinstance(node.func, _ast.Name):
            raise ValueError("AR: only bare function calls (SIN(...), LOG(...)) allowed")


def _fn_ar_expr(p1, p2=None, p3=None, p4=None, formula: str = "P1"):
    """SPIDER AR — evaluate an arithmetic expression per pixel.

    The formula uses SPIDER's variable convention:
      P1        = input image (required)
      P2..P4    = optional additional images (wire more inputs to combine)
      [x], [y]  = pixel coordinates (0-indexed)
      SIN/COS/EXP/LOG/SQRT/ABS/… + PI, E constants (case-insensitive)
      `^` and `**` both work as power.

    Examples (all valid, matching SPIDER manual):
      (P1 - 2.0) / 2.0 + SIN(P1 + 2)
      (P1 / 512.0) + 0.0
      EXP(LOG(P1 / 512.0) + [x] + [y])
      P1 * P2                          (two-image multiply — same as MU)
      where(P1 > 0.5, P1, 0)           (thresholding)
    """
    a = h(p1).astype(np.float32)
    ns: dict = dict(_AR_FUNCS)
    ns.update(_AR_CONSTS)
    ns["P1"] = a
    if p2 is not None: ns["P2"] = h(p2).astype(np.float32)
    if p3 is not None: ns["P3"] = h(p3).astype(np.float32)
    if p4 is not None: ns["P4"] = h(p4).astype(np.float32)

    # Coordinate grids: 2D by default; for 3D input use (y, x) of the middle slice.
    if a.ndim == 2:
        y, x = np.mgrid[:a.shape[0], :a.shape[1]].astype(np.float32)
    elif a.ndim == 3:
        y, x = np.mgrid[:a.shape[-2], :a.shape[-1]].astype(np.float32)
    else:
        raise ValueError(f"AR: input must be 2D or 3D, got shape {a.shape}")
    ns["x"], ns["y"] = x, y

    expr = _ar_translate(formula or "P1").strip()
    if not expr:
        expr = "P1"
    tree = _ast.parse(expr, mode="eval")
    _ar_validate(tree)
    code = compile(tree, "<AR-formula>", "eval")
    out = eval(code, {"__builtins__": {}}, ns)  # noqa: S307
    return {"image": d(np.asarray(out, dtype=np.float32))}


# =========================================================================
# Statistics.
# =========================================================================

def _fn_stats(image):
    a = h(image).astype(np.float32)
    # Shape convention: 2D → (ny, nx); 3D stack/volume → (nz, ny, nx).
    # Undefined dims are reported as 1 so the port is always wireable.
    if a.ndim == 3:
        nz, ny, nx = int(a.shape[0]), int(a.shape[1]), int(a.shape[2])
    elif a.ndim == 2:
        nz = 1
        ny, nx = int(a.shape[0]), int(a.shape[1])
    else:
        nz, ny, nx = 1, 1, int(a.size)
    return {"min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "sigma": float(a.std()),
            "sum": float(a.sum()),
            "nx": nx, "ny": ny, "nz": nz}

def _fn_histogram(image, bins: int = 64, lo: float = 0.0, hi: float = 0.0):
    a = h(image).ravel().astype(np.float32)
    if float(lo) == 0.0 and float(hi) == 0.0:
        rng = None
    else:
        rng = (float(lo), float(hi))
    hist, edges = np.histogram(a, bins=int(bins), range=rng)
    # Render histogram as a 2D image (bar chart).
    H, W = 64, 256
    img = np.zeros((H, W), np.float32)
    if hist.max() > 0:
        norm = (hist / hist.max() * H).astype(int)
        for i, hi in enumerate(norm):
            col = int(i * W / len(hist))
            colw = max(1, W // len(hist))
            img[H - hi : H, col : col + colw] = 1.0
    return {"image": d(img), "min": float(edges[0]), "max": float(edges[-1])}


# =========================================================================
# Geometry (SPIDER RT SQ / SH / IP / WI / PD).
# =========================================================================

# Map named interpolation kinds → scipy / skimage spline order.
_INTERP_ORDER = {"nearest": 0, "linear": 1, "cubic": 3, "quintic": 5}

def _interp_order(name) -> int:
    """Accept either a name ('nearest'/'linear'/'cubic'/'quintic') or the
    raw int order (backwards-compat with old graphs)."""
    if isinstance(name, (int, float)):
        return int(name)
    return _INTERP_ORDER.get(str(name).lower(), 3)


def _fn_rotate(image, angle_deg: float = 0.0, cval: float = 0.0,
               interpolation: str = "cubic"):
    """SPIDER RO — rotate image in the (y, x) plane.
    Stack-aware: `ndi.rotate` rotates the last 2 axes when `axes=(-2,-1)`."""
    order = _interp_order(interpolation)
    return {"image": d(ndi.rotate(h(image), float(angle_deg),
                                   axes=(-2, -1),
                                   reshape=False, order=order,
                                   mode="constant", cval=float(cval)).astype(np.float32))}

def _fn_shift(image, dy: float = 0.0, dx: float = 0.0, cval: float = 0.0,
              interpolation: str = "cubic"):
    """SPIDER SH — translate image in the (y, x) plane.
    Stack-aware: for a 3D input the frame axis gets shift 0."""
    a = h(image)
    order = _interp_order(interpolation)
    sh = (float(dy), float(dx)) if a.ndim == 2 else (0.0, float(dy), float(dx))
    return {"image": d(ndi.shift(a, sh, order=order, mode="constant",
                                  cval=float(cval)).astype(np.float32))}

def _fn_rot_shift(image, angle_deg: float = 0.0, dy: float = 0.0, dx: float = 0.0,
                  cval: float = 0.0, interpolation: str = "cubic"):
    """Rotate-then-shift composed into ONE interpolation pass via
    ``ndi.affine_transform``. Mathematically equivalent to
    ``shift(rotate(a, angle), dy, dx)`` but with a single spline
    resample — avoids the extra low-pass blur the back-to-back
    ``ndi.rotate`` + ``ndi.shift`` sequence incurs. Stack-aware: for
    a 3D input the frame axis is untouched."""
    a = h(image)
    order = _interp_order(interpolation)
    theta = np.deg2rad(float(angle_deg))
    c, s = np.cos(theta), np.sin(theta)
    # Pullback matrix matching ndi.rotate's internal convention
    # (verified: ndi.affine_transform(a, [[c,s],[-s,c]], offset=...)
    # reproduces ndi.rotate(a, angle_deg) exactly when offset accounts
    # for the rotation centre).
    M = np.array([[c, s], [-s, c]], dtype=np.float64)
    shift_v = np.array([float(dy), float(dx)], dtype=np.float64)

    def _one(frame: np.ndarray) -> np.ndarray:
        center = np.array([(frame.shape[0] - 1) / 2.0,
                           (frame.shape[1] - 1) / 2.0], dtype=np.float64)
        offset = center - M @ (center + shift_v)
        return ndi.affine_transform(frame, M, offset=offset, order=order,
                                     mode="constant", cval=float(cval))

    if a.ndim == 2:
        out = _one(a)
    elif a.ndim == 3:
        out = np.stack([_one(a[i]) for i in range(a.shape[0])], axis=0)
    else:
        raise ValueError(f"rot_shift: input must be 2D or 3D, got {a.shape}")
    return {"image": d(out.astype(np.float32))}

def _fn_resize(image, ny: int = 128, nx: int = 128,
               interpolation: str = "cubic",
               anti_alias: str = "auto"):
    """Resample the image plane to (ny, nx). Stack-aware: applies per frame."""
    a = h(image)
    order = _interp_order(interpolation)
    aa = None if anti_alias == "auto" else (anti_alias == "yes")
    if a.ndim == 2:
        return {"image": d(sk_resize(a, (int(ny), int(nx)),
                                      order=order, preserve_range=True,
                                      anti_aliasing=aa).astype(np.float32))}
    if a.ndim == 3:
        frames = [sk_resize(a[i], (int(ny), int(nx)),
                            order=order, preserve_range=True,
                            anti_aliasing=aa).astype(np.float32)
                  for i in range(a.shape[0])]
        return {"image": d(np.stack(frames, axis=0))}
    raise ValueError(f"resize: input must be 2D or 3D, got shape {a.shape}")

def _fn_crop(image, y: int = 0, x: int = 0, ny: int = 64, nx: int = 64,
             center: bool = True):
    """SPIDER WI — extract a sub-window ``(ny, nx)`` from an image.

    ``center = True`` (default): the crop is auto-centred on the array
    centre — ``(y, x)`` are ignored. This is what you want for Fourier
    cropping (FT → WI → IFT) because FT output has DC at the centre; a
    corner-anchored crop would shift DC off-centre and produce a phase
    ramp / wrap-around in the inverse transform. It's also the natural
    "give me a smaller box around the object" behaviour for real-space
    images.

    ``center = False``: the crop starts at pixel ``(y, x)`` — the
    traditional SPIDER WI corner-anchored crop. Use this when you want
    to sample a specific off-centre region of a real-space image.

    Stack-aware: a 3D input ``(N, H, W)`` is cropped frame by frame; only
    the last two axes are sliced.
    """
    a = h(image)
    ny, nx = int(ny), int(nx)
    if bool(center):
        # Compute the top-left corner such that the crop is centred on the
        # array centre (matches np.fft.fftshift convention).
        H, W = a.shape[-2:]
        y = max(0, (H - ny) // 2)
        x = max(0, (W - nx) // 2)
    else:
        y, x = int(y), int(x)
    # Preserve dtype: don't force float32, otherwise a complex Fourier crop
    # loses its imaginary part and IFT reconstructs a symmetric mess.
    dt = a.dtype if np.iscomplexobj(a) else np.float32
    if a.ndim == 2:
        return {"image": d(np.asarray(a[y : y + ny, x : x + nx], dtype=dt))}
    if a.ndim == 3:
        return {"image": d(np.asarray(a[:, y : y + ny, x : x + nx], dtype=dt))}
    raise ValueError(f"WI: input must be 2D or 3D, got shape {a.shape}")

def _pad_frame(a: np.ndarray, ny: int, nx: int,
               pad_type: str, background: float) -> np.ndarray:
    ay, ax = a.shape
    py, px = max(0, ny - ay), max(0, nx - ax)
    pad_width = ((py // 2, py - py // 2), (px // 2, px - px // 2))
    if pad_type == "circular":
        return np.pad(a, pad_width, mode="wrap")
    if pad_type == "set":
        fill = float(background)
    elif pad_type == "border":
        top    = a[0, :].mean() if a.shape[0] else 0.0
        bottom = a[-1, :].mean() if a.shape[0] else 0.0
        left   = a[:, 0].mean() if a.shape[1] else 0.0
        right  = a[:, -1].mean() if a.shape[1] else 0.0
        fill = float((top + bottom + left + right) / 4.0)
    elif pad_type == "min":
        fill = float(a.min()) if a.size else 0.0
    else:  # "average" (default)
        fill = float(a.mean()) if a.size else 0.0
    return np.pad(a, pad_width, mode="constant", constant_values=fill)


def _fn_pad(image, ny: int = 256, nx: int = 256,
            pad_type: str = "average", background: float = 0.0):
    """SPIDER PD pad-type options:
      average  — fill with the input's overall mean (SPIDER 'A')
      set      — fill with the ``background`` value (SPIDER 'S')
      border   — fill with the perimeter-pixel average (SPIDER 'B')
      min      — fill with the input's minimum value (SPIDER 'M')
      circular — wrap the extended region around the input (SPIDER 'C')

    A 3D stack input ``(N, ny, nx)`` is padded frame by frame — fill values
    (mean / min / border) are computed per frame so each panel is treated
    independently. Output is a stack of the same length.
    """
    a = h(image).astype(np.float32)
    tny, tnx = int(ny), int(nx)
    if a.ndim == 2:
        return {"image": d(_pad_frame(a, tny, tnx, pad_type, background))}
    if a.ndim == 3:
        frames = [_pad_frame(a[i], tny, tnx, pad_type, background)
                  for i in range(a.shape[0])]
        return {"image": d(np.stack(frames, axis=0))}
    raise ValueError(f"PD: input must be 2D or 3D, got shape {a.shape}")

def _fn_po(image, inner_radius: int = 1, outer_radius: int = 0,
           center_x: float = -1.0, center_y: float = -1.0):
    """SPIDER PO — rectangular → polar coordinate representation.

    Output layout (per SPIDER convention):
      * one row per integer ring, from inner_radius (row 0) to outer_radius
        (last row); (outer - inner + 1) rows total.
      * columns are angular samples along each ring, starting at the +x axis
        of a horizontal line halfway down the input and sweeping clockwise;
        line length = round(outer_radius * 2π) columns.
      * sampled with cubic spline interpolation, out-of-bounds → 0.

    Defaults: inner_radius = 1, outer_radius = 0 (means "half of the shorter
    image side, auto"). Handy so the node works on any input without tuning.

    If a 3D array is passed (e.g. a stack from PJ 3Q), the first frame is
    used — pick a specific frame with `stack.index` for finer control.
    """
    a = h(image).astype(np.float32)
    if a.ndim == 3:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"PO needs a 2D image; got shape {tuple(a.shape)}")
    ny, nx = a.shape
    cx = float(center_x) if center_x >= 0 else (nx - 1) / 2.0
    cy = float(center_y) if center_y >= 0 else (ny - 1) / 2.0

    r_in = max(0, int(inner_radius))
    r_out = int(outer_radius) if outer_radius > 0 else min(ny, nx) // 2
    r_out = max(r_in + 1, r_out)
    n_r = r_out - r_in + 1
    n_theta = max(1, int(round(r_out * 2.0 * np.pi)))

    r_vals = np.arange(r_in, r_out + 1, dtype=np.float32)                # (n_r,)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False,
                        dtype=np.float32)                                # (n_theta,)
    R, T = np.meshgrid(r_vals, theta, indexing="ij")                     # (n_r, n_theta)
    # y increases downward → +sin(θ) makes θ sweep clockwise visually.
    xs = cx + R * np.cos(T)
    ys = cy + R * np.sin(T)

    coords = np.stack([ys.ravel(), xs.ravel()], axis=0)
    polar = ndi.map_coordinates(a, coords, order=3, mode="constant", cval=0.0)
    return {"image": d(polar.reshape(n_r, n_theta).astype(np.float32))}


def _fn_flip_h(image):
    """Flip left-right in the image plane; per-frame for 3D stacks."""
    a = h(image)
    return {"image": d(np.ascontiguousarray(np.flip(a, axis=-1)).astype(np.float32))}

def _fn_flip_v(image):
    """Flip top-bottom in the image plane; per-frame for 3D stacks."""
    a = h(image)
    return {"image": d(np.ascontiguousarray(np.flip(a, axis=-2)).astype(np.float32))}

def _fn_transpose(image):
    """Swap x/y axes of the image plane; leaves frame axis alone for stacks."""
    a = h(image)
    return {"image": d(np.ascontiguousarray(np.swapaxes(a, -2, -1)).astype(np.float32))}


# =========================================================================
# Masking (SPIDER MA / TH M / TM).
# =========================================================================

def _radial_dist(shape: tuple[int, int]) -> np.ndarray:
    ny, nx = shape
    y, x = np.mgrid[:ny, :nx].astype(np.float32)
    return np.hypot(y - (ny - 1) / 2, x - (nx - 1) / 2)

def _mask_preview_with_profile(mask_2d: np.ndarray, size: int = 220) -> np.ndarray:
    """Render a mask as a grayscale silhouette with a yellow horizontal
    cross-section profile overlaid at the bottom third — so the shape of
    the edge (hard, gaussian, cosine, ...) is instantly visible.
    Returns (H, W, 3) uint8 for the RGB preview pipeline."""
    m = np.clip(np.asarray(mask_2d, dtype=np.float32), 0.0, 1.0)
    ny, nx = m.shape[-2:]
    # Downsample to `size` for the preview so a 512² mask doesn't create
    # a huge preview image. Simple stride sampling — mask is smooth so
    # aliasing is not a concern.
    if max(ny, nx) > size:
        step_y = max(1, ny // size)
        step_x = max(1, nx // size)
        m = m[..., ::step_y, ::step_x]
        ny, nx = m.shape[-2:]
    top_h = int(ny * 0.72)
    prof_h = ny - top_h
    rgb = np.zeros((ny, nx, 3), dtype=np.uint8)
    # Top: mask silhouette (brighter inside, dark outside)
    gray = (m * 220.0).clip(0, 255).astype(np.uint8)
    rgb[:top_h, :, 0] = gray[:top_h]
    rgb[:top_h, :, 1] = gray[:top_h]
    rgb[:top_h, :, 2] = gray[:top_h]
    # Bottom: dark plot area
    rgb[top_h:, :, :] = 22
    # Faint horizontal 0 and 1 reference lines
    rgb[ny - 2, :, :] = 55                                 # y = 0
    rgb[top_h + 1, :, :] = 55                              # y = 1
    # Faint vertical centre line + edges
    for x in (0, nx // 2, nx - 1):
        rgb[top_h:, x, :] = 55
    # Yellow profile curve: horizontal centre line of the mask
    profile = m[ny // 2, :]
    # y-axis: value 0 at bottom, 1 at top of plot area
    ys = (top_h + 2 + (1.0 - profile) * (prof_h - 4)).astype(np.int32)
    ys = np.clip(ys, top_h + 1, ny - 2)
    for x in range(nx):
        y = ys[x]
        rgb[y - 1:y + 2, x, 0] = 255
        rgb[y - 1:y + 2, x, 1] = 220
        rgb[y - 1:y + 2, x, 2] = 0
    return rgb


def _fn_circle_mask(image, outer_radius: float = 0.0,
                    inner_radius: float = 0.0,
                    edge: str = "disk",
                    background_mode: str = "average",
                    background: float = 0.0,
                    edge_width: float = 3.0):
    """SPIDER MA (masks with radii in pixels).

    ``outer_radius = 0`` is interpreted as "auto = half of the shorter image
    side", so the node works on any input without manual tuning.

    edge:  disk (D) sharp, cosine (C) cosine roll-off outside outer_radius,
           gauss_edge (G) Gaussian roll-off outside outer_radius,
           true_gauss (T) Gaussian centred at origin with sigma = outer_radius.
    background_mode: average (A) input mean, circumference (C) perimeter mean,
           external (E) uses the ``background`` value.
    edge_width: transition width (pixels) for cosine / gauss_edge profiles.
    """
    a = h(image)
    # PRESERVE complex dtype — FT data flowing through the mask must keep
    # its imaginary part. Casting to float32 would leave a Hermitian-real
    # spectrum whose iFT is symmetric about the centre (the "clock's 12
    # showing up at 6 too" bug). Real inputs stay real.
    if np.iscomplexobj(a):
        a = np.asarray(a, dtype=np.complex64)
    else:
        a = np.asarray(a, dtype=np.float32)
    plane = a.shape[-2:]  # last 2 axes: works for 2D and stacks
    r = _radial_dist(plane)
    ro = float(outer_radius)
    if ro <= 0.0:
        ro = float(min(plane) // 2)
    ri = float(inner_radius)
    w = max(1e-4, float(edge_width))

    if edge == "true_gauss":
        # Pure Gaussian mask across the whole image; ro plays the role of sigma.
        mask = np.exp(-0.5 * (r / max(1e-4, ro)) ** 2).astype(np.float32)
    else:
        # Radial pass region [ri, ro], possibly with soft outer edge.
        outer = np.ones_like(r, np.float32)
        if edge == "cosine":
            t = np.clip((r - ro) / w, 0.0, 1.0)
            outer = 0.5 * (np.cos(np.pi * t) + 1.0)
            outer = np.where(r <= ro, 1.0, outer)
        elif edge == "gauss_edge":
            beyond = np.exp(-0.5 * ((r - ro) / w) ** 2)
            outer = np.where(r <= ro, 1.0, beyond)
        else:  # "disk"
            outer = (r <= ro).astype(np.float32)
        inner = (r >= ri).astype(np.float32) if ri > 0 else 1.0
        mask = (outer * inner).astype(np.float32)

    # For complex (Fourier) input, "background" makes no physical sense —
    # you don't fill masked-out frequencies with a mean amplitude. Force
    # bg = 0 so the mask is pure multiplication (zero out rejected bins).
    if np.iscomplexobj(a):
        bg = 0.0
    elif background_mode == "average":
        bg = float(a.mean()) if a.size else 0.0
    elif background_mode == "circumference":
        # Perimeter of the (y, x) plane. `...` selects "all leading axes" so
        # a 3D stack yields four (N,)-length arrays; averaging then reduces
        # to a scalar bg (one bg per whole run — consistent with SPIDER's
        # single 'C' fill).
        top    = a[..., 0, :].mean() if a.shape[-2] else 0.0
        bottom = a[..., -1, :].mean() if a.shape[-2] else 0.0
        left   = a[..., :, 0].mean() if a.shape[-1] else 0.0
        right  = a[..., :, -1].mean() if a.shape[-1] else 0.0
        bg = float((top + bottom + left + right) / 4.0)
    else:  # "external"
        bg = float(background)

    # Broadcast the 2D mask across the frame axis for stack input. The node
    # previews the masked output directly (= what downstream nodes consume),
    # matching the standard "see what the node produced" convention.
    return {
        "image": d(a * mask + bg * (1.0 - mask)),
    }

def _fn_gaussian_mask(image, sigma: float = 0.25, background: float = 0.0):
    a = h(image)
    # Preserve complex dtype (see mask.circle for the "12 shows at 6" bug).
    if np.iscomplexobj(a):
        a = np.asarray(a, dtype=np.complex64)
    else:
        a = np.asarray(a, dtype=np.float32)
    plane = a.shape[-2:]
    r = _radial_dist(plane) / (min(plane) / 2)
    m = np.exp(-0.5 * (r / max(1e-4, sigma)) ** 2).astype(np.float32)
    bg = 0.0 if np.iscomplexobj(a) else float(background)
    return {
        "image": d(a * m + bg * (1.0 - m)),
    }

def _fn_threshold(image, direction: str = "both",
                  upper: float = 1.0, lower: float = 0.0):
    """SPIDER TH direction:
      above  (A) — clip values above `upper` down to `upper`
      below  (B) — clip values below `lower` up to `lower`
      both   (C) — clip both sides into [lower, upper]
    """
    a = h(image).astype(np.float32)
    if direction == "above":
        out = np.minimum(a, float(upper))
    elif direction == "below":
        out = np.maximum(a, float(lower))
    else:  # "both"
        out = np.clip(a, float(lower), float(upper))
    return {"image": d(out)}

def _fn_binarize(image, threshold: float = 0.5,
                 low_value: float = 0.0, high_value: float = 1.0):
    a = h(image).astype(np.float32)
    return {"image": d(np.where(a > float(threshold),
                                float(high_value), float(low_value))
                       .astype(np.float32))}


# =========================================================================
# Filter (existing SPIDER FQ).
# =========================================================================

# SPIDER FQ has ten IOPT variants; each family exposes different knobs:
#   fc     — sharp / Gaussian: single cutoff
#   fermi  — Fermi: cutoff fc + edge temperature
#   band   — Butterworth / raised-cosine: passband fp + stopband fs
_FQ = [
    ("tophat-lp", "TopHat LP", 1,  "fc"),
    ("tophat-hp", "TopHat HP", 2,  "fc"),
    ("gauss-lp",  "Gaussian LP", 3, "fc"),
    ("gauss-hp",  "Gaussian HP", 4, "fc"),
    ("fermi-lp",  "Fermi LP",  5, "fermi"),
    ("fermi-hp",  "Fermi HP",  6, "fermi"),
    ("butter-lp", "Butterworth LP", 7,  "band"),
    ("butter-hp", "Butterworth HP", 8,  "band"),
    ("raised-cos-lp", "Raised-Cos LP", 9,  "band"),
    ("raised-cos-hp", "Raised-Cos HP", 10, "band"),
]

_FQ_DOCS = {
    "tophat-lp": "Sharp low-pass. Passes frequencies below fc, cuts everything above. Introduces ringing.",
    "tophat-hp": "Sharp high-pass. Removes low-frequency content (background / gradients) below fc.",
    "gauss-lp":  "Gaussian low-pass with smooth attenuation exp(-(f/fc)²/2). No ringing.",
    "gauss-hp":  "Gaussian high-pass (1 − gauss-lp). Suppresses slow variations.",
    "fermi-lp":  "Fermi (sigmoid) low-pass. Edge centred at radius fc, width controlled by temp (larger = softer).",
    "fermi-hp":  "Fermi high-pass. Complementary to fermi-lp.",
    "butter-lp": "Butterworth low-pass. Order is fitted from the passband/stopband pair (fp, fs).",
    "butter-hp": "Butterworth high-pass. Same order-fitting rule, complementary shape.",
    "raised-cos-lp": "Raised-cosine low-pass. Cosine roll-off across the [fp, fs] transition band.",
    "raised-cos-hp": "Raised-cosine high-pass (1 − raised-cos-lp).",
}


def _mk_fq(kind: str, label: str, iopt: int, param_kind: str) -> NodeSpec:
    pad_param = ("padding", "select", "border", ["border", "none"])
    if param_kind == "fc":
        params = [("fc", "float", 0.1, None, 0.0, 0.5), pad_param]
        def fn(image, fc: float = 0.1, padding: str = "border"):
            return {"image": d(filter_image(h(image), kind,
                                            fc=float(fc), padding=padding).astype(np.float32))}
    elif param_kind == "fermi":
        params = [("fc", "float", 0.1, None, 0.0, 0.5),
                  ("temp", "float", 0.02, None, 0.001, 0.5), pad_param]
        def fn(image, fc: float = 0.1, temp: float = 0.02, padding: str = "border"):
            return {"image": d(filter_image(h(image), kind,
                                            fc=float(fc), temp=float(temp),
                                            padding=padding).astype(np.float32))}
    else:  # band-limited (fp / fs)
        params = [("fp", "float", 0.10, None, 0.0, 0.5),
                  ("fs", "float", 0.15, None, 0.0, 0.5), pad_param]
        def fn(image, fp: float = 0.10, fs: float = 0.15, padding: str = "border"):
            return {"image": d(filter_image(h(image), kind,
                                            fp=float(fp), fs=float(fs),
                                            padding=padding).astype(np.float32))}
    doc = (f"SPIDER FQ · {label} (IOPT {iopt}). {_FQ_DOCS[kind]} "
           f"Frequencies are in cycles per pixel (0.5 = Nyquist). "
           f"padding='border' extends the image with the perimeter mean before FFT "
           f"(SPIDER's default), 'none' reproduces FQ NP.")
    return N(f"filter.{kind.replace('-', '_')}", label, "Filter",
             "image:image", "image:image", params, doc, fn, manual="FQ")


# =========================================================================
# Convolution / correlation extras.
# =========================================================================

def _fn_autocorrelate(image):
    return {"image": d(cross_correlate(h(image), h(image)))}

def _fn_ccc(a, b, mask=None):
    """SPIDER CC C — Pearson coefficient plus per-image mean/std and Euclidean
    distance.  If a mask image is wired in, only pixels where mask>0 are used
    (the SPIDER convention for the optional mask file).
    """
    xa = h(a).astype(np.float64).ravel()
    xb = h(b).astype(np.float64).ravel()
    if mask is not None:
        mv = h(mask).astype(np.float64).ravel() > 0
        xa, xb = xa[mv], xb[mv]
    av1, av2 = float(xa.mean()), float(xb.mean())
    sd1, sd2 = float(xa.std()),  float(xb.std())
    xc = xa - av1; yc = xb - av2
    n1 = np.linalg.norm(xc); n2 = np.linalg.norm(yc)
    ccc = float((xc @ yc) / (n1 * n2 + 1e-30))
    eud = float(np.linalg.norm(xa - xb))
    return {"ccc": ccc, "av1": av1, "sd1": sd1, "av2": av2, "sd2": sd2, "eud": eud}


# =========================================================================
# Doc-file I/O and simple utilities (SPIDER CP / FI / UD / SD).
# =========================================================================

from pyspider import read_header as _read_header


def _fn_cp(image):
    """Pass-through copy — useful as a junction point in graphs."""
    return {"image": image}


def _fn_fi(path: str):
    hdr = _read_header(path)
    return {
        "nx": int(hdr.nx), "ny": int(hdr.ny), "nz": int(hdr.nz),
        "fmax": float(hdr.fmax), "fmin": float(hdr.fmin),
        "av": float(hdr.av), "sig": float(hdr.sig),
    }


def _fn_ud(path: str, key: int = 1):
    """SPIDER UD — read the row with matching key from a text doc file.

    Doc-file format follows SPIDER's plain-text convention:
        <key>  <ncol>  <v1> <v2> ... <vN>
    or a legacy variant:
        <key>  <v1> <v2> ... <vN>
    Comments (leading ';') and blank lines are ignored.  If several rows
    share the same key, the LAST one wins (matches SPIDER UD).
    """
    target = int(key)
    hit: list[float] | None = None
    try:
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith(";"):
                    continue
                parts = line.split()
                try:
                    k = int(float(parts[0]))
                except ValueError:
                    continue
                if k != target:
                    continue
                # Optional column count: if parts[1] is an integer AND matches
                # the remaining count, treat it as ncol.
                rest = parts[1:]
                if len(rest) >= 1:
                    try:
                        ncol = int(rest[0])
                        if ncol == len(rest) - 1:
                            rest = rest[1:]
                    except ValueError:
                        pass
                hit = [float(x) for x in rest]
    except FileNotFoundError:
        return {"v1": 0.0, "v2": 0.0, "v3": 0.0, "found": 0}
    if hit is None:
        return {"v1": 0.0, "v2": 0.0, "v3": 0.0, "found": 0}
    vals = hit + [0.0, 0.0, 0.0]
    return {"v1": float(vals[0]), "v2": float(vals[1]), "v3": float(vals[2]),
            "found": 1}


def _fn_sd(path: str, key: int = 1, n_cols: int = 3,
           v1: float = 0.0, v2: float = 0.0, v3: float = 0.0, v4: float = 0.0,
           v5: float = 0.0, v6: float = 0.0, v7: float = 0.0, v8: float = 0.0,
           append: str = "yes"):
    """SPIDER SD — append a row ``key ncol v1 v2 ... v_ncol`` to a text doc.

    ``n_cols`` selects how many of v1..v8 are actually written. Wire the
    scalar outputs you care about (e.g. OR SH's angle_deg, dy, dx, peak,
    mirror into v1..v5, with n_cols=5). Extra inputs above n_cols are
    ignored — you don't need to disconnect them.
    """
    n = max(0, min(8, int(n_cols)))
    vals = [v1, v2, v3, v4, v5, v6, v7, v8][:n]
    cells = "".join(f" {float(v):14.6g}" for v in vals)
    row = f"{int(key):6d} {n:2d}{cells}\n"
    mode = "a" if append == "yes" else "w"
    with open(path, mode) as fh:
        fh.write(row)
    return {}


# =========================================================================
# Model image generator (SPIDER MO).
# =========================================================================


def _fn_mo(nx: int = 128, ny: int = 128, kind: str = "circle",
           density: float = 1.0, radius: float = 20.0, radius_y: float = 20.0,
           width: float = 30.0, height: float = 30.0,
           center_x: float = -1.0, center_y: float = -1.0,
           mean: float = 0.0, std: float = 0.35, seed: int = 42,
           waves_csv: str = ""):
    """SPIDER MO — generate a model 2D image.

    kind:
      blank           — SPIDER B: constant `density` everywhere
      circle          — SPIDER C: filled disk of `radius` at (cx, cy)
      gaussian        — legacy: 2D Gaussian, max = `density`, sigma = `radius`
      rectangle       — filled rectangle `width` × `height` centred on (cx, cy)
      ring            — annulus of outer `radius`, thickness `width`
      gaussian_sum    — SPIDER G: 2D Gaussian ellipse (radii `radius`, `radius_y`)
                        normalized so pixel sum = 1
      gaussian_g1     — SPIDER G1: first-order Gaussian ellipse mask, max = 1
      gaussian_g2     — SPIDER G2: second-order super-Gaussian mask
      gaussian_g3     — SPIDER G3: third-order super-Gaussian mask
      random_uniform  — SPIDER R (N): uniform [0,1) noise; `seed` reproducible
      random_gaussian — SPIDER R (Y): Gaussian noise (`mean`, `std`); `seed` reproducible
      sine_waves      — SPIDER S: sum of sine waves from `waves_csv`
                        (one row per wave: amp,phase_deg,kx,ky; up to 20 rows)
      test_sine       — SPIDER T: diagonal sine wave, spatial frequency (2, 2)
      wedge           — SPIDER W: diagonal density wedge, 1.0 (top-left) → 0.0 (bottom-right)
    center coordinates default (-1) to the image centre.
    """
    nx, ny = int(nx), int(ny)
    cx = center_x if center_x >= 0 else (nx - 1) / 2.0
    cy = center_y if center_y >= 0 else (ny - 1) / 2.0
    y, x = np.mgrid[:ny, :nx].astype(np.float32)

    if kind == "blank":
        img = np.full((ny, nx), float(density), dtype=np.float32)
    elif kind == "circle":
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        img = np.where(r2 <= float(radius) ** 2, float(density), 0.0).astype(np.float32)
    elif kind == "gaussian":
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        sig = max(1e-3, float(radius))
        img = (float(density) * np.exp(-0.5 * r2 / (sig * sig))).astype(np.float32)
    elif kind == "soft_circle":
        # Flat disk of `radius` with a Gaussian-tapered edge of σ = `width`.
        # Inside the disk the value is `density`; outside it falls off as
        # density · exp(-(dist-r)² / (2σ²)). Set width ≈ 2–5 for a visibly
        # soft edge; width = 0 degenerates to a hard-edged circle.
        # (Avoid naming the local `d` — that's the host→device helper
        # imported at module top.)
        dist = np.hypot(x - cx, y - cy).astype(np.float32)
        r = float(radius); sig = max(1e-3, float(width))
        edge = np.exp(-0.5 * np.maximum(dist - r, 0.0) ** 2 / (sig * sig))
        img = (float(density) * edge).astype(np.float32)
    elif kind == "rectangle":
        w2, h2 = float(width) / 2.0, float(height) / 2.0
        inside = (np.abs(x - cx) <= w2) & (np.abs(y - cy) <= h2)
        img = np.where(inside, float(density), 0.0).astype(np.float32)
    elif kind == "ring":
        r = np.hypot(x - cx, y - cy)
        ro, wid = float(radius), float(width)
        img = np.where((r <= ro) & (r >= max(0.0, ro - wid)),
                       float(density), 0.0).astype(np.float32)
    elif kind in ("gaussian_sum", "gaussian_g1", "gaussian_g2", "gaussian_g3"):
        a = max(1e-3, float(radius))
        b = max(1e-3, float(radius_y))
        rr = ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2
        if kind == "gaussian_sum":
            g = np.exp(-rr).astype(np.float32)
            s = float(g.sum())
            img = (g / s).astype(np.float32) if s > 0 else g
        else:
            order = {"gaussian_g1": 1, "gaussian_g2": 2, "gaussian_g3": 3}[kind]
            img = np.exp(-(rr ** order)).astype(np.float32)
    elif kind == "random_uniform":
        rng = np.random.default_rng(int(seed))
        img = rng.random((ny, nx)).astype(np.float32)
    elif kind == "random_gaussian":
        rng = np.random.default_rng(int(seed))
        img = rng.normal(float(mean), float(std), size=(ny, nx)).astype(np.float32)
    elif kind == "sine_waves":
        img = np.zeros((ny, nx), np.float32)
        count = 0
        for line in (waves_csv or "").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.replace(";", ",").split(",") if p.strip()]
            if len(parts) < 4:
                continue
            try:
                amp = float(parts[0])
                phase_deg = float(parts[1])
                kx = float(parts[2])
                ky = float(parts[3])
            except ValueError:
                continue
            phi = np.deg2rad(phase_deg)
            img += (amp * np.sin(2.0 * np.pi * (kx * x / nx + ky * y / ny) + phi)).astype(np.float32)
            count += 1
            if count >= 20:  # SPIDER max
                break
    elif kind == "test_sine":
        img = np.sin(2.0 * np.pi * (2.0 * x / nx + 2.0 * y / ny)).astype(np.float32)
    elif kind == "wedge":
        img = (1.0 - (x / max(1.0, nx - 1) + y / max(1.0, ny - 1)) / 2.0).astype(np.float32)
    else:
        img = np.zeros((ny, nx), np.float32)
    return {"image": d(img)}


# =========================================================================
# Even Euler-angle generator (SPIDER VO EA).
# =========================================================================


def _fn_voea(delta_theta: float = 15.0,
             theta_min: float = 0.0, theta_max: float = 90.0,
             phi_min: float = 0.0, phi_max: float = 359.9,
             symmetry: str = "c1"):
    """SPIDER VO EA — quasi-even Eulerian angles on a spherical patch.

    Algorithm (Penczek et al., matching SPIDER's VO EA):
      for theta in [theta_min, theta_max] step delta_theta:
        n_phi = round((phi_max - phi_min) * sin(theta) / delta_theta)
        distribute n_phi angles evenly across [phi_min, phi_max]
    psi is set to 0 (SPIDER's VO EA convention; use OR SH / AP SH to refine).

    ``symmetry`` restricts the output to the *asymmetric unit* of the given
    point group, matching what BP 3F / BP CG expect when they replicate the
    projections around the symmetry axes internally:
      c1        — full sphere (phi ∈ [phi_min, phi_max], theta ∈ [theta_min, theta_max])
      cN (N≥2)  — phi restricted to [0, 360/N); theta unrestricted
      dN (N≥1)  — phi restricted to [0, 360/N); theta restricted to upper hemisphere [0, 90]
    When the user has ALREADY set a tighter phi/theta window than the ASU
    would allow, we keep the user's tighter window (no silent widening).
    """
    sym = symmetry.strip().lower()
    p_lo, p_hi = float(phi_min), float(phi_max)
    t_lo, t_hi = float(theta_min), float(theta_max)
    if sym.startswith("c") and sym[1:].isdigit() and int(sym[1:]) > 1:
        n = int(sym[1:])
        p_hi = min(p_hi, 360.0 / n - 1e-6)
    elif sym.startswith("d") and sym[1:].isdigit() and int(sym[1:]) >= 1:
        n = int(sym[1:])
        p_hi = min(p_hi, 360.0 / n - 1e-6)
        t_hi = min(t_hi, 90.0)

    dt = max(1e-4, float(delta_theta))
    rows: list[tuple[float, float, float]] = []
    theta = t_lo
    while theta <= t_hi + 1e-6:
        s = max(1e-6, np.sin(np.deg2rad(theta)))
        n_phi = max(1, int(round((p_hi - p_lo) * s / dt)))
        if theta == 0.0 or theta == 180.0:
            rows.append((0.0, theta, 0.0))
        else:
            for j in range(n_phi):
                phi = p_lo + (p_hi - p_lo) * j / n_phi
                rows.append((phi, theta, 0.0))
        theta += dt
    csv = "\n".join(f"{p:.4f},{t:.4f},{s:.4f}" for p, t, s in rows)
    return {"eulers_csv": csv, "num_angles": len(rows)}


# =========================================================================
# OR SH — combined rotation+translation alignment (single-image AP SH).
# =========================================================================


def _fn_or_sh(reference, image, first_ring: int = 2, last_ring: int = 15,
              search_range: int = 6, step: int = 1,
              check_mirror: str = "yes"):
    """SPIDER OR SH — determine the rotation, translation (and optional
    mirror flip) that best aligns ``image`` to ``reference``.  This is the
    single-image counterpart of AP SH.

    Strategy:
      1. Optional mirror: also try flip_h(image) and pick the better peak.
      2. Rotational alignment via polar rings [first_ring, last_ring].
      3. Rotate the image back, then translational refinement (CC + PK)
         restricted to ±search_range pixels in each direction.
    """
    ref = h(reference).astype(np.float32)
    img = h(image).astype(np.float32)
    if ref.ndim != 2 or img.ndim != 2:
        raise ValueError(
            f"OR SH: needs a pair of 2D images, got reference {ref.shape} "
            f"and image {img.shape}. Insert `stack.index` if you meant one "
            "frame from a stack, or use AP SH for stack-vs-reference batch "
            "alignment."
        )

    def _try(im):
        raw_ang, pk = rotational_angle(
            ref, im,
            n_rings=max(1, int(last_ring) - int(first_ring) + 1),
            n_angles=360, sub_pixel=True)
        # Sign fix: `rotational_angle` returns the polar-space shift k such
        # that polar(ref) matches polar(im) shifted by +k — this is the
        # rotation that moves REF to IM. To align IM to REF we need the
        # inverse rotation when calling scipy.rotate (which treats positive
        # angle as CCW on the image). Without this negation scipy ends up
        # rotating in the wrong direction and alignment stalls far from the
        # true peak (verified empirically against an exhaustive scan).
        ang = -float(raw_ang)
        rot = ndi.rotate(im, ang, reshape=False, order=3,
                         mode="constant", cval=0.0)
        cc = cross_correlate(ref, rot)
        ny, nx = cc.shape
        cy, cx = ny // 2, nx // 2
        sr = max(1, int(search_range))
        y0, y1 = max(0, cy - sr), min(ny, cy + sr + 1)
        x0, x1 = max(0, cx - sr), min(nx, cx + sr + 1)
        window = cc[y0:y1, x0:x1]
        peaks = find_peaks(window, n_peaks=1, sub_pixel=True)
        if not peaks:
            return ang, 0.0, 0.0, float(pk)
        p = peaks[0]
        dy = float(p.iy + p.dy) + y0 - cy
        dx = float(p.ix + p.dx) + x0 - cx
        h_pk = float(p.refined_height if p.refined_height is not None else p.height)
        return ang, dy, dx, h_pk

    ang, dy, dx, pk = _try(img)
    mirror = 0
    if check_mirror == "yes":
        img_m = img[:, ::-1].copy()
        ang_m, dy_m, dx_m, pk_m = _try(img_m)
        if pk_m > pk:
            ang, dy, dx, pk, mirror = ang_m, dy_m, dx_m, pk_m, 1
    return {"angle_deg": float(ang), "dy": float(dy), "dx": float(dx),
            "peak": float(pk), "mirror": int(mirror)}


# =========================================================================
# Stack + volume constructors (utility, not SPIDER-native).
# =========================================================================


def _fn_mo3d(nz: int = 32, ny: int = 32, nx: int = 32,
             kind: str = "sphere", density: float = 1.0,
             radius: float = 8.0,
             center_z: float = -1.0, center_y: float = -1.0,
             center_x: float = -1.0):
    """3D counterpart of MO — generate a synthetic volume.

    kind: blank | sphere | gaussian | cube.
    """
    nz, ny, nx = int(nz), int(ny), int(nx)
    cz = center_z if center_z >= 0 else (nz - 1) / 2.0
    cy = center_y if center_y >= 0 else (ny - 1) / 2.0
    cx = center_x if center_x >= 0 else (nx - 1) / 2.0
    z, y, x = np.mgrid[:nz, :ny, :nx].astype(np.float32)
    r2 = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2
    if kind == "blank":
        vol = np.full((nz, ny, nx), float(density), np.float32)
    elif kind == "gaussian":
        sig = max(1e-3, float(radius))
        vol = (float(density) * np.exp(-0.5 * r2 / (sig * sig))).astype(np.float32)
    elif kind == "cube":
        r = float(radius)
        inside = (np.abs(z - cz) <= r) & (np.abs(y - cy) <= r) & (np.abs(x - cx) <= r)
        vol = np.where(inside, float(density), 0.0).astype(np.float32)
    else:  # sphere
        vol = np.where(r2 <= float(radius) ** 2, float(density), 0.0).astype(np.float32)
    return {"volume": d(vol)}


def _resize_image_2d(img: np.ndarray, target_size: int) -> np.ndarray:
    """Resize a square 2D image to (target_size, target_size).

    - Downscale (target < current): Fourier crop — ideal low-pass + subsample,
      no aliasing.
    - Upscale (target > current): cubic B-spline zoom via ndi.zoom (order=3).
    Mirrors the volume-resize path used by fetch_emdb.
    """
    img = np.asarray(img, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"resize_image_2d needs 2D input, got {img.shape}")
    n = int(target_size)
    if n <= 0 or img.shape == (n, n):
        return img.astype(np.float32)
    src_n = min(img.shape)
    if n < src_n:
        # Fourier crop: keep central (n, n) block of fftshift(fft2(img)).
        F = np.fft.fftshift(np.fft.fft2(img))
        y0 = (F.shape[0] - n) // 2
        x0 = (F.shape[1] - n) // 2
        Fc = F[y0:y0 + n, x0:x0 + n]
        out = np.fft.ifft2(np.fft.ifftshift(Fc)).real
        # Rescale so a constant array survives with the same DC level.
        scale = (float(n) * n) / (float(F.shape[0]) * F.shape[1])
        return (out * scale).astype(np.float32)
    factors = (n / img.shape[0], n / img.shape[1])
    return ndi.zoom(img, factors, order=3,
                    mode="constant", cval=0.0).astype(np.float32)


# --- data.betagal — bundled β-galactosidase demo data ----------------------
#
# Three variants shipped in `backend/bundled_data/betagal/` (all derived
# from the publicly-distributed RELION 4.0 tutorial dataset — courtesy of
# Kato / Namba group, Osaka University, JEOL Cryo-ARM 200 kV, 0.885 Å/pix):
#
#   average       — one 2D class average (Class2D/job008 iter 25, class #39).
#                   Clean, high SNR, β-gal tetramer immediately recognisable.
#                   Native 64×64 px. Ideal for demonstrating masks / filters
#                   / alignment on a "textbook" image.
#   raw_particle  — one un-averaged extracted particle from
#                   20170629_00021_frameImage.mrcs. Native 256×256.
#                   Realistic cryo-EM SNR — nearly noise dominated, matches
#                   what a raw particle actually looks like.
#   stack_100     — 100 raw particles evenly sampled from the same movie's
#                   extracted stack. Native 100×256×256. Suitable for demos
#                   of AS S (stack averaging), Class2D (`cls.pca`), AP MR
#                   (multi-reference align), etc.

_BETAGAL_DIR = Path(__file__).resolve().parent / "bundled_data" / "betagal"
# `__file__` = backend/nodes.py, so `.parent` = backend/. The bundled data
# ships alongside the code, not under the (transient) tutorial_data/ tree.
_BETAGAL_VARIANTS = {
    "average":      _BETAGAL_DIR / "class_averages.mrcs",  # full 50-class stack
    "raw_particle": _BETAGAL_DIR / "raw_particle.mrc",
    "stack_100":    _BETAGAL_DIR / "stack_100.mrc",
    # Real β-galactosidase 3D reference reconstruction, native 128³ voxels
    # (from the RELION 3.0 tutorial dataset — courtesy of the Namba lab and
    # RELION consortium). Resampled to the user's chosen `size` at load time.
    "map3d":        _BETAGAL_DIR / "map3d_128.mrc",
}

# Module-level counter for `class_index = -1` (auto-cycle). Each Run graph
# that touches the average variant advances by one and wraps modulo 50, so
# the user can browse every class average by hitting Run repeatedly.
_BETAGAL_CYCLE = 0


def _load_betagal_map3d(size: int = 128,
                         lp_resolution_A: float = 5.0,
                         apix_A: float = 1.0) -> np.ndarray:
    """Load the bundled REAL β-galactosidase 3D reconstruction (RELION 3.0
    tutorial reference, native 128³ voxels at 1 Å/px), apply a soft
    Fourier low-pass at `lp_resolution_A` Å, then optionally resample
    (cubic B-spline) to `size × size × size`. The low-pass removes
    high-frequency noise that dominates a raw refined map so the 3D
    viewer / projection / back-projection demos look clean."""
    src = _BETAGAL_VARIANTS["map3d"]
    if not src.exists():
        raise FileNotFoundError(
            f"β-gal map3d file missing at {src}. "
            "Run the bundled-data prep step to populate it.")
    vol = formats.read_array(str(src)).astype(np.float32)
    if vol.ndim != 3:
        raise ValueError(f"β-gal map3d: expected 3D volume, got {vol.shape}")

    # Cosine-edged Fourier low-pass. cutoff_freq [cycles/Å] = 1 / resolution.
    # Transition band ~ 1 Å wider on each side so the edge isn't a hard
    # brick-wall (which would introduce ripples in real space).
    F = np.fft.fftshift(np.fft.fftn(vol))
    N = vol.shape[0]
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=float(apix_A)))
    kz, ky, kx = np.meshgrid(freqs, freqs, freqs, indexing="ij")
    kr = np.sqrt(kx * kx + ky * ky + kz * kz).astype(np.float32)
    cutoff = 1.0 / float(lp_resolution_A)             # cycles/Å (pass band)
    trans_lo = 1.0 / (float(lp_resolution_A) + 1.0)   # transition start
    trans_hi = 1.0 / (float(lp_resolution_A) - 1.0) if lp_resolution_A > 1.5 else cutoff * 1.2
    mask = np.ones_like(kr, dtype=np.float32)
    mask[kr > trans_hi] = 0.0
    in_trans = (kr >= trans_lo) & (kr <= trans_hi)
    denom = max(1e-6, trans_hi - trans_lo)
    mask[in_trans] = 0.5 * (1.0 + np.cos(np.pi * (kr[in_trans] - trans_lo) / denom))
    vol = np.real(np.fft.ifftn(np.fft.ifftshift(F * mask))).astype(np.float32)
    del F, mask, kr, kx, ky, kz

    tgt = int(size)
    if tgt > 0 and vol.shape != (tgt, tgt, tgt):
        from scipy import ndimage as ndi
        factors = tuple(tgt / s for s in vol.shape)
        vol = ndi.zoom(vol, factors, order=3, mode="constant", cval=0.0)
    return vol.astype(np.float32)


def _fn_betagal(variant: str = "average", class_index: int = -1,
                size: int = 128):
    """Load a bundled β-galactosidase image / stack from the RELION 4.0
    tutorial dataset.

    variant:
      average       ONE 2D class average from the bundled 50-class stack.
                    ``class_index = -1`` (default) auto-cycles: each Run
                    graph shows the next class in order (0 → 1 → … → 49 →
                    0 …), so you can browse them all just by hitting Run.
                    Set ``class_index`` to a specific 0–49 to pin one.
      raw_particle  one un-averaged extracted particle — realistic
                    cryo-EM SNR, appropriate for showing what averaging
                    buys you.
      stack_100     100 raw particles as a 3D stack — feed into AS S for
                    an on-the-fly class average, or into Class2D / AP MR
                    for a mini refinement demo.

    ``size`` resizes each frame's last two axes to the given square box
    (Fourier crop when smaller than native, cubic B-spline zoom when
    larger). Native: 64 (average), 256 (raw_particle, stack_100).
    """
    v = str(variant).lower().strip()

    # Real β-gal 3D reconstruction — bundled from the RELION 3.0 tutorial
    # reference (native 128³ @ 1 Å/px). Low-passed at 5 Å to remove noise,
    # then resampled to the user's chosen `size` (default = 128 = native).
    if v == "map3d":
        vol = _load_betagal_map3d(int(size) if int(size) > 0 else 128)
        return {"volume": d(vol)}

    src = _BETAGAL_VARIANTS.get(v)
    if src is None:
        raise ValueError(
            f"β-gal variant {v!r} unknown. Use one of: map3d, "
            + ", ".join(_BETAGAL_VARIANTS)
        )
    if not src.exists():
        raise FileNotFoundError(
            f"β-gal variant {v!r} missing at {src}. "
            "Run the bundled-data prep step to populate it."
        )
    arr = formats.read_array(str(src)).astype(np.float32)

    # For the class-averages variant, pick which class to show.
    #   class_index = -1  → auto-cycle each Run (module-level counter)
    #   class_index >= 0  → pin that class (0..n-1, wraps modulo n)
    used_idx = int(class_index)
    if v == "average" and arr.ndim == 3:
        global _BETAGAL_CYCLE
        n_classes = arr.shape[0]
        if used_idx < 0:
            used_idx = _BETAGAL_CYCLE % n_classes
            _BETAGAL_CYCLE = (used_idx + 1) % n_classes
        else:
            used_idx = used_idx % n_classes
        arr = arr[used_idx]
        # Restandardise per class so SNR is comparable across selections.
        arr = (arr - arr.mean()) / (arr.std() + 1e-9)
        # Emit which class was actually shown — visible in the Inspector log.
        try:
            from backend.progress import report_progress as _rp
            _rp(1.0, f"β-gal average → class #{used_idx}")
        except Exception:
            pass
    elif arr.ndim == 3 and arr.shape[0] == 1:
        # Trivial length-1 stacks → 2D image so downstream nodes don't broadcast.
        arr = arr[0]

    tgt = int(size)
    if tgt > 0:
        if arr.ndim == 2 and arr.shape != (tgt, tgt):
            arr = _resize_image_2d(arr, tgt)
        elif arr.ndim == 3 and arr.shape[-2:] != (tgt, tgt):
            arr = np.stack(
                [_resize_image_2d(arr[i], tgt) for i in range(arr.shape[0])],
                axis=0,
            )
    # Route to the right output port so downstream sees the correct wire
    # kind: `stack` when the variant returns >1 frame, `image` otherwise.
    if arr.ndim == 3 and arr.shape[0] > 1:
        return {"stack": d(arr)}
    return {"image": d(arr)}


def _fn_stack_build(a, b, c=None, d_=None):
    """Combine 2–4 images into a stack along a new leading axis."""
    frames = [h(a).astype(np.float32), h(b).astype(np.float32)]
    if c is not None:
        frames.append(h(c).astype(np.float32))
    if d_ is not None:
        frames.append(h(d_).astype(np.float32))
    return {"stack": d(np.stack(frames, axis=0))}


def _parse_eulers_list(eulers_csv: str) -> list[tuple[float, float, float]]:
    """Parse `phi,theta,psi` rows out of a doc-file style string. Ignores
    blank rows and comments.

    Format detection:
      - 3+ columns  → columns 0,1,2 are (phi, theta, psi)   [VO EA, sim.projections]
      - 7+ columns  → ml_angle format `idx, phi, theta, psi, dy, dx, ll, entropy`;
                       columns 1,2,3 are the Eulers.
    This lets the student wire align.ml_angle's assignments_csv directly
    into pj3q without a column-extracting intermediate node.
    """
    out: list[tuple[float, float, float]] = []
    for r in eulers_csv.strip().splitlines():
        r = r.strip()
        if not r or r.startswith("#"):
            continue
        cols = r.split("#", 1)[0].split(",")
        try:
            if len(cols) >= 7:
                out.append((float(cols[1]), float(cols[2]), float(cols[3])))
            else:
                out.append((float(cols[0]), float(cols[1]), float(cols[2])))
        except (ValueError, IndexError):
            continue
    return out


def _project_batch_fourier(vol: np.ndarray,
                           eulers: np.ndarray) -> np.ndarray:
    """Fast batch projection via the central-slice theorem.

    Complexity: **one** 3D FFT on the volume, then per angle a 2D
    trilinear slice extraction + one 2D inverse FFT. For a 256³ box and
    ~600 projections this is ~20-50× faster than real-space per-frame
    rotation (which allocates a 256⁴ coordinate grid each time).
    Falls back to `_recon.project` for non-cubic volumes since the
    central slice needs a common size.
    """
    vol = np.asarray(vol, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"volume must be 3D, got {vol.shape}")
    nz, ny, nx = vol.shape
    if not (nz == ny == nx):
        return np.stack([_recon.project(vol, *e) for e in eulers], axis=0)
    N = nz

    # For the central-slice theorem to give the same projection as the
    # real-space `_recon.project` (which rotates about the geometric centre
    # (N-1)/2), we run the FFT in a **centred** convention: shift V so its
    # centre-of-grid is at index (0, 0, 0), FFT, then fftshift so DC is at
    # (N//2, N//2, N//2). Symmetrically, we ifft2 the extracted slice and
    # fftshift the real-space result. This eliminates the global phase-ramp
    # artefact that manifested as horizontal wrap-around at even N.
    V_c = np.fft.ifftshift(vol)
    F = np.fft.fftshift(np.fft.fftn(V_c))
    Fr = np.ascontiguousarray(F.real.astype(np.float32))
    Fi = np.ascontiguousarray(F.imag.astype(np.float32))

    centre = int(N // 2)
    kk = (np.arange(N) - centre).astype(np.float32)
    KY, KX = np.meshgrid(kk, kk, indexing="ij")
    zero = np.zeros_like(KY)
    pts = np.stack([zero, KY, KX], axis=-1)          # (N, N, 3)

    n_e = len(eulers)
    out = np.empty((n_e, N, N), dtype=np.float32)
    # `_recon.project` rotates about (N-1)/2 (half-integer for even N) while
    # our Fourier method's implicit centre is N//2 (integer). For rotations
    # that mix the z axis with y or x, this shows as a 1-pixel translation
    # of the projection. Correct via a per-angle Fourier phase ramp:
    #   shift = 0.5 - (R^T @ (0.5, 0.5, 0.5))[y,x]
    # is the (Δy, Δx) needed to align our output with `_recon.project`.
    half = np.float32(0.5)
    for i in range(n_e):
        R = _recon._euler_zyz(float(eulers[i][0]),
                              float(eulers[i][1]),
                              float(eulers[i][2])).astype(np.float32)
        rot = pts @ R.T                              # (N, N, 3)
        zp = rot[..., 0] + centre
        yp = rot[..., 1] + centre
        xp = rot[..., 2] + centre
        # DFT samples are periodic — use map_coordinates with mode='wrap' so
        # sample points at frequency ±N/2 (which land on index N after the
        # centre offset, one past the array) fold correctly to index 0.
        sr = ndi.map_coordinates(Fr, [zp, yp, xp], order=1, mode="wrap")
        si = ndi.map_coordinates(Fi, [zp, yp, xp], order=1, mode="wrap")
        slice_ft = sr.astype(np.complex64) + 1j * si.astype(np.complex64)

        # Per-angle centring correction.
        A = R.T @ np.array([half, half, half], dtype=np.float32)
        dy_shift = half - float(A[1])
        dx_shift = half - float(A[2])
        # Shift the real-space projection by (-Δy, -Δx) via a Fourier phase
        # ramp so the output matches `_recon.project`'s centred rotation.
        phase = np.exp(
            2j * np.pi * (KY * dy_shift + KX * dx_shift) / float(N)
        ).astype(np.complex64)
        slice_ft = slice_ft * phase

        # Undo the centring: ifftshift moves DC back to (0, 0), ifft2, and
        # fftshift centres the resulting real-space projection at N//2.
        out[i] = np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(slice_ft)).real
        ).astype(np.float32)
    return out


def _fn_project_batch(volume, eulers_csv: str = "0,0,0",
                      method: str = "fourier"):
    """Project a volume at every Euler triple in eulers_csv → stack.

    method:
      fourier — central-slice extraction from fftn(volume); one FFT +
                cheap per-angle slice. **Default**; ~20-50× faster for
                dense batches at box ≥ 128.
      real    — original per-angle real-space rotate-and-sum. Slower
                but simpler to reason about; kept as an escape hatch.
    """
    from backend.progress import report_progress as _rp
    vol = h(volume).astype(np.float32)
    triples = _parse_eulers_list(eulers_csv)
    total = len(triples)
    if total == 0:
        return {"stack": d(np.zeros((0,) + vol.shape[1:], np.float32))}
    eulers = np.asarray(triples, dtype=np.float32)

    if method == "fourier":
        _rp(0.02, f"FFT volume  ({vol.shape})")
        # Compute in one shot; the slice-extraction loop is fast enough
        # that a real per-angle heartbeat there is more noise than help.
        projs = _project_batch_fourier(vol, eulers)
        _rp(1.0, f"projected {total}/{total}  (Fourier central-slice)")
        return {"stack": d(projs.astype(np.float32))}

    projs = np.empty((total, vol.shape[-2], vol.shape[-1]), dtype=np.float32)
    step = max(1, total // 50)
    for idx, (phi, theta, psi) in enumerate(triples):
        projs[idx] = _recon.project(vol, phi, theta, psi).astype(np.float32)
        if (idx + 1) % step == 0 or (idx + 1) == total:
            _rp((idx + 1) / total,
                f"projected {idx + 1}/{total}  (phi={phi:.1f} theta={theta:.1f})")
    return {"stack": d(projs)}


# =========================================================================
# CN / CN N — convolution family (SPIDER CN, CN N).
# =========================================================================

from scipy.signal import fftconvolve as _fftconvolve


def _stack_kernel_pair(xa: np.ndarray, xb: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Align a possibly-3D input with a possibly-2D kernel for use with
    ``scipy.signal.fftconvolve(..., axes=(-2, -1))`` — which requires equal
    ndim on both operands.  A 2D kernel gets a leading length-1 frame axis
    so it broadcasts across the stack's frame axis.  Returns the promoted
    arrays plus the conv axes.
    """
    # Convolution is commutative; normalise so the stack (if any) is `xa`.
    if xa.ndim == 2 and xb.ndim == 3:
        xa, xb = xb, xa
    if xa.ndim == 3 and xb.ndim == 2:
        xb = xb[np.newaxis, ...]
    axes = (-2, -1)
    return xa, xb, axes


def _fn_cn(a, b):
    """SPIDER CN — real-space 2D convolution via FFT.

    Stack-aware: pairing a 3D stack ``(N, ny, nx)`` with a 2D kernel
    ``(ny, nx)`` convolves every frame with the same kernel (last-2-axis
    convolution + broadcast). Both inputs 3D with matching frame count also
    works — per-frame pairwise convolution.

    Note: for CTF correction you probably want `Apply CTF` (single node) or
    `FT → MU × CTF → IFT` — CN treats its kernel as a real-space PSF, so
    feeding a Fourier-space CTF directly to CN is not the standard workflow.
    """
    xa = h(a).astype(np.float32)
    xb = h(b).astype(np.float32)
    xa, xb, axes = _stack_kernel_pair(xa, xb)
    return {"image": d(_fftconvolve(xa, xb, mode="same", axes=axes).astype(np.float32))}


def _fn_cnn(a, b):
    """SPIDER CN N — normalized convolution.

    Standard normalisation: numerator = a * b (FFT convolution);
    denominator = sqrt(local energy of a in the support of b) * ||b||_2.
    Matches the local template-matching formulation used by SPIDER CN N.
    Stack-aware in the same way as ``CN``.
    """
    xa = h(a).astype(np.float32)
    xb = h(b).astype(np.float32)
    xa, xb, axes = _stack_kernel_pair(xa, xb)
    num = _fftconvolve(xa, xb, mode="same", axes=axes)
    kernel = np.ones_like(xb, dtype=np.float32)
    loc_e = _fftconvolve(xa * xa, kernel, mode="same", axes=axes)
    bn = float(np.linalg.norm(xb))
    norm = np.sqrt(np.maximum(loc_e, 0.0)) * (bn if bn > 0 else 1.0)
    out = np.divide(num, norm, out=np.zeros_like(num, np.float32),
                    where=norm > 1e-30)
    return {"image": d(out.astype(np.float32))}


# =========================================================================
# PT — pattern / line / rectangle-outline image (SPIDER PT).
# =========================================================================


def _fn_pt(nx: int = 128, ny: int = 128, pattern: str = "point",
           x1: float = 32.0, y1: float = 32.0,
           x2: float = 96.0, y2: float = 96.0,
           x3: float = 32.0, y3: float = 96.0,
           radius: float = 16.0,
           intensity: float = 1.0, background: float = 0.0,
           subtract: bool = False):
    """SPIDER PT — build a 2D pattern / mask image.

    pattern (matches SPIDER PT codes):
      point            — P:   single pixel at (x1, y1)
      line             — L:   Bresenham line (x1,y1) → (x2,y2)
      filled_circle    — C:   solid disk, centre (x1,y1), `radius`
      outlined_circle  — CL:  circle outline, centre (x1,y1), `radius`
      circle_3pts      — CJ:  unique circle through (x1,y1),(x2,y2),(x3,y3)
      filled_triangle  — T:   solid triangle with vertices at the three points
      outlined_triangle— TL:  triangle outline (three lines)
      filled_box       — B:   solid rectangle from (x1,y1) to (x2,y2)
      outlined_box     — BL:  rectangle outline (same corners)
      cross            — (extra, not SPIDER): full-image cross-hair at (x1,y1)
      rect_outline     — alias of outlined_box (kept for backward compat)
    subtract=True is SPIDER's 'S' suffix: writes `background` where the shape
    would have written `intensity` (erases into an existing mask). Coordinates
    are 0-indexed pixels.
    """
    nx, ny = int(nx), int(ny)
    img = np.full((ny, nx), float(background), np.float32)
    p1 = (int(round(x1)), int(round(y1)))
    p2 = (int(round(x2)), int(round(y2)))
    p3 = (int(round(x3)), int(round(y3)))
    rad = max(0.0, float(radius))
    val = float(background) if subtract else float(intensity)

    def _put(x, y):
        if 0 <= x < nx and 0 <= y < ny:
            img[y, x] = val

    def _line(a, b):
        x, y = a
        dx = abs(b[0] - a[0]); dy = abs(b[1] - a[1])
        sx = 1 if a[0] < b[0] else -1
        sy = 1 if a[1] < b[1] else -1
        err = dx - dy
        while True:
            _put(x, y)
            if x == b[0] and y == b[1]: break
            e2 = 2 * err
            if e2 > -dy: err -= dy; x += sx
            if e2 <  dx: err += dx; y += sy

    def _fill_disk(cx, cy, r):
        # Vectorised: build a boolean mask and assign val.
        yy, xx = np.mgrid[:ny, :nx]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        img[mask] = val

    def _circle_outline(cx, cy, r):
        # Bresenham's midpoint circle algorithm (integer-only).
        cxi, cyi, ri = int(round(cx)), int(round(cy)), int(round(r))
        if ri <= 0:
            _put(cxi, cyi); return
        x, y, err = ri, 0, 0
        while x >= y:
            for (dx, dy) in ((x, y), (y, x), (-y, x), (-x, y),
                             (-x, -y), (-y, -x), (y, -x), (x, -y)):
                _put(cxi + dx, cyi + dy)
            y += 1
            if err <= 0:
                err += 2 * y + 1
            if err > 0:
                x -= 1
                err -= 2 * x + 1

    def _fill_triangle(a, b, c):
        # Barycentric test on the bounding box — plenty fast for typical
        # frame sizes and keeps the code short.
        (ax, ay), (bx, by), (cx, cy) = a, b, c
        x0, x1_ = max(0, min(ax, bx, cx)), min(nx - 1, max(ax, bx, cx))
        y0, y1_ = max(0, min(ay, by, cy)), min(ny - 1, max(ay, by, cy))
        if x0 > x1_ or y0 > y1_:
            return
        yy, xx = np.mgrid[y0:y1_ + 1, x0:x1_ + 1]
        denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if denom == 0:
            _line(a, b); _line(b, c); _line(c, a); return
        u = ((by - cy) * (xx - cx) + (cx - bx) * (yy - cy)) / denom
        v = ((cy - ay) * (xx - cx) + (ax - cx) * (yy - cy)) / denom
        w = 1.0 - u - v
        mask = (u >= 0) & (v >= 0) & (w >= 0)
        sub = img[y0:y1_ + 1, x0:x1_ + 1]
        sub[mask] = val

    def _circumcircle(a, b, c):
        (ax, ay), (bx, by), (cx, cy) = a, b, c
        D = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if D == 0:
            return None
        a2, b2, c2 = ax*ax + ay*ay, bx*bx + by*by, cx*cx + cy*cy
        ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / D
        uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / D
        r = float(np.hypot(ax - ux, ay - uy))
        return ux, uy, r

    if pattern == "point":
        _put(*p1)
    elif pattern == "line":
        _line(p1, p2)
    elif pattern == "cross":
        for x in range(nx): _put(x, p1[1])
        for y in range(ny): _put(p1[0], y)
    elif pattern == "filled_circle":
        _fill_disk(x1, y1, rad)
    elif pattern == "outlined_circle":
        _circle_outline(x1, y1, rad)
    elif pattern == "circle_3pts":
        cc = _circumcircle((float(x1), float(y1)),
                           (float(x2), float(y2)),
                           (float(x3), float(y3)))
        if cc is not None:
            _circle_outline(cc[0], cc[1], cc[2])
    elif pattern == "filled_triangle":
        _fill_triangle(p1, p2, p3)
    elif pattern == "outlined_triangle":
        _line(p1, p2); _line(p2, p3); _line(p3, p1)
    elif pattern == "filled_box":
        x0, y0 = max(0, min(p1[0], p2[0])), max(0, min(p1[1], p2[1]))
        x1i = min(nx - 1, max(p1[0], p2[0]))
        y1i = min(ny - 1, max(p1[1], p2[1]))
        if x0 <= x1i and y0 <= y1i:
            img[y0:y1i + 1, x0:x1i + 1] = val
    elif pattern in ("outlined_box", "rect_outline"):
        x0, y0 = min(p1[0], p2[0]), min(p1[1], p2[1])
        x1i, y1i = max(p1[0], p2[0]), max(p1[1], p2[1])
        for x in range(x0, x1i + 1): _put(x, y0); _put(x, y1i)
        for y in range(y0, y1i + 1): _put(x0, y); _put(x1i, y)
    return {"image": d(img)}


# =========================================================================
# AS S — stack average with variance (SPIDER AS S).
# =========================================================================


def _fn_ass(stack, mode: str = "average_and_variance"):
    """SPIDER AS S — average + variance images of a stack.

    mode:
      average              — return only the mean image
      variance             — return only the per-pixel variance
      average_and_variance — return both (matches SPIDER 'A V' menu)
    """
    s = h(stack).astype(np.float32)
    avg = s.mean(axis=0).astype(np.float32)
    var = s.var(axis=0).astype(np.float32)
    if mode == "average":
        return {"average": d(avg)}
    if mode == "variance":
        return {"variance": d(var)}
    return {"average": d(avg), "variance": d(var)}


# --- CA series (PCA / CORAN eigenimage analysis) ---------------------------

def _fn_ca_s(stack, method: str = "pca",
             n_factors: int = 10, subtract_mean: bool = True):
    """SPIDER CA S — factor analysis on an image stack.

    Two analysis modes matching SPIDER's CA S:
      - **pca**   : classical principal component analysis (mean centring,
                    SVD on the covariance).
      - **coran** : Correspondence Analysis. Each row / column is normalised
                    by its total (Chi-square metric), which is the classical
                    metric for count-like non-negative data (SPIDER's
                    default for image series where pixels are ≥ 0).

    Returns:
      eigenimages   : the top `n_factors` factor images (2D each), stacked
      eigenvalues   : scalar array of length `n_factors`
      scores_csv    : one row per input frame, giving its projection on
                      each retained factor (comma-separated)
      explained_ratio: cumulative fraction of variance kept
    """
    s = h(stack).astype(np.float64)
    if s.ndim != 3:
        raise ValueError(f"CA S: needs a 3D stack (N, ny, nx); got {s.shape}")
    n, ny, nx = s.shape
    X = s.reshape(n, ny * nx)  # (n, p)

    mean_img = X.mean(axis=0)
    if method == "coran":
        # Correspondence analysis with Chi-square normalisation.
        eps = 1e-12
        row_totals = X.sum(axis=1, keepdims=True) + eps
        col_totals = X.sum(axis=0, keepdims=True) + eps
        total = X.sum() + eps
        # Z_ij = (x_ij / total) / sqrt(r_i c_j) - sqrt(r_i c_j) / total
        Z = (X / total) / np.sqrt(row_totals * col_totals / total ** 2 + eps)
        centred = Z - Z.mean(axis=0, keepdims=True)
    else:
        centred = X - mean_img if subtract_mean else X

    # SVD on the centred data matrix. Since typically n << p, this is cheap.
    U, S_, Vt = np.linalg.svd(centred, full_matrices=False)
    k = min(int(n_factors), Vt.shape[0])
    eigenvalues = (S_[:k] ** 2 / max(1, n - 1)).astype(np.float32)
    total_var = float((S_ ** 2).sum() / max(1, n - 1))
    explained = float(eigenvalues.sum() / total_var) if total_var > 0 else 0.0

    factors = Vt[:k].reshape(k, ny, nx).astype(np.float32)
    scores = (U[:, :k] * S_[:k]).astype(np.float32)
    scores_csv = "\n".join(
        ",".join(f"{v:.6f}" for v in row) for row in scores
    )

    return {
        "eigenimages": d(factors),
        "eigenvalues": d(eigenvalues),
        "mean_image":  d(mean_img.reshape(ny, nx).astype(np.float32)),
        "scores_csv":  scores_csv,
        "explained_ratio": explained,
    }


def _fn_ca_si(stack, n_factors: int = 10, subtract_mean: bool = True):
    """SPIDER CA SI — CORAN / PCA that emits its **eigenimages only**
    (SPIDER's "solution + eigenimages" variant). Convenience wrapper
    around CA S with method='pca'."""
    r = _fn_ca_s(stack, method="pca", n_factors=int(n_factors),
                 subtract_mean=bool(subtract_mean))
    return {"eigenimages": r["eigenimages"],
            "eigenvalues": r["eigenvalues"]}


def _fn_ca_sme(eigenvalues, top_n: int = 10):
    """SPIDER CA SME — print / return the top-n eigenvalues.

    Emits a doc-file style CSV (`index, value, fraction`) suitable for
    piping into `Plot · Line` (Plot the scree curve).
    """
    ev = np.asarray(h(eigenvalues), dtype=np.float64).ravel()
    if ev.size == 0:
        return {"scree_csv": "", "top_sum": 0.0}
    k = min(int(top_n), ev.size)
    total = float(ev.sum())
    lines = []
    for i in range(k):
        frac = float(ev[i] / total) if total > 0 else 0.0
        lines.append(f"{i},{float(ev[i]):.6f},{frac:.6f}")
    return {"scree_csv": "\n".join(lines),
            "top_sum": float(ev[:k].sum())}


def _fn_ca_sr(eigenimages, scores_csv: str = "", mean_image=None,
              frame_index: int = 0, factors_used: str = ""):
    """SPIDER CA SR — reconstitute an approximation of an original image
    from selected factors.

    ``scores_csv`` is the per-frame factor loadings emitted by CA S. Pick
    which frame to reconstruct with ``frame_index``. ``factors_used`` is a
    comma-separated list of factor indices to keep (empty = all).
    Result = mean + Σ_k score[frame, k] * eigenimage_k over the chosen k.
    """
    E = h(eigenimages).astype(np.float32)          # (k, ny, nx)
    rows = [r for r in scores_csv.strip().splitlines() if r.strip()]
    if not rows:
        raise ValueError("CA SR: scores_csv is empty")
    fi = max(0, min(int(frame_index), len(rows) - 1))
    scores = np.asarray([float(v) for v in rows[fi].split(",")], dtype=np.float32)
    k_avail = min(E.shape[0], scores.shape[0])

    if factors_used.strip():
        ks = [int(x) for x in factors_used.replace(";", ",").split(",")
              if x.strip().lstrip("-").isdigit()]
        ks = [k for k in ks if 0 <= k < k_avail]
    else:
        ks = list(range(k_avail))

    ny, nx = E.shape[-2:]
    out = (np.zeros((ny, nx), dtype=np.float32) if mean_image is None
           else h(mean_image).astype(np.float32))
    for k in ks:
        out = out + scores[k] * E[k]
    return {"image": d(out.astype(np.float32))}


def _fn_ca_srd(eigenimages, eigenvalues, factor_index: int = 0):
    """SPIDER CA SRD — differential (eigen) image for a selected factor.

    Simply returns eigenimage[factor_index] scaled by sqrt(eigenvalue) so
    the magnitudes are comparable across factors (as CA SRD historically
    reports).
    """
    E = h(eigenimages).astype(np.float32)
    ev = np.asarray(h(eigenvalues), dtype=np.float32).ravel()
    k = max(0, min(int(factor_index), E.shape[0] - 1))
    scale = float(np.sqrt(max(0.0, ev[k]))) if k < ev.size else 1.0
    return {"image": d((E[k] * scale).astype(np.float32))}


# --- AS — Sum a stack (SPIDER AS: sum + variance measures) -----------------

def _fn_as_sum(stack, mode: str = "sum_and_variance"):
    """SPIDER AS — sum every frame of a stack + variance measures.

    mode:
      sum                — total pixel sum (image = per-pixel sum)
      variance           — per-pixel variance across frames
      sum_and_variance   — both (default)

    Also emits ``mean_of_variance`` (average variance across pixels) as a
    scalar — matches SPIDER AS 'avgoff' register.
    """
    s = h(stack).astype(np.float32)
    total = s.sum(axis=0).astype(np.float32)
    var = s.var(axis=0).astype(np.float32)
    m_var = float(var.mean()) if var.size else 0.0
    if mode == "sum":
        return {"sum": d(total), "mean_of_variance": m_var}
    if mode == "variance":
        return {"variance": d(var), "mean_of_variance": m_var}
    return {"sum": d(total), "variance": d(var), "mean_of_variance": m_var}


# --- AS R — Average a stack (SPIDER AS R: mean + variance, optional halves)

def _fn_asr(stack, mode: str = "average_and_variance", seed: int = 42):
    """SPIDER AS R — average a stack + variance measures.

    Similar to AS S but also supports the SPIDER 'split into two subsets'
    mode used for FSC-style comparison.

    mode:
      average_and_variance — mean + per-pixel variance (SPIDER default)
      average              — mean only
      variance             — per-pixel variance only
      two_subsets          — random half-split; returns two averages
    """
    s = h(stack).astype(np.float32)
    if mode == "two_subsets":
        n = s.shape[0]
        rng = np.random.default_rng(int(seed))
        idx = rng.permutation(n)
        half = n // 2
        a = s[idx[:half]].mean(axis=0).astype(np.float32) if half else np.zeros(s.shape[1:], np.float32)
        b = s[idx[half:]].mean(axis=0).astype(np.float32) if (n - half) else np.zeros(s.shape[1:], np.float32)
        return {"average_a": d(a), "average_b": d(b), "n_a": half, "n_b": n - half}
    avg = s.mean(axis=0).astype(np.float32)
    var = s.var(axis=0).astype(np.float32)
    if mode == "average":
        return {"average": d(avg)}
    if mode == "variance":
        return {"variance": d(var)}
    return {"average": d(avg), "variance": d(var)}


# --- ADD family (SPIDER ADD / ADD 2 / ADD F / ADD R) -----------------------

def _fn_add(stack):
    """SPIDER ADD — sum every frame of a stack (or 2 – 4 wired images
    combined via `stack.build`) into one image. Frame count is arbitrary."""
    s = h(stack).astype(np.float32)
    if s.ndim == 2:
        return {"image": d(s)}
    if s.ndim == 3:
        return {"image": d(s.sum(axis=0).astype(np.float32))}
    raise ValueError(f"ADD: expected 2D image or 3D stack, got {s.shape}")


def _fn_add2(a, b):
    """SPIDER ADD 2 — sum of two images / volumes / Fourier arrays.

    Same as MU · Multiply but for addition; equivalent to `AR · Expression`
    with formula `P1 + P2`, and to `AD · Add` (arith.add). Kept as its own
    node because ADD 2 is the SPIDER-canonical name.
    """
    xa = h(a); xb = h(b)
    return {"image": d((np.asarray(xa) + np.asarray(xb)).astype(np.float32))}


def _fn_addf(a, b, factor_a: float = 1.0, factor_b: float = 1.0):
    """SPIDER ADD F — weighted sum: `P3 = P1 * factor_a + P2 * factor_b`."""
    xa = h(a); xb = h(b)
    out = np.asarray(xa) * float(factor_a) + np.asarray(xb) * float(factor_b)
    return {"image": d(out.astype(np.float32))}


def _fn_addr(a, b, factor_a: float = 1.0, factor_b: float = 1.0,
             eps: float = 1e-12):
    """SPIDER ADD R — ratio-weighted sum:
    ``P3 = P1 * factor_a + (P1 / P2) * factor_b``.

    Guards `P2 == 0` with ``eps`` (SPIDER silently produces inf; we don't).
    """
    xa = np.asarray(h(a), dtype=np.float32)
    xb = np.asarray(h(b), dtype=np.float32)
    safe = np.where(np.abs(xb) > float(eps), xb, np.copysign(float(eps), xb))
    out = xa * float(factor_a) + (xa / safe) * float(factor_b)
    return {"image": d(out.astype(np.float32))}


# --- AR IF / AR SCA (conditional arithmetic + range rescale) ---------------

def _fn_ar_if(image, threshold: float = 0.0, comparison: str = ">",
              then_value: float = 1.0, else_value: float = 0.0):
    """SPIDER AR IF — per-pixel conditional arithmetic.

    A simple thresholding variant: for each pixel, if `comparison` holds
    against `threshold`, output `then_value`, else `else_value`. Also
    supported by the more general `AR · Expression` via
    `where(P1 > threshold, then_value, else_value)`; this node is kept for
    quick binarisation / segmentation.
    """
    a = np.asarray(h(image), dtype=np.float32)
    t = float(threshold)
    cmp_map = {">":  a >  t, "<":  a <  t,
               ">=": a >= t, "<=": a <= t,
               "==": a == t, "!=": a != t}
    if comparison not in cmp_map:
        raise ValueError(f"AR IF: unknown comparison {comparison!r}")
    out = np.where(cmp_map[comparison], float(then_value), float(else_value))
    return {"image": d(out.astype(np.float32))}


def _fn_ar_sca(image, new_min: float = 0.0, new_max: float = 1.0):
    """SPIDER AR SCA — linearly rescale the image so its min → `new_min`
    and its max → `new_max`."""
    a = np.asarray(h(image), dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi > lo:
        out = (a - lo) * (float(new_max) - float(new_min)) / (hi - lo) + float(new_min)
    else:
        out = np.full_like(a, float(new_min))
    return {"image": d(out.astype(np.float32))}


# --- BP 32F — BP 3F with two random half-set reconstructions ---------------

def _fn_bp32f(stack, eulers_csv: str = "0,0,0", symmetry: str = "c1",
              nz: int = 0, ny: int = 0, nx: int = 0, seed: int = 42):
    """SPIDER BP 32F — BP 3F on the full stack plus two random half-set
    reconstructions (for FSC-style resolution assessment).

    Returns three volumes:
      volume     — reconstruction from ALL projections
      half_a     — reconstruction from a random half of the projections
      half_b     — reconstruction from the complementary half
      n_a / n_b  — how many frames went into each half
    """
    s = h(stack)
    n = s.shape[0]
    eulers = _parse_eulers(eulers_csv)
    if len(eulers) != n:
        eulers = (np.tile(eulers[:1], (n, 1)) if len(eulers)
                  else np.zeros((n, 3), np.float32))

    def _bp(frames, angles):
        if len(frames) == 0:
            return np.zeros((max(1, nz or 1), max(1, ny or 1), max(1, nx or 1)),
                            np.float32)
        shape = (int(nz), int(ny), int(nx)) if nz and ny and nx else None
        sym = symmetry.strip().lower()
        f_ = frames
        e_ = angles
        if sym.startswith("c") and sym[1:].isdigit() and int(sym[1:]) > 1:
            k = int(sym[1:])
            all_f = [f_]
            all_e = [e_]
            for j in range(1, k):
                e2 = e_.copy()
                e2[:, 0] = (e2[:, 0] + 360.0 * j / k) % 360.0
                all_f.append(f_)
                all_e.append(e2)
            f_ = np.concatenate(all_f, axis=0)
            e_ = np.concatenate(all_e, axis=0)
        return _recon.bp3f(f_, e_, volume_shape=shape).astype(np.float32)

    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(n)
    half = n // 2
    ia, ib = idx[:half], idx[half:]

    v_full = _bp(np.asarray(s), eulers)
    v_a    = _bp(np.asarray(s)[ia], eulers[ia])
    v_b    = _bp(np.asarray(s)[ib], eulers[ib])

    return {"volume": d(v_full), "half_a": d(v_a), "half_b": d(v_b),
            "n_a": int(half), "n_b": int(n - half)}


# =========================================================================
# VA — volume analysis (3D counterpart of FS).
# =========================================================================


def _fn_va(volume):
    a = h(volume).astype(np.float32)
    return {"min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "sigma": float(a.std()),
            "sum": float(a.sum()),
            "nz": int(a.shape[0]) if a.ndim >= 1 else 0,
            "ny": int(a.shape[1]) if a.ndim >= 2 else 0,
            "nx": int(a.shape[2]) if a.ndim >= 3 else 0}


# =========================================================================
# Stack batch helpers (implements DO-loop equivalents for common ops).
# =========================================================================


def _fn_stack_filter(stack, kind: str = "gauss-lp",
                     fc: float = 0.1, temp: float = 0.02,
                     fp: float = 0.10, fs: float = 0.15,
                     padding: str = "border"):
    """DO loop analog — apply an FQ filter to every frame of the stack."""
    s = h(stack).astype(np.float32)
    if kind.startswith("gauss") or kind.startswith("tophat"):
        kwargs = dict(fc=float(fc), padding=padding)
    elif kind.startswith("fermi"):
        kwargs = dict(fc=float(fc), temp=float(temp), padding=padding)
    else:  # butter / raised-cos
        kwargs = dict(fp=float(fp), fs=float(fs), padding=padding)
    out = np.stack([filter_image(f, kind, **kwargs).astype(np.float32) for f in s])
    return {"stack": d(out)}


def _fn_stack_rotate(stack, angle_deg: float = 0.0, cval: float = 0.0,
                     interpolation: str = "cubic"):
    """Rotate every frame of a stack by the same angle."""
    s = h(stack).astype(np.float32)
    order = _interp_order(interpolation)
    out = np.stack([ndi.rotate(f, float(angle_deg), reshape=False,
                                order=order, mode="constant",
                                cval=float(cval)).astype(np.float32) for f in s])
    return {"stack": d(out)}


def _fn_stack_shift(stack, dy: float = 0.0, dx: float = 0.0,
                    cval: float = 0.0, interpolation: str = "cubic"):
    """Shift every frame of a stack by the same (dy, dx)."""
    s = h(stack).astype(np.float32)
    order = _interp_order(interpolation)
    out = np.stack([ndi.shift(f, (float(dy), float(dx)),
                               order=order, mode="constant",
                               cval=float(cval)).astype(np.float32) for f in s])
    return {"stack": d(out)}


def _fn_stack_apply_align(stack, eulers_csv: str = "",
                          cval: float = 0.0,
                          interpolation: str = "cubic"):
    """Apply per-frame rotation + shift + mirror to a stack.

    Reads AP SHC / AP SR-style output where each line is
    `phi,theta,psi  # dy=<v> dx=<v> mirror=<0|1>`.  For 2D alignment
    only `phi` (in-plane rotation) and the trailing dy/dx/mirror are
    consumed. The stack must have the SAME frame count as the CSV.
    """
    import re as _re
    s = h(stack).astype(np.float32)
    n = int(s.shape[0])
    lines = [ln for ln in (eulers_csv or "").strip().splitlines() if ln.strip()]
    if len(lines) != n:
        raise ValueError(
            f"stack.apply_align: eulers_csv has {len(lines)} rows but "
            f"stack has {n} frames — must match 1:1")
    order = _interp_order(interpolation)
    out_frames: list[np.ndarray] = []
    for i, ln in enumerate(lines):
        body, _, comment = ln.partition("#")
        cols = [c.strip() for c in body.split(",")]
        if not cols or not cols[0]:
            raise ValueError(f"stack.apply_align: row {i} has no phi angle")
        phi = float(cols[0])
        dy = dx = 0.0
        mirror = 0
        # Extract dy=... dx=... mirror=... from the trailing comment.
        for key, tgt in (("dy", "dy"), ("dx", "dx"), ("mirror", "mirror")):
            m = _re.search(rf"{key}\s*=\s*(-?[\d.eE+-]+)", comment)
            if m:
                if tgt == "mirror":
                    mirror = int(float(m.group(1)))
                elif tgt == "dy":
                    dy = float(m.group(1))
                else:
                    dx = float(m.group(1))
        f = s[i]
        if mirror:
            f = np.ascontiguousarray(f[:, ::-1])
        rot = ndi.rotate(f, phi, reshape=False, order=order,
                          mode="constant", cval=float(cval))
        shifted = ndi.shift(rot, (dy, dx), order=order,
                             mode="constant", cval=float(cval))
        out_frames.append(shifted.astype(np.float32))
    return {"stack": d(np.stack(out_frames, axis=0))}


# =========================================================================
# AP SHC — iterative reference-updated multi-image alignment (SPIDER AP SHC).
# =========================================================================


def _fn_ap_shc(stack, reference,
               n_iterations: int = 3,
               first_ring: int = 2, last_ring: int = 15,
               search_range: int = 6,
               check_mirror: str = "yes"):
    """SPIDER AP SHC — batch OR SH with reference updating between iterations.

    For each iteration:
      1. Align every frame to the current reference (OR SH per frame).
      2. Average the aligned frames to produce a new reference.
    Returns the final aligned stack, the updated reference, and per-frame
    alignment parameters encoded as one 'phi,theta,psi' triple per line
    (theta=0 for pure in-plane alignment) so downstream nodes can consume
    them the same way as VO EA output.
    """
    s0 = h(stack).astype(np.float32)
    ref = h(reference).astype(np.float32).copy()

    aligned = s0.copy()
    per_frame: list[tuple[float, float, float, int]] = []

    for _ in range(max(1, int(n_iterations))):
        aligned_frames = []
        params = []
        for frame in s0:
            res = _fn_or_sh(ref, frame,
                            first_ring=first_ring, last_ring=last_ring,
                            search_range=search_range,
                            check_mirror=check_mirror)
            ang = res["angle_deg"]; dy = res["dy"]; dx = res["dx"]; mir = res["mirror"]
            src = frame if mir == 0 else np.ascontiguousarray(frame[:, ::-1])
            rot = ndi.rotate(src, ang, reshape=False, order=3,
                             mode="constant", cval=0.0)
            shifted = ndi.shift(rot, (dy, dx), order=3, mode="constant", cval=0.0)
            aligned_frames.append(shifted.astype(np.float32))
            params.append((ang, dy, dx, mir))
        aligned = np.stack(aligned_frames, axis=0)
        ref = aligned.mean(axis=0).astype(np.float32)
        per_frame = params

    eulers_csv = "\n".join(
        f"{ang:.4f},0.0000,0.0000  # dy={dy:.3f} dx={dx:.3f} mirror={mir}"
        for (ang, dy, dx, mir) in per_frame
    )
    return {"aligned": d(aligned), "reference": d(ref),
            "eulers_csv": eulers_csv,
            "n_frames": int(len(per_frame))}


# =========================================================================
# AP ITER — iterative multi-reference alignment with per-class reference
#           updating. Generalises SPIDER AP SHC (K=1) to K>=1 so the same
#           node drives both 2D class averaging and K-means-like iterative
#           classification without a hand-rolled DO + select + mean script.
# =========================================================================


def _fn_ap_iter(stack, references, n_iterations: int = 5,
                first_ring: int = 2, last_ring: int = 15,
                search_range: int = 6, check_mirror: str = "yes",
                reference_eulers_csv: str = ""):
    """Iterative multi-reference alignment with per-class reference update.

    Per iteration:
      1. Align every particle to the current ``references`` via the
         shared ``_multi_ref_align`` engine (polar-CC shortlist + NCC
         shift refinement, same as AP MR).
      2. Group the aligned particles by their chosen ``ref_index`` and
         replace each reference by the mean of its assigned particles.
         Empty classes keep the old reference (they'd otherwise collapse).
      3. Feed the fresh references into the next iteration.

    K = 1: equivalent to SPIDER AP SHC (single-reference class average
    with reference updating).
    K > 1: K-means-style iterative classification — each reference
    converges to the mean of the particles that pick it.

    ``reference_eulers_csv`` is passed through so the final ``eulers_csv``
    output is BP-ready (phi, theta, psi per particle) when the references
    started from VO EA + pj3q.
    """
    s = h(stack).astype(np.float32)
    if s.ndim == 2:
        s = s[None, ...]
    refs = h(references).astype(np.float32).copy()
    if refs.ndim == 2:
        refs = refs[None, ...]   # degenerate K = 1 case
    if s.ndim != 3 or refs.ndim != 3:
        raise ValueError(
            f"AP ITER: need stack-shaped inputs, got stack {s.shape} refs {refs.shape}"
        )
    K = int(refs.shape[0])

    aligned = s.copy()
    ref_idx = np.zeros(s.shape[0], dtype=np.int32)
    ang = dy = dx = pk = None
    mir = None
    assignments_csv = ""
    eulers_csv_out = ""

    for _ in range(max(1, int(n_iterations))):
        r = _multi_ref_align(s, refs, int(first_ring), int(last_ring),
                             int(search_range), do_shift=True,
                             check_mirror=(check_mirror == "yes"),
                             reference_eulers_csv=reference_eulers_csv)
        aligned = np.asarray(h(r["aligned"]), dtype=np.float32)
        ref_idx = np.asarray(h(r["ref_index"]), dtype=np.int32)
        assignments_csv = r["assignments_csv"]
        eulers_csv_out  = r["eulers_csv"]

        # Per-class mean → new references. Empty classes keep their old
        # content so they can still attract particles next iteration
        # (zeroing them would make NCC with any particle near 0).
        new_refs = refs.copy()
        for k in range(K):
            mask = (ref_idx == k)
            if int(mask.sum()) > 0:
                new_refs[k] = aligned[mask].mean(axis=0)
        refs = new_refs

    return {
        "aligned":         d(aligned.astype(np.float32)),
        "references":      d(refs.astype(np.float32)),
        "assignments_csv": assignments_csv,
        "eulers_csv":      eulers_csv_out,
        "ref_index":       d(ref_idx),
        "n_matched":       int(s.shape[0]),
        "n_classes":       K,
    }


# =========================================================================
# AP MR / AP MQ / AP MH / AP MD — multi-reference alignment (SPIDER).
# =========================================================================


def _align_one_pair(ref: np.ndarray, img: np.ndarray,
                    first_ring: int, last_ring: int,
                    search_range: int,
                    do_shift: bool,
                    check_mirror: bool
                    ) -> tuple[float, float, float, int, float]:
    """Align `img` to `ref` — rotation + optional shift + optional mirror.
    Returns (angle_deg, dy, dx, mirror, peak). Shared engine for the AP MR
    family: AP MR / AP MQ use `do_shift=True`, AP MH / AP MD use False.

    The returned peak is now a sum of per-ring Pearson coefficients
    (``normalize_rings=True`` inside ``rotational_angle``), matching
    SPIDER's AP MR normalised cross-correlation. Without ring-level
    normalisation, high-variance references dominate the score purely
    because their rings carry more total energy — not because they're
    better 3D matches."""
    def _try(im):
        raw_ang, pk = rotational_angle(
            ref, im,
            n_rings=max(1, int(last_ring) - int(first_ring) + 1),
            n_angles=360, sub_pixel=True)
        # Sign fix (see _fn_or_sh): rotational_angle returns the polar-space
        # shift that maps ref → im; scipy.rotate's positive-CCW convention
        # needs the inverse to align im → ref.
        ang = -float(raw_ang)
        if not do_shift:
            return ang, 0.0, 0.0, float(pk)
        rot = ndi.rotate(im, ang, reshape=False, order=3,
                         mode="constant", cval=0.0)
        # Normalised CC: zero-mean, unit-std both operands. Raw `cross_correlate`
        # is just `ifft(F[a] · conj(F[b]))` which scales with ||a||·||b||, so a
        # high-energy reference (e.g. the β-gal side-view which integrates more
        # mass than the top view) dominates the peak height regardless of
        # pattern fit — in multi-reference alignment this collapses every
        # particle onto the single strongest reference. Subtracting the mean
        # and dividing by std turns the peak into NCC, identical in spirit to
        # the per-ring Pearson scoring used on the rotation path above.
        def _zn(a: np.ndarray) -> np.ndarray:
            m = float(a.mean())
            s = float(a.std()) + 1e-8
            return ((a - m) / s).astype(np.float32)
        cc = cross_correlate(_zn(ref), _zn(rot)) / float(ref.size)
        ny, nx = cc.shape
        cy, cx = ny // 2, nx // 2
        sr = max(1, int(search_range))
        y0, y1 = max(0, cy - sr), min(ny, cy + sr + 1)
        x0, x1 = max(0, cx - sr), min(nx, cx + sr + 1)
        window = cc[y0:y1, x0:x1]
        peaks = find_peaks(window, n_peaks=1, sub_pixel=True)
        if not peaks:
            # No local maximum inside the shift search window. Use the
            # window's max as the match score so the returned `pk` stays on
            # the same NCC scale as the normal path — falling back to the
            # raw ring-pearson `pk` from rotational_angle would mix scales
            # across refs and silently break the outer "best peak" sweep in
            # _multi_ref_align.
            return ang, 0.0, 0.0, float(window.max())
        p = peaks[0]
        dy = float(p.iy + p.dy) + y0 - cy
        dx = float(p.ix + p.dx) + x0 - cx
        h_pk = float(p.refined_height if p.refined_height is not None else p.height)
        return ang, dy, dx, h_pk

    ang, dy, dx, pk = _try(img)
    mirror = 0
    if check_mirror:
        ang_m, dy_m, dx_m, pk_m = _try(img[:, ::-1].copy())
        if pk_m > pk:
            ang, dy, dx, pk, mirror = ang_m, dy_m, dx_m, pk_m, 1
    return float(ang), float(dy), float(dx), int(mirror), float(pk)


def _multi_ref_align(stack, references,
                     first_ring: int, last_ring: int, search_range: int,
                     do_shift: bool, check_mirror: bool,
                     reference_eulers_csv: str = ""):
    """Common engine for the AP MR family. For each frame of ``stack``,
    picks the best-matching reference from ``references`` and reports its
    index, in-plane rotation, shift, mirror flag, and CC peak.

    Fast path: precompute each reference's polar transform and 1D FFT
    once, then score all refs against a given particle via one batched
    1D CCC sweep (ndarray broadcasting). Only the top-K refs by polar
    score pay for the full shift/NCC refinement, so the inner loop
    shrinks from N_img · N_ref full pair-alignments to
    N_img · K pair-alignments (~20× fewer when N_ref is in the hundreds).
    """
    from pyspider.align import polar_sample
    s = h(stack).astype(np.float32)
    r = h(references).astype(np.float32)
    if s.ndim == 2:
        s = s[None, ...]
    if r.ndim == 2:
        r = r[None, ...]
    if s.ndim != 3 or r.ndim != 3:
        raise ValueError(
            f"AP MR: need 2D or stack inputs, got stack {s.shape} refs {r.shape}"
        )
    n_img, ny, nx = s.shape
    n_ref = r.shape[0]
    n_rings  = max(1, int(last_ring) - int(first_ring) + 1)
    n_angles = 360
    # Enough top candidates to tolerate ties and near-equivalents without
    # the shift refinement changing the identity-case winner. 8 handles
    # D2-symmetric molecules (4 D2 partners × 2 for mirror); cap to n_ref.
    K = min(n_ref, 8)

    # --- Precompute reference polar transforms + 1D rFFT. One pass.
    # Per-ring z-score (subtract ring mean, divide by ring std) so the
    # summed ring CC is a bounded Pearson score and not dominated by
    # high-energy refs (side views) — otherwise the top-K short-list
    # clusters on brightness, not shape.
    ref_polar = np.stack(
        [polar_sample(r[j], n_rings=n_rings, n_angles=n_angles) for j in range(n_ref)],
        axis=0,
    )  # (n_ref, n_rings, n_angles)
    def _zn_rings(p):
        m = p.mean(axis=-1, keepdims=True)
        sd = p.std(axis=-1, keepdims=True) + 1e-8
        return ((p - m) / sd).astype(np.float32)
    ref_polar_zn = _zn_rings(ref_polar)
    ref_polar_fft_conj = np.conj(np.fft.rfft(ref_polar_zn, axis=-1))

    ref_idx = np.zeros(n_img, np.int32)
    ang     = np.zeros(n_img, np.float32)
    dy      = np.zeros(n_img, np.float32)
    dx      = np.zeros(n_img, np.float32)
    mir     = np.zeros(n_img, np.int32)
    pk      = np.full(n_img, -np.inf, np.float32)

    from backend.progress import report_progress as _rp
    for i in range(n_img):
        im = s[i]
        # --- Polar score every ref against this particle in one shot.
        im_polar = polar_sample(im, n_rings=n_rings, n_angles=n_angles)
        im_polar_fft = np.fft.rfft(_zn_rings(im_polar), axis=-1)
        cc = np.fft.irfft(ref_polar_fft_conj * im_polar_fft[None, :, :],
                           n=n_angles, axis=-1).sum(axis=1)  # (n_ref, n_angles)
        polar_peak = cc.max(axis=-1)  # (n_ref,)
        # Short-list of promising refs.
        topk_norm = np.argsort(polar_peak)[::-1][:K]
        candidates = set(int(j) for j in topk_norm.tolist())
        if check_mirror:
            im_m_polar = polar_sample(im[:, ::-1].copy(),
                                       n_rings=n_rings, n_angles=n_angles)
            im_m_polar_fft = np.fft.rfft(_zn_rings(im_m_polar), axis=-1)
            cc_m = np.fft.irfft(ref_polar_fft_conj * im_m_polar_fft[None, :, :],
                                 n=n_angles, axis=-1).sum(axis=1)
            polar_peak_m = cc_m.max(axis=-1)
            topk_m = np.argsort(polar_peak_m)[::-1][:K]
            candidates.update(int(j) for j in topk_m.tolist())

        # --- Full shift/NCC refinement — only for the short-listed refs.
        for j in candidates:
            a_ij, dy_ij, dx_ij, m_ij, p_ij = _align_one_pair(
                r[j], im, first_ring, last_ring, search_range,
                do_shift, check_mirror)
            if p_ij > pk[i]:
                ref_idx[i] = j
                ang[i], dy[i], dx[i], mir[i], pk[i] = a_ij, dy_ij, dx_ij, m_ij, p_ij
        _rp((i + 1) / max(1, n_img),
            f"frame {i + 1}/{n_img}  best ref = {int(ref_idx[i])}  peak = {pk[i]:.3g}")

    # Build a doc-file style CSV (one row per experimental image).
    lines = []
    for i in range(n_img):
        lines.append(f"{ref_idx[i]},{ang[i]:.4f},{dy[i]:.4f},{dx[i]:.4f},"
                     f"{mir[i]},{pk[i]:.6f}")
    csv = "\n".join(lines)

    # Also build the aligned stack: rotate → shift → optional mirror.
    aligned = np.empty_like(s)
    for i in range(n_img):
        im = s[i][:, ::-1].copy() if mir[i] else s[i]
        rot = ndi.rotate(im, float(ang[i]), reshape=False, order=3,
                         mode="constant", cval=0.0)
        aligned[i] = ndi.shift(rot, (float(dy[i]), float(dx[i])),
                                order=3, mode="constant", cval=0.0)

    # First-frame assignment, exposed as scalars so students don't have to
    # parse the CSV. For single-particle angle-assignment workflows these
    # are what you actually care about; for multi-frame runs they're a quick
    # peek at frame 0 (full detail is still in assignments_csv).
    best_ref_idx = int(ref_idx[0]) if n_img > 0 else 0
    best_peak    = float(pk[0])    if n_img > 0 else 0.0
    best_angle   = float(ang[0])   if n_img > 0 else 0.0
    best_dy      = float(dy[0])    if n_img > 0 else 0.0
    best_dx      = float(dx[0])    if n_img > 0 else 0.0

    # Standardised 3-column (phi, theta, psi) eulers_csv — populated only
    # when the caller provides the per-reference Euler list. Looking up
    # the chosen ref_idx in that list and folding the in-plane rotation
    # into psi yields a per-particle Euler triple that bp / pj3q accept
    # directly, so the whole ap_m → bp pipeline no longer needs a bespoke
    # column-join node.
    ref_eulers = _parse_eulers_list(reference_eulers_csv or "")
    eulers_csv = ""
    if ref_eulers and len(ref_eulers) >= int(ref_idx.max() + 1 if n_img else 0):
        rows3 = []
        for i in range(n_img):
            phi_r, theta_r, psi_r = ref_eulers[int(ref_idx[i])]
            # SPIDER convention: psi of the particle = psi of the reference
            # + in-plane angle returned by AP M (angle to rotate image onto
            # the reference, which is the particle's psi in its own frame).
            psi_total = float(psi_r) + float(ang[i])
            rows3.append(f"{float(phi_r):.4f},{float(theta_r):.4f},{psi_total:.4f}")
        eulers_csv = "\n".join(rows3)

    return {
        "aligned": d(aligned.astype(np.float32)),
        "assignments_csv": csv,
        "eulers_csv": eulers_csv,
        "ref_index": d(ref_idx),
        "n_matched": int(n_img),
        "best_ref_idx": best_ref_idx,
        "best_peak":    best_peak,
        "best_angle_deg": best_angle,
        "best_dy":      best_dy,
        "best_dx":      best_dx,
    }


def _fn_ap_mr(stack, references,
              first_ring: int = 2, last_ring: int = 15,
              search_range: int = 6, check_mirror: str = "yes",
              reference_eulers_csv: str = ""):
    """SPIDER AP MR — multi-reference alignment (rotation + shift)."""
    return _multi_ref_align(stack, references,
                            int(first_ring), int(last_ring), int(search_range),
                            do_shift=True, check_mirror=(check_mirror == "yes"),
                            reference_eulers_csv=reference_eulers_csv)


def _fn_ap_mq(stack, references,
              first_ring: int = 2, last_ring: int = 15,
              search_range: int = 6, check_mirror: str = "yes",
              reference_eulers_csv: str = ""):
    """SPIDER AP MQ — modern replacement for AP MR (same operational goal:
    rot + shift multi-reference matching)."""
    return _multi_ref_align(stack, references,
                            int(first_ring), int(last_ring), int(search_range),
                            do_shift=True, check_mirror=(check_mirror == "yes"),
                            reference_eulers_csv=reference_eulers_csv)


def _fn_ap_mh(stack, references,
              first_ring: int = 2, last_ring: int = 15,
              check_mirror: str = "yes",
              reference_eulers_csv: str = ""):
    """SPIDER AP MH — rotational differences vs a reference series (no shift)."""
    return _multi_ref_align(stack, references,
                            int(first_ring), int(last_ring), 0,
                            do_shift=False, check_mirror=(check_mirror == "yes"),
                            reference_eulers_csv=reference_eulers_csv)


def _fn_ap_md(stack, references,
              first_ring: int = 2, last_ring: int = 15,
              check_mirror: str = "no",
              reference_eulers_csv: str = ""):
    """SPIDER AP MD — rotational differences (mirror-check off by default).
    Same core as AP MH; SPIDER's AP MD historically skipped the mirror test."""
    return _multi_ref_align(stack, references,
                            int(first_ring), int(last_ring), 0,
                            do_shift=False, check_mirror=(check_mirror == "yes"),
                            reference_eulers_csv=reference_eulers_csv)


# --- Reference-free alignment (AP RA / SA / SR) ----------------------------

def _apply_align(im: np.ndarray, ang: float, dy: float, dx: float,
                 mirror: int) -> np.ndarray:
    x = im[:, ::-1].copy() if mirror else im
    r = ndi.rotate(x, float(ang), reshape=False, order=3,
                   mode="constant", cval=0.0)
    return ndi.shift(r, (float(dy), float(dx)), order=3,
                     mode="constant", cval=0.0)


def _ref_free_align(stack, n_iterations: int, do_rotation: bool, do_shift: bool,
                    first_ring: int, last_ring: int, search_range: int,
                    check_mirror: bool):
    """Iterative reference-free alignment: use the current stack mean as the
    reference each pass, align every frame to it, average again, repeat.
    ``do_rotation`` and ``do_shift`` gate which transforms are applied per
    iteration (AP RA = rotation only, AP SA = shift only, AP SR = both)."""
    s = h(stack).astype(np.float32)
    if s.ndim == 2:
        s = s[None, ...]
    if s.ndim != 3:
        raise ValueError(f"AP R*: needs stack input, got shape {s.shape}")
    aligned = s.copy()

    from backend.progress import report_progress as _rp
    total_iters = max(1, int(n_iterations))
    for _it in range(total_iters):
        ref = aligned.mean(axis=0)
        next_stack = np.empty_like(aligned)
        for i in range(aligned.shape[0]):
            if do_rotation and do_shift:
                a, dy, dx, mir, _ = _align_one_pair(
                    ref, aligned[i], first_ring, last_ring, search_range,
                    do_shift=True, check_mirror=check_mirror)
            elif do_rotation:
                a, dy, dx, mir, _ = _align_one_pair(
                    ref, aligned[i], first_ring, last_ring, 0,
                    do_shift=False, check_mirror=check_mirror)
            else:
                # Shift only — cross-correlate directly.
                cc = cross_correlate(ref, aligned[i])
                ny, nx = cc.shape
                cy, cx = ny // 2, nx // 2
                sr = max(1, int(search_range))
                y0, y1 = max(0, cy - sr), min(ny, cy + sr + 1)
                x0, x1 = max(0, cx - sr), min(nx, cx + sr + 1)
                peaks = find_peaks(cc[y0:y1, x0:x1], n_peaks=1, sub_pixel=True)
                if peaks:
                    p = peaks[0]
                    dy = float(p.iy + p.dy) + y0 - cy
                    dx = float(p.ix + p.dx) + x0 - cx
                else:
                    dy, dx = 0.0, 0.0
                a, mir = 0.0, 0
            next_stack[i] = _apply_align(aligned[i], a, dy, dx, mir)
        aligned = next_stack
        _rp((_it + 1) / total_iters,
            f"pass {_it + 1}/{total_iters}  frames = {aligned.shape[0]}")

    ref_final = aligned.mean(axis=0)
    return {"aligned": d(aligned.astype(np.float32)),
            "reference": d(ref_final.astype(np.float32)),
            "n_iterations": int(n_iterations)}


def _fn_ap_ra(stack, n_iterations: int = 5, first_ring: int = 2,
              last_ring: int = 15, check_mirror: str = "yes"):
    """SPIDER AP RA — reference-free rotational alignment."""
    return _ref_free_align(stack, int(n_iterations), True, False,
                           int(first_ring), int(last_ring), 0,
                           check_mirror == "yes")


def _fn_ap_sa(stack, n_iterations: int = 5, search_range: int = 6):
    """SPIDER AP SA — reference-free shift alignment."""
    return _ref_free_align(stack, int(n_iterations), False, True,
                           2, 15, int(search_range), False)


def _fn_ap_sr(stack, n_iterations: int = 5,
              first_ring: int = 2, last_ring: int = 15,
              search_range: int = 6, check_mirror: str = "yes"):
    """SPIDER AP SR — reference-free combined shift + rotational alignment."""
    return _ref_free_align(stack, int(n_iterations), True, True,
                           int(first_ring), int(last_ring), int(search_range),
                           check_mirror == "yes")


# --- Align + classify simultaneously (AP C / AP CA / AP CM) ----------------

def _align_and_classify(stack, k: int, n_iterations: int,
                        check_mirror: bool,
                        first_ring: int, last_ring: int,
                        search_range: int, do_shift: bool,
                        seed: int):
    """K-means-in-image-space + rotational (or full) alignment. Each
    iteration: (1) align every frame to the closest class centre with the
    given transform search; (2) reassign classes to whichever centre gives
    the best CC peak; (3) recompute centres as the average of their
    aligned members. Convergence hint via `n_iterations`."""
    s = h(stack).astype(np.float32)
    if s.ndim == 2:
        s = s[None, ...]
    n = s.shape[0]
    k = max(1, min(int(k), n))
    rng = np.random.default_rng(int(seed))
    seed_indices = rng.choice(n, size=k, replace=False)
    centres = s[seed_indices].copy()

    assigned = np.zeros(n, dtype=np.int32)
    per_frame = np.zeros((n, 5), dtype=np.float32)  # ang, dy, dx, mirror, peak

    from backend.progress import report_progress as _rp
    total_iters = max(1, int(n_iterations))
    for _it in range(total_iters):
        # Assign + align.
        for i in range(n):
            best_pk = -np.inf
            for j in range(k):
                a, dy, dx, mir, pk = _align_one_pair(
                    centres[j], s[i], first_ring, last_ring, search_range,
                    do_shift=do_shift, check_mirror=check_mirror)
                if pk > best_pk:
                    best_pk = pk
                    assigned[i] = j
                    per_frame[i] = (a, dy, dx, mir, pk)
        _rp((_it + 1) / total_iters,
            f"pass {_it + 1}/{total_iters}  k={k}  n={n}")
        # Recompute centres from aligned members.
        new_centres = np.zeros_like(centres)
        counts = np.zeros(k, dtype=np.int32)
        for i in range(n):
            j = int(assigned[i])
            a, dy, dx, mir, _ = per_frame[i]
            new_centres[j] += _apply_align(s[i], a, dy, dx, int(mir))
            counts[j] += 1
        for j in range(k):
            if counts[j] > 0:
                new_centres[j] /= counts[j]
            else:
                # Empty class → reseed from a random frame to avoid dead classes.
                new_centres[j] = s[rng.integers(0, n)]
        centres = new_centres

    # Build aligned stack (transformed to its class centre).
    aligned = np.empty_like(s)
    for i in range(n):
        a, dy, dx, mir, _ = per_frame[i]
        aligned[i] = _apply_align(s[i], a, dy, dx, int(mir))

    csv_rows = [
        f"{int(assigned[i])},{per_frame[i, 0]:.4f},"
        f"{per_frame[i, 1]:.4f},{per_frame[i, 2]:.4f},"
        f"{int(per_frame[i, 3])},{per_frame[i, 4]:.6f}"
        for i in range(n)
    ]

    return {
        "class_averages": d(centres.astype(np.float32)),
        "aligned":        d(aligned.astype(np.float32)),
        "assignments_csv": "\n".join(csv_rows),
        "n_classes":      int(k),
    }


def _fn_ap_c(stack, k: int = 4, n_iterations: int = 5,
             check_mirror: str = "yes",
             first_ring: int = 2, last_ring: int = 15):
    """SPIDER AP C — rotational alignment + classification into K groups."""
    return _align_and_classify(stack, int(k), int(n_iterations),
                               check_mirror == "yes",
                               int(first_ring), int(last_ring), 0,
                               do_shift=False, seed=42)


def _fn_ap_ca(stack, k: int = 4, n_iterations: int = 5,
              first_ring: int = 2, last_ring: int = 15):
    """SPIDER AP CA — rotational alignment (no mirror check) + K-means
    classification into K groups."""
    return _align_and_classify(stack, int(k), int(n_iterations),
                               check_mirror=False,
                               first_ring=int(first_ring),
                               last_ring=int(last_ring),
                               search_range=0, do_shift=False, seed=42)


def _fn_ap_cm(stack, k: int = 4, n_iterations: int = 5,
              first_ring: int = 2, last_ring: int = 15,
              search_range: int = 6):
    """SPIDER AP CM — rotational + shift alignment with mirror check +
    K-means classification (full alignment+clustering combo)."""
    return _align_and_classify(stack, int(k), int(n_iterations),
                               check_mirror=True,
                               first_ring=int(first_ring),
                               last_ring=int(last_ring),
                               search_range=int(search_range),
                               do_shift=True, seed=42)


# =========================================================================
# BP CG — conjugate-gradient 3D reconstruction (SPIDER BP CG).
# =========================================================================

from scipy.sparse.linalg import cg as _cg, LinearOperator as _LinOp


def _fn_bp_cg(stack, eulers_csv: str = "0,0,0",
              iterations: int = 20,
              regularization: float = 0.01,
              radius: int = 0):
    """SPIDER BP CG — conjugate-gradient reconstruction with Tikhonov reg.

    Solves for a volume V that minimises
        ||P V − Y||² + λ ||V||²
    where P is the block of projection operators (one per Euler triple),
    Y is the observed projection stack, and λ = regularization.
    Uses scipy's cg on the normal equations.

    ``radius`` (pixels) optionally masks the reconstruction to a sphere;
    0 disables the mask. Only supports cubic volumes (ny == nx).
    """
    s = h(stack).astype(np.float32)
    n, ny, nx = s.shape
    if ny != nx:
        raise ValueError(f"BP CG requires square projections, got {ny}x{nx}")

    rows = [row for row in eulers_csv.strip().splitlines() if row.strip()]
    eulers = np.array(
        [[float(v) for v in row.split("#", 1)[0].split(",")[:3]] for row in rows],
        dtype=np.float32,
    )
    if len(eulers) != n:
        eulers = (np.tile(eulers[:1], (n, 1)) if len(eulers)
                  else np.zeros((n, 3), np.float32))

    nz = ny
    vshape = (nz, ny, nx)
    N = nz * ny * nx

    def A(v: np.ndarray) -> np.ndarray:
        vol = v.reshape(vshape).astype(np.float32)
        out = np.empty((n, ny, nx), np.float32)
        for i, e in enumerate(eulers):
            out[i] = _recon.project(vol, float(e[0]), float(e[1]), float(e[2]))
        return out.ravel()

    def AT(y: np.ndarray) -> np.ndarray:
        projs = y.reshape(n, ny, nx).astype(np.float32)
        return _recon.bp3f(projs, eulers, volume_shape=vshape).ravel()

    lam = float(regularization)

    def AtA(v: np.ndarray) -> np.ndarray:
        return AT(A(v)) + lam * v

    op = _LinOp((N, N), matvec=AtA, dtype=np.float32)
    b = AT(s.ravel())

    from backend.progress import report_progress as _rp
    max_it = int(iterations)
    cg_state = {"i": 0}
    def _cb(_xk):
        cg_state["i"] += 1
        _rp(min(1.0, cg_state["i"] / max(1, max_it)),
            f"CG iter {cg_state['i']}/{max_it}")
    x, info = _cg(op, b, maxiter=max_it, rtol=1e-4, callback=_cb)
    vol = x.reshape(vshape).astype(np.float32)

    if int(radius) > 0:
        r = int(radius)
        z, y_, x_ = np.mgrid[:nz, :ny, :nx].astype(np.float32)
        cz, cy, cx = (nz - 1) / 2, (ny - 1) / 2, (nx - 1) / 2
        rr = np.sqrt((z - cz) ** 2 + (y_ - cy) ** 2 + (x_ - cx) ** 2)
        vol = np.where(rr <= r, vol, 0.0).astype(np.float32)

    return {"volume": d(vol), "info": int(info), "iters": int(iterations)}


# =========================================================================
# CTF (SPIDER TF C / TF CT / apply / Wiener).
# =========================================================================

_CTF_PARAMS = [
    ("voltage_kV",   "float", 300.0,  None,  80.0,   1000.0),
    ("cs_mm",        "float", 2.7,    None,   0.0,     10.0),
    ("defocus_A",    "float", 10000.0,None,   0.0, 100000.0),
    ("amp_contrast", "float", 0.1,    None,   0.0,      1.0),
    ("pixel_A",      "float", 1.0,    None,   0.1,     10.0),
    ("astigmatism_A","float", 0.0,    None, -20000.0, 20000.0),
    ("azimuth_deg",  "float", 0.0,    None, -180.0,   180.0),
    ("b_factor",     "float", 0.0,    None,   0.0,    500.0),
]

def _build_ctf(shape, **kw):
    """rfft2 half-plane CTF (ny, nx // 2 + 1). Used internally by apply /
    phase_flip / wiener where the CTF is consumed by rfft2 + multiply + irfft2
    and never exposed to the user."""
    return _ctf.ctf_2d(shape=shape,
                       pixel_size=float(kw["pixel_A"]),
                       defocus=float(kw["defocus_A"]),
                       voltage_kV=float(kw["voltage_kV"]),
                       cs_mm=float(kw["cs_mm"]),
                       amplitude_contrast=float(kw["amp_contrast"]),
                       astigmatism=float(kw.get("astigmatism_A", 0.0)),
                       azimuth_deg=float(kw.get("azimuth_deg", 0.0)),
                       b_factor=float(kw.get("b_factor", 0.0)))


def _build_ctf_full(shape, **kw):
    """Build the 2D CTF on the full (ny, nx) plane with DC pre-shifted to
    (ny // 2, nx // 2) — the fftshift-centred convention used by FT / IFT.

    A CTF built this way has the same shape and centring as an FT'd image,
    so it can be wired directly into a MU (`arith.mul`) alongside the FT
    output, and the product fed straight into IFT — no manual `fftshift`
    dance and no rfft2 half-plane cropping.
    """
    ny, nx = int(shape[0]), int(shape[1])
    pixel = float(kw["pixel_A"])
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=pixel)).astype(np.float32)[None, :]
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=pixel)).astype(np.float32)[:, None]

    voltage_kV   = float(kw["voltage_kV"])
    defocus_A    = float(kw["defocus_A"])
    cs_mm        = float(kw["cs_mm"])
    amp_contrast = float(kw["amp_contrast"])
    astigmatism  = float(kw.get("astigmatism_A", 0.0))
    azimuth_deg  = float(kw.get("azimuth_deg", 0.0))
    b_factor     = float(kw.get("b_factor", 0.0))

    lambda_A = _ctf.electron_wavelength(voltage_kV)
    cs_A = cs_mm * 1.0e7
    W = _ctf._wgh_phase(amp_contrast)

    if astigmatism == 0.0:
        k2 = kx * kx + ky * ky
        gamma = _ctf._gamma(k2, defocus_A, lambda_A, cs_A)
    else:
        # Mirror ctf_2d's astigmatism formula (SPIDER trafc.f convention).
        azr = np.deg2rad(azimuth_deg)
        az = np.arctan2(ky, np.where(kx == 0, 1e-30, kx)) + 0.5 * np.pi
        dz_eff = defocus_A + 0.5 * astigmatism * np.sin(2.0 * (az - azr))
        k2 = kx * kx + ky * ky
        gamma = (0.5 * np.pi * cs_A * lambda_A ** 3 * k2 * k2
                 - np.pi * dz_eff * lambda_A * k2)

    H = np.sin(gamma - W).astype(np.float32)
    if b_factor != 0.0:
        env = np.exp(-b_factor * (kx * kx + ky * ky) / 4.0).astype(np.float32)
        H = H * env
    return H


def _fn_ctf_model(nx: int = 256, ny: int = 256, **kw):
    H = _build_ctf_full((int(ny), int(nx)), **kw)
    return {"ctf": d(np.asarray(H, np.float32))}

def _fn_apply_ctf(image, **kw):
    a = h(image).astype(np.float32)
    # Build the CTF from the last two axes so a 3D stack (N, ny, nx) applies
    # the same CTF to every frame via broadcasting.
    H = _build_ctf(a.shape[-2:], **kw)
    return {"image": d(_ctf.apply_ctf(a, H).astype(np.float32))}

def _tile_avg_power(a: np.ndarray, tile: int, overlap: float
                    ) -> tuple[np.ndarray, int]:
    """Average |FFT(tile)|² over overlapping Hann-windowed tiles.

    Standard CTFFIND-style pre-processing — averaging N tiles reduces the
    variance of the power spectrum by ~N, making Thon rings emerge from
    micrograph shot noise. Returns (power2D, effective_tile_size).
    """
    ny, nx = a.shape
    T = int(min(tile, ny, nx))
    if T < 32:
        F = np.fft.fftshift(np.fft.fft2(a - a.mean()))
        return (np.abs(F) ** 2).astype(np.float32), min(ny, nx)
    step = max(1, int(T * (1.0 - max(0.0, min(0.95, overlap)))))
    # Hann 2D window
    w1 = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(T) / (T - 1))
    W = np.outer(w1, w1).astype(np.float32)
    acc = np.zeros((T, T), dtype=np.float64)
    n = 0
    for y in range(0, ny - T + 1, step):
        for x in range(0, nx - T + 1, step):
            t = a[y:y + T, x:x + T].astype(np.float32)
            t = (t - t.mean()) * W
            F = np.fft.fftshift(np.fft.fft2(t))
            acc += np.abs(F) ** 2
            n += 1
    if n == 0:
        F = np.fft.fftshift(np.fft.fft2(a - a.mean()))
        return (np.abs(F) ** 2).astype(np.float32), min(ny, nx)
    return (acc / n).astype(np.float32), T


def _fn_ctffind(image,
                voltage_kV: float = 300.0,
                cs_mm: float = 2.7,
                amp_contrast: float = 0.1,
                pixel_A: float = 1.0,
                defocus_min_A: float = 5000.0,
                defocus_max_A: float = 50000.0,
                defocus_step_A: float = 500.0,
                ring_min_frac: float = 0.05,
                ring_max_frac: float = 0.4,
                fit_astigmatism: str = "no",
                tile_size: int = 512,
                tile_overlap: float = 0.5):
    """CTFFIND-style defocus estimation.

    Given a micrograph (or any image whose power spectrum has visible Thon
    rings), search a defocus range and pick the value whose theoretical
    CTF² best matches the observed radial power spectrum. Optionally
    refines an astigmatism (Δf_u, Δf_v, angle) via a coarse 2D grid.

    Returns the fitted parameters plus a theoretical CTF² image and the
    observed / theoretical 1D profiles overlaid as a PNG-friendly array.
    """
    from scipy.ndimage import gaussian_filter1d

    a = h(image).astype(np.float32)
    if a.ndim == 3:
        a = a[a.shape[0] // 2]
    if a.ndim != 2:
        raise ValueError(f"CTFFIND needs a 2D image; got {a.shape}")

    # Average |FFT|² over overlapping Hann-windowed tiles (CTFFIND-style
    # pre-processing). Small tile = smoother rings but coarser freq bins.
    # tile_size = 0 falls back to one whole-image FFT.
    T_target = int(tile_size) if tile_size and tile_size > 0 else max(a.shape)
    P_lin, N = _tile_avg_power(a, T_target, float(tile_overlap))
    P = np.log1p(P_lin).astype(np.float32)
    ny_p, nx_p = P.shape  # power-spectrum shape (may be smaller than input)

    # Radial pixel distance from the DC pixel of the (tile-sized) spectrum.
    y = np.arange(ny_p) - ny_p // 2
    x = np.arange(nx_p) - nx_p // 2
    Y, X = np.meshgrid(y, x, indexing="ij")
    r_pix = np.hypot(Y, X)

    # 1D rotational average of the observed power.
    r_int = r_pix.astype(np.int32)
    max_r = int(r_int.max()) + 1
    prof_obs = np.bincount(r_int.ravel(), weights=P.ravel(),
                           minlength=max_r).astype(np.float64)
    counts = np.bincount(r_int.ravel(), minlength=max_r).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    prof_obs /= counts

    # Frequency (Å^-1) for each radial bin: bin k corresponds to
    # spatial frequency k / (N * pixel_A).
    k_1d = np.arange(max_r, dtype=np.float32) / (N * float(pixel_A))
    k_nyq = 0.5 / float(pixel_A)
    mask = (k_1d >= ring_min_frac * k_nyq) & (k_1d <= ring_max_frac * k_nyq)
    if mask.sum() < 8:
        raise ValueError("fit ring is too narrow — widen ring_min/max_frac")
    k_fit = k_1d[mask].astype(np.float32)
    obs = prof_obs[mask].astype(np.float32)

    # Detrend the observation with a slowly-varying baseline so we're fitting
    # the oscillation, not the DC background.
    sig = max(3.0, len(obs) / 30.0)
    obs_sig = obs - gaussian_filter1d(obs, sigma=sig)
    obs_norm = obs_sig / (np.linalg.norm(obs_sig) + 1e-9)

    # CTF² formula (isotropic case).
    lam = 12.2643 / np.sqrt(1000.0 * voltage_kV
                            * (1.0 + 0.9785e-6 * 1000.0 * voltage_kV))
    Cs_A = float(cs_mm) * 1e7
    ac = float(amp_contrast)
    w1 = float(np.sqrt(max(0.0, 1.0 - ac * ac)))

    def ctf2_1d(defocus_A: float) -> np.ndarray:
        gamma = 2.0 * np.pi * (0.5 * lam * k_fit ** 2 * defocus_A
                               - 0.25 * Cs_A * lam ** 3 * k_fit ** 4)
        ctf = -w1 * np.sin(gamma) - ac * np.cos(gamma)
        return (ctf * ctf).astype(np.float32)

    # 1D defocus search — high-density (step_A defocus).
    dfs = np.arange(float(defocus_min_A),
                    float(defocus_max_A) + 1e-6,
                    float(defocus_step_A), dtype=np.float64)
    from backend.progress import report_progress as _rp
    best_score = -np.inf
    best_df = float(defocus_min_A)
    for k_i, df in enumerate(dfs):
        th = ctf2_1d(df)
        th_sig = th - gaussian_filter1d(th, sigma=sig)
        th_norm = th_sig / (np.linalg.norm(th_sig) + 1e-9)
        score = float(np.dot(obs_norm, th_norm))
        if score > best_score:
            best_score = score
            best_df = float(df)
        if (k_i + 1) % max(1, len(dfs) // 20) == 0 or k_i + 1 == len(dfs):
            _rp((k_i + 1) / len(dfs),
                f"defocus scan {k_i + 1}/{len(dfs)}  best = {best_df:.0f} A  "
                f"score = {best_score:.3f}")

    defocus_u = best_df
    defocus_v = best_df
    azimuth_deg = 0.0

    # Optional 2D astigmatism refinement (coarse grid around best_df).
    if fit_astigmatism == "yes":
        # Observed detrended 2D power in the fit annulus.
        R = np.hypot(Y, X).astype(np.float32) / (N * float(pixel_A))
        annulus = ((R >= ring_min_frac * k_nyq) &
                   (R <= ring_max_frac * k_nyq))
        P_bg = gaussian_filter1d(prof_obs, sigma=sig)  # radial baseline
        P_bg_2d = P_bg[np.clip(r_int, 0, max_r - 1)]
        obs2 = (P - P_bg_2d).astype(np.float32)
        obs2 *= annulus
        obs2_norm = obs2 / (np.linalg.norm(obs2) + 1e-9)

        # Azimuth of each pixel in the plot's coordinates.
        phi = np.arctan2(Y, X).astype(np.float32)

        best2 = best_score
        span = float(defocus_step_A) * 6
        for du in np.arange(best_df - span, best_df + span + 1, defocus_step_A):
            for dv in np.arange(best_df - span, best_df + span + 1, defocus_step_A):
                if dv > du:
                    continue
                for az in np.arange(0, 180, 15, dtype=np.float32):
                    az_r = np.deg2rad(az)
                    df_theta = (0.5 * (du + dv)
                                + 0.5 * (du - dv) * np.cos(2.0 * (phi - az_r)))
                    gamma = 2.0 * np.pi * (0.5 * lam * R ** 2 * df_theta
                                           - 0.25 * Cs_A * lam ** 3 * R ** 4)
                    ctf = -w1 * np.sin(gamma) - ac * np.cos(gamma)
                    ctf2 = (ctf * ctf) * annulus
                    ctf2_bg = gaussian_filter1d(
                        np.bincount(r_int.ravel(), weights=ctf2.ravel(),
                                    minlength=max_r) / counts,
                        sigma=sig)
                    ctf2_bg_2d = ctf2_bg[np.clip(r_int, 0, max_r - 1)]
                    th2 = (ctf2 - ctf2_bg_2d) * annulus
                    th2_norm = th2 / (np.linalg.norm(th2) + 1e-9)
                    s = float((obs2_norm * th2_norm).sum())
                    if s > best2:
                        best2 = s
                        defocus_u, defocus_v, azimuth_deg = float(du), float(dv), float(az)
        best_score = best2

    # Synthetic CTF² image at best fit (for visual comparison).
    R = np.hypot(Y, X).astype(np.float32) / (N * float(pixel_A))
    phi = np.arctan2(Y, X).astype(np.float32)
    az_r = np.deg2rad(azimuth_deg)
    df_theta = (0.5 * (defocus_u + defocus_v)
                + 0.5 * (defocus_u - defocus_v) * np.cos(2.0 * (phi - az_r)))
    gamma = 2.0 * np.pi * (0.5 * lam * R ** 2 * df_theta
                           - 0.25 * Cs_A * lam ** 3 * R ** 4)
    ctf2_full = (-w1 * np.sin(gamma) - ac * np.cos(gamma)) ** 2

    return {
        "defocus_u_A":     float(defocus_u),
        "defocus_v_A":     float(defocus_v),
        "azimuth_deg":     float(azimuth_deg),
        "score":           float(best_score),
        "theoretical_ctf2": d(ctf2_full.astype(np.float32)),
    }


def _fn_phase_flip(image, **kw):
    a = h(image).astype(np.float32)
    H = _build_ctf(a.shape[-2:], **kw)
    return {"image": d(_ctf.phase_flip(a, H).astype(np.float32))}

def _fn_wiener(image, snr: float = 1.0, **kw):
    a = h(image).astype(np.float32)
    H = _build_ctf(a.shape[-2:], **kw)
    return {"image": d(_ctf.wiener_correct(a, H, snr=float(snr)).astype(np.float32))}


# =========================================================================
# Reconstruction / Projection.
# =========================================================================

def _fn_ml_angle(stack, reference, candidate_angles_csv: str = "0,0,0",
                 sigma: float = 1.0, do_shifts: bool = True,
                 search_range: int = 6, mask_radius: float = 0.0,
                 search_rotation: bool = True,
                 first_ring: int = 2, last_ring: int = 15,
                 reference_mode: str = "volume"):
    """Maximum-likelihood angle estimation per experimental image.

    For every image y_i in `stack` and every candidate Euler triple
    θ_j in `candidate_angles_csv` (typically VO EA output), we compute
    the Gaussian log-likelihood

        log p(y_i | V, θ_j)  ∝  −0.5 · ||y_i − S(P_{θ_j} V)||² / σ²

    where P_{θ} V is the projection of the reference `volume` at that
    Euler triple (Fourier central-slice, fast) and S(·) is the optimal
    in-plane translation (via cross-correlation, restricted to
    ±`search_range` pixels; skipped when `do_shifts=False`).

    Log-likelihoods across candidate angles are turned into a posterior
    p_j by softmax; per image we emit:
      map_idx        — argmax posterior (candidate-angle index)
      phi/theta/psi  — the MAP angle triple
      dy, dx         — the best in-plane shift at the MAP angle
      log_lik_max    — the raw (unnormalised) log-likelihood at MAP
      entropy        — H(p) = −Σ p_j log p_j, in nats.
                       0 = concentrated posterior (confident);
                       log(K) = uniform (indeterminate).

    ``sigma`` scales the noise assumption. Too small → the posterior
    collapses to argmax (behaves like AP MR). Too large → posterior is
    flat and entropy saturates at log(K); a good starting point is the
    std of a background patch of the image. ``mask_radius`` (in pixels)
    optionally restricts the residual to a central disk — helpful when
    the background contributes significant noise.

    To compare against real-space projection matching with CCC on the
    same inputs: wire the same volume + candidate_angles through
    `Project volume` (batch) → `Multi-reference align` (AP MR), then
    diff the `ref_index` column of AP MR's `assignments_csv` against
    `map_idx` here.
    """
    from backend.progress import report_progress as _rp

    s = h(stack).astype(np.float32)
    if s.ndim == 2:
        s = s[None, ...]
    if s.ndim != 3:
        raise ValueError(f"ML angle: stack must be 2D or 3D, got {s.shape}")

    triples = _parse_eulers_list(candidate_angles_csv)

    # `reference` is either a 3D density volume (mode=volume → project
    # internally at each candidate angle) or a stack of pre-computed
    # projections (mode=stack → use them directly, saving one 3D-FFT).
    ref_host = np.asarray(h(reference), dtype=np.float32)
    if ref_host.ndim != 3:
        raise ValueError(f"ML angle: reference must be 3D, got {ref_host.shape}")

    mode = str(reference_mode).lower()
    if mode == "volume":
        # Volume mode requires explicit candidate angles — otherwise we have
        # nothing to project at.
        if not triples:
            raise ValueError(
                "ML angle (volume mode): candidate_angles_csv is empty. "
                "Wire VO EA's eulers_csv in, or switch reference_mode to "
                "'stack' if you want to supply pre-computed projections."
            )
        vol_host = ref_host
        eulers = np.asarray(triples, dtype=np.float32)
        _rp(0.05, f"projecting reference at {len(triples)} candidate angles")
        refs = _project_batch_fourier(vol_host, eulers)
    elif mode == "stack":
        # Pre-computed reference stack — angle labels are optional. If
        # candidate_angles_csv is missing we fill (phi, theta, psi) with
        # zeros and only report `best_idx` meaningfully; wire it in to get
        # the full phi/theta/psi back out.
        vol_host = None
        refs = ref_host
        K = refs.shape[0]
        if triples:
            if len(triples) != K:
                raise ValueError(
                    f"ML angle (stack mode): reference stack has {K} frames "
                    f"but candidate_angles_csv has {len(triples)} rows — "
                    "they must match index-wise."
                )
            eulers = np.asarray(triples, dtype=np.float32)
        else:
            eulers = np.zeros((K, 3), dtype=np.float32)
        _rp(0.05, f"using supplied reference stack ({refs.shape[0]} angles)")
    else:
        raise ValueError(
            f"reference_mode must be 'volume' or 'stack', got {reference_mode!r}"
        )

    n, ny, nx = s.shape
    K = refs.shape[0]

    # Optional soft mask restricting the residual to a central disk.
    mask = None
    if mask_radius > 0.0:
        yy, xx = np.mgrid[:ny, :nx].astype(np.float32)
        r = np.hypot(yy - (ny - 1) / 2.0, xx - (nx - 1) / 2.0)
        mask = (r <= float(mask_radius)).astype(np.float32)

    sig2 = 2.0 * max(float(sigma), 1e-6) ** 2

    map_idx  = np.zeros(n, np.int32)
    map_dy   = np.zeros(n, np.float32)
    map_dx   = np.zeros(n, np.float32)
    map_psi  = np.zeros(n, np.float32)  # additional in-plane rotation (if searched)
    map_ll   = np.full(n, -np.inf, np.float32)
    entropy  = np.zeros(n, np.float32)

    sr = max(0, int(search_range))
    do_shift = bool(do_shifts) and sr > 0
    do_rot = bool(search_rotation)

    for i in range(n):
        y = s[i]
        log_lik = np.empty(K, dtype=np.float64)
        best_dy = np.zeros(K, dtype=np.float32)
        best_dx = np.zeros(K, dtype=np.float32)
        # Precompute psi_fit for every candidate so we can batch the
        # re-projection in one Fourier pass (184 slice extractions, 1 FFT)
        # instead of per-candidate (184 FFTs).
        psi_fits = np.zeros(K, dtype=np.float32)
        if do_rot:
            n_rings = max(1, int(last_ring) - int(first_ring) + 1)
            for j in range(K):
                psi_fits[j], _ = rotational_angle(refs[j], y,
                                                   n_rings=n_rings,
                                                   n_angles=360,
                                                   sub_pixel=True)
        # Build per-candidate refs at the fitted psi.
        if vol_host is not None:
            # Volume mode: batch re-project via Fourier central-slice —
            # exact, no scipy.rotate loss.
            fitted_eulers = np.stack(
                [eulers[:, 0], eulers[:, 1], psi_fits], axis=1
            ).astype(np.float32)
            refs_at_psi = _project_batch_fourier(vol_host, fitted_eulers)
        else:
            # Stack mode: no volume to re-project from, so rotate the
            # pre-computed references in-plane via scipy. This is the lossy
            # path (particularly near 180°); the student is responsible for
            # sampling psi densely enough in their reference stack.
            refs_at_psi = np.empty_like(refs)
            for j in range(K):
                refs_at_psi[j] = ndi.rotate(
                    refs[j], float(psi_fits[j]), reshape=False, order=3,
                    mode="constant", cval=0.0,
                ).astype(np.float32)
        best_psi = np.zeros(K, dtype=np.float32)
        for j in range(K):
            # Reference already includes the fitted psi (batched above).
            p_at_psi = refs_at_psi[j]
            psi_extra = float(psi_fits[j])
            # 2. Optional translation search between (fixed) particle and the
            # re-projected reference. We only translate — never rotate — the
            # particle, so the comparison picture stays faithful.
            if do_shift:
                cc = cross_correlate(p_at_psi, y)
                cy, cx = cc.shape[0] // 2, cc.shape[1] // 2
                y0, y1 = max(0, cy - sr), min(cc.shape[0], cy + sr + 1)
                x0, x1 = max(0, cx - sr), min(cc.shape[1], cx + sr + 1)
                peaks = find_peaks(cc[y0:y1, x0:x1], n_peaks=1, sub_pixel=True)
                if peaks:
                    pk = peaks[0]
                    dy_ = float(pk.iy + pk.dy) + y0 - cy
                    dx_ = float(pk.ix + pk.dx) + x0 - cx
                    y_aligned = ndi.shift(y, (-dy_, -dx_), order=1,
                                           mode="constant", cval=0.0)
                else:
                    dy_ = dx_ = 0.0
                    y_aligned = y
            else:
                dy_ = dx_ = 0.0
                y_aligned = y
            resid = y_aligned - p_at_psi
            if mask is not None:
                resid = resid * mask
            sq = float((resid * resid).sum())
            # Gaussian log-likelihood (dropping constant Z(σ)):
            #   log p = −||y − P||² / (2σ²)
            # sig2 above = 2σ², so we divide `sq` by `sig2`.
            log_lik[j] = -sq / sig2
            best_dy[j] = dy_
            best_dx[j] = dx_
            best_psi[j] = float(psi_extra)

        # Softmax to get a numerically stable posterior.
        m = float(log_lik.max())
        p_post = np.exp(log_lik - m)
        p_post /= p_post.sum()
        j_map = int(np.argmax(p_post))
        map_idx[i] = j_map
        map_dy[i]  = best_dy[j_map]
        map_dx[i]  = best_dx[j_map]
        map_psi[i] = best_psi[j_map]
        map_ll[i]  = float(log_lik[j_map])
        # Entropy in nats; guard against log(0) and floating-point negatives.
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -float((p_post * np.where(p_post > 0, np.log(p_post), 0.0)).sum())
        entropy[i] = max(0.0, ent)

        if (i + 1) % max(1, n // 30) == 0 or (i + 1) == n:
            _rp(0.05 + 0.9 * (i + 1) / n,
                f"image {i + 1}/{n}  MAP idx {j_map}  entropy {ent:.3f} "
                f"(uniform = log(K) = {float(np.log(K)):.3f})")

    lines = []
    for i in range(n):
        phi, theta, psi_base = eulers[map_idx[i]]
        # Report the effective psi (candidate value + in-plane refinement).
        psi_total = float(psi_base) + float(map_psi[i])
        lines.append(
            f"{int(map_idx[i])},{float(phi):.4f},{float(theta):.4f},"
            f"{psi_total:.4f},{float(map_dy[i]):.4f},{float(map_dx[i]):.4f},"
            f"{float(map_ll[i]):.4f},{float(entropy[i]):.4f}"
        )
    csv = "\n".join(lines)

    # Build aligned stack for visual comparison — rotate + shift the
    # experimental image so it looks like the MAP reference. Apply the
    # rotation-search refinement first (matches the residual computation).
    aligned = np.empty_like(s)
    for i in range(n):
        y_rot = ndi.rotate(s[i], -float(map_psi[i]), reshape=False,
                            order=3, mode="constant", cval=0.0) if do_rot else s[i]
        aligned[i] = ndi.shift(
            y_rot, (-float(map_dy[i]), -float(map_dx[i])), order=1,
            mode="constant", cval=0.0,
        ) if do_shift else y_rot

    # Frame-0 assignment as scalars so students don't have to parse the CSV.
    # These are wireable into recon.pj3q (phi/theta/psi/eulers_csv) directly.
    if n > 0:
        _i0 = int(map_idx[0])
        _phi0, _theta0, _psi_base0 = eulers[_i0]
        _psi_total0 = float(_psi_base0) + float(map_psi[0])
        best_idx = _i0
        best_phi = float(_phi0)
        best_theta = float(_theta0)
        best_psi = float(_psi_total0)
        best_dy = float(map_dy[0])
        best_dx = float(map_dx[0])
        best_ll = float(map_ll[0])
    else:
        best_idx = 0
        best_phi = best_theta = best_psi = 0.0
        best_dy = best_dx = 0.0
        best_ll = 0.0

    return {
        "assignments_csv": csv,
        "aligned": d(aligned.astype(np.float32)),
        "n_frames": int(n),
        "n_candidates": int(K),
        "best_idx": best_idx,
        "best_phi": best_phi,
        "best_theta": best_theta,
        "best_psi": best_psi,
        "best_dy": best_dy,
        "best_dx": best_dx,
        "best_log_lik": best_ll,
    }


def _fn_pj3q(volume, batch: bool = True, eulers_csv: str = "0,0,0",
             phi: float = 0.0, theta: float = 0.0, psi: float = 0.0,
             method: str = "fourier"):
    """Merged PJ 3Q — batch (many angles from CSV) or single (phi/theta/psi).

    Always returns a ``stack``. In single-angle mode it is a length-1 stack.
    The frontend edge-classifier treats stack → image as a fine connection
    (engine auto-unwraps to the middle frame, which for length-1 is that
    frame), so downstream 2D nodes wire in without warnings.

    method (batch only): 'fourier' (central-slice, fast) or 'real' (per-frame
    rotate-and-sum, slower but simpler). Single-angle mode always uses real.
    """
    if batch:
        return _fn_project_batch(volume, eulers_csv, method=method)
    img = h(_fn_project(volume, phi=phi, theta=theta, psi=psi)["image"]).astype(np.float32)
    return {"stack": d(img[np.newaxis, ...])}


def _fn_project(volume, phi: float = 0.0, theta: float = 0.0, psi: float = 0.0):
    p = _recon.project(h(volume), float(phi), float(theta), float(psi))
    return {"image": d(p.astype(np.float32))}

# --- BP 3D / BP 3E — real-space back-projection variants -------------------

def _fn_bp3d(stack, eulers_csv: str = "0,0,0", symmetry: str = "c1",
             nz: int = 0, ny: int = 0, nx: int = 0):
    """SPIDER BP 3D — direct (unfiltered) real-space back-projection.

    Same voxel-smearing kernel as BP 3F but without the ramp filter, so the
    output has the classical low-pass bias of raw back-projection. Useful
    when you plan to apply your own reconstruction filter downstream (e.g.
    Wiener) or for teaching / debugging.
    """
    return _bp_variant(stack, eulers_csv, symmetry, nz, ny, nx,
                       ramp_filter=False)


def _fn_bp3e(stack, eulers_csv: str = "0,0,0", symmetry: str = "c1",
             existing_volume=None,
             nz: int = 0, ny: int = 0, nx: int = 0):
    """SPIDER BP 3E — weighted back-projection with optional accumulation.

    Same as BP 3D, but if an ``existing_volume`` is wired in the newly
    computed contribution is added to it (SPIDER's 'add on to existing
    volume' semantics for building an aggregate reconstruction).
    """
    v = _bp_variant(stack, eulers_csv, symmetry, nz, ny, nx,
                    ramp_filter=False, return_ndarray=True)
    if existing_volume is not None:
        base = h(existing_volume).astype(np.float32)
        if base.shape == v.shape:
            v = v + base
    return {"volume": d(v.astype(np.float32))}


def _bp_variant(stack, eulers_csv, symmetry, nz, ny, nx,
                ramp_filter: bool, return_ndarray: bool = False):
    """Shared setup for the real-space BP family (BP 3D / BP 3E / BP 3F).
    Applies Cn symmetry via projection duplication (matching BP 3F)."""
    s = h(stack)
    n = s.shape[0]
    rows = [row for row in eulers_csv.strip().splitlines() if row.strip()]
    eulers = np.array([[float(v) for v in row.split(",")[:3]] for row in rows],
                      dtype=np.float32)
    if len(eulers) != n:
        eulers = (np.tile(eulers[:1], (n, 1)) if len(eulers)
                  else np.zeros((n, 3), np.float32))
    sym = symmetry.strip().lower()
    if sym.startswith("c") and sym[1:].isdigit() and int(sym[1:]) > 1:
        k = int(sym[1:])
        all_s, all_e = [s], [eulers]
        for i in range(1, k):
            e2 = eulers.copy()
            e2[:, 0] = (e2[:, 0] + 360.0 * i / k) % 360.0
            all_s.append(s); all_e.append(e2)
        s = np.concatenate(all_s, axis=0)
        eulers = np.concatenate(all_e, axis=0)
    shape = (int(nz), int(ny), int(nx)) if nz and ny and nx else None
    v = _recon.bp3f(s, eulers, volume_shape=shape,
                    ramp_filter=ramp_filter).astype(np.float32)
    if return_ndarray:
        return v
    return {"volume": d(v)}


# --- BP 3N / BP 3G — Fourier-domain reconstruction -------------------------

def _bp_fourier(stack, eulers_csv: str, symmetry: str, nz: int, ny: int, nx: int,
                gridding: bool):
    """Fourier-domain 3D reconstruction — insert each projection's FT onto
    the corresponding central slice of the 3D FT, then invert.

    ``gridding=False`` — nearest-neighbour insertion (SPIDER BP 3N).
    ``gridding=True``  — Kaiser-Bessel gridding with tri-linear splat, the
    standard non-uniform-FFT variant (SPIDER BP 3G).
    """
    s = h(stack).astype(np.float32)
    n, ay, ax = s.shape
    if ax != ay:
        raise ValueError(f"BP 3{'G' if gridding else 'N'}: needs square projections; got {ay}x{ax}")

    rows = [r for r in eulers_csv.strip().splitlines() if r.strip()]
    eulers = np.array([[float(v) for v in r.split(",")[:3]] for r in rows],
                      dtype=np.float32)
    if len(eulers) != n:
        eulers = (np.tile(eulers[:1], (n, 1)) if len(eulers)
                  else np.zeros((n, 3), np.float32))
    sym = symmetry.strip().lower()
    if sym.startswith("c") and sym[1:].isdigit() and int(sym[1:]) > 1:
        k = int(sym[1:])
        all_s, all_e = [s], [eulers]
        for i in range(1, k):
            e2 = eulers.copy()
            e2[:, 0] = (e2[:, 0] + 360.0 * i / k) % 360.0
            all_s.append(s); all_e.append(e2)
        s = np.concatenate(all_s, axis=0)
        eulers = np.concatenate(all_e, axis=0)

    N = int(nx) if nx else ax
    n_all = s.shape[0]

    # Central-slice theorem: FT of each projection = central slice of the
    # 3D FT taken perpendicular to the projection direction. Insert every
    # projection into a padded 3D Fourier volume at the appropriate slice.
    F = np.zeros((N, N, N), dtype=np.complex64)
    W = np.zeros((N, N, N), dtype=np.float32)  # accumulation weights

    # Frequency labels must match the SHIFTED FFT convention used below —
    # we `fftshift(fft2(.))` each projection, so F2[i, j] carries the
    # frequency (i - ay//2, j - ax//2). Using plain `fftfreq(ax)*ax`
    # (unshifted ordering) was the historical bug: F2 values landed in
    # the wrong central-slice positions and the reconstruction came out
    # as noise. See _fn_bp3n / _fn_bp3g regression test.
    ky = np.arange(ax, dtype=np.float32) - (ax // 2)
    kx = np.arange(ax, dtype=np.float32) - (ax // 2)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    ones_slice = np.ones_like(KY, dtype=np.float32)

    for i in range(n_all):
        # ifftshift the projection before fft2 so the IMAGE centre
        # (pixel (ay//2, ax//2)) is treated as the physical origin — the
        # same convention we use for the 3D volume. Without this step the
        # inserted slices carry a residual (k0+k1) phase ramp, so when
        # several views overlap they add destructively and the inverse
        # FFT returns noise, not the reconstruction.
        F2 = np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(s[i]))
        ).astype(np.complex64)
        R = _recon._euler_zyz(*eulers[i]).astype(np.float32)
        # 3D coordinates for each slice pixel: (0, ky, kx) → R (0, ky, kx)^T
        # then shift to centre of the 3D grid. The target volume F is
        # also fftshifted (we invert with ifftshift → ifftn), so DC
        # lives at index N//2 along each axis.
        pts = np.stack([np.zeros_like(KY), KY, KX], axis=-1)  # (ay, ax, 3)
        rot_pts = pts @ R.T  # apply R to each 3-vector
        cz = cy_ = cx_ = N // 2
        gz = rot_pts[..., 0] + cz
        gy = rot_pts[..., 1] + cy_
        gx = rot_pts[..., 2] + cx_

        if not gridding:
            iz = np.round(gz).astype(np.int32)
            iy = np.round(gy).astype(np.int32)
            ix = np.round(gx).astype(np.int32)
            valid = (iz >= 0) & (iz < N) & (iy >= 0) & (iy < N) & (ix >= 0) & (ix < N)
            np.add.at(F, (iz[valid], iy[valid], ix[valid]), F2[valid])
            np.add.at(W, (iz[valid], iy[valid], ix[valid]), ones_slice[valid])
        else:
            # Trilinear splat with a Kaiser-Bessel-like (β = 2π * 1.5) taper.
            iz0 = np.floor(gz).astype(np.int32); iz1 = iz0 + 1
            iy0 = np.floor(gy).astype(np.int32); iy1 = iy0 + 1
            ix0 = np.floor(gx).astype(np.int32); ix1 = ix0 + 1
            fz = gz - iz0; fy = gy - iy0; fx = gx - ix0
            for dz, wz in ((0, 1 - fz), (1, fz)):
                for dy, wy in ((0, 1 - fy), (1, fy)):
                    for dx_, wx_ in ((0, 1 - fx), (1, fx)):
                        zi = iz0 + dz; yi = iy0 + dy; xi = ix0 + dx_
                        w = (wz * wy * wx_).astype(np.float32)
                        m = ((zi >= 0) & (zi < N) &
                             (yi >= 0) & (yi < N) &
                             (xi >= 0) & (xi < N))
                        if not np.any(m):
                            continue
                        np.add.at(F, (zi[m], yi[m], xi[m]), F2[m] * w[m])
                        np.add.at(W, (zi[m], yi[m], xi[m]), w[m])

    # Density-compensate and invert. `fftshift` on the output re-centres
    # the reconstruction so the object sits at voxel (N//2, N//2, N//2),
    # matching the convention of the real-space BP family.
    F = np.where(W > 0, F / np.maximum(W, 1e-8), F)
    vol = np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(F)).real
    ).astype(np.float32)
    if gridding:
        # De-apodize: trilinear splat in Fourier is a convolution with the
        # tent kernel, which in real space multiplies the reconstruction by
        # sinc²(x/N) · sinc²(y/N) · sinc²(z/N). Without compensation the
        # particle's periphery fades out (low-pass bias that grows with
        # radius). Divide by the envelope to recover a flat reconstruction.
        x = (np.arange(N, dtype=np.float32) - (N // 2)) / float(N)
        # np.sinc is normalised sinc, i.e. sin(πx)/(πx); floor at a tiny
        # positive to avoid dividing by near-zero at the far edges.
        env1 = np.maximum(np.sinc(x) ** 2, 1e-6)
        env = (env1[:, None, None] * env1[None, :, None]
               * env1[None, None, :]).astype(np.float32)
        vol = (vol / env).astype(np.float32)
    return vol


# ---------------------------------------------------------------------------
# 2D Radon-transform versions of random_projections and BP.
#
# These are the n-dimension analogues of their 3D counterparts, used by
# `data.sim.projections` / `recon.bp` when `mode='2d'`. In 2D the "stack
# of projections" is a set of 1D arrays; we store them as (N, 1, W) so
# the data stays shaped like a stack (and still flows through nodes
# whose port kind is `stack`). The in-plane rotation angle plays the
# role of the full ZYZ Euler triple.
# ---------------------------------------------------------------------------


def _sim_random_projections_2d(image, n: int, seed: int,
                               shift_sigma: float, noise_sigma: float):
    """Build `n` 1D projections of a 2D image at random angles.

    Returns a tuple ``(stack, angles)`` where
        stack  : (n, 1, W) float32  — each frame is one 1D line projection
        angles : (n,)      float32  — the projection direction in degrees
                                       (angle from +x axis, counter-clockwise)
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"image must be 2D, got {img.shape}")
    H, W = img.shape
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    rng = np.random.default_rng(int(seed))
    angles = rng.uniform(0.0, 360.0, size=n).astype(np.float32)
    rng_shift = np.random.default_rng(int(seed) + 1)
    rng_noise = np.random.default_rng(int(seed) + 2)
    out = np.empty((n, 1, W), dtype=np.float32)
    for i in range(n):
        # Rotate so the projection direction aligns with the +y axis of
        # the rotated frame — then integrate along y gives us the 1D
        # projection parameterised by the perpendicular coordinate (x).
        rotated = ndi.rotate(img, float(angles[i]), reshape=False, order=1,
                             mode="constant", cval=0.0)
        proj = rotated.sum(axis=0)                              # (W,)
        if shift_sigma > 0:
            dx = float(rng_shift.standard_normal()) * float(shift_sigma)
            proj = ndi.shift(proj, dx, order=1,
                             mode="constant", cval=0.0)
        if noise_sigma > 0:
            proj = proj + float(noise_sigma) * rng_noise.standard_normal(W).astype(np.float32)
        out[i, 0] = proj.astype(np.float32)
    return out, angles


def _bp_2d_real(stack_1d, angles_deg, size: int, ramp_filter: bool):
    """Real-space 2D back-projection with optional |k| ramp filter.

    stack_1d : (N, W)        each row is a 1D projection
    angles   : (N,)          projection angle (deg, counter-clockwise from +x)
    size     : int           output image side (N × N → size × size)
    ramp_filter : bool       filtered back-projection when True
    """
    stack_1d = np.asarray(stack_1d, dtype=np.float32)
    n, W = stack_1d.shape
    N = int(size) if size else W
    # Pre-filter: ramp filter in 1D Fourier space (standard FBP).
    if ramp_filter:
        k = np.fft.rfftfreq(W)
        kmag = np.abs(k).astype(np.float32)
        filtered = np.empty_like(stack_1d)
        for i in range(n):
            spec = np.fft.rfft(stack_1d[i]) * kmag
            filtered[i] = np.fft.irfft(spec, n=W).astype(np.float32)
        stack_1d = filtered

    out = np.zeros((N, N), dtype=np.float32)
    for i in range(n):
        # Smear the 1D line along the integration axis (y of the
        # rotated frame) to a 2D "slab" of width W, height N.
        slab = np.broadcast_to(stack_1d[i][None, :], (N, W)).astype(np.float32)
        # Pad / crop to the target square (N, N) first — the projection
        # width W may differ from the reconstruction side N.
        if W != N:
            # Centre the slab horizontally.
            if W < N:
                pad = (N - W) // 2
                tmp = np.zeros((N, N), dtype=np.float32)
                tmp[:, pad:pad + W] = slab
                slab = tmp
            else:
                crop = (W - N) // 2
                slab = slab[:, crop:crop + N]
        # Un-rotate by the projection angle so the slab lands at the
        # correct orientation in the output frame.
        rotated = ndi.rotate(slab, -float(angles_deg[i]),
                             reshape=False, order=1,
                             mode="constant", cval=0.0)
        out += rotated
    return (out / float(n)).astype(np.float32)


def _bp_2d_fourier(stack_1d, angles_deg, size: int, gridding: bool):
    """2D Fourier-insert reconstruction from 1D projections.

    Each 1D FT is inserted into the 2D FT along the line perpendicular
    to its projection direction (central-slice theorem in 2D), then the
    accumulation is density-compensated and inverted.
    """
    stack_1d = np.asarray(stack_1d, dtype=np.float32)
    n, W = stack_1d.shape
    N = int(size) if size else W
    F = np.zeros((N, N), dtype=np.complex64)
    Wt = np.zeros((N, N), dtype=np.float32)
    # Shifted 1D frequency labels (match the fftshift convention below).
    k_perp = np.arange(W, dtype=np.float32) - (W // 2)
    cy = cx = N // 2
    for i in range(n):
        # `fft(ifftshift(.))` → image centre as the physical origin, so
        # the inserted spectra don't carry a residual phase ramp.
        F1 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(stack_1d[i])))
        theta = np.deg2rad(float(angles_deg[i]))
        # `_sim_random_projections_2d` applies ndi.rotate(img, θ) and sums
        # over axis 0, which — tracing scipy's CCW convention in array
        # coords — projects along (dy, dx) = (cos θ, -sin θ) in the
        # original image frame. The 2D-FT central line is therefore the
        # PERPENDICULAR direction, (sin θ, cos θ) in (ky, kx). Inserting
        # k_perp samples along (-sin θ, cos θ) (the earlier form) swaps
        # ky ↔ kx, which inverted the reconstruction upside-down.
        dy = np.sin(theta)
        dx = np.cos(theta)
        gx = k_perp * dx + cx
        gy = k_perp * dy + cy
        if not gridding:
            ix = np.round(gx).astype(np.int32)
            iy = np.round(gy).astype(np.int32)
            m = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N)
            np.add.at(F, (iy[m], ix[m]), F1[m].astype(np.complex64))
            np.add.at(Wt, (iy[m], ix[m]), np.ones(np.count_nonzero(m), dtype=np.float32))
        else:
            ix0 = np.floor(gx).astype(np.int32); ix1 = ix0 + 1
            iy0 = np.floor(gy).astype(np.int32); iy1 = iy0 + 1
            fx = gx - ix0; fy = gy - iy0
            for dj, wj in ((0, 1 - fy), (1, fy)):
                for di, wi in ((0, 1 - fx), (1, fx)):
                    yi = iy0 + dj; xi = ix0 + di
                    w = (wj * wi).astype(np.float32)
                    m = ((xi >= 0) & (xi < N) & (yi >= 0) & (yi < N))
                    if not np.any(m):
                        continue
                    np.add.at(F, (yi[m], xi[m]),
                              (F1[m] * w[m]).astype(np.complex64))
                    np.add.at(Wt, (yi[m], xi[m]), w[m])
    F = np.where(Wt > 0, F / np.maximum(Wt, 1e-8), F)
    img = np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(F)).real
    ).astype(np.float32)
    return img


def _fn_bp3n(stack, eulers_csv: str = "0,0,0", symmetry: str = "c1",
             nz: int = 0, ny: int = 0, nx: int = 0):
    """SPIDER BP 3N — Fourier-domain 3D reconstruction with **nearest-
    neighbour** central-slice insertion.

    Very fast; low angular sampling → visible interpolation striations.
    Use BP 3G for smoother results at the cost of some speed.
    """
    v = _bp_fourier(stack, eulers_csv, symmetry, nz, ny, nx, gridding=False)
    return {"volume": d(v)}


def _fn_bp3g(stack, eulers_csv: str = "0,0,0", symmetry: str = "c1",
             nz: int = 0, ny: int = 0, nx: int = 0):
    """SPIDER BP 3G — Fourier-domain reconstruction using **gridding**
    (trilinear splat with a KB-like taper).

    Higher-fidelity than BP 3N when the angular sampling is sparse — the
    tri-linear weights spread each projection over a small Fourier
    neighbourhood, reducing the aliasing striations of nearest-neighbour
    insertion.
    """
    v = _bp_fourier(stack, eulers_csv, symmetry, nz, ny, nx, gridding=True)
    return {"volume": d(v)}


# --- BP MEM 2 — Maximum Entropy 2D reconstruction --------------------------

def _fn_bpmem2(stack, eulers_csv: str = "0,0,0",
               iterations: int = 30, prior_strength: float = 1.0,
               ny: int = 0, nx: int = 0):
    """SPIDER BP MEM 2 — 2D reconstruction from projections via a simple
    multiplicative Maximum-Entropy scheme (Cornwell-Evans-style).

    We reconstruct a 2D image whose forward projections match the input
    stack, biased toward a flat prior. Each iteration:

        residual = observed - forward_project(current)
        current *= exp( (learning_rate / prior_strength) * back_project(residual) )

    Assumes the projection direction encoded by each ``psi`` value in the
    eulers_csv (SPIDER 2D case: only the in-plane angle matters). Output is
    a 2D image (SPIDER BP MEM 2 = 2D counterpart of the 3D MEM variants).
    """
    s = h(stack).astype(np.float32)
    if s.ndim != 3:
        raise ValueError(f"BP MEM 2: needs a stack of 1D projections in 2D imagery form; got {s.shape}")
    n, ay, ax = s.shape
    if ax != ay:
        raise ValueError("BP MEM 2: projections must be square")
    N = int(nx) if nx else ax

    rows = [r for r in eulers_csv.strip().splitlines() if r.strip()]
    psi = []
    for r in rows:
        parts = r.split(",")
        if len(parts) >= 3:
            psi.append(float(parts[2]))
        elif parts:
            psi.append(float(parts[0]))
    if len(psi) != n:
        psi = (psi * ((n // max(1, len(psi))) + 1))[:n] if psi else [0.0] * n
    psi = np.asarray(psi, dtype=np.float32)

    # Use the middle row of each 2D projection as its 1D shadow (typical
    # SPIDER convention for the 2D MEM variant, since 'BP MEM 2' works on
    # 1D projections stacked in a 2D file).
    shadows = s[:, ay // 2, :].astype(np.float32)  # (n, N)

    # Current estimate — start from uniform positive image.
    current = np.full((N, N), 1.0 / N, dtype=np.float32)
    yy, xx = np.mgrid[:N, :N].astype(np.float32)
    cy, cx = (N - 1) / 2, (N - 1) / 2

    for _it in range(int(iterations)):
        # Forward-project current at each psi (rotate then sum along y).
        residual_bp = np.zeros_like(current)
        for i in range(n):
            rot = ndi.rotate(current, -float(psi[i]), reshape=False, order=1,
                             mode="constant", cval=0.0)
            fp = rot.sum(axis=0)  # (N,)
            # Interpolate observed shadow to the same 1D grid if sizes differ.
            if fp.shape[0] != shadows[i].shape[0]:
                obs = np.interp(
                    np.linspace(0, 1, fp.shape[0]),
                    np.linspace(0, 1, shadows[i].shape[0]),
                    shadows[i],
                )
            else:
                obs = shadows[i]
            resid = obs - fp
            # Smear the 1D residual back along y, then rotate back.
            resid_2d = np.broadcast_to(resid[None, :], (N, N)).astype(np.float32)
            back = ndi.rotate(resid_2d, float(psi[i]), reshape=False, order=1,
                              mode="constant", cval=0.0)
            residual_bp += back / n

        lam = 0.05 / max(float(prior_strength), 1e-3)
        # Multiplicative update — MEM prior of a flat image.
        current = current * np.exp(lam * residual_bp)
        current = np.clip(current, 1e-10, None)

    return {"image": d(current.astype(np.float32))}


def _fn_bp3f(stack, eulers_csv: str = "0,0,0", symmetry: str = "c1",
             nz: int = 0, ny: int = 0, nx: int = 0):
    """SPIDER BP 3F with optional Cn cyclic symmetry around Z.

    ``symmetry`` accepts "c1" (none), "c2", "c3", ... .  For Cn the input
    projection is used at each of n rotations 360/n apart (phi + 360k/n);
    this is the simplest symmetrisation matching SPIDER's symmetries-doc
    convention when all entries are pure phi rotations.
    """
    s = h(stack)
    n = s.shape[0]
    # Use the shared parser so ml_angle's 8-column format and the 3-column
    # VO EA / sim.projections / ap_m.eulers_csv format all feed through
    # without a column-extracting intermediate.
    triples = _parse_eulers_list(eulers_csv)
    eulers = (np.asarray(triples, dtype=np.float32) if triples
              else np.zeros((0, 3), np.float32))
    if len(eulers) != n:
        eulers = np.tile(eulers[:1], (n, 1)) if len(eulers) else np.zeros((n, 3), np.float32)

    sym = symmetry.strip().lower()
    if sym.startswith("c") and sym[1:].isdigit() and int(sym[1:]) > 1:
        k = int(sym[1:])
        extra_stacks = [s]
        extra_eulers = [eulers]
        for i in range(1, k):
            e2 = eulers.copy()
            e2[:, 0] = (e2[:, 0] + 360.0 * i / k) % 360.0
            extra_stacks.append(s)
            extra_eulers.append(e2)
        s = np.concatenate(extra_stacks, axis=0)
        eulers = np.concatenate(extra_eulers, axis=0)

    shape = (int(nz), int(ny), int(nx)) if nz and ny and nx else None
    v = _recon.bp3f(s, eulers, volume_shape=shape)
    return {"volume": d(v.astype(np.float32))}


# ---- SGD-based ab-initio 3D (not in SPIDER; modern cryo-SPARC style) ------

def _parse_eulers(eulers_csv: str) -> np.ndarray:
    rows = [r for r in eulers_csv.strip().splitlines()
            if r.strip() and not r.strip().startswith("#")]
    if not rows:
        return np.zeros((0, 3), np.float32)
    out = []
    for r in rows:
        parts = r.split("#", 1)[0].split(",")
        if len(parts) < 3:
            continue
        try:
            out.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except ValueError:
            continue
    return np.asarray(out, dtype=np.float32)


def _random_eulers(n: int, seed: int) -> np.ndarray:
    """Uniform-on-sphere Euler triples (phi, theta, psi)."""
    rng = np.random.default_rng(int(seed))
    phi = rng.uniform(0.0, 360.0, n)
    theta = np.degrees(np.arccos(rng.uniform(-1.0, 1.0, n)))
    psi = rng.uniform(0.0, 360.0, n)
    return np.stack([phi, theta, psi], axis=1).astype(np.float32)


def _fn_sgd_init(stack, eulers_csv: str = "",
                 n_iters: int = 30, batch_size: int = 10, lr: float = 0.05,
                 seed: int = 0):
    """Ab-initio 3D initial model via mini-batch SGD.

    Not a SPIDER command — this is a modern cryo-SPARC/RELION-style ab-initio
    routine. Each iteration: pick a random subset of particles, forward-project
    the current volume at their (assumed) angles, compute residuals, and
    back-project those residuals to form the gradient. Update V by -lr * grad.

    Inputs:
      stack       — N experimental projection images (ny × nx each).
      eulers_csv  — one Euler triple per line matching N. If empty, uniform
                    random angles on the sphere are used (only useful as a
                    stress test — true ab-initio would need pose search too).

    Outputs:
      volume        — the fitted 3D density (nz = ny).
      loss_history  — comma-separated per-iteration mean squared residual.
      final_loss    — scalar, last iteration's loss.
    """
    projs = h(stack).astype(np.float32)
    if projs.ndim != 3:
        raise ValueError(f"stack must be 3D (N,ny,nx); got shape {projs.shape}")
    N, ny, nx = projs.shape
    nz = ny
    if N == 0:
        return {"volume": d(np.zeros((nz, ny, nx), np.float32)),
                "loss_history": "", "final_loss": 0.0}

    eulers = _parse_eulers(eulers_csv) if eulers_csv.strip() else _random_eulers(N, seed)
    if len(eulers) < N:
        eulers = np.vstack([eulers, _random_eulers(N - len(eulers), seed + 1)])
    eulers = eulers[:N]

    rng_v = np.random.default_rng(int(seed) + 999)
    V = rng_v.normal(0.0, 0.01, (nz, ny, nx)).astype(np.float32)

    rng_b = np.random.default_rng(int(seed) + 42)
    bs = max(1, min(int(batch_size), N))
    losses: list[float] = []

    from backend.progress import report_progress as _rp
    total = max(1, int(n_iters))
    for i in range(total):
        idx = rng_b.choice(N, size=bs, replace=False)
        preds = np.stack([_recon.project(V, *eulers[i2]) for i2 in idx], axis=0)
        residuals = (preds - projs[idx]).astype(np.float32)
        loss = float((residuals * residuals).mean())
        losses.append(loss)
        # For SGD, the gradient of ||P(V) - y||² wrt V is P^T(residual) — i.e.
        # the *unfiltered* back-projection (voxel-smear sum). `bp3f` with the
        # ramp filter is FBP (an approximate inverse), which as a "gradient"
        # kills convergence. Use `ramp_filter=False` for the true adjoint.
        # Also: bp3f already divides by the number of projections inside, so
        # a further `/ bs` here would shrink the effective learning rate to
        # `lr / bs²` — do NOT re-divide.
        grad = _recon.bp3f(residuals, eulers[idx], volume_shape=V.shape,
                           ramp_filter=False)
        V -= float(lr) * grad.astype(np.float32)
        _rp((i + 1) / total, f"iter {i + 1}/{total}  loss = {loss:.6g}")

    return {
        "volume": d(V.astype(np.float32)),
        "loss_history": ",".join(f"{l:.4g}" for l in losses),
        "final_loss": float(losses[-1]) if losses else 0.0,
    }


# =========================================================================
# Classification (SPIDER CA S / CL KM).
# =========================================================================

def _fn_pca(stack, n_components: int = 8, method: str = "pca",
            additive_constant: float = 0.0):
    """SPIDER CA S — factor analysis of an image stack.

    method ∈ {"pca","coran","iterative","skip"}.  For CoRAN (SPIDER's
    "correspondence analysis") each pixel is shifted by additive_constant
    to make it strictly positive, then the χ²-metric decomposition of the
    row/column-normalized table is done via SVD — same result the SPIDER
    CoRAN prescription gives on non-negative data.  "iterative" and "skip"
    are aliases that fall through to plain PCA here.
    """
    s = h(stack).astype(np.float32)
    feat = _classify.flatten_stack(s)
    m = method.strip().lower()
    if m == "coran":
        feat = feat + float(additive_constant)
        row_m = feat.mean(axis=1, keepdims=True)
        col_m = feat.mean(axis=0, keepdims=True)
        total = feat.mean()
        # χ²-centering: (X - r c / n) / sqrt(r c / n)
        exp = row_m * col_m / max(total, 1e-30)
        std = np.sqrt(np.maximum(exp, 1e-30))
        feat = (feat - exp) / std
    res = _classify.pca(feat, int(n_components))
    ny, nx = s.shape[1], s.shape[2]
    comp0 = res.components[0].reshape(ny, nx)
    return {"scores": d(res.scores.astype(np.float32)),
            "eigenimage": d(comp0.astype(np.float32))}

def _fn_kmeans(scores, n_clusters: int = 4, n_init: int = 10,
               max_iter: int = 200, seed: int = 0):
    s = h(scores).astype(np.float32)
    res = _classify.kmeans(s, int(n_clusters),
                           n_init=int(n_init), max_iter=int(max_iter),
                           seed=int(seed))
    return {"labels": d(res.labels.astype(np.int32))}

def _fn_class_avg(stack, labels, n_classes: int = 4):
    s = h(stack).astype(np.float32)
    lb = h(labels).astype(np.int64)
    avgs = _classify.class_averages(s, lb, n_classes=int(n_classes))
    return {"stack": d(avgs.astype(np.float32))}


# =========================================================================
# Stack helpers.
# =========================================================================

def _fn_stack_average(stack):
    s = h(stack).astype(np.float32)
    return {"image": d(s.mean(axis=0))}

def _fn_stack_variance(stack):
    s = h(stack).astype(np.float32)
    return {"image": d(s.var(axis=0))}

def _fn_stack_index(stack, index: int = 0):
    s = h(stack).astype(np.float32)
    i = int(index) % s.shape[0]
    return {"image": d(s[i])}


# =========================================================================
# Alignment (existing pyspider).
# =========================================================================

def _fn_math_expr(formula: str = "P1", n_inputs: int = 1, **kwargs):
    """Evaluate a NumPy-vectorised expression on wired inputs P1…Pn.

    The `formula` string is evaluated with each wired `Pi` bound to its
    incoming value (scalar or array) and with a small set of common math
    functions in scope: sqr, sqrt, log, log2, log10, exp, sin, cos, tan,
    asin, acos, atan, atan2, abs, floor, ceil, round, sum, mean, std,
    min (element-wise), max (element-wise), pi, e, np (full numpy).

    Everything is plain NumPy: shapes broadcast automatically, so
    `P1 * 4` works for a scalar OR an image OR a stack. `P1 + P2` works
    when the two inputs are broadcast-compatible; a scalar Pi combines
    naturally with an image Pi.

    A leading `P=` (with optional spaces) in the formula is tolerated
    so the user's shorthand ('P= P1 * 4') behaves the same as 'P1 * 4'.
    """
    import re as _re
    expr = (formula or "").strip()
    # Tolerate a leading 'P =' or 'P=' — user writes 'P = P1 * 4'.
    m = _re.match(r"^\s*p\s*=\s*(.*)$", expr, _re.IGNORECASE)
    if m:
        expr = m.group(1).strip()
    if not expr:
        raise ValueError("math.expr: formula is empty")
    n = max(1, min(8, int(n_inputs)))
    ns: dict[str, Any] = {}
    for i in range(1, n + 1):
        k = f"P{i}"
        v = kwargs.get(k)
        if v is None:
            raise ValueError(
                f"math.expr: input {k} is not wired (n_inputs={n} but {k} "
                f"has no incoming edge)")
        # Convert arrays to numpy on the numpy backend so downstream math
        # broadcasts cleanly; leave Python scalars alone.
        if hasattr(v, "ndim"):
            ns[k] = np.asarray(h(v))
        else:
            ns[k] = v
    ns.update({
        "np": np, "numpy": np,
        "sqr": lambda x: np.square(x),
        "sqrt": np.sqrt, "log": np.log, "log2": np.log2, "log10": np.log10,
        "exp": np.exp, "abs": np.abs,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
        "atan2": np.arctan2,
        "floor": np.floor, "ceil": np.ceil, "round": np.round,
        "min": np.minimum, "max": np.maximum,
        "sum": np.sum, "mean": np.mean, "std": np.std,
        "pi": float(np.pi), "e": float(np.e),
    })
    try:
        result = eval(expr, {"__builtins__": {}}, ns)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"math.expr: failed to evaluate {expr!r}: {exc}") from exc
    # Single unified output — `result` port. Downstream image-kind
    # inputs accept both arrays and 0-d scalars (0-d numpy behaves like
    # a Python scalar). classifyEdge permits image → scalar so wiring
    # the result into a scalar param slot (RT SQ's angle_deg, etc.)
    # stays green.
    if isinstance(result, (int, float, np.integer, np.floating)):
        return {"result": np.array(float(result), dtype=np.float32)}
    arr = np.asarray(result, dtype=np.float32)
    if arr.ndim == 0:
        return {"result": arr}
    return {"result": d(arr)}


def _fn_polar_angle(polar_map, dx: float = 0.0, sign: str = "invert"):
    """Convert a polar-CC peak's dx (in pixel-columns) to a rotation
    angle in degrees. The polar image has one column per angular sample
    around 360°, so `angle_deg = dx · 360 / N_columns`. `sign="invert"`
    (default) negates the result — a peak at +dx means the query image
    is rotated CLOCKWISE by that angle relative to the reference, so
    rotating the query by -angle brings it back into alignment."""
    a = h(polar_map)
    if a.ndim < 2:
        raise ValueError("polar_angle: needs the polar-CC or polar image "
                          "so its column count is known")
    ncols = int(a.shape[-1])
    if ncols <= 0:
        raise ValueError("polar_angle: polar image has no columns")
    ang = float(dx) * 360.0 / ncols
    if sign == "invert":
        ang = -ang
    return {"angle_deg": float(ang), "n_cols": int(ncols)}


def _fn_scalar_linear(value: float = 0.0, a: float = 1.0, b: float = 0.0):
    """Compute `a * value + b`. Handy for unit conversion or trivial
    scaling between wired scalars."""
    return {"result": float(a) * float(value) + float(b)}


def _fn_cc(a, b, log_display: bool = True):
    """Cross-correlate two images. If `log_display` is True the returned map
    is signed-log-compressed (`sign(cc) * log1p(|cc|)`) so a huge central
    peak doesn't wash out the smaller side lobes — this only changes the
    map that gets rendered, side-peak POSITIONS are unaffected so PK still
    picks the right shift/angle."""
    cc = cross_correlate(h(a), h(b))
    if log_display:
        cc_np = np.asarray(cc, dtype=np.float32)
        cc = d(np.sign(cc_np) * np.log1p(np.abs(cc_np)))
    else:
        cc = d(cc)
    return {"cc": cc}

def _fn_align_shift(a, b):
    dy, dx, pk = translational_shift(h(a), h(b))
    return {"dy": float(dy), "dx": float(dx), "peak": float(pk)}


def _fn_find_peaks(image, number_of_peaks: int = 1,
                   center_origin_override: int = 0,
                   new_origin_y: int = 0, new_origin_x: int = 0,
                   peak_number_for_ratio: int = 1,
                   sub_pixel: bool = True):
    """SPIDER PK — locate strict local maxima in a 2D map.

    Outputs mirror SPIDER's register variables:
      xo, yo   — top peak absolute coordinates
      pk       — top peak height (parabola-refined)
      dx, dy   — top peak offset from the image centre (nx // 2, ny // 2).
                 Signed: negative = left/up of centre, positive = right/down.
                 On a CC map this is exactly the (dy, dx) shift that aligns
                 the two inputs; on a polar-CC map it's the rotation angle
                 in pixel-column units.
      ratio    — pk / (peak-number-for-ratio's height)
      xs, ys   — coordinates of the peak selected by peak_number_for_ratio
      pks      — that peak's height
      n_found  — how many strict local maxima were kept

    ``center_origin_override``: 0 = report absolute (xo, yo); 1 = report
    xo/yo relative to new_origin_(y,x). Independent of this flag, `dx`/`dy`
    are always computed against the image centre — the natural convention
    for CC maps.
    """
    a = h(image).astype(np.float32)
    if a.ndim != 2:
        raise ValueError(
            f"PK: expects a 2D peak map, got shape {a.shape}. Insert "
            "`stack.index` to pick a specific frame first."
        )
    ny, nx = a.shape
    n_req = max(1, int(number_of_peaks))
    peaks = find_peaks(a, n_peaks=n_req, sub_pixel=bool(sub_pixel))
    if not peaks:
        return {"xo": 0.0, "yo": 0.0, "pk": 0.0,
                "dx": 0.0, "dy": 0.0,
                "xs": 0.0, "ys": 0.0, "pks": 0.0,
                "ratio": 0.0, "n_found": 0}

    if int(center_origin_override):
        ox = float(new_origin_x) if int(new_origin_x) != 0 else nx // 2
        oy = float(new_origin_y) if int(new_origin_y) != 0 else ny // 2
    else:
        ox = 0.0
        oy = 0.0

    # Image centre — always used for the dx / dy offsets so a CC map's peak
    # position is directly readable as the (dx, dy) that aligns a with b.
    cx, cy = nx // 2, ny // 2

    def _abs(p):
        return float(p.iy + p.dy), float(p.ix + p.dx)
    def _h(p):
        return float(p.refined_height if p.refined_height is not None else p.height)

    y0, x0 = _abs(peaks[0])
    h0 = _h(peaks[0])
    idx = max(1, min(len(peaks), int(peak_number_for_ratio))) - 1
    y1, x1 = _abs(peaks[idx])
    h1 = _h(peaks[idx])

    # Build an RGB preview: input CC map as grayscale + yellow dashed
    # centre crosshair + red dashed circle around the detected top peak.
    # Returned as (H, W, 3) uint8; preview.py handles this shape.
    pmn = float(a.min()) if a.size else 0.0
    pmx = float(a.max()) if a.size else 1.0
    if pmx > pmn:
        gray = ((a - pmn) / (pmx - pmn) * 255.0)
    else:
        gray = np.zeros_like(a)
    rgb = np.stack([gray, gray, gray], axis=-1).clip(0, 255).astype(np.uint8)

    # Dashed pattern helper — every ~4 pixels ON, ~4 pixels OFF.
    def _dashed(mask_1d):
        # Convert a 1-D coordinate array into an on/off dash pattern.
        return ((mask_1d // 4) % 2 == 0)

    YELLOW = np.array([255, 220, 0], dtype=np.uint8)
    RED    = np.array([255, 40, 40], dtype=np.uint8)

    # Yellow dashed centre lines (image centre = nx//2, ny//2).
    xs = np.arange(nx)
    ys = np.arange(ny)
    # Horizontal line at y = cy
    hmask = _dashed(xs)
    rgb[cy, xs[hmask]] = YELLOW
    # Vertical line at x = cx
    vmask = _dashed(ys)
    rgb[ys[vmask], cx] = YELLOW

    # Red dashed ring around the detected peak (radius scales with map).
    ring_r = max(4.0, min(nx, ny) * 0.04)
    yy, xx = np.ogrid[:ny, :nx]
    dist = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    ring_mask = np.abs(dist - ring_r) < 0.8
    # Dashed by angle: keep every other ~15° arc.
    ang = np.degrees(np.arctan2(yy - y0, xx - x0))
    dash_arc = ((ang.astype(np.int32) // 15) % 2 == 0)
    ring_dashed = ring_mask & dash_arc
    rgb[ring_dashed] = RED

    return {
        "xo": x0 - ox, "yo": y0 - oy, "pk": h0,
        "dx": x0 - cx, "dy": y0 - cy,
        "xs": x1 - ox, "ys": y1 - oy, "pks": h1,
        "ratio": h0 / (h1 + 1e-30),
        "n_found": len(peaks),
        "preview": rgb,
    }

def _fn_rot_ang(a, b, n_rings: int = 32, n_angles: int = 360,
                sub_pixel: str = "yes"):
    ang, pk = rotational_angle(h(a), h(b),
                               n_rings=int(n_rings), n_angles=int(n_angles),
                               sub_pixel=(sub_pixel == "yes"))
    return {"angle_deg": float(ang), "peak": float(pk)}


# =========================================================================
# Test-data (pyspider.testdata).
# =========================================================================

def _mk_td(id_, label, fn_call, extra_params=None):
    p = [("size", "int", 128, None, 32, 1024)]
    if extra_params:
        p += extra_params
    def fn(**kw):
        return {"image": d(fn_call(**kw))}
    return N(id_, label, "Data", "", "image:image", p,
             f"Test image: {label}", fn)


def _clock_image(size: int = 256, hour: int = 10, minute: int = 10) -> np.ndarray:
    """Clock-face test image with hour ticks, radially-oriented digits at
    12/3/6/9, and hour + minute hands at (hour, minute). Radially-readable
    digits (top of each glyph points OUTWARD from centre) make the polar
    transform obvious: rotation of the whole clock = horizontal shift of
    every feature in polar space by the same number of columns.

    Default clock time is 10:10 (the classic ad-photo pose) so the two
    hands are widely separated and easy to see."""
    from PIL import Image, ImageDraw, ImageFont
    import os as _os

    n = int(size)
    img = Image.new("L", (n, n), 0)
    d = ImageDraw.Draw(img)
    cx = cy = n // 2
    r = int(n * 0.44)

    # Face disk (mid-grey) so bright ticks / digits stand out.
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=60)
    # Outer bezel
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=210, width=2)

    # 12 hour ticks — long tick every 3 hours (at digit positions), short
    # elsewhere. Angle convention: math radians, 0 = +x (3 o'clock), grows
    # CCW. So 12 o'clock = -π/2, 3 = 0, 6 = +π/2, 9 = π.
    for h in range(12):
        ang = -np.pi / 2 + h * (2 * np.pi / 12)
        is_major = h % 3 == 0
        r_in = r * (0.82 if is_major else 0.88)
        r_out = r * 0.96
        w = 3 if is_major else 1
        d.line([
            (cx + r_in * np.cos(ang),  cy + r_in * np.sin(ang)),
            (cx + r_out * np.cos(ang), cy + r_out * np.sin(ang)),
        ], fill=230, width=w)

    # Font lookup — try common system paths, fall back to PIL's built-in.
    def _find_font(sz):
        for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                  "/System/Library/Fonts/Helvetica.ttc",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                  "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                  "Arial Bold.ttf", "DejaVuSans-Bold.ttf"):
            try:
                return ImageFont.truetype(p, sz)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()
    font = _find_font(max(14, n // 12))

    # Digits at 12, 3, 6, 9 — orient the digit so its top points AWAY
    # from the centre (radially readable: 12 upright at top, 3 rotated
    # so its top points right, 6 upside-down at bottom, 9 rotated so its
    # top points left). Draw text on a small transparent tile, rotate,
    # then paste onto the clock face.
    digit_positions = [
        # (text, position_angle_deg_from_12_CW, rotation_deg_to_make_top_outward)
        ("12", 0,   0),    # 12 o'clock, no rotation
        ("3",  90,  270),  # 3 o'clock, top points right = rotate -90 = 270 CCW
        ("6",  180, 180),  # 6 o'clock, upside down
        ("9",  270, 90),   # 9 o'clock, top points left
    ]
    tile_w = max(28, n // 8)
    r_txt = r * 0.72   # radial position of digit centre
    for text, pos_deg, rot_deg in digit_positions:
        # Convert clock-face angle (0=up, CW) → math angle (0=right, CCW).
        pos_math = -np.pi / 2 + np.deg2rad(pos_deg)
        tx = cx + r_txt * np.cos(pos_math)
        ty = cy + r_txt * np.sin(pos_math)
        tile = Image.new("L", (tile_w, tile_w), 0)
        td = ImageDraw.Draw(tile)
        # Draw centred using anchor="mm" (middle-middle).
        try:
            td.text((tile_w // 2, tile_w // 2), text, fill=255,
                    anchor="mm", font=font)
        except TypeError:
            # Old PIL without anchor= — estimate centre manually.
            bbox = td.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            td.text(((tile_w - tw) // 2 - bbox[0],
                     (tile_w - th) // 2 - bbox[1]), text, fill=255, font=font)
        tile_rot = tile.rotate(rot_deg, resample=Image.BICUBIC)
        img.paste(tile_rot, (int(tx) - tile_w // 2,
                              int(ty) - tile_w // 2), mask=tile_rot)

    # Hour hand — short, thick. Position slides between hours by minute:
    # `hour_angle = -90 + 30·(hour%12) + 0.5·minute` degrees (clock-face).
    def _to_math(deg_from_12_cw):
        return np.deg2rad(deg_from_12_cw - 90.0)
    h_ang_deg = 30.0 * (int(hour) % 12) + 0.5 * int(minute)
    m_ang_deg = 6.0  * int(minute)
    h_ang = _to_math(h_ang_deg)
    m_ang = _to_math(m_ang_deg)
    h_len = r * 0.50
    m_len = r * 0.78
    d.line([(cx, cy), (cx + h_len * np.cos(h_ang), cy + h_len * np.sin(h_ang))],
           fill=255, width=6)   # hour = thick
    d.line([(cx, cy), (cx + m_len * np.cos(m_ang), cy + m_len * np.sin(m_ang))],
           fill=255, width=3)   # minute = thinner but longer
    # Central pivot dot
    d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=255)

    return np.asarray(img, dtype=np.float32) / 255.0


# ---- 3D test volumes (one-click presets, 3D counterparts of camera/coins). ----

def _shepp_logan_3d(size: int = 64) -> np.ndarray:
    """Modified 3D Shepp-Logan phantom (10 ellipsoids in a [-1,1]^3 cube)."""
    N = int(size)
    z, y, x = np.mgrid[:N, :N, :N].astype(np.float32)
    x = 2.0 * x / (N - 1) - 1.0
    y = 2.0 * y / (N - 1) - 1.0
    z = 2.0 * z / (N - 1) - 1.0
    ellipsoids = [
        # (cx, cy, cz, a, b, c, phi_deg (rot about z), intensity)
        (0.0,   0.0,   0.0,   0.69,   0.90,  0.92,  0.0,   1.0),
        (0.0,   0.0,   0.0,   0.6624, 0.874, 0.88,  0.0,  -0.8),
        (-0.22, 0.0,  -0.25,  0.41,   0.16,  0.21,  108.0,-0.2),
        (0.22,  0.0,  -0.25,  0.31,   0.11,  0.22,  72.0, -0.2),
        (0.0,   0.35, -0.25,  0.21,   0.25,  0.5,   0.0,   0.2),
        (0.0,   0.1,  -0.25,  0.046,  0.046, 0.046, 0.0,   0.2),
        (-0.08,-0.605,-0.25,  0.046,  0.023, 0.02,  0.0,   0.1),
        (0.06, -0.605,-0.25,  0.023,  0.023, 0.02,  90.0,  0.1),
        (0.06, -0.105, 0.625, 0.056,  0.04,  0.1,   90.0,  0.2),
        (0.0,   0.1,   0.625, 0.056,  0.056, 0.1,   0.0,  -0.2),
    ]
    vol = np.zeros((N, N, N), np.float32)
    for cx, cy, cz, a, b, c, phi_deg, rho in ellipsoids:
        phi = np.deg2rad(phi_deg)
        cp, sp = np.cos(phi), np.sin(phi)
        xp =  cp * (x - cx) + sp * (y - cy)
        yp = -sp * (x - cx) + cp * (y - cy)
        zp =  z - cz
        r = (xp / a) ** 2 + (yp / b) ** 2 + (zp / c) ** 2
        vol[r <= 1.0] += rho
    return vol.astype(np.float32)


def _blobs_3d(size: int = 64, seed: int = 0) -> np.ndarray:
    """Cluster of 6 Gaussian blobs, soft-enveloped — synthetic macromolecule."""
    N = int(size)
    rng = np.random.default_rng(int(seed))
    z, y, x = np.mgrid[:N, :N, :N].astype(np.float32)
    center = (N - 1) / 2.0
    vol = np.zeros((N, N, N), np.float32)
    positions = rng.uniform(-N * 0.22, N * 0.22, size=(6, 3))
    for i in range(6):
        cz, cy, cx = center + positions[i]
        sigma = N * float(rng.uniform(0.06, 0.12))
        amp = float(rng.uniform(0.6, 1.0))
        r2 = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2
        vol += amp * np.exp(-0.5 * r2 / (sigma * sigma))
    r_env = np.sqrt((z - center) ** 2 + (y - center) ** 2 + (x - center) ** 2)
    envelope = 1.0 / (1.0 + np.exp((r_env - N * 0.35) / max(1.0, N * 0.03)))
    return (vol * envelope).astype(np.float32)


def _helix_3d(size: int = 64, n_turns: float = 3.0) -> np.ndarray:
    """DNA-like helix — Gaussian beads along a helical path."""
    N = int(size)
    center = (N - 1) / 2.0
    R = 0.25 * N
    sig = max(0.5, 0.05 * N)
    n_beads = max(8, N)
    z, y, x = np.mgrid[:N, :N, :N].astype(np.float32)
    vol = np.zeros((N, N, N), np.float32)
    for i in range(n_beads):
        t = i / max(1, n_beads - 1)
        cz = center + (t - 0.5) * (N * 0.8)
        angle = 2.0 * np.pi * float(n_turns) * t
        cy = center + R * np.sin(angle)
        cx = center + R * np.cos(angle)
        r2 = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2
        vol += np.exp(-0.5 * r2 / (sig * sig))
    return vol.astype(np.float32)


def _two_beads_3d(size: int = 64, separation_frac: float = 0.3) -> np.ndarray:
    """Two Gaussian beads along z — classic resolution demo."""
    N = int(size)
    center = (N - 1) / 2.0
    d0 = 0.5 * float(separation_frac) * N
    sig = max(0.5, 0.08 * N)
    z, y, x = np.mgrid[:N, :N, :N].astype(np.float32)
    r_a = (z - center - d0) ** 2 + (y - center) ** 2 + (x - center) ** 2
    r_b = (z - center + d0) ** 2 + (y - center) ** 2 + (x - center) ** 2
    return (np.exp(-0.5 * r_a / (sig * sig)) +
            np.exp(-0.5 * r_b / (sig * sig))).astype(np.float32)


def _torus_3d(size: int = 64, R_frac: float = 0.28, r_frac: float = 0.08) -> np.ndarray:
    """Filled torus (major R_frac, minor r_frac, in units of size)."""
    N = int(size)
    center = (N - 1) / 2.0
    R0 = float(R_frac) * N
    r0 = float(r_frac) * N
    z, y, x = np.mgrid[:N, :N, :N].astype(np.float32)
    rho = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    d2 = (rho - R0) ** 2 + (z - center) ** 2
    return np.where(d2 <= r0 * r0, 1.0, 0.0).astype(np.float32)


def _mk_td3d(id_, label, fn_call, extra_params=None):
    p = [("size", "int", 128, None, 8, 256)]
    if extra_params:
        p += extra_params
    def fn(**kw):
        return {"volume": d(fn_call(**kw))}
    return N(id_, label, "Data", "", "volume:volume", p,
             f"Test volume: {label}", fn)


# =========================================================================
# Catalog assembly.
# =========================================================================

def build_catalog() -> dict[str, NodeSpec]:
    # All built-in nodes have been split into backend/plugins/builtin_*.py
    # so that every node in the app goes through the same plugin registration
    # path. CATALOG starts empty here and is filled by _load_plugins_into_catalog().
    return {}


CATALOG: dict[str, NodeSpec] = build_catalog()


def _load_plugins_into_catalog() -> None:
    """Discover drop-in plugins and merge their nodes into CATALOG.

    Plugins live in backend/plugins/, $PYSPIDER_PLUGIN_DIR entries, or
    ~/.pyspider/plugins/. If a plugin registers an id that collides with
    a built-in, the built-in wins (silently — we log the collision).
    """
    try:
        from . import plugin_api, plugin_loader
    except Exception:  # noqa: BLE001
        return
    plugin_loader.load_plugins()
    for spec in plugin_api.registered_nodes():
        if spec.id in CATALOG:
            continue
        _fold_manual(spec)
        CATALOG[spec.id] = spec


_load_plugins_into_catalog()


def public_catalog() -> list[dict]:
    return [s.to_public() for s in CATALOG.values()]

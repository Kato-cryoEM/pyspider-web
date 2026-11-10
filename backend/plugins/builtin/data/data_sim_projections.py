"""Built-in node — data.sim.projections (Sim · Random Projections) — Data.

Random projections of a volume (3D → stack of 2D projections) OR of an
image (2D → stack of 1D line projections, 2D Radon transform). The mode
is **auto-detected from the wired input** — connect either `volume` or
`image` and the node figures out what to do.

Preview adapts to the output type:
  - 2D projections (volume input): the stack's filmstrip (per-frame
    thumbnails) is the primary preview.
  - 1D projections (image input): a line-plot overlay of ~10 projections
    is emitted on a dedicated `preview` image port so a 1D-line signal
    is visualised as a line graph, not a sliver image.

`eulers_csv` wires directly to `recon.bp` for a round-trip
(image → projections → BP) in either dimensionality.
"""

from __future__ import annotations

import io
import numpy as np

from backend.plugin_api import register, N, Param, Port, h, d
from backend.nodes import _sim_random_projections_2d, _project_batch_fourier
from pyspider import simulate
import scipy.ndimage as _ndi


_PREVIEW_H = 320
_PREVIEW_W = 480


def _render_1d_projections_preview(stack_1d: np.ndarray,
                                   angles: np.ndarray) -> np.ndarray:
    """Render an overlay line-plot of up to ~10 1D projections. Returns
    a grayscale float32 image (H, W) suitable for a plain `image` port."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n, W = stack_1d.shape
    max_lines = min(10, n)
    step = max(1, n // max_lines)
    picks = list(range(0, n, step))[:max_lines]

    fig = plt.figure(figsize=(_PREVIEW_W / 100.0, _PREVIEW_H / 100.0), dpi=100)
    ax = fig.add_subplot(111)
    x = np.arange(W)
    for i in picks:
        ax.plot(x, stack_1d[i], linewidth=0.9, alpha=0.65,
                label=f"{float(angles[i]):.0f}°")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("position (pixel)")
    ax.set_ylabel("integrated intensity")
    ax.set_title(f"1D projections (showing {len(picks)} of {n})")
    if len(picks) <= 8:
        ax.legend(fontsize=7, ncol=4, loc="upper right",
                  frameon=False, labelspacing=0.2)
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    plt.close(fig)
    gray = (0.299 * buf[..., 0]
          + 0.587 * buf[..., 1]
          + 0.114 * buf[..., 2]).astype(np.float32)
    return gray


def _fn_sim_projections(volume=None, image=None,
                        n: int = 16, seed: int = 0,
                        uniform_sphere: str = "yes",
                        shift_sigma: float = 0.0, noise_sigma: float = 0.0):
    # ------------------------------------------------------------------
    # Auto-detect mode from the wired inputs.
    # If BOTH are wired, prefer whichever is actually shape-appropriate.
    # ------------------------------------------------------------------
    vol = h(volume) if volume is not None else None
    img = h(image)  if image  is not None else None

    if img is not None and img.ndim == 2:
        is_2d = True
    elif vol is not None and vol.ndim == 3:
        is_2d = False
    elif vol is not None and vol.ndim == 2:
        # User wired a 2D image into the `volume` port by habit —
        # treat it as a 2D image.
        is_2d = True
        img = vol
    elif img is not None and img.ndim == 3:
        # 3D data landed on the `image` port — treat as volume.
        is_2d = False
        vol = img
    else:
        raise ValueError(
            "sim.projections: wire either a 2D image (→ 1D projections) "
            "or a 3D volume (→ 2D projections) into the respective input."
        )

    if is_2d:
        stk, angles = _sim_random_projections_2d(
            img, n=int(n), seed=int(seed),
            shift_sigma=float(shift_sigma), noise_sigma=float(noise_sigma),
        )
        eulers_csv = "\n".join(
            f"{a:.4f},0.0000,0.0000" for a in angles.tolist()
        )
        shifts_csv = ""
        preview = _render_1d_projections_preview(stk[:, 0, :], angles)
        return {
            "stack": d(stk),
            "eulers_csv": eulers_csv,
            "shifts_csv": shifts_csv,
            "num_frames": int(n),
            "preview": d(preview),
        }

    # 3D case — use the central-slice Fourier batch projector. One 3D FFT
    # of the volume is shared across all `n` angles, then each projection
    # is just a 2D slice extraction + 2D IFFT. Benchmarks on the current
    # box sizes see 40–60× speedups vs the per-frame real-space rotate-
    # and-sum loop that `simulate.random_projections` uses, so for large
    # n or large volumes this is the only practical path.
    eulers = simulate.random_euler_angles(
        int(n), seed=int(seed), uniform_sphere=(uniform_sphere == "yes"))
    rng = np.random.default_rng(int(seed) + 1)
    if shift_sigma > 0:
        shifts = (rng.standard_normal((int(n), 2)) * float(shift_sigma)
                  ).astype(np.float32)
    else:
        shifts = np.zeros((int(n), 2), dtype=np.float32)

    projs = _project_batch_fourier(vol, eulers)      # (n, ny, nx) float32

    if shift_sigma > 0:
        for i in range(int(n)):
            projs[i] = _ndi.shift(projs[i],
                                  (float(shifts[i, 0]), float(shifts[i, 1])),
                                  order=1, mode="constant",
                                  cval=0.0).astype(np.float32)
    if noise_sigma > 0:
        ny, nx = projs.shape[-2], projs.shape[-1]
        noise = rng.standard_normal((int(n), ny, nx)).astype(np.float32)
        projs = projs + float(noise_sigma) * noise

    eulers_csv = "\n".join(
        f"{p:.4f},{t:.4f},{s:.4f}" for p, t, s in eulers.tolist()
    )
    shifts_csv = "\n".join(
        f"{dy:.4f},{dx:.4f}" for dy, dx in shifts.tolist()
    )
    first_proj = projs[0].astype(np.float32)
    return {
        "stack": d(projs.astype(np.float32)),
        "eulers_csv": eulers_csv,
        "shifts_csv": shifts_csv,
        "num_frames": int(n),
        "preview": d(first_proj),
    }


register(N(
    "data.sim.projections", "Sim · Random Projections", "Data",
    [
        Port("volume", "volume", optional=True),
        Port("image",  "image",  optional=True),
    ],
    [
        Port("stack",      "stack"),
        Port("eulers_csv", "string"),
        Port("shifts_csv", "string"),
        Port("num_frames", "scalar"),
        Port("preview",    "image"),
    ],
    [
        Param("n", "int", 16, None, 1, 2048),
        Param("seed", "int", 0, None, 0, 1_000_000),
        Param("uniform_sphere", "select", "yes", ["yes", "no"]),
        Param("shift_sigma", "float", 0.0, None, 0.0, 100.0),
        Param("noise_sigma", "float", 0.0, None, 0.0, 10.0),
    ],
    "Random projections. **The dimensionality is auto-detected** — wire "
    "either a `volume` (3D) or an `image` (2D), the other input stays "
    "empty, and the node routes accordingly.\n\n"
    "**volume input (3D)** → stack of `n` random SPIDER ZYZ 2D "
    "projections. `uniform_sphere=yes` draws cos(theta) uniformly so "
    "directions are evenly distributed on S^2. `eulers_csv` has one "
    "`phi,theta,psi` per row (deg).\n\n"
    "**image input (2D)** → 2D Radon transform: `n` 1D line projections "
    "at random in-plane angles. Stack shape is `(n, 1, W)` so it still "
    "flows through `stack` wires; `eulers_csv` encodes the projection "
    "angle as `phi,0,0` (deg from +x axis, CCW). `uniform_sphere` is "
    "ignored.\n\n"
    "The `preview` port is adaptive: 2D projections show the first "
    "frame as an image; 1D projections show up to 10 of them as a line "
    "plot overlay so SpiderNode's auto-preview renders a graph instead "
    "of a 1-pixel-tall sliver. `shift_sigma` adds a random translation "
    "to each output (pixels; 2D case shifts along the perpendicular "
    "axis only). `noise_sigma` adds per-pixel Gaussian noise after "
    "shifting.",
    _fn_sim_projections,
))

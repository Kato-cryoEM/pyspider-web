"""Built-in node — data.sim.rotations (Sim · Random Rotations) — category: Data.

Wraps :func:`pyspider.simulate.random_rotations` — a 2D image is replicated
``n`` times with random in-plane angles (and optional shifts / Gaussian noise).
``angles_csv`` reports the ground truth so alignment demos can score recovery.
"""

from __future__ import annotations

from backend.plugin_api import register, N, h, d
from pyspider import simulate


def _fn_sim_rotations(image, n: int = 8, seed: int = 0,
                      angle_min: float = 0.0, angle_max: float = 360.0,
                      shift_sigma: float = 0.0, noise_sigma: float = 0.0):
    img = h(image)
    stk = simulate.random_rotations(
        img, n=int(n), seed=int(seed),
        angle_range=(float(angle_min), float(angle_max)),
        shift_sigma=float(shift_sigma), noise_sigma=float(noise_sigma),
    )
    rows = [
        f"{a:.4f},{dy:.4f},{dx:.4f}"
        for a, (dy, dx) in zip(stk.angles.tolist(),
                               [tuple(r) for r in stk.shifts.tolist()])
    ]
    return {
        "stack": d(stk.images),
        "angles_csv": "\n".join(rows),
        "num_frames": int(n),
    }


register(N(
    "data.sim.rotations", "Sim · Random Rotations", "Data",
    "image:image",
    "stack:stack,angles_csv:string,num_frames:scalar",
    [
        ("n", "int", 8, None, 1, 1024),
        ("seed", "int", 0, None, 0, 1_000_000),
        ("angle_min", "float", 0.0, None, -360.0, 360.0),
        ("angle_max", "float", 360.0, None, -360.0, 720.0),
        ("shift_sigma", "float", 0.0, None, 0.0, 100.0),
        ("noise_sigma", "float", 0.0, None, 0.0, 10.0),
    ],
    "Build a stack of random in-plane rotations of a 2D image. "
    "angles_csv lists `angle_deg,dy,dx` per frame (the ground truth "
    "transform that was applied). Set shift_sigma / noise_sigma > 0 "
    "to add Gaussian translation / per-pixel noise.",
    _fn_sim_rotations,
))

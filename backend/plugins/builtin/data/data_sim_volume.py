"""Built-in node — data.sim.volume (Sim · Random Volume) — category: Data.

Wraps :func:`pyspider.simulate.random_volume` — a reproducible, soft-enveloped
blob particle that drops straight into reconstruction / projection demos.
"""

from __future__ import annotations

from backend.plugin_api import register, N, d
from pyspider import simulate


def _fn_sim_volume(size: int = 64, n_blobs: int = 10, seed: int = 0,
                   soft_edge: str = "yes"):
    vol = simulate.random_volume(
        N=int(size), n_blobs=int(n_blobs), seed=int(seed),
        soft_edge=(soft_edge == "yes"),
    )
    return {"volume": d(vol)}


register(N(
    "data.sim.volume", "Sim · Random Volume", "Data",
    "", "volume:volume",
    [
        ("size", "int", 128, None, 8, 256),
        ("n_blobs", "int", 10, None, 0, 50),
        ("seed", "int", 0, None, 0, 1_000_000),
        ("soft_edge", "select", "yes", ["yes", "no"]),
    ],
    "Random cubic blob volume (central body + off-centre Gaussians). "
    "Reproducible per seed, soft-enveloped so the particle sits clear "
    "of the box walls — pair with Sim · Random Projections for a "
    "self-contained reconstruction demo.",
    _fn_sim_volume,
))

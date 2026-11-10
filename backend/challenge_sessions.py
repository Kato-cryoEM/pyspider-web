"""Challenge mode — session state, fixture generation, judging.

A challenge session caches the two (or more) ground-truth fixture images for
the duration of one puzzle attempt. Fixture nodes placed on the student's
graph look up their image here by (session_id, name). The judge endpoint
runs the student's graph, pulls the designated answer image, and compares
it to the reference fixture.

Sessions are in-memory only — a server restart invalidates them, which is
fine: the student just presses "Start challenge" again. We intentionally
avoid persistence to keep the attempt surface attack-free.
"""

from __future__ import annotations

import json
import secrets
import time
from pathlib import Path
from typing import Any

import numpy as np

# {session_id: {"challenge_id": str, "created": float, "fixtures": {name: ndarray}}}
_SESSIONS: dict[str, dict[str, Any]] = {}

# Oldest-first eviction so a leftover tab doesn't balloon memory.
_MAX_SESSIONS = 128


# ---------------------------------------------------------------------------
# Fixture generation — tiny recipe DSL. Each recipe dict declares HOW to
# build a float32 2D image, either from scratch (testdata) or by transforming
# a previously-built fixture in the same challenge.

def _apply_recipe(recipe: dict, built: dict[str, np.ndarray]) -> np.ndarray:
    # Imported lazily so this module doesn't force `.nodes` to load during
    # plugin discovery (which runs *inside* nodes.py — a circular import trap).
    from scipy import ndimage as ndi
    if "testdata" in recipe:
        name = recipe["testdata"]
        if name == "clock":
            from .nodes import _clock_image
            size = int(recipe.get("size", 128))
            hour = int(recipe.get("hour", 10))
            minute = int(recipe.get("minute", 10))
            return _clock_image(size, hour, minute).astype(np.float32)
        from pyspider import testdata as td
        fn = getattr(td, name, None)
        if fn is None:
            raise ValueError(f"unknown testdata function: {name!r}")
        kwargs = {k: v for k, v in recipe.items() if k != "testdata"}
        return np.asarray(fn(**kwargs), dtype=np.float32)
    if "shift_of" in recipe:
        src = recipe["shift_of"]
        if src not in built:
            raise ValueError(
                f"shift_of refers to {src!r} which is not built yet "
                "(order fixtures so dependencies come first)"
            )
        dy = float(recipe.get("dy", 0.0))
        dx = float(recipe.get("dx", 0.0))
        order = int(recipe.get("order", 3))
        return ndi.shift(built[src], (dy, dx), order=order,
                         mode="constant", cval=0.0).astype(np.float32)
    if "rotate_of" in recipe:
        src = recipe["rotate_of"]
        if src not in built:
            raise ValueError(
                f"rotate_of refers to {src!r} which is not built yet "
                "(order fixtures so dependencies come first)"
            )
        angle_deg = float(recipe.get("angle_deg", 0.0))
        order = int(recipe.get("order", 3))
        return ndi.rotate(built[src], angle_deg, reshape=False, order=order,
                          mode="constant", cval=0.0).astype(np.float32)
    if "sim_volume" in recipe:
        # Random blob 3D volume (pyspider.simulate.random_volume).
        from pyspider import simulate
        size = int(recipe.get("size", 64))
        seed = int(recipe.get("seed", 0))
        n_blobs = int(recipe.get("n_blobs", 10))
        return simulate.random_volume(N=size, n_blobs=n_blobs, seed=seed).astype(np.float32)
    if "helix_volume" in recipe:
        # DNA-like 3D helix with Gaussian beads — pedagogically obvious shape.
        from .nodes import _helix_3d
        size = int(recipe.get("size", 64))
        n_turns = float(recipe.get("n_turns", 3.0))
        return _helix_3d(size, n_turns).astype(np.float32)
    if "pdb_volume" in recipe:
        # Load a bundled PDB-derived density map from backend/bundled_data/betagal/.
        # `source` selects which .mrc file (allowlisted to the betagal/ folder
        # so a stray recipe can't read arbitrary paths).
        from pathlib import Path
        from . import formats
        src_name = str(recipe.get("source", "pdb_1jz8_2A"))
        if "/" in src_name or ".." in src_name:
            raise ValueError(f"pdb_volume source {src_name!r} must be a bare filename stem")
        src = (Path(__file__).resolve().parent / "bundled_data" / "betagal"
               / f"{src_name}.mrc")
        if not src.exists():
            raise FileNotFoundError(
                f"bundled map {src} not found. Available: pdb_1jz8_2A, map3d_128"
            )
        return formats.read_array(str(src)).astype(np.float32)
    if ("betagal_micrograph" in recipe
            or "betagal_truth_mask" in recipe):
        # β-galactosidase (RCSB 1JZ8) atomic coordinates → bundled density
        # map (`pdb_1jz8_2A.mrc`, 104³ at 2 Å/px, Gaussian-rasterised from
        # atoms so the background is exactly zero). Placed as N
        # non-overlapping random projections on a 1024² canvas.
        kind = "micrograph" if "betagal_micrograph" in recipe else "truth_mask"
        from pathlib import Path
        from . import formats
        from pyspider.reconstruct import project as _project

        size = int(recipe.get("size", 1024))
        n_particles = int(recipe.get("n_particles", 30))
        seed = int(recipe.get("seed", 42))
        noise = float(recipe.get("noise", 0.0))
        spot_sigma = float(recipe.get("spot_sigma", 25.0))

        rng_pos = np.random.default_rng(seed * 1000 + 1)
        rng_rot = np.random.default_rng(seed * 1000 + 2)
        rng_noise = np.random.default_rng(seed * 1000 + 3)

        src = (Path(__file__).resolve().parent / "bundled_data" / "betagal"
               / "pdb_1jz8_2A.mrc")
        if not src.exists():
            raise FileNotFoundError(
                f"PDB-derived β-gal map missing at {src}. "
                "Regenerate with the PDB → density rasteriser."
            )
        vol = formats.read_array(str(src)).astype(np.float32)
        particle = vol.shape[0]
        half = particle // 2

        # Reject-sample 30 positions with min pairwise distance ≈ particle
        # diameter so particles don't overlap.
        min_dist2 = int(particle * 1.1) ** 2
        positions: list[tuple[int, int]] = []
        max_tries = 20000
        tries = 0
        while len(positions) < n_particles and tries < max_tries:
            y = int(rng_pos.integers(half + 2, size - half - 2))
            x = int(rng_pos.integers(half + 2, size - half - 2))
            if all((y - yy) ** 2 + (x - xx) ** 2 >= min_dist2
                   for yy, xx in positions):
                positions.append((y, x))
            tries += 1

        # Random orientations on S² (uniform in solid angle: cos(theta) uniform).
        eulers = []
        for _ in positions:
            phi = float(rng_rot.uniform(0.0, 360.0))
            theta = float(np.rad2deg(np.arccos(rng_rot.uniform(-1.0, 1.0))))
            psi = float(rng_rot.uniform(0.0, 360.0))
            eulers.append((phi, theta, psi))

        if kind == "micrograph":
            out = np.zeros((size, size), dtype=np.float32)
            for (y, x), (phi, theta, psi) in zip(positions, eulers):
                proj = _project(vol, phi, theta, psi).astype(np.float32)
                y0 = y - half; x0 = x - half
                out[y0:y0 + particle, x0:x0 + particle] += proj
            out += noise * rng_noise.standard_normal((size, size)).astype(np.float32)
            return out.astype(np.float32)
        # truth_mask
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        mask = np.zeros((size, size), dtype=np.float32)
        for (y, x) in positions:
            r2 = (xx - float(x)) ** 2 + (yy - float(y)) ** 2
            mask += np.exp(-r2 / (2.0 * spot_sigma * spot_sigma))
        return mask.astype(np.float32)
    if ("micrograph_from_shape" in recipe
            or "truth_mask_from_shape" in recipe):
        # Build a synthetic micrograph (or its matching ground-truth mask)
        # directly from a Gaussian-blob particle shape — no template fixture
        # needed, the student rebuilds the template themselves with data.mo.
        # Positions are drawn from a seeded RNG so the two fixtures match.
        kind = "micrograph" if "micrograph_from_shape" in recipe else "truth_mask"

        size = int(recipe.get("size", 1024))
        n_particles = int(recipe.get("n_particles", 30))
        seed = int(recipe.get("seed", 42))
        noise = float(recipe.get("noise", 0.05))
        particle_r = float(recipe.get("particle_radius", 18.0))
        spot_sigma = float(recipe.get("spot_sigma", particle_r * 0.8))

        rng_pos = np.random.default_rng(seed * 1000 + 1)
        rng_rot = np.random.default_rng(seed * 1000 + 2)
        rng_noise = np.random.default_rng(seed * 1000 + 3)

        margin = int(particle_r * 2.5) + 2
        xs = rng_pos.integers(margin, size - margin, n_particles)
        ys = rng_pos.integers(margin, size - margin, n_particles)
        # Rotation stream is advanced even though a radially-symmetric blob
        # ignores it — keeps the RNG trace deterministic for future variants
        # that place anisotropic particles here.
        rng_rot.uniform(0.0, 360.0, n_particles)

        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        if kind == "micrograph":
            out = np.zeros((size, size), dtype=np.float32)
            for i in range(n_particles):
                r2 = (xx - float(xs[i])) ** 2 + (yy - float(ys[i])) ** 2
                out += np.exp(-r2 / (2.0 * particle_r * particle_r))
            out += noise * rng_noise.standard_normal((size, size)).astype(np.float32)
            return out.astype(np.float32)
        # truth_mask: Gaussian spot per centre (broader than particle so a
        # thresholded CC blob and the ground-truth spot overlap generously).
        mask = np.zeros((size, size), dtype=np.float32)
        for i in range(n_particles):
            r2 = (xx - float(xs[i])) ** 2 + (yy - float(ys[i])) ** 2
            mask += np.exp(-r2 / (2.0 * spot_sigma * spot_sigma))
        return mask.astype(np.float32)
    if "project_of" in recipe:
        # Project an existing volume fixture along given ZYZ Eulers (degrees).
        from pyspider import reconstruct
        src = recipe["project_of"]
        if src not in built:
            raise ValueError(
                f"project_of refers to {src!r} which is not built yet"
            )
        vol = built[src]
        if vol.ndim != 3:
            raise ValueError(
                f"project_of source {src!r} must be 3D, got {vol.shape}"
            )
        phi = float(recipe.get("phi", 0.0))
        theta = float(recipe.get("theta", 0.0))
        psi = float(recipe.get("psi", 0.0))
        return reconstruct.project(vol, phi, theta, psi).astype(np.float32)
    if "axis_project" in recipe:
        # Straight axis-aligned sum (no Euler rotation): pedagogically clear
        # pair for the slice theorem — axis='y' gives sum_y V = (nz, nx).
        src = recipe["axis_project"]
        if src not in built:
            raise ValueError(
                f"axis_project refers to {src!r} which is not built yet"
            )
        vol = built[src]
        if vol.ndim != 3:
            raise ValueError(f"axis_project source must be 3D, got {vol.shape}")
        axis = recipe.get("axis", "z")
        axis_index = {"z": 0, "y": 1, "x": 2}.get(axis)
        if axis_index is None:
            raise ValueError(f"axis must be 'z' / 'y' / 'x', got {axis!r}")
        return vol.sum(axis=axis_index).astype(np.float32)
    if "sim_project" in recipe:
        # Build a random volume, project along given ZYZ Eulers, return 2D image.
        # Convenience shorthand for sim_volume + project_of in one shot.
        from pyspider import simulate, reconstruct
        size = int(recipe.get("size", 64))
        seed = int(recipe.get("seed", 0))
        n_blobs = int(recipe.get("n_blobs", 10))
        phi = float(recipe.get("phi", 0.0))
        theta = float(recipe.get("theta", 0.0))
        psi = float(recipe.get("psi", 0.0))
        vol = simulate.random_volume(N=size, n_blobs=n_blobs, seed=seed)
        return reconstruct.project(vol, phi, theta, psi).astype(np.float32)
    if ("ica_clean" in recipe or "ica_stack" in recipe):
        # Iterative class-average fixtures. Shared seed so clean + stack
        # agree on the base particle; distinct sub-recipes decide which
        # fixture is emitted (clean reference vs. rotated noisy stack).
        kind = "clean" if "ica_clean" in recipe else "stack"
        from pyspider import simulate as _sim
        size = int(recipe.get("size", 128))
        seed = int(recipe.get("seed", 42))
        n = int(recipe.get("n_frames", 30))
        shift_sigma = float(recipe.get("shift_sigma", 3.0))
        noise_sigma = float(recipe.get("noise_sigma", 0.3))

        # Compact asymmetric multi-blob particle. Chosen so (i) the
        # radial projection isn't close enough to the full image for the
        # raw stack mean to already be a perfect answer, (ii) random
        # rotations produce visually noisy frames even at modest σ.
        yy, xx = np.mgrid[:size, :size].astype(np.float32)
        cy, cx = size / 2.0, size / 2.0
        def _g(y0, x0, sig, amp):
            return amp * np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2)
                                / (2.0 * sig * sig))
        base = (
            _g(cy,       cx - size * 0.16, size * 0.08, 0.80)
          + _g(cy - size * 0.12, cx + size * 0.08, size * 0.055, 0.50)
          + _g(cy + size * 0.16, cx + size * 0.12, size * 0.040, 0.70)
          + _g(cy + size * 0.04, cx,              size * 0.12,  0.30)
        ).astype(np.float32)
        if kind == "stack":
            stk = _sim.random_rotations(base, n=n, seed=seed,
                                        angle_range=(0.0, 360.0),
                                        shift_sigma=shift_sigma,
                                        noise_sigma=noise_sigma)
            return stk.images.astype(np.float32)
        # kind == "clean": shortcut for the "class-average look" target —
        # add independent Gaussian noise to the base image, average many
        # copies, no rotation. Costs milliseconds. We use more copies than
        # the student-visible stack (`clean_averages`, default 500) so the
        # intrinsic noise on the comparison target (σ/√N) is low enough
        # that the student's converged mean can plausibly correlate above
        # the pass threshold — the student's output has its own residual
        # noise + alignment interpolation blur, so the clean side needs
        # headroom.
        n_clean = int(recipe.get("clean_averages", 2000))
        rng = np.random.default_rng(seed)
        acc = np.zeros_like(base, dtype=np.float32)
        for _ in range(n_clean):
            acc += base + rng.normal(0.0, noise_sigma, base.shape).astype(np.float32)
        return (acc / float(n_clean)).astype(np.float32)
    if ("mcls_stack" in recipe or "mcls_truth" in recipe
            or "mcls_labels" in recipe):
        # Multi-class classification fixtures. One 3D particle is sampled
        # from K viewing directions to produce K distinct 2D projections —
        # these are the "classes" the student has to disentangle. The
        # mixed stack randomly draws from the K classes and hits each
        # frame with random in-plane rotation + small shift + noise.
        # `mcls_truth` returns the clean K-projection stack (judge
        # reference); `mcls_stack` returns the noisy mixed stack. Shared
        # seed so the two agree on the base volume and its orientations.
        if "mcls_truth" in recipe:
            kind = "truth"
        elif "mcls_labels" in recipe:
            kind = "labels"
        else:
            kind = "stack"
        from pyspider import simulate as _sim
        from pyspider import reconstruct as _rec
        size = int(recipe.get("size", 128))
        seed = int(recipe.get("seed", 42))
        k_classes = int(recipe.get("k_classes", 4))
        n_per_class = int(recipe.get("n_per_class", 25))
        shift_sigma = float(recipe.get("shift_sigma", 3.0))
        noise_sigma = float(recipe.get("noise_sigma", 0.9))

        # Build one 3D volume and pick K viewing directions (phi, theta).
        # psi is left at 0 so random in-plane rotations handle the roll —
        # the student recovers an in-plane-aligned class average that still
        # matches any truth orientation modulo rotation (judge uses rot_ncc).
        # Default: a helix of beads. Random blob volumes give projections
        # too similar to each other under rot_ncc (central-blob dominated,
        # off-diag correlations stay around 0.95); a helix has a strongly
        # orientation-dependent silhouette (axial view = circle, side view
        # = sinusoid), giving off-diag correlations down to ~0.6-0.75.
        if recipe.get("use_random_volume", False):
            vol = _sim.random_volume(
                N=size, n_blobs=int(recipe.get("n_blobs", 8)),
                seed=seed, soft_edge=True)
        else:
            n_turns   = float(recipe.get("helix_turns",       2.0))
            r_frac    = float(recipe.get("helix_radius_frac", 0.25))
            bead_frac = float(recipe.get("helix_bead_frac",   0.045))
            n_beads   = int(  recipe.get("helix_beads",       24))
            length_frac = float(recipe.get("helix_length_frac", 0.5))
            zz, yy, xx = np.mgrid[:size, :size, :size].astype(np.float32)
            cz = cy = cx = size / 2.0
            radius = size * r_frac
            sig = size * bead_frac
            vol = np.zeros_like(zz)
            for i in range(n_beads):
                t = i / max(1, n_beads - 1)
                z = cz + (t - 0.5) * size * length_frac
                theta_h = 2 * np.pi * n_turns * t
                y = cy + radius * np.cos(theta_h)
                x = cx + radius * np.sin(theta_h)
                vol = vol + np.exp(
                    -((zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2)
                    / (2 * sig * sig))
            vol = vol.astype(np.float32)
        # (phi, theta) under rot_ncc: phi rotates around z, which is a pure
        # in-plane rotation when theta=0, so (0,0) and (90,0) are
        # indistinguishable to the judge. These presets vary theta widely
        # — axial / tilted / side views — giving silhouettes of distinctly
        # different outlines.
        preset_angles = [
            (0.0,  0.0),     # axial (down z) — circle for the helix
            (0.0, 90.0),     # pure side view — sinusoid
            (0.0, 45.0),     # oblique
            (90.0, 90.0),    # side view 90° around z — different silhouette
            (0.0,  60.0),
            (45.0, 45.0),
            (180.0, 90.0),
            (120.0, 60.0),
        ]
        if k_classes <= len(preset_angles):
            angles = preset_angles[:k_classes]
        else:
            # Seeded random fallback for K > 8.
            rng = np.random.default_rng(seed + 7777)
            angles = list(preset_angles)
            for _ in range(k_classes - len(preset_angles)):
                angles.append((float(rng.uniform(0, 360)),
                               float(rng.uniform(0, 180))))
        bases = np.stack([
            _rec.project(vol, phi, theta, 0.0).astype(np.float32)
            for (phi, theta) in angles
        ], axis=0)
        # Normalise each projection to roughly [0, 1] so noise_sigma means
        # the same thing as in the single-class challenge (where the base
        # was a Gaussian-blob composite peaking at ~1.0). Without this the
        # projection sum peaks at ~50, making noise_sigma=0.9 practically
        # invisible.
        peaks = bases.reshape(k_classes, -1).max(axis=1)
        peaks = np.where(peaks > 0, peaks, 1.0).astype(np.float32)
        bases = bases / peaks[:, None, None]

        if kind == "truth":
            # Clean-average style per-class target: for each base, add noise
            # to many independent copies and average. No rotation, no
            # alignment — matches the pedagogical shortcut used by the
            # single-class challenge.
            n_clean = int(recipe.get("clean_averages", 2000))
            rng = np.random.default_rng(seed)
            out = np.zeros_like(bases)
            for c in range(k_classes):
                acc = np.zeros_like(bases[c])
                for _ in range(n_clean):
                    acc += bases[c] + rng.normal(
                        0.0, noise_sigma, bases[c].shape).astype(np.float32)
                out[c] = acc / float(n_clean)
            return out.astype(np.float32)

        # kind == "stack" or "labels": build the per-frame labels first so
        # both branches agree. Labels are deterministic in `seed` and the
        # labels fixture lets the judge compare clustering agreement rather
        # than the (brittle) averaged-image appearance.
        total = k_classes * n_per_class
        rng = np.random.default_rng(seed)
        labels = np.repeat(np.arange(k_classes), n_per_class)
        rng.shuffle(labels)
        labels = labels.astype(np.int32)
        if kind == "labels":
            return labels
        frames = np.empty((total, size, size), dtype=np.float32)
        for i in range(total):
            # Each frame gets its own random seed (independent of others)
            # so the global stack is deterministic in `seed`.
            sub_seed = int(rng.integers(0, 2**31 - 1))
            stk = _sim.random_rotations(
                bases[int(labels[i])], n=1, seed=sub_seed,
                angle_range=(0.0, 360.0),
                shift_sigma=shift_sigma, noise_sigma=noise_sigma)
            frames[i] = stk.images[0].astype(np.float32)
        return frames
    if "fft_amplitude_of" in recipe:
        # 2D FFT amplitude, fftshifted to match ft.forward + ft.amplitude.
        src = recipe["fft_amplitude_of"]
        if src not in built:
            raise ValueError(
                f"fft_amplitude_of refers to {src!r} which is not built yet"
            )
        arr = built[src]
        return np.abs(np.fft.fftshift(np.fft.fft2(arr))).astype(np.float32)
    raise ValueError(f"unsupported recipe keys: {sorted(recipe.keys())}")


def build_fixtures(spec: dict) -> dict[str, np.ndarray]:
    """Return {fixture_name: float32 ndarray} for every fixture in `spec`."""
    built: dict[str, np.ndarray] = {}
    for item in spec.get("fixtures", []):
        name = item["name"]
        built[name] = _apply_recipe(item["recipe"], built)
    return built


# ---------------------------------------------------------------------------
# Session lifecycle.

def start_session(spec: dict) -> dict:
    """Build fixtures, cache them under a fresh id, return a client payload."""
    sid = secrets.token_hex(8)
    fixtures = build_fixtures(spec)
    _SESSIONS[sid] = {
        "challenge_id": spec["id"],
        "created": time.time(),
        "fixtures": fixtures,
    }
    if len(_SESSIONS) > _MAX_SESSIONS:
        oldest = min(_SESSIONS, key=lambda k: _SESSIONS[k]["created"])
        _SESSIONS.pop(oldest, None)
    # Expose per-fixture `kind` so the frontend can place the right plugin
    # node (data.challenge_input vs data.challenge_input_volume). The
    # optional `judge_only` flag tells the frontend NOT to drop a canvas
    # node for this fixture — it's visible only as a preview in the
    # challenge panel so the student can see the target but cannot wire
    # it as a graph input (defeats the "just connect the answer" cheese).
    fixture_meta = [
        {"name": item["name"],
         "kind": item.get("kind", "image"),
         "judge_only": bool(item.get("judge_only", False))}
        for item in spec.get("fixtures", [])
    ]
    return {
        "session_id": sid,
        "challenge_id": spec["id"],
        "title": spec.get("title", spec["id"]),
        "prompt": spec.get("prompt", ""),
        "fixtures": fixture_meta,
        "fixture_names": [f["name"] for f in fixture_meta],  # back-compat
        "allowed_nodes": spec.get("allowed_nodes", []),
        "hint": spec.get("hint", ""),
    }


def get_fixture(session_id: str, name: str) -> np.ndarray | None:
    sess = _SESSIONS.get(session_id)
    if sess is None:
        return None
    return sess["fixtures"].get(name)


def end_session(session_id: str) -> None:
    _SESSIONS.pop(session_id, None)


# ---------------------------------------------------------------------------
# Judging.

def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation of two same-shape arrays, in [-1, 1]."""
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a = a - a.mean(); b = b - b.mean()
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float((a @ b) / (na * nb))


def _rot_ncc(a: np.ndarray, b: np.ndarray, step_deg: float = 2.0) -> float:
    """Rotation-invariant NCC: rotate `a` through 0..360° in `step_deg`
    increments and return the best plain NCC against `b`. For class-average
    challenges the student's answer can land at an arbitrary in-plane
    orientation (= whichever frame the DO loop locked onto first), so a
    pose-sensitive metric would penalise valid solutions."""
    from scipy.ndimage import rotate as _rot
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    best = -1.0
    for ang in np.arange(0.0, 360.0, float(step_deg)):
        r = _rot(a, float(ang), reshape=False, order=1, mode="constant")
        s = _ncc(r, b)
        if s > best:
            best = s
    return float(best)


def _cluster_accuracy(labels_pred: np.ndarray,
                      labels_true: np.ndarray) -> float:
    """Best-permutation clustering accuracy. Build the K×K confusion
    matrix between predicted and true labels, then pick the label
    permutation (via the Hungarian assignment) that maximises the diagonal
    — that's the fraction of frames whose "true cluster" matches the
    student's cluster, independent of how clusters are numbered. Random
    clustering → 1/K; perfect → 1.0."""
    p = np.asarray(labels_pred).ravel().astype(np.int32)
    t = np.asarray(labels_true).ravel().astype(np.int32)
    if p.shape != t.shape:
        raise ValueError(
            f"cluster_accuracy: label-count mismatch "
            f"(pred {p.shape} vs truth {t.shape})"
        )
    k = int(max(p.max(initial=0), t.max(initial=0)) + 1)
    cm = np.zeros((k, k), dtype=np.int64)
    for pi, ti in zip(p, t):
        cm[int(ti), int(pi)] += 1
    try:
        from scipy.optimize import linear_sum_assignment
        row, col = linear_sum_assignment(-cm)
        return float(cm[row, col].sum() / max(1, len(p)))
    except ImportError:
        from itertools import permutations
        best = 0
        for perm in permutations(range(k)):
            best = max(best, int(cm[np.arange(k), np.asarray(perm)].sum()))
        return float(best / max(1, len(p)))


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float32) - b.astype(np.float32)
    return float((d * d).mean())


def _multi_rot_ncc(answer_stack: np.ndarray, truth_stack: np.ndarray,
                   step_deg: float = 2.0) -> float:
    """Rotation-invariant best-permutation match for a K-class problem.
    Both inputs have shape (K, ny, nx). For each (student class i, truth
    class j) pair compute rot_ncc, then pick the permutation that maximises
    the sum. Returns the mean rot_ncc under that permutation — insensitive
    to class ordering, so the student doesn't have to produce classes in
    the "correct" order."""
    k_ans = int(answer_stack.shape[0])
    k_ref = int(truth_stack.shape[0])
    if k_ans != k_ref:
        raise ValueError(
            f"multi_rot_ncc: answer has {k_ans} classes but truth has "
            f"{k_ref} — must match"
        )
    k = k_ans
    cc = np.zeros((k, k), dtype=np.float32)
    for i in range(k):
        for j in range(k):
            cc[i, j] = _rot_ncc(answer_stack[i], truth_stack[j],
                                step_deg=step_deg)
    # Best permutation via Hungarian (SciPy). Fall back to a greedy brute
    # force if SciPy isn't available (K is small — K! for K ≤ 8 is fine).
    try:
        from scipy.optimize import linear_sum_assignment
        row, col = linear_sum_assignment(-cc)
        return float(cc[row, col].mean())
    except ImportError:
        from itertools import permutations
        best = -1.0
        for perm in permutations(range(k)):
            s = float(cc[np.arange(k), np.asarray(perm)].mean())
            if s > best:
                best = s
        return best


def judge(spec: dict, session_id: str, answer_image: np.ndarray) -> dict:
    """Score `answer_image` against the challenge's `compare_to` fixture."""
    ref_name = spec.get("compare_to", "reference")
    ref = get_fixture(session_id, ref_name)
    if ref is None:
        return {"ok": False, "pass": False, "score": None,
                "error": f"session {session_id!r} has no fixture {ref_name!r} "
                         "(start the challenge first)"}
    metric = spec.get("metric", "ncc")
    threshold = float(spec.get("pass_score", 0.95))
    # cluster_accuracy compares 1D int label arrays, not float images.
    if metric == "cluster_accuracy":
        ans = np.asarray(answer_image).ravel().astype(np.int32)
        truth = np.asarray(ref).ravel().astype(np.int32)
        if ans.shape != truth.shape:
            return {"ok": False, "pass": False, "score": None,
                    "error": (f"answer has {ans.size} labels, truth has "
                              f"{truth.size} — must match")}
        score = _cluster_accuracy(ans, truth)
        return {"ok": True, "pass": bool(score >= threshold),
                "score": float(score), "metric": metric,
                "threshold": threshold, "compared_to": ref_name}
    ans = np.asarray(answer_image, dtype=np.float32)
    if ans.shape != ref.shape:
        return {"ok": False, "pass": False, "score": None,
                "error": f"answer shape {ans.shape} != reference shape {ref.shape}"}
    if metric == "ncc":
        score = _ncc(ans, ref)
        passed = score >= threshold
    elif metric == "rot_ncc":
        score = _rot_ncc(ans, ref, step_deg=float(spec.get("rot_step_deg", 2.0)))
        passed = score >= threshold
    elif metric == "multi_rot_ncc":
        score = _multi_rot_ncc(ans, ref,
                               step_deg=float(spec.get("rot_step_deg", 2.0)))
        passed = score >= threshold
    elif metric == "neg_mse":
        # smaller MSE = better; map to a 1 / (1 + MSE) score
        score = 1.0 / (1.0 + _mse(ans, ref))
        passed = score >= threshold
    else:
        return {"ok": False, "pass": False, "score": None,
                "error": f"unknown metric {metric!r}"}
    return {"ok": True, "pass": bool(passed), "score": float(score),
            "metric": metric, "threshold": threshold,
            "compared_to": ref_name}


# ---------------------------------------------------------------------------
# Loader (parallels tutorials loader in main.py).

_BUILTIN_CHALLENGE_DIR = Path(__file__).resolve().parent / "challenges"
_USER_CHALLENGE_DIR = Path.home() / ".pyspider" / "challenges"


def load_challenges() -> tuple[list[dict], list[dict]]:
    """Return (specs, load_errors). Specs retain full JSON for the judge."""
    specs: list[dict] = []
    errors: list[dict] = []
    seen: set[str] = set()
    for d in (_BUILTIN_CHALLENGE_DIR, _USER_CHALLENGE_DIR):
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.json")):
            try:
                obj = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                errors.append({"file": str(f), "error": str(e)})
                continue
            if not (isinstance(obj, dict)
                    and isinstance(obj.get("id"), str)
                    and isinstance(obj.get("fixtures"), list)
                    and isinstance(obj.get("allowed_nodes"), list)):
                errors.append({"file": str(f),
                               "error": "missing id / fixtures / allowed_nodes"})
                continue
            if obj["id"] in seen:
                errors.append({"file": str(f),
                               "error": f"duplicate challenge id {obj['id']!r}"})
                continue
            seen.add(obj["id"])
            specs.append(obj)
    return specs, errors


def get_spec(challenge_id: str) -> dict | None:
    """Look up one spec by id (re-reads from disk every call so edits land)."""
    specs, _ = load_challenges()
    for s in specs:
        if s["id"] == challenge_id:
            return s
    return None

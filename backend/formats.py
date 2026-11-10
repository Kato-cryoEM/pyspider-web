"""Format registry for on-disk array I/O.

The rest of the app never calls format-specific readers/writers directly.
It calls ``read_array(path)`` / ``write_array(path, arr)`` which dispatch on
file extension. This is the single point where new file formats can be
plugged in — plugins call ``register_reader(".tif", my_tiff_reader)`` and
`io.load`, `io.save`, `save.py` all pick it up transparently.

Built-in support:
  * SPIDER  — .spi, .dat  (via pyspider.format.{read,write}_image)
  * MRC     — .mrc, .map, .ccp4 (self-contained reader/writer below)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

Reader = Callable[[Path], np.ndarray]
Writer = Callable[[Path, np.ndarray], None]

_READERS: dict[str, Reader] = {}
_WRITERS: dict[str, Writer] = {}


# ---- Public registry API --------------------------------------------------

def register_reader(ext: str, fn: Reader) -> None:
    """Register a reader for a file extension (e.g. ``".tif"``)."""
    _READERS[_norm(ext)] = fn


def register_writer(ext: str, fn: Writer) -> None:
    _WRITERS[_norm(ext)] = fn


def readers() -> list[str]:
    return sorted(_READERS.keys())


def writers() -> list[str]:
    return sorted(_WRITERS.keys())


def has_reader(ext: str) -> bool:
    return _norm(ext) in _READERS


def has_writer(ext: str) -> bool:
    return _norm(ext) in _WRITERS


def read_array(path: str | Path) -> np.ndarray:
    """Read an on-disk array, dispatching on file extension."""
    p = Path(path).expanduser()
    ext = _norm(p.suffix)
    if ext not in _READERS:
        raise ValueError(
            f"no reader registered for extension {ext!r} "
            f"(known: {readers()})"
        )
    return _READERS[ext](p)


def write_array(path: str | Path, data: np.ndarray) -> None:
    p = Path(path).expanduser()
    ext = _norm(p.suffix)
    if ext not in _WRITERS:
        raise ValueError(
            f"no writer registered for extension {ext!r} "
            f"(known: {writers()})"
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    _WRITERS[ext](p, np.ascontiguousarray(data, dtype=np.float32))


def _norm(ext: str) -> str:
    e = ext.lower().strip()
    if e and not e.startswith("."):
        e = "." + e
    return e


# ---- MRC/CCP4 (mode 2, float32) -------------------------------------------

def _read_mrc(path: Path) -> np.ndarray:
    """Minimal MRC/CCP4 reader.  Supports modes 0/1/2/6, auto-detects
    endianness via MACHST, skips extended header."""
    with path.open("rb") as f:
        header = f.read(1024)

    def _try(endian: str):
        try:
            nx, ny, nz, mode = np.frombuffer(header[:16], dtype=endian + "i4")
            nsymbt = int(np.frombuffer(header[92:96], dtype=endian + "i4")[0])
        except Exception:
            return None
        if not (1 <= nx <= 8192 and 1 <= ny <= 8192 and 1 <= nz <= 8192):
            return None
        if mode not in (0, 1, 2, 6):
            return None
        return int(nx), int(ny), int(nz), int(mode), nsymbt

    machst = header[212:216]
    if machst[:2] == b"\x44\x44":
        endian = "<"
    elif machst[:2] == b"\x11\x11":
        endian = ">"
    else:
        endian = "<"

    parsed = _try(endian) or _try("<") or _try(">")
    if parsed is None:
        raise ValueError("could not parse MRC header (unsupported mode / bad dims)")
    nx, ny, nz, mode, nsymbt = parsed
    dtype_map = {0: "i1", 1: "i2", 2: "f4", 6: "u2"}
    with path.open("rb") as f:
        f.seek(1024 + nsymbt)
        raw = f.read()
    data = np.frombuffer(raw, dtype=endian + dtype_map[mode])
    need = nz * ny * nx
    if data.size < need:
        raise ValueError(f"MRC data truncated: got {data.size}, need {need}")
    return data[:need].reshape(nz, ny, nx).astype(np.float32)


def _write_mrc(path: Path, data: np.ndarray) -> None:
    """Write ``data`` as MRC (mode 2, float32, little-endian)."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    if data.ndim == 2:
        ny, nx = data.shape
        nz = 1
    elif data.ndim == 3:
        nz, ny, nx = data.shape
    else:
        raise ValueError(f"MRC needs 2D or 3D data; got shape {data.shape}")

    header = bytearray(1024)
    def _i32(off: int, val: int) -> None:
        header[off:off + 4] = np.int32(val).tobytes()
    def _f32(off: int, val: float) -> None:
        header[off:off + 4] = np.float32(val).tobytes()

    _i32(  0, nx);  _i32(  4, ny);  _i32(  8, nz);  _i32( 12, 2)
    _i32( 16, 0);   _i32( 20, 0);   _i32( 24, 0)
    _i32( 28, nx);  _i32( 32, ny);  _i32( 36, nz)
    _f32( 40, nx);  _f32( 44, ny);  _f32( 48, nz)
    _f32( 52, 90);  _f32( 56, 90);  _f32( 60, 90)
    _i32( 64, 1);   _i32( 68, 2);   _i32( 72, 3)
    dmin  = float(data.min())  if data.size else 0.0
    dmax  = float(data.max())  if data.size else 0.0
    dmean = float(data.mean()) if data.size else 0.0
    _f32( 76, dmin); _f32( 80, dmax); _f32( 84, dmean)
    _i32( 88, 0);   _i32( 92, 0)
    header[208:212] = b"MAP "
    header[212:216] = bytes([0x44, 0x44, 0x00, 0x00])
    _f32(216, float(data.std()) if data.size else 0.0)
    _i32(220, 0)
    with path.open("wb") as f:
        f.write(bytes(header))
        f.write(data.tobytes())


# ---- SPIDER via pyspider.format --------------------------------------------

def _read_spider(path: Path) -> np.ndarray:
    from pyspider import read_image
    _, arr = read_image(str(path))
    return np.asarray(arr, dtype=np.float32)


def _write_spider(path: Path, data: np.ndarray) -> None:
    from pyspider import write_image
    write_image(str(path), np.ascontiguousarray(data, dtype=np.float32))


# ---- Register built-ins ---------------------------------------------------

for _ext in (".mrc", ".mrcs", ".map", ".ccp4"):
    # `.mrcs` = MRC-stack (RELION convention). Identical on-disk layout to
    # `.mrc`, just a filename convention flagging that the array is a
    # 3D stack of 2D frames rather than a 3D volume — we treat both the
    # same and let downstream nodes decide.
    register_reader(_ext, _read_mrc)
    register_writer(_ext, _write_mrc)

for _ext in (".spi", ".dat", ".spider"):
    register_reader(_ext, _read_spider)
    register_writer(_ext, _write_spider)

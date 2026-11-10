"""SPIDER image file format.

A SPIDER image, volume, or stack is an unformatted, direct-access file of
4-byte floats.  Each "record" is NX floats (NX*4 bytes).  The header occupies
LABREC = ceil(1024 / (NX*4)) records, so its total size LABBYT is always a
multiple of NX*4 and at least 1024 bytes.  Integers in the header (NX, NY,
NZ, IREC, ...) are stored as IEEE-754 floats and converted on read.

References:
  - /Applications/spider/docs/image_doc.html
  - /Applications/spider/src/lunsethdr.f
  - Frank J. et al., J. Struct. Biol. 116:190 (1996)

The first NUM_HEADER_FLOATS = 256 header positions form the "label record".
Position indexing in the SPIDER documentation and Fortran source is 1-based;
this module also offers a 1-based dictionary view through SpiderHeader.

Endianness: SPIDER files are written in the host's native byte order.  We
detect byte order on read using the standard heuristic: NZ at position 1
should be a small positive integer when interpreted correctly.
"""

from __future__ import annotations

import math
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np

# Number of floats in the label record (header).  See LUNHDR.INC: LENHDR = 260.
# Only the first 256 are documented as the standard header; positions 257-260
# are SPIDER-internal scratch.  We always write/read 256 documented positions.
NUM_HEADER_FLOATS = 256

# IFORM values per the file-format spec.
IFORM_2D_IMAGE = 1
IFORM_3D_VOLUME = 3
IFORM_2D_FOURIER_ODD = -11
IFORM_2D_FOURIER_EVEN = -12
IFORM_3D_FOURIER_ODD = -21
IFORM_3D_FOURIER_EVEN = -22

_FOURIER_IFORMS = frozenset(
    (IFORM_2D_FOURIER_ODD, IFORM_2D_FOURIER_EVEN,
     IFORM_3D_FOURIER_ODD, IFORM_3D_FOURIER_EVEN)
)


def _fourier_iform(real_nx: int, ndim: int) -> int:
    """IFORM tag for the Fourier file produced from a real image of the given
    NX and dimensionality (2 or 3).
    """
    odd = (real_nx % 2) != 0
    if ndim == 2:
        return IFORM_2D_FOURIER_ODD if odd else IFORM_2D_FOURIER_EVEN
    if ndim == 3:
        return IFORM_3D_FOURIER_ODD if odd else IFORM_3D_FOURIER_EVEN
    raise ValueError(f"ndim must be 2 or 3, got {ndim}")


def _packed_nx(real_nx: int) -> int:
    """File-side row length (NNNN) for a Fourier file derived from real_nx.

    Matches `NX-MOD(-IFORM,10)` inverse in src/four1.f:109:
      odd  -> NNNN = real_nx + 1
      even -> NNNN = real_nx + 2
    """
    return real_nx + 1 if (real_nx % 2) else real_nx + 2


def _real_nx_from_iform(packed_nx: int, iform: int) -> int:
    """Recover the real-space NX from the Fourier file's NX field (NNNN)."""
    if iform not in _FOURIER_IFORMS:
        raise ValueError(f"iform {iform} is not a Fourier IFORM")
    # NXO = NX - MOD(-IFORM, 10) in src/four1.f:109
    return packed_nx - (-iform % 10)

_DTYPE_LE = np.dtype("<f4")
_DTYPE_BE = np.dtype(">f4")


def _label_layout(nx: int) -> tuple[int, int, int]:
    """Return (LENBYT, LABREC, LABBYT) for an image with row length NX.

    Matches the formula in /Applications/spider/docs/image_doc.html and
    in src/lunsethdr.f.
    """
    if nx <= 0:
        raise ValueError(f"NX must be positive, got {nx}")
    lenbyt = nx * 4
    labrec = 1024 // lenbyt
    if 1024 % lenbyt != 0:
        labrec += 1
    labbyt = labrec * lenbyt
    return lenbyt, labrec, labbyt


@dataclass
class SpiderHeader:
    """SPIDER image header, decoded.

    Field names mirror the documented SPIDER header positions so a Fortran
    reader can locate the same value.  Lengths in bytes/records:
        LENBYT = NX * 4
        LABREC = ceil(1024 / LENBYT)
        LABBYT = LABREC * LENBYT
    """

    # Geometry
    nx: int
    ny: int
    nz: int = 1
    iform: int = IFORM_2D_IMAGE

    # Statistics (set when populated by SPIDER's FS or by our writer)
    imami: int = 0
    fmax: float = 0.0
    fmin: float = 0.0
    av: float = 0.0
    sig: float = -1.0

    # Stack
    istack: int = 0   # 0 simple image; >0 stack; <0 indexed stack (MAXINDX)
    maxim: int = 0    # highest used image in stack
    imgnum: int = 0   # this image's position in a stack
    lastindx: int = 0

    # Tilt / projection angles & translations
    iangle: int = 0
    phi: float = 0.0
    theta: float = 0.0
    gamma: float = 0.0
    xoff: float = 0.0
    yoff: float = 0.0
    zoff: float = 0.0
    scale: float = 1.0

    # Additional rotation angles (slots 31..37)
    kangle: int = 0
    phi1: float = 0.0
    theta1: float = 0.0
    psi1: float = 0.0
    phi2: float = 0.0
    theta2: float = 0.0
    psi2: float = 0.0

    # Acquisition
    pixsiz: float = 0.0   # pos 38
    ev: float = 0.0       # pos 39, electron voltage
    proj: int = 0         # pos 40
    mic: int = 0          # pos 41
    num: int = 0          # pos 42
    glonum: int = 0       # pos 43

    # Provenance text (label slots are zero-padded with spaces)
    cdat: str = ""        # pos 212-214, 11 chars
    ctim: str = ""        # pos 215-216, 8 chars
    ctit: str = ""        # pos 217-256, 160 chars

    # Raw label record kept around for round-trip fidelity.
    raw: np.ndarray = field(default_factory=lambda: np.zeros(NUM_HEADER_FLOATS, dtype=np.float32))

    # ---- derived ----
    @property
    def lenbyt(self) -> int:
        return self.nx * 4

    @property
    def labrec(self) -> int:
        return _label_layout(self.nx)[1]

    @property
    def labbyt(self) -> int:
        return _label_layout(self.nx)[2]

    @property
    def irec(self) -> int:
        """Total records (header + data) per image.  Matches header[3]."""
        return self.labrec + self.ny * self.nz

    @property
    def is_stack(self) -> bool:
        return self.istack > 0

    # ---- 1-based dictionary view ----
    def as_position_dict(self) -> dict[int, float]:
        """Return the documented header positions as {pos: value}, 1-based."""
        return {i + 1: float(self.raw[i]) for i in range(NUM_HEADER_FLOATS)}


def _detect_endianness(first_bytes: bytes) -> str:
    """Pick byte order using NZ (position 1) and NY (position 2) sanity.

    SPIDER never produces a file with non-positive NZ or NY, and these
    fields are reasonably small.  If both endianness interpretations look
    plausible, prefer native because that is what SPIDER itself writes.
    """
    le_nz, le_ny = struct.unpack("<ff", first_bytes[:8])
    be_nz, be_ny = struct.unpack(">ff", first_bytes[:8])

    def looks_valid(nz: float, ny: float) -> bool:
        return (
            nz == int(nz)
            and ny == int(ny)
            and 1 <= int(nz) <= 100_000
            and 1 <= int(ny) <= 1_000_000
        )

    le_ok = looks_valid(le_nz, le_ny)
    be_ok = looks_valid(be_nz, be_ny)
    if le_ok and not be_ok:
        return "<"
    if be_ok and not le_ok:
        return ">"
    # Ambiguous or both fail: default to native, then little-endian.
    return "=" if sys.byteorder == "little" else ">"


def _decode_label_text(raw: np.ndarray, start_pos: int, n_floats: int) -> str:
    """Decode a CHARACTER buffer stored in n_floats consecutive header floats.

    SPIDER stores character data by EQUIVALENCE-ing it onto the floating point
    label record, so we recover bytes from the float32 buffer.
    """
    section = raw[start_pos - 1 : start_pos - 1 + n_floats]
    raw_bytes = section.astype(np.float32).tobytes()
    return raw_bytes.rstrip(b"\x00").decode("latin-1", errors="replace").rstrip()


def _encode_label_text(text: str, n_floats: int) -> np.ndarray:
    """Pack a string into n_floats float32 slots, space-padded."""
    target_len = n_floats * 4
    encoded = text.encode("latin-1", errors="replace")
    if len(encoded) > target_len:
        encoded = encoded[:target_len]
    encoded = encoded + b" " * (target_len - len(encoded))
    return np.frombuffer(encoded, dtype=np.float32).copy()


def _header_from_raw(raw: np.ndarray) -> SpiderHeader:
    """Decode the documented fields from a 256-float header array."""
    h = raw
    hdr = SpiderHeader(
        nx=int(h[11]),
        ny=int(h[1]),
        nz=int(h[0]),
        iform=int(h[4]),
        imami=int(h[5]),
        fmax=float(h[6]),
        fmin=float(h[7]),
        av=float(h[8]),
        sig=float(h[9]),
        iangle=int(h[13]),
        phi=float(h[14]),
        theta=float(h[15]),
        gamma=float(h[16]),
        xoff=float(h[17]),
        yoff=float(h[18]),
        zoff=float(h[19]),
        scale=float(h[20]),
        istack=int(h[23]),
        maxim=int(h[25]),
        imgnum=int(h[26]),
        lastindx=int(h[27]),
        kangle=int(h[30]),
        phi1=float(h[31]),
        theta1=float(h[32]),
        psi1=float(h[33]),
        phi2=float(h[34]),
        theta2=float(h[35]),
        psi2=float(h[36]),
        pixsiz=float(h[37]),
        ev=float(h[38]),
        proj=int(h[39]),
        mic=int(h[40]),
        num=int(h[41]),
        glonum=int(h[42]),
        cdat=_decode_label_text(h, 212, 3),
        ctim=_decode_label_text(h, 215, 2),
        ctit=_decode_label_text(h, 217, 40),
        raw=h.copy(),
    )
    return hdr


def _raw_from_header(hdr: SpiderHeader) -> np.ndarray:
    """Build the 256-float label record from a SpiderHeader."""
    h = np.zeros(NUM_HEADER_FLOATS, dtype=np.float32)
    _, labrec, labbyt = _label_layout(hdr.nx)
    h[0] = hdr.nz
    h[1] = hdr.ny
    h[2] = labrec + hdr.ny * hdr.nz                # IREC
    h[4] = hdr.iform
    h[5] = hdr.imami
    h[6] = hdr.fmax
    h[7] = hdr.fmin
    h[8] = hdr.av
    h[9] = hdr.sig
    h[11] = hdr.nx
    h[12] = labrec
    h[13] = hdr.iangle
    h[14] = hdr.phi
    h[15] = hdr.theta
    h[16] = hdr.gamma
    h[17] = hdr.xoff
    h[18] = hdr.yoff
    h[19] = hdr.zoff
    h[20] = hdr.scale
    h[21] = labbyt
    h[22] = hdr.nx * 4                              # LENBYT
    h[23] = hdr.istack
    h[25] = hdr.maxim
    h[26] = hdr.imgnum
    h[27] = hdr.lastindx
    h[30] = hdr.kangle
    h[31] = hdr.phi1
    h[32] = hdr.theta1
    h[33] = hdr.psi1
    h[34] = hdr.phi2
    h[35] = hdr.theta2
    h[36] = hdr.psi2
    h[37] = hdr.pixsiz
    h[38] = hdr.ev
    h[39] = hdr.proj
    h[40] = hdr.mic
    h[41] = hdr.num
    h[42] = hdr.glonum
    h[211:214] = _encode_label_text(hdr.cdat, 3)
    h[214:216] = _encode_label_text(hdr.ctim, 2)
    h[216:256] = _encode_label_text(hdr.ctit, 40)
    return h


def read_header(path: str | Path) -> SpiderHeader:
    """Read only the header from a SPIDER file."""
    path = Path(path)
    with path.open("rb") as f:
        first = f.read(8)
        if len(first) < 8:
            raise ValueError(f"{path}: file too short to contain a SPIDER header")
        order = _detect_endianness(first)
        dtype = np.dtype("=f4") if order == "=" else np.dtype(order + "f4")
        f.seek(0)
        nx_bytes = f.read(48 * 4)  # need at least up to NX (pos 12)
        peek = np.frombuffer(nx_bytes, dtype=dtype)
        nx = int(peek[11])
        if nx <= 0:
            raise ValueError(f"{path}: header reports NX={nx}")
        _, _, labbyt = _label_layout(nx)
        f.seek(0)
        raw = np.frombuffer(f.read(labbyt), dtype=dtype, count=NUM_HEADER_FLOATS)
        # Always return native-endian floats so callers can compare freely.
        raw = raw.astype(np.float32)
        return _header_from_raw(raw)


def read_image(path: str | Path) -> tuple[SpiderHeader, np.ndarray]:
    """Read a simple SPIDER image or volume into a NumPy array.

    Returns (header, data) with data shape (NY, NX) for 2D or (NZ, NY, NX)
    for 3D.  Stacks must be read via ``read_stack``.
    """
    path = Path(path)
    with path.open("rb") as f:
        head = f.read(48 * 4)
        order = _detect_endianness(head[:8])
        dtype = np.dtype("=f4") if order == "=" else np.dtype(order + "f4")
        peek = np.frombuffer(head, dtype=dtype)
        nx = int(peek[11])
        _, _, labbyt = _label_layout(nx)
        f.seek(0)
        raw = np.frombuffer(f.read(labbyt), dtype=dtype, count=NUM_HEADER_FLOATS).astype(np.float32)
        hdr = _header_from_raw(raw)
        if hdr.is_stack:
            raise ValueError(
                f"{path}: is a SPIDER stack (ISTACK={hdr.istack}); use read_stack"
            )
        n_floats = hdr.ny * hdr.nx * max(hdr.nz, 1)
        data_bytes = f.read(n_floats * 4)
        if len(data_bytes) < n_floats * 4:
            raise ValueError(
                f"{path}: expected {n_floats * 4} data bytes, got {len(data_bytes)}"
            )
        data = np.frombuffer(data_bytes, dtype=dtype).astype(np.float32, copy=True)
        if hdr.nz > 1:
            data = data.reshape(hdr.nz, hdr.ny, hdr.nx)
        else:
            data = data.reshape(hdr.ny, hdr.nx)
        return hdr, data


def read_stack(path: str | Path) -> Iterator[tuple[SpiderHeader, np.ndarray]]:
    """Yield (header, data) tuples for each image in a SPIDER stack."""
    path = Path(path)
    with path.open("rb") as f:
        head = f.read(48 * 4)
        order = _detect_endianness(head[:8])
        dtype = np.dtype("=f4") if order == "=" else np.dtype(order + "f4")
        peek = np.frombuffer(head, dtype=dtype)
        nx = int(peek[11])
        _, labrec, labbyt = _label_layout(nx)
        f.seek(0)
        raw = np.frombuffer(f.read(labbyt), dtype=dtype, count=NUM_HEADER_FLOATS).astype(np.float32)
        overall = _header_from_raw(raw)
        if not overall.is_stack:
            raise ValueError(f"{path}: not a SPIDER stack (ISTACK={overall.istack})")
        ny = overall.ny
        nz = max(overall.nz, 1)
        # An indexed stack (ISTACK < 0) precedes its images with an index
        # block of INDXREC records; we do not support indexed stacks here.
        if overall.istack < 0:
            raise NotImplementedError(
                f"{path}: indexed SPIDER stacks are not supported yet"
            )
        per_image_floats = labrec * nx + ny * nx * nz
        for _ in range(overall.maxim):
            img_raw = np.frombuffer(
                f.read(labbyt), dtype=dtype, count=NUM_HEADER_FLOATS
            ).astype(np.float32)
            img_hdr = _header_from_raw(img_raw)
            n_data_floats = ny * nx * nz
            data = np.frombuffer(f.read(n_data_floats * 4), dtype=dtype).astype(
                np.float32, copy=True
            )
            if nz > 1:
                data = data.reshape(nz, ny, nx)
            else:
                data = data.reshape(ny, nx)
            yield img_hdr, data


def write_image(
    path: str | Path,
    data: np.ndarray,
    header: SpiderHeader | None = None,
    *,
    compute_stats: bool = True,
    title: str = "",
) -> SpiderHeader:
    """Write a 2D image or 3D volume in SPIDER format.

    If ``header`` is None, a fresh one is built from ``data``.  Statistics
    (FMAX, FMIN, AV, SIG, IMAMI) are computed automatically unless
    compute_stats is False.
    """
    path = Path(path)
    data = np.ascontiguousarray(data, dtype=np.float32)
    if data.ndim == 2:
        ny, nx = data.shape
        nz = 1
        iform = IFORM_2D_IMAGE
    elif data.ndim == 3:
        nz, ny, nx = data.shape
        iform = IFORM_3D_VOLUME
    else:
        raise ValueError(f"data must be 2D or 3D, got shape {data.shape}")

    if header is None:
        now = time.localtime()
        cdat = time.strftime("%d-%b-%Y", now).upper()  # 11 chars: DD-MMM-YYYY
        ctim = time.strftime("%H:%M:%S", now)          # 8 chars
        header = SpiderHeader(
            nx=nx, ny=ny, nz=nz, iform=iform, cdat=cdat, ctim=ctim, ctit=title
        )
    else:
        # Adopt incoming geometry from data.
        header = SpiderHeader(**{**header.__dict__, "nx": nx, "ny": ny, "nz": nz, "iform": iform})

    if compute_stats and data.size > 0:
        header.fmax = float(data.max())
        header.fmin = float(data.min())
        header.av = float(data.mean())
        header.sig = float(data.std(ddof=0))
        header.imami = 1

    raw = _raw_from_header(header)
    header.raw = raw

    with path.open("wb") as f:
        f.write(raw.tobytes())
        # Pad remaining bytes in the label area with zeros.
        extra = header.labbyt - raw.nbytes
        if extra > 0:
            f.write(b"\x00" * extra)
        f.write(data.tobytes())
    return header


def read_fourier(path: str | Path) -> tuple[SpiderHeader, np.ndarray]:
    """Read a SPIDER Fourier file (IFORM -11/-12/-21/-22) as complex64.

    Returns (header, spectrum) where ``spectrum`` has shape (NY, NX/2+1)
    for 2D or (NZ, NY, NX/2+1) for 3D, indexed as ``numpy.fft.rfft2`` /
    ``rfftn`` would produce: axis 0 in DC-first FFT order, the last axis
    in rfft-half order from KX=0 to KX=NX/2.

    Sign convention: SPIDER's FT uses ``exp(+2πi k·x / N)`` (FFTW backward
    sign) while ``numpy.fft.rfft2`` uses ``exp(-2πi k·x / N)``.  We
    complex-conjugate on read so the returned array matches NumPy's
    convention and can be fed directly to ``numpy.fft.irfft2``.

    The returned ``header.nx`` is the *real-space* NX (recovered from the
    Fourier IFORM and the file's NNNN).
    """
    path = Path(path)
    with path.open("rb") as f:
        head = f.read(48 * 4)
        order = _detect_endianness(head[:8])
        dtype_in = np.dtype("=f4") if order == "=" else np.dtype(order + "f4")
        peek = np.frombuffer(head, dtype=dtype_in)
        packed_nx = int(peek[11])
        iform = int(peek[4])
        if iform not in _FOURIER_IFORMS:
            raise ValueError(
                f"{path}: IFORM={iform} is not a Fourier format "
                f"(expected one of {sorted(_FOURIER_IFORMS)})"
            )
        _, _, labbyt = _label_layout(packed_nx)
        f.seek(0)
        raw = np.frombuffer(
            f.read(labbyt), dtype=dtype_in, count=NUM_HEADER_FLOATS
        ).astype(np.float32)
        hdr = _header_from_raw(raw)
        # Decode dimensions: header stores packed NX, real NY, real NZ.
        real_nx = _real_nx_from_iform(packed_nx, iform)
        ny = hdr.ny
        nz = max(hdr.nz, 1)
        n_floats = packed_nx * ny * nz
        data_bytes = f.read(n_floats * 4)
        if len(data_bytes) < n_floats * 4:
            raise ValueError(
                f"{path}: expected {n_floats * 4} data bytes, got {len(data_bytes)}"
            )
        floats = np.frombuffer(data_bytes, dtype=dtype_in).astype(np.float32, copy=True)
        # The packed format is exactly Re,Im,Re,Im,... per row, the same
        # memory layout as numpy.complex64 (real then imaginary).  View it.
        cmplx = floats.view(np.complex64)
        half_nx = packed_nx // 2
        if iform in (IFORM_3D_FOURIER_ODD, IFORM_3D_FOURIER_EVEN):
            cmplx = cmplx.reshape(nz, ny, half_nx)
        else:
            cmplx = cmplx.reshape(ny, half_nx)
        # SPIDER uses the opposite sign convention from numpy.fft.rfft2.
        cmplx = np.conjugate(cmplx)
        # Patch the header so the caller sees real-space NX, not NNNN.
        hdr.nx = real_nx
        return hdr, cmplx


def write_fourier(
    path: str | Path,
    spectrum: np.ndarray,
    *,
    real_nx: int | None = None,
    title: str = "",
) -> SpiderHeader:
    """Write a complex spectrum as a SPIDER Fourier file.

    ``spectrum`` should be the output of ``numpy.fft.rfft2`` (2D, shape
    ``(NY, NX/2+1)``) or ``rfftn`` (3D, shape ``(NZ, NY, NX/2+1)``).  The
    real-space NX is inferred from the last-axis length:

      ``NX_half = NX/2 + 1``  →  real NX is either ``2*(NX_half-1)`` (even)
      or ``2*(NX_half-1)+1`` (odd).  Pass ``real_nx`` explicitly to
      disambiguate.

    The spectrum is interpreted in NumPy's sign convention and conjugated
    on the way out to match SPIDER's ``exp(+i)`` convention.  See the
    matching note on ``read_fourier``.
    """
    path = Path(path)
    spectrum = np.ascontiguousarray(spectrum, dtype=np.complex64)
    if spectrum.ndim == 2:
        ny, half_nx = spectrum.shape
        nz = 1
        ndim = 2
    elif spectrum.ndim == 3:
        nz, ny, half_nx = spectrum.shape
        ndim = 3
    else:
        raise ValueError(f"spectrum must be 2D or 3D, got shape {spectrum.shape}")

    if real_nx is None:
        # Default to even — by far the most common case in cryo-EM.
        real_nx = 2 * (half_nx - 1)
    elif (real_nx + 2) // 2 != half_nx and (real_nx + 1) // 2 != half_nx:
        # ``(real_nx + 2) // 2 == half_nx`` when real_nx is even.
        # ``(real_nx + 1) // 2 == half_nx`` when real_nx is odd.
        raise ValueError(
            f"real_nx={real_nx} does not match spectrum half size {half_nx}"
        )

    packed_nx = _packed_nx(real_nx)
    if packed_nx // 2 != half_nx:
        raise ValueError(
            f"packed_nx ({packed_nx}) and half size ({half_nx}) disagree; "
            f"check real_nx={real_nx}"
        )
    iform = _fourier_iform(real_nx, ndim)

    now = time.localtime()
    cdat = time.strftime("%d-%b-%Y", now).upper()
    ctim = time.strftime("%H:%M:%S", now)
    header = SpiderHeader(
        nx=packed_nx,  # the *file's* NX is NNNN, per the spec
        ny=ny,
        nz=nz,
        iform=iform,
        cdat=cdat,
        ctim=ctim,
        ctit=title,
    )

    raw = _raw_from_header(header)
    header.raw = raw

    # Emit as packed floats: viewing complex64 as float32 gives Re,Im,Re,Im...
    # Conjugate first so SPIDER's exp(+i...) convention sees the spectrum we
    # describe in NumPy's exp(-i...) convention.  This is the inverse of
    # the conjugation in read_fourier.
    floats = np.ascontiguousarray(np.conjugate(spectrum), dtype=np.complex64).view(np.float32)
    with path.open("wb") as f:
        f.write(raw.tobytes())
        extra = header.labbyt - raw.nbytes
        if extra > 0:
            f.write(b"\x00" * extra)
        f.write(floats.tobytes())

    # Return a header presenting *real-space* NX so callers see the same
    # semantics as read_fourier.
    header.nx = real_nx
    return header


def inspect(path: str | Path) -> str:
    """Return a human-readable summary of a SPIDER file's header."""
    hdr = read_header(path)
    kind = {
        IFORM_2D_IMAGE: "2D image",
        IFORM_3D_VOLUME: "3D volume",
        IFORM_2D_FOURIER_ODD: "2D Fourier (odd)",
        IFORM_2D_FOURIER_EVEN: "2D Fourier (even)",
        IFORM_3D_FOURIER_ODD: "3D Fourier (odd)",
        IFORM_3D_FOURIER_EVEN: "3D Fourier (even)",
    }.get(hdr.iform, f"unknown (iform={hdr.iform})")
    lines = [
        f"{path}",
        f"  kind:       {kind}",
        f"  geometry:   NX={hdr.nx}  NY={hdr.ny}  NZ={hdr.nz}",
        f"  header:     LABREC={hdr.labrec}  LABBYT={hdr.labbyt}  LENBYT={hdr.lenbyt}",
        f"  stack:      ISTACK={hdr.istack}  MAXIM={hdr.maxim}  IMGNUM={hdr.imgnum}",
    ]
    if hdr.imami:
        lines.append(
            f"  stats:      min={hdr.fmin:g}  max={hdr.fmax:g}  "
            f"av={hdr.av:g}  sig={hdr.sig:g}"
        )
    else:
        lines.append("  stats:      not computed")
    if hdr.cdat or hdr.ctim:
        lines.append(f"  created:    {hdr.cdat} {hdr.ctim}")
    if hdr.ctit.strip():
        lines.append(f"  title:      {hdr.ctit.strip()}")
    return "\n".join(lines)

"""SPIDER manual excerpt loader.

Reads the local SPIDER man-page HTML tree (default:
``/Applications/spider/docs/man/``) and pulls just the short
reference-level sections needed for the node inspector:

  * PURPOSE — a 1–3 sentence factual summary of what the command does.
  * SEE ALSO — a list of ``(cmd_name, short_description)`` cross-refs.

The rest of the manual (USAGE, NOTES, EXAMPLES, etc.) is deliberately
left in the browser man pages; we only surface the reference stubs so
the node UI stays uncluttered.

Extraction is done with tolerant regex over the well-known SPIDER
manual layout — the HTML is machine-generated and consistent across
600+ pages, so we don't need a full parser.
"""

from __future__ import annotations

import html as _html
import os
import re
from dataclasses import dataclass
from pathlib import Path


def _resolve_default_man_dir() -> Path:
    """Locate the SPIDER man-page HTML tree.

    Resolution order:
      1. ``$SPIDER_MAN_DIR`` env var, if set to an existing directory.
      2. ``<project_root>/docs/man`` (i.e. next to the backend package).
      3. ``/Applications/spider/docs/man`` for the original dev machine.

    Returns the first hit. If none exist the returned Path may not exist,
    which the loader treats as "no manual available" (nodes still work,
    they just don't carry a SPIDER excerpt).
    """
    env = os.environ.get("SPIDER_MAN_DIR")
    if env and Path(env).is_dir():
        return Path(env)
    # backend/man.py -> backend/ -> project_root/
    here = Path(__file__).resolve().parent.parent
    local = here / "docs" / "man"
    if local.is_dir():
        return local
    return Path("/Applications/spider/docs/man")


DEFAULT_MAN_DIR = _resolve_default_man_dir()


@dataclass
class ManualEntry:
    cmd: str
    purpose: str
    see_also: list[tuple[str, str]]  # (cmd, one-line desc)

    def as_text(self) -> str:
        parts = []
        if self.purpose:
            parts.append(self.purpose)
        if self.see_also:
            refs = ", ".join(f"{c} ({d})" if d else c
                              for c, d in self.see_also)
            parts.append("See also: " + refs)
        return "\n\n".join(parts)


# --- HTML → plain text helpers --------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_EXAMPLE_LINK_RE = re.compile(r"&nbsp;?\s*<a[^>]*>Example</a>\.?",
                              flags=re.IGNORECASE)


def _strip_html(chunk: str) -> str:
    """Turn an HTML fragment into a single normalised whitespace string."""
    chunk = _EXAMPLE_LINK_RE.sub("", chunk)
    text = _TAG_RE.sub(" ", chunk)
    text = _html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


# The SPIDER man pages have this consistent skeleton:
#     <dl><dt><strong>PURPOSE</strong> ... <dd>...</dl>
#     <dl><dt><strong>SEE ALSO</strong> ... <dd><table>...</table></dl>

_PURPOSE_RE = re.compile(
    r"<dt>\s*<strong>\s*PURPOSE\s*</strong>.*?<dd>(?P<body>.*?)</dl>",
    flags=re.IGNORECASE | re.DOTALL,
)
_SEEALSO_RE = re.compile(
    r"<dt>\s*<strong>\s*SEE\s*ALSO\s*</strong>.*?<dd>(?P<body>.*?)</dl>",
    flags=re.IGNORECASE | re.DOTALL,
)
_SEEALSO_ROW_RE = re.compile(
    r"<a[^>]*>\s*<strong>\s*(?P<cmd>[^<]+?)\s*</strong>\s*</a>\s*"
    r"</td>\s*<td>\s*\[?(?P<desc>[^\]<]*?)\]?\s*</td>",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_page(html_text: str, cmd: str = "") -> ManualEntry:
    m = _PURPOSE_RE.search(html_text)
    purpose = _strip_html(m.group("body")) if m else ""

    see_also: list[tuple[str, str]] = []
    m = _SEEALSO_RE.search(html_text)
    if m:
        for row in _SEEALSO_ROW_RE.finditer(m.group("body")):
            name = _strip_html(row.group("cmd"))
            desc = _strip_html(row.group("desc"))
            see_also.append((name, desc))

    return ManualEntry(cmd=cmd, purpose=purpose, see_also=see_also)


class ManualCatalog:
    """Lazy read-once cache of SPIDER man pages."""

    def __init__(self, root: Path = DEFAULT_MAN_DIR):
        self.root = Path(root)
        self._cache: dict[str, ManualEntry | None] = {}

    def get(self, cmd: str) -> ManualEntry | None:
        """Look up a manual entry for a SPIDER command like 'CC' or 'AP SH'.

        The lookup normalises the command to the SPIDER man-page filename
        stem (lowercase, no spaces): 'AP SH' → 'apsh', 'BP 3F' → 'bp3f'.
        Returns None if the page is not found or has no Purpose section.
        """
        cmd_norm = cmd.strip()
        if cmd_norm in self._cache:
            return self._cache[cmd_norm]

        stem = cmd_norm.lower().replace(" ", "").replace("-", "")
        path = self.root / f"{stem}.html"
        if not path.exists():
            self._cache[cmd_norm] = None
            return None

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            self._cache[cmd_norm] = None
            return None

        entry = parse_page(text, cmd=cmd_norm)
        # If the Purpose parse failed, treat as missing.
        if not entry.purpose:
            self._cache[cmd_norm] = None
            return None
        self._cache[cmd_norm] = entry
        return entry


MANUAL = ManualCatalog()

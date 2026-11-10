"""CLI entrypoint — `pyspider-web` on your PATH after `pip install`.

Starts uvicorn serving `backend.main:app` on 127.0.0.1:8000 by default,
opens the browser, prints the URL. No extra dependencies at runtime
beyond what pyproject.toml declares.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="pyspider-web",
        description="Visual node editor for cryo-EM image processing.",
    )
    p.add_argument("--host", default="127.0.0.1",
                   help="Bind address (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8000,
                   help="Port (default: 8000)")
    p.add_argument("--no-browser", action="store_true",
                   help="Do not auto-open a browser window on startup.")
    p.add_argument("--reload", action="store_true",
                   help="Enable uvicorn auto-reload (dev mode).")
    args = p.parse_args(argv)

    # Sanity: make sure the packaged UI exists — either as backend/_static/
    # (installed wheel) or frontend/dist/ (source checkout).
    from backend.main import _FRONTEND_DIST
    if not _FRONTEND_DIST.is_dir():
        print(f"[pyspider-web] warning: UI directory {_FRONTEND_DIST} not "
              f"found — API will start but the / page will 404.\n"
              f"  Build with:   cd frontend && npm install && npm run build",
              file=sys.stderr)

    url = f"http://{args.host}:{args.port}/"
    print(f"[pyspider-web] starting on {url}")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    import uvicorn
    uvicorn.run("backend.main:app", host=args.host, port=args.port,
                reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())

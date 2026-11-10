# Installation guide (English)

Two install paths. Pick based on how you want to use pyspider.

| Path | Who it's for | Needs |
|---|---|---|
| **A. Web build (Pyodide)** | Most students / instructors — just want to click through challenges and tutorials. | A modern browser only. |
| **B. Local server (FastAPI)** | Developers, custom plugins, large datasets, optional CUDA. | Python 3.11+, Node 20+. |

The sections below show macOS and Windows instructions side-by-side.
Japanese version: [INSTALL.ja.md](INSTALL.ja.md).

---

## A. Web build (Pyodide) — easiest path

### A-1. Serve from a built release

Assuming you already have a `pyspider_web_dist/frontend/dist/` directory
(either downloaded or produced by `npm run build`), you just need to
serve the directory with any static HTTP server.

#### macOS

```sh
cd path/to/pyspider_web_dist/frontend/dist
python3 -m http.server 8080
```

Open http://127.0.0.1:8080/ in Safari / Chrome / Firefox.

Python 3 is pre-installed on recent macOS; if not, install it from
https://www.python.org/downloads/mac-osx/.

#### Windows

```powershell
cd path\to\pyspider_web_dist\frontend\dist
python -m http.server 8080
```

Open http://127.0.0.1:8080/ in Edge / Chrome / Firefox.

If `python` is missing: install from https://www.python.org/downloads/windows/
and tick **"Add python.exe to PATH"** during setup.

### A-2. First-run notes

- The page downloads Pyodide + numpy / scipy / Pillow / scikit-image on
  first load (~45 MB total). The browser caches them, so subsequent loads
  are fast.
- After the boot log reaches `ready`, the full UI appears.
- No files are written to disk from the browser (Pyodide runs in a
  sandboxed virtual filesystem).

### A-3. Building from source

If you cloned the repo without a pre-built `dist/`:

```sh
# Both platforms. Needs Node 20+.
cd frontend
npm install
npm run build
cd ..
bash pyodide_poc/build_bundle.sh   # creates frontend/dist/bundle.zip
```

On Windows, run the shell script from Git Bash or WSL — or do it manually:

```powershell
Remove-Item -Force frontend\dist\bundle.zip -ErrorAction SilentlyContinue
# zip backend + pyspider into frontend\dist\bundle.zip
powershell -Command "Compress-Archive -Path backend,pyspider -DestinationPath frontend\dist\bundle.zip -Force"
```

---

## B. Local server (FastAPI) — development mode

### B-1. Install Python & Node

#### macOS

```sh
# Via Homebrew
brew install python@3.12 node@20
```

#### Windows

- Python: https://www.python.org/downloads/windows/ (tick "Add to PATH")
- Node: https://nodejs.org/ (LTS)

Verify in a new terminal:

```sh
python --version    # should report 3.11+
node --version      # should report 20.x+
npm --version
```

### B-2. Install Python dependencies

```sh
cd path/to/pyspider_web_dist
python -m venv .venv

# Activate venv:
#   macOS / Linux
source .venv/bin/activate
#   Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### B-3. Start the backend

```sh
uvicorn backend.main:app --reload --port 8000
```

Keep this terminal open.

### B-4. Start the frontend (dev mode)

Open a second terminal:

```sh
cd path/to/pyspider_web_dist/frontend
npm install        # first time only
npm run dev
```

Open http://127.0.0.1:5173/ in your browser.

### B-5. Optional — CUDA acceleration

In the backend venv:

```sh
pip install cupy-cuda12x    # CUDA 12
# or
pip install cupy-cuda11x    # CUDA 11
```

Switch backend at runtime from the UI (top-right of the canvas).

---

## Troubleshooting

### `python` command not found (Windows)

The installer didn't add Python to PATH. Rerun the installer and tick
**"Add python.exe to PATH"**, or manually add
`C:\Users\<you>\AppData\Local\Programs\Python\Python312\` to the system PATH.

### `permission denied` from `python3 -m http.server` (macOS)

You picked a port under 1024. Use 8080 / 8000 / 5173 instead.

### Slow first load in the browser

Normal — Pyodide + packages are ~45 MB. Subsequent loads use the cache.

### "worker not initialised" error (Pyodide mode)

Hard-refresh the page (⌘-Shift-R / Ctrl-Shift-R). The worker failed to
load — usually because the server didn't serve `pyodideWorker.js` or
`bundle.zip`. Check both URLs return 200:
- `http://127.0.0.1:8080/pyodideWorker.js`
- `http://127.0.0.1:8080/bundle.zip`

### `npm install` fails on Windows

Delete `frontend\node_modules` and `frontend\package-lock.json`, then rerun
`npm install`. Also ensure Node 20+ (older versions miss required APIs).

### CORS errors in local-server mode

Check that the backend is running on port 8000 and the frontend on 5173;
Vite's proxy is wired for those exact ports. If you must change them,
edit `frontend/vite.config.ts`.

---

## Where files live

| | macOS | Windows |
|---|---|---|
| App directory | `/Applications/pyspider_web_dist/` or anywhere you like | anywhere; keep the path short to avoid long-path issues |
| Python venv | `.venv/` inside the app directory | `.venv\` inside the app directory |
| Frontend build | `frontend/dist/` | `frontend\dist\` |
| Backend static mount | `backend/_static/` (FastAPI serves it as `/`) | same |
| Browser cache | your browser's regular cache | same |

---

## Uninstall

### Web build
Delete the directory. That's it — the browser cache automatically cleans
up in a few weeks, or clear it manually from the browser's settings.

### Local server
```sh
# Deactivate and remove the venv
deactivate
rm -rf .venv            # macOS / Linux
Remove-Item -Recurse -Force .venv   # Windows

# Optionally remove node_modules
rm -rf frontend/node_modules
```

No files are written outside the app directory (except the browser
cache). No system-wide registry entries, no services.

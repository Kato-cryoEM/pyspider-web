# pyspider · Node-based cryo-EM playground

Hands-on, browser-based cryo-EM image-processing environment built on top
of the SPIDER operator set. Compose nodes on a canvas (React Flow), edit
parameters, and run the graph — the engine executes real numpy / scipy
code and renders previews of every intermediate result.

Two deployment modes ship from the same codebase:

- **Web build (Pyodide)** — one directory of static files (`frontend/dist`).
  Open it with any HTTP server and the full Python backend runs **inside the
  browser** via a Pyodide Web Worker. No install, no server process.
- **Local server (FastAPI)** — classic client/server setup for development,
  larger datasets, and GPU work via CuPy.

Both modes drive the same UI and the same node catalog.

## Features

- **Node catalog (100+)** — projections, FFT (2D/3D with auto-dims),
  filters, masks, alignment (`align.ap` covers multi-ref / single-ref /
  rotation-only / class assignment), back-projection (`bp3f` / `bp3d` /
  `bp3g` / `bp3n`), CTF, iterative loops (`flow.do` + `flow.state`).
- **DO loops with feedback state** — iterative alignment, class averaging,
  and SGD-style reconstruction work out of the box.
- **Live previews** — every node renders a PNG thumbnail; stacks show
  filmstrips, volumes show rotatable Three.js views, Fourier shows
  HSV-domain-coloured amplitudes.
- **Challenges** — 8 bite-size puzzles (shift undo, particle picking, class
  averaging, angle assignment, …) with automatic NCC / cluster-accuracy
  judging.
- **Tutorials** — 17 interactive walkthroughs with mobile-game-style
  highlighting (Basic 1–12, Comparison 1–5).

## Quick start — Web build (recommended)

Prerequisites: a modern browser (Chrome / Edge / Safari / Firefox, last
~3 years). No Python, pip, Node, or numpy install on the user side.

```sh
# From a pre-built release:
cd frontend/dist
python3 -m http.server 8080   # or any static server
open http://127.0.0.1:8080/

# From source:
cd frontend
npm install
npm run build
bash ../pyodide_poc/build_bundle.sh   # bundles backend + pyspider into dist/bundle.zip
cd dist && python3 -m http.server 8080
```

The first load downloads Pyodide + numpy / scipy / Pillow / scikit-image
(~45 MB, cached after). Every subsequent action runs in-browser.

See [INSTALL.md](INSTALL.md) for Windows / macOS step-by-step guides.

## Quick start — Local server

Prerequisites: Python 3.11+, Node 20+.

```sh
# Backend (port 8000)
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000

# Frontend (port 5173, proxies /api → 8000)
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173/.

## Repository layout

```
backend/              FastAPI app + plugin system + DAG engine
  engine.py           topological executor + DO loops + streaming
  nodes.py            core numeric kernels
  plugins/builtin/    one .py per node (drop in / drop out = enable / disable)
  tutorials/          interactive walkthroughs (JSON)
  challenges/         8 cryo-EM puzzles (JSON)
  api_shim.py         Pyodide-mode entrypoints (mirror FastAPI)
  bundled_data/       β-gal reference maps (MRC)
frontend/
  src/                React + React Flow UI
    pyodideEngine.ts  Web Worker client for Pyodide
  public/
    pyodideWorker.js  Pyodide Web Worker (loads numpy/scipy, runs engine)
  dist/               Static site (after `npm run build`)
pyspider/             Pure-numpy SPIDER kernels (shared by both modes)
pyodide_poc/
  build_bundle.sh     Packages backend + pyspider into bundle.zip
```

## Challenges

| # | Title | Metric | Pass |
|---|---|---|---|
| 1 | Undo the shift | NCC | 0.9 |
| 2 | Undo the rotation | NCC | 0.9 |
| 3 | Auto-pick particles from a micrograph | NCC | 0.5 |
| 4 | Projection in real space (no FFT) | NCC | 0.9 |
| 5 | Projection–slice theorem (Fourier) | NCC | 0.99 |
| 6 | Recover K class averages from a noisy stack | cluster accuracy | 0.80 |
| 7 | Recover a clean average from a noisy stack | rot-NCC | 0.85 |
| 8 | Assign an orientation to a single particle | NCC | 0.9 |

Each challenge restricts the palette to the nodes you need and ships with
a hint that lays out the intended pipeline.

## Contributing

Node plugins live at `backend/plugins/builtin/<category>/<name>.py`. One
file per node — copy an existing sibling as a template, register with
`register(N(id, label, category, inputs, outputs, [params], doc, fn))`.
Restart the backend (or just reload the browser in Pyodide mode) to see
it in the palette.

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgements

- The **SPIDER** system (Frank et al., Wadsworth) for the algorithms and
  40 years of cryo-EM tooling this project draws from.
- The **RELION** consortium for the tutorial β-gal dataset bundled as
  test data.
- **React Flow** for the canvas / node / edge library the UI is built on.
- **Pyodide** for making a full CPython + numpy + scipy stack run in the
  browser.

---

日本語版は [README.ja.md](README.ja.md) にあります.

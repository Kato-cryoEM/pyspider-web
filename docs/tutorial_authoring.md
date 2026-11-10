# Tutorial authoring guide

The interactive tutorial you see in the app when you press **📖 Tutorial** is
driven by plain JSON files. Anyone can add a new lesson — no TypeScript,
no rebuild, just drop a `.json` file in the right directory and reload the
browser.

## Where lessons live

Two discovery locations, scanned every request:

| Location | Purpose |
|---|---|
| `backend/tutorials/*.json` | Bundled with the app — reference / official lessons. |
| `~/.pyspider/tutorials/*.json` | User-local — your custom lessons, survive upgrades. |

The lesson `id` field must be unique across both directories. A duplicate is
skipped and reported under `errors` in the `/api/tutorials` response.

Filenames are irrelevant to the app (they're sorted alphabetically so a
`01_`, `02_` prefix controls display order in the picker); the lesson `id`
inside the JSON is what identifies it.

## Lesson schema

Every JSON file is one lesson:

```json
{
  "id": "masking",
  "title": "1 · Soft masking",
  "summary": "Drop a synthetic disk, apply a Gaussian mask, watch the edge soften.",
  "steps": [ ... ]
}
```

| Field | Type | Purpose |
|---|---|---|
| `id` | string | Unique, stable identifier used for `localStorage` progress. `[a-z0-9_-]+`. |
| `title` | string | Shown in the lesson picker and on the tutorial card header. |
| `summary` | string | One-line description in the picker. |
| `steps` | array | The linear sequence of steps (see below). |

## Step schema

```json
{
  "id": "drag-mo",
  "title": "Drop a Model image",
  "body": "In the Palette on the left (Data category), drag \"Model image\" onto the canvas.",
  "target": {
    "tutorId": "palette:data.mo",
    "cardSide": "right",
    "expandPaletteCategory": "Data"
  },
  "action": { "kind": "add_node", "nodeType": "data.mo" }
}
```

| Field | Type | Purpose |
|---|---|---|
| `id` | string | Unique within the lesson. |
| `title` | string | Bold heading on the tutorial card. |
| `body` | string | The instruction. Split into paragraphs with `\n\n`. |
| `target` | object (optional) | Which UI element to spotlight. Omit for a global info card. |
| `action` | object | What the runner is waiting for. |

### `target` fields

| Field | Purpose |
|---|---|
| `tutorId` | CSS selector against `data-tutor-id="..."`. See the "Available targets" table below. |
| `cardSide` | Where to place the info card relative to the target: `"left"`, `"right"`, `"top"`, `"bottom"`, or `"auto"` (default). |
| `expandPaletteCategory` | If the target is inside a collapsible Palette category (e.g. `"Data"`, `"Masking"`, `"Reconstruction"`, ...), the runner auto-expands it before spotlighting. |

### `action` kinds

The step advances when its `action` fires. All actions:

| `kind` | Extra fields | Advances when… |
|---|---|---|
| `info` | — | User clicks **Next →**. |
| `add_node` | `nodeType` | Any node with `spec.id === nodeType` appears on the canvas. |
| `set_param` | `nodeName`, `value` | Any node's `params[nodeName]` equals `String(value)`. |
| `select_node` | `nodeType` | The currently-selected node has `spec.id === nodeType`. |
| `run_node` | `nodeType` | Any node of that type finishes running (`status === "success"`). |
| `click_target` | — | (Reserved — currently behaves like `info`.) |

If you need "wait for two things", split into two steps.

## Available targets

Every UI element the tutorial can point at carries a stable `data-tutor-id`
attribute. Current tags:

| `tutorId` template | Points to |
|---|---|
| `palette:<node-id>` | A palette entry, e.g. `palette:data.mo`. Node IDs are in `GET /api/nodes`. |
| `param:<name>` | An Inspector parameter row for the currently-selected node, e.g. `param:kind`. |
| `node:<node-id>` | A node body on the canvas by type, e.g. `node:mask.circle` (highlights the first instance). |
| `run-button` | The green Run graph / red Stop button in the top bar. |

**Adding new tutor IDs** is a code change — set `data-tutor-id="my-thing"`
on a JSX element and it becomes targetable.

### Discovering targets

Open the browser DevTools and search the DOM tree for `data-tutor-id`. Any
element with that attribute is a valid `tutorId`. Nothing else works.

## Complete example

`backend/tutorials/01_masking.json` (bundled) walks through the whole
data → mask → run → preview flow. Read it end-to-end when authoring your
first lesson — it exercises every action kind and target pattern.

## Testing a lesson

1. Save your `.json` file.
2. In the app, click **📖 Tutorial** (the picker re-reads from
   `/api/tutorials` every time it opens).
3. Select your lesson. If it doesn't appear, check
   `GET /api/tutorials` in the browser DevTools Network tab —
   the `errors` array names files that failed to parse.

Any invalid JSON is dropped with the offending file in `errors`; a bad
lesson never breaks the working ones.

## Tips

- **Prefer many small steps.** Each `body` should be one clear instruction.
  Two-step "select node, then edit param" reads better than one crowded step.
- **Auto-advance where possible.** Users lose momentum on Next buttons.
  Reserve `info` for framing sentences and wiring steps (which the runner
  can't observe automatically).
- **Use `cardSide` to keep the card out of the way.** `"auto"` picks the
  side opposite the viewport centre from the target, which is right most
  of the time — but for Inspector params (right side of screen) explicitly
  say `"left"` so the card doesn't cover the input.
- **Every `set_param` step should be preceded by a `select_node` step**
  targeting the same node, otherwise the `param:*` target won't be visible
  in the Inspector.
- **Palette items are hidden inside collapsed categories.** Always set
  `expandPaletteCategory` on palette targets.
- **User-local tutorials** in `~/.pyspider/tutorials/` never get overwritten
  when you update the app. Publish drafts there, promote to
  `backend/tutorials/` for wider distribution.

## Contributing back

To share your tutorial:

1. Test it locally under `~/.pyspider/tutorials/`.
2. When happy, move (or copy) the JSON file into
   `backend/tutorials/` and commit it. No code changes needed — the loader
   scans the directory at each `/api/tutorials` request.
3. Number the filename so it sorts into the intended position among the
   bundled lessons (`03_rotation.json`, etc.).

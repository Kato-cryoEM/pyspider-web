/**
 * Browser-build stub for the host-filesystem picker.
 *
 * In the Pyodide build we run entirely inside the browser sandbox, so we
 * cannot browse the user's filesystem. Keep the component signature so
 * the Settings modal still compiles; render a short notice instead.
 */

interface Props {
  initialPath?: string;
  onSelect: (path: string) => void;
  onClose: () => void;
}

export default function FileBrowser({ initialPath = "", onClose }: Props) {
  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-dialog file-browser"
           onClick={(e) => e.stopPropagation()}>
        <h3>Directory picker unavailable</h3>
        <p style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.5 }}>
          This build runs entirely in your browser (Pyodide) and cannot read the
          host filesystem. Node outputs cannot be auto-saved to disk here —
          the <code>default_save_dir</code> setting has no effect.
        </p>
        {initialPath && (
          <p style={{ fontSize: 11, color: "var(--muted)" }}>
            Current value: <code>{initialPath}</code>
          </p>
        )}
        <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 12 }}>
          <button onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}

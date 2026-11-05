import { useCallback, useEffect, useRef, useState } from "react";
import { api, fileToPngBase64 } from "../api";

export interface MaskValue {
  image?: string; // engine path of the picture
  mask?: string;  // engine path of the mask (white = redraw)
}

type Tool = "brush" | "erase" | "rect";
type Point = { x: number; y: number };

// The mask is shown in a colour that stands out on most pictures and exported white on black.
const INK = "#ff4d6d";

// Paint layers by uploaded mask path, so reopening the editor shows the mask again (this tab only).
const painted = new Map<string, string>();

// A picture and a mask painted over it (brush, eraser, rectangle; undo/redo, invert), uploaded as a
// white-on-black PNG shortly after each change. The picture is picked here (and uploaded too), or
// given as `source` (e.g. the output of an earlier step). `feather` is mentioned under the picture.
export function MaskPainter(props: { value?: MaskValue; onChange(v: MaskValue): void; source?: string; feather?: number }) {
  const [src, setSrc] = useState<string | undefined>(props.source);
  const [size, setSize] = useState<{ w: number; h: number }>();
  const [tool, setTool] = useState<Tool>("brush");
  const [brush, setBrush] = useState(48);
  const [showMask, setShowMask] = useState(true);
  const [state, setState] = useState<"" | "saving" | "saved">(props.value?.mask ? "saved" : "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const [cursor, setCursor] = useState<Point & { scale: number }>();
  const [history, setHistory] = useState({ undo: 0, redo: 0 });
  const base = useRef<HTMLCanvasElement>(null);
  const paint = useRef<HTMLCanvasElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const drawing = useRef<{ last?: Point; start?: Point; before?: ImageData } | undefined>(undefined);
  const undoStack = useRef<ImageData[]>([]);
  const redoStack = useRef<ImageData[]>([]);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const file = useRef<HTMLInputElement>(null);
  const valueRef = useRef(props.value);
  valueRef.current = props.value;
  const onChangeRef = useRef(props.onChange);
  onChangeRef.current = props.onChange;

  useEffect(() => setSrc(props.source ?? src), [props.source]); // eslint-disable-line react-hooks/exhaustive-deps

  async function pick(f: File | undefined) {
    if (!f) return;
    setBusy(true);
    setError(undefined);
    try {
      const png = await fileToPngBase64(f, 1024);
      const { path } = await api.upload(f.name, png.data);
      setSrc(`data:image/png;base64,${png.data}`);
      props.onChange({ image: path });
      setState("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  // Load the picture; its size sets the canvases' size.
  useEffect(() => {
    if (!src) return;
    const img = new Image();
    img.onload = () => setSize({ w: img.naturalWidth, h: img.naturalHeight });
    img.onerror = () => setError("Resim yüklenemedi");
    img.src = src;
  }, [src]);
  useEffect(() => {
    if (!src || !size || !base.current || !paint.current) return;
    const img = new Image();
    img.onload = () => base.current?.getContext("2d")?.drawImage(img, 0, 0, size.w, size.h);
    img.src = src;
    const p = paint.current.getContext("2d")!;
    p.clearRect(0, 0, size.w, size.h);
    undoStack.current = [];
    redoStack.current = [];
    setHistory({ undo: 0, redo: 0 });
    const prev = valueRef.current?.mask ? painted.get(valueRef.current.mask) : undefined;
    if (prev) {
      const m = new Image();
      m.onload = () => p.drawImage(m, 0, 0, size.w, size.h);
      m.src = prev;
    }
  }, [src, size]);

  const ctx = () => paint.current!.getContext("2d", { willReadFrequently: true })!;
  const snapshot = () => ctx().getImageData(0, 0, size!.w, size!.h);

  // Upload the mask a moment after the last change.
  const save = useCallback(() => {
    if (!size || !paint.current) return;
    clearTimeout(saveTimer.current);
    setState("saving");
    saveTimer.current = setTimeout(async () => {
      try {
        const layer = paint.current!.toDataURL("image/png");
        const c = document.createElement("canvas");
        c.width = size.w;
        c.height = size.h;
        const m = c.getContext("2d")!;
        m.drawImage(paint.current!, 0, 0);
        m.globalCompositeOperation = "source-in"; // painted pixels -> white, keeping soft edges
        m.fillStyle = "#fff";
        m.fillRect(0, 0, size.w, size.h);
        m.globalCompositeOperation = "destination-over";
        m.fillStyle = "#000";
        m.fillRect(0, 0, size.w, size.h);
        const { path } = await api.upload("mask.png", c.toDataURL("image/png").split(",")[1]);
        painted.set(path, layer);
        onChangeRef.current({ image: valueRef.current?.image, mask: path });
        setState("saved");
      } catch (e) {
        setError((e as Error).message);
        setState("");
      }
    }, 500);
  }, [size]);
  useEffect(() => () => clearTimeout(saveTimer.current), []);

  function commit(before: ImageData) {
    undoStack.current = [...undoStack.current.slice(-29), before];
    redoStack.current = [];
    setHistory({ undo: undoStack.current.length, redo: 0 });
    save();
  }
  function undo() {
    const prev = undoStack.current.pop();
    if (!prev) return;
    redoStack.current.push(snapshot());
    ctx().putImageData(prev, 0, 0);
    setHistory({ undo: undoStack.current.length, redo: redoStack.current.length });
    save();
  }
  function redo() {
    const next = redoStack.current.pop();
    if (!next) return;
    undoStack.current.push(snapshot());
    ctx().putImageData(next, 0, 0);
    setHistory({ undo: undoStack.current.length, redo: redoStack.current.length });
    save();
  }
  function invert() {
    const before = snapshot();
    const c = ctx();
    c.globalCompositeOperation = "xor";
    c.fillStyle = INK;
    c.fillRect(0, 0, size!.w, size!.h);
    c.globalCompositeOperation = "source-over";
    commit(before);
  }
  function clear() {
    const before = snapshot();
    ctx().clearRect(0, 0, size!.w, size!.h);
    commit(before);
  }

  // Keyboard: [ ] brush size, B / E / R tools, Ctrl+Z / Ctrl+Shift+Z (or Ctrl+Y), I invert.
  useEffect(() => {
    if (!size) return;
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) return;
      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (e.shiftKey) redo();
        else undo();
      } else if (mod && e.key.toLowerCase() === "y") {
        e.preventDefault();
        redo();
      } else if (!mod && e.key === "[") setBrush((b) => Math.max(4, Math.round(b / 1.25)));
      else if (!mod && e.key === "]") setBrush((b) => Math.min(300, Math.round(b * 1.25)));
      else if (!mod && e.key.toLowerCase() === "b") setTool("brush");
      else if (!mod && e.key.toLowerCase() === "e") setTool("erase");
      else if (!mod && e.key.toLowerCase() === "r") setTool("rect");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  function at(e: React.PointerEvent<HTMLElement>): Point & { scale: number } {
    const r = paint.current!.getBoundingClientRect();
    const scale = size!.w / r.width;
    return { x: (e.clientX - r.left) * scale, y: (e.clientY - r.top) * scale, scale };
  }
  function line(from: Point, to: Point) {
    const c = ctx();
    c.globalCompositeOperation = tool === "erase" ? "destination-out" : "source-over";
    c.strokeStyle = INK;
    c.lineCap = "round";
    c.lineJoin = "round";
    c.lineWidth = brush;
    c.beginPath();
    c.moveTo(from.x, from.y);
    c.lineTo(to.x + 0.01, to.y);
    c.stroke();
    c.globalCompositeOperation = "source-over";
  }
  function rect(a: Point, b: Point) {
    const c = ctx();
    c.fillStyle = INK;
    c.fillRect(Math.min(a.x, b.x), Math.min(a.y, b.y), Math.abs(a.x - b.x), Math.abs(a.y - b.y));
  }

  function down(e: React.PointerEvent<HTMLCanvasElement>) {
    e.currentTarget.setPointerCapture(e.pointerId);
    const p = at(e);
    drawing.current = { before: snapshot(), last: p, start: p };
    if (tool !== "rect") line(p, p);
  }
  function move(e: React.PointerEvent<HTMLCanvasElement>) {
    const p = at(e);
    setCursor(p);
    const d = drawing.current;
    if (!d) return;
    if (tool === "rect") {
      ctx().putImageData(d.before!, 0, 0);
      rect(d.start!, p);
    } else {
      line(d.last ?? p, p);
      d.last = p;
    }
  }
  function up() {
    const d = drawing.current;
    drawing.current = undefined;
    if (d?.before) commit(d.before);
  }

  return (
    <div className="mask-wrap">
      <div className="toolbar mask-tools">
        {props.source ? null : (
          <>
            <button type="button" className="btn small" onClick={() => file.current?.click()}>
              {src ? "Başka resim" : "Resim seç"}
            </button>
            <input ref={file} type="file" accept="image/*" hidden onChange={(e) => void pick(e.target.files?.[0] ?? undefined)} />
          </>
        )}
        {src && size ? (
          <>
            <div className="segmented" role="group" aria-label="Araç">
              <button type="button" aria-pressed={tool === "brush"} title="Fırça (B)" onClick={() => setTool("brush")}>Fırça</button>
              <button type="button" aria-pressed={tool === "erase"} title="Silgi (E)" onClick={() => setTool("erase")}>Silgi</button>
              <button type="button" aria-pressed={tool === "rect"} title="Dikdörtgen (R)" onClick={() => setTool("rect")}>Kutu</button>
            </div>
            <label className="range-row help" style={{ minWidth: 190 }} title="[ ve ] ile de değişir">
              Fırça {brush}px
              <input type="range" min={4} max={300} value={brush} onChange={(e) => setBrush(Number(e.target.value))} />
            </label>
            <button type="button" className="btn small ghost" disabled={!history.undo} title="Geri al (Ctrl+Z)" onClick={undo}>↶</button>
            <button type="button" className="btn small ghost" disabled={!history.redo} title="Yinele (Ctrl+Shift+Z)" onClick={redo}>↷</button>
            <button type="button" className="btn small ghost" onClick={invert}>Ters çevir</button>
            <button type="button" className="btn small ghost" onClick={clear}>Temizle</button>
            <button type="button" className="btn small ghost" aria-pressed={showMask} onClick={() => setShowMask((v) => !v)}>
              {showMask ? "Maskeyi gizle" : "Maskeyi göster"}
            </button>
            <span className="note">{state === "saving" ? "Kaydediliyor…" : state === "saved" ? "Maske kaydedildi" : "Değiştirmek istediğin yeri boya"}</span>
          </>
        ) : null}
        {busy ? <span className="note">Yükleniyor…</span> : null}
      </div>
      {src && size ? (
        <div ref={stage} className="mask-stage" style={{ width: Math.min(720, Math.max(size.w, 480)) }}>
          <canvas ref={base} width={size.w} height={size.h} />
          <canvas
            ref={paint}
            className="paint"
            width={size.w}
            height={size.h}
            aria-label="Maske"
            style={{ opacity: showMask ? 0.55 : 0, cursor: tool === "rect" ? "crosshair" : "none" }}
            onPointerDown={down}
            onPointerMove={move}
            onPointerUp={up}
            onPointerCancel={up}
            onPointerLeave={() => setCursor(undefined)}
          />
          {cursor && tool !== "rect" ? (
            <span className="brush-cursor" style={{
              left: `${(cursor.x / size.w) * 100}%`, top: `${(cursor.y / size.h) * 100}%`,
              width: brush / cursor.scale, height: brush / cursor.scale,
              borderStyle: tool === "erase" ? "dashed" : "solid",
            }} />
          ) : null}
        </div>
      ) : props.source ? (
        <div className="note">{error ?? "Resim yükleniyor…"}</div>
      ) : (
        <div className="drop" onClick={() => file.current?.click()} role="button" tabIndex={0}
          onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && file.current?.click()}>
          Yeniden çizilecek resmi seç, sonra değiştirmek istediğin yeri boya.
        </div>
      )}
      {error && src ? <span className="error-text">{error}</span> : null}
      {props.feather ? <span className="help">Kenarlar {props.feather}px yumuşatılır; boyanan alanın kenarında sonuç orijinalle harmanlanır.</span> : null}
    </div>
  );
}

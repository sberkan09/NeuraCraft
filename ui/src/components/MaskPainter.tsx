import { useEffect, useRef, useState } from "react";
import { api, fileToPngBase64 } from "../api";

export interface MaskValue {
  image?: string; // engine path of the picture
  mask?: string;  // engine path of the mask (white = redraw)
}

// Picture + a painted mask (brush / eraser), both uploaded as PNG for the inpaint step.
export function MaskPainter(props: { value?: MaskValue; onChange(v: MaskValue): void }) {
  const [src, setSrc] = useState<string>();
  const [size, setSize] = useState<{ w: number; h: number }>();
  const [brush, setBrush] = useState(48);
  const [erase, setErase] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const base = useRef<HTMLCanvasElement>(null);
  const paint = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const last = useRef<{ x: number; y: number } | undefined>(undefined);
  const file = useRef<HTMLInputElement>(null);

  async function pick(f: File | undefined) {
    if (!f) return;
    setBusy(true);
    setError(undefined);
    try {
      const png = await fileToPngBase64(f, 1024);
      const { path } = await api.upload(f.name, png.data);
      setSrc(`data:image/png;base64,${png.data}`);
      setSize({ w: png.width, h: png.height });
      props.onChange({ image: path });
      setDirty(false);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (!src || !size || !base.current || !paint.current) return;
    const img = new Image();
    img.onload = () => base.current!.getContext("2d")!.drawImage(img, 0, 0, size.w, size.h);
    img.src = src;
    const p = paint.current.getContext("2d")!;
    p.clearRect(0, 0, size.w, size.h);
  }, [src, size]);

  function at(e: React.PointerEvent<HTMLCanvasElement>) {
    const r = e.currentTarget.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * (size?.w ?? 1), y: ((e.clientY - r.top) / r.height) * (size?.h ?? 1) };
  }
  function stroke(to: { x: number; y: number }) {
    const ctx = paint.current!.getContext("2d")!;
    ctx.globalCompositeOperation = erase ? "destination-out" : "source-over";
    ctx.strokeStyle = "#ffffff";
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = brush;
    ctx.beginPath();
    const from = last.current ?? to;
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x + 0.01, to.y);
    ctx.stroke();
    last.current = to;
    setDirty(true);
  }

  async function useMask() {
    if (!size || !paint.current) return;
    setBusy(true);
    try {
      // White strokes on black, as the inpaint step expects.
      const c = document.createElement("canvas");
      c.width = size.w;
      c.height = size.h;
      const ctx = c.getContext("2d")!;
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, size.w, size.h);
      ctx.drawImage(paint.current, 0, 0);
      const { path } = await api.upload("mask.png", c.toDataURL("image/png").split(",")[1]);
      props.onChange({ image: props.value?.image, mask: path });
      setDirty(false);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mask-wrap">
      <div className="toolbar">
        <button type="button" className="btn small" onClick={() => file.current?.click()}>
          {src ? "Başka resim" : "Resim seç"}
        </button>
        <input ref={file} type="file" accept="image/*" hidden onChange={(e) => void pick(e.target.files?.[0] ?? undefined)} />
        {src ? (
          <>
            <div className="segmented" role="group" aria-label="Araç">
              <button type="button" aria-pressed={!erase} onClick={() => setErase(false)}>Fırça</button>
              <button type="button" aria-pressed={erase} onClick={() => setErase(true)}>Silgi</button>
            </div>
            <label className="range-row help" style={{ minWidth: 200 }}>
              Boyut {brush}px
              <input type="range" min={4} max={200} value={brush} onChange={(e) => setBrush(Number(e.target.value))} />
            </label>
            <button type="button" className="btn small ghost" onClick={() => {
              paint.current?.getContext("2d")?.clearRect(0, 0, size!.w, size!.h);
              setDirty(true);
            }}>Temizle</button>
            <button type="button" className="btn small primary" disabled={busy || !dirty} onClick={() => void useMask()}>
              {props.value?.mask && !dirty ? "Maske hazır" : "Maskeyi kullan"}
            </button>
          </>
        ) : null}
        {busy ? <span className="note">Yükleniyor…</span> : null}
      </div>
      {src && size ? (
        <div className="mask-stage" style={{ width: Math.min(size.w, 640) }}>
          <canvas ref={base} width={size.w} height={size.h} />
          <canvas
            ref={paint}
            className="paint"
            width={size.w}
            height={size.h}
            onPointerDown={(e) => {
              e.currentTarget.setPointerCapture(e.pointerId);
              drawing.current = true;
              last.current = undefined;
              stroke(at(e));
            }}
            onPointerMove={(e) => drawing.current && stroke(at(e))}
            onPointerUp={() => {
              drawing.current = false;
              last.current = undefined;
            }}
          />
        </div>
      ) : (
        <div className="drop" onClick={() => file.current?.click()} role="button" tabIndex={0}
          onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && file.current?.click()}>
          Yeniden çizilecek resmi seç, sonra değiştirmek istediğin yeri boya.
        </div>
      )}
      {error ? <span className="error-text">{error}</span> : null}
    </div>
  );
}

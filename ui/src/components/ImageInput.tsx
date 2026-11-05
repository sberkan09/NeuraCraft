import { useRef, useState } from "react";
import { api, fileToPngBase64 } from "../api";

// Pick or drop a picture: converted to PNG in the browser, saved by the engine, returns its path.
export function ImageInput(props: { value?: string; onChange(path: string | undefined, preview?: string): void; label: string; previewUrl?: string }) {
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const [thumb, setThumb] = useState<string | undefined>(props.previewUrl);
  const input = useRef<HTMLInputElement>(null);

  async function take(file: File | undefined) {
    if (!file) return;
    setBusy(true);
    setError(undefined);
    try {
      const png = await fileToPngBase64(file);
      const { path } = await api.upload(file.name, png.data);
      const url = URL.createObjectURL(file);
      setThumb(url);
      props.onChange(path, url);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="field">
      <div
        className={`drop${over ? " over" : ""}`}
        role="button"
        tabIndex={0}
        aria-label={`${props.label}: resim seç`}
        onClick={() => input.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          void take(e.dataTransfer.files[0]);
        }}
      >
        {thumb && props.value ? <img src={thumb} alt="" /> : null}
        <span>
          {busy ? "Yükleniyor…" : props.value ? <span className="mono">{props.value}</span> : "Resmi buraya sürükle ya da tıklayıp seç"}
        </span>
        {props.value ? (
          <button
            type="button"
            className="btn small ghost"
            onClick={(e) => {
              e.stopPropagation();
              setThumb(undefined);
              props.onChange(undefined);
            }}
          >
            Kaldır
          </button>
        ) : null}
      </div>
      <input ref={input} type="file" accept="image/*" hidden onChange={(e) => void take(e.target.files?.[0] ?? undefined)} />
      {error ? <span className="error-text">{error}</span> : null}
    </div>
  );
}

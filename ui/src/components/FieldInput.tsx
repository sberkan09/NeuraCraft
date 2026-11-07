import type { ModelInfo, WorkflowField } from "../api";
import { formatBytes } from "../api";

// Turkish labels for the field names users meet most; others show the schema name.
const labels: Record<string, string> = {
  model: "Model", prompt: "Prompt", negative: "Negatif prompt", steps: "Adım", seed: "Seed", guidance: "Guidance",
  width: "Genişlik", height: "Yükseklik", strength: "Güç", sampler: "Örnekleyici", int8: "int8 ağırlıklar", lora: "LoRA",
  tiny_vae: "TAESD", controlnet: "ControlNet", control: "Kontrol resmi", control_scale: "ControlNet gücü",
  ip_adapter: "IP-Adapter", image_encoder: "CLIP vision", image_prompt: "Resim promptu", ip_scale: "Resim promptu gücü",
  frames: "Kare", init: "Başlangıç", prompt_at: "Prompt geçişleri", context: "Pencere", overlap: "Örtüşme", motion: "Hareket modülü",
  in: "Girdi", out: "Çıktı adı", path: "Dosya", dir: "Klasör", fps: "fps", quality: "Kalite", mask: "Maske", feather: "Kenar yumuşatma",
  prompts: "Promptlar", seeds: "Seed'ler", loop: "Döngü", zoom: "Yakınlaşma", keyframes: "Anahtar kare", shrink: "Küçülme",
  zoom_in: "İçeri doğru", low: "Alt eşik", high: "Üst eşik", size: "Ağ boyutu", amplitude: "Salınım", passes: "Geçiş", mode: "Mod",
  weight: "Ağırlık", beta_schedule: "Beta planı", grid: "Izgara", depth: "Derinlik", edges: "Kenar kesme", turn: "Dönüş",
  yaw: "Açı", pitch: "Yükseklik açısı", fov: "Görüş açısı", background: "Arka plan", lighting: "Işıklandırma",
  texture: "Doku boyutu", faces: "Üçgen sayısı", layout: "Düzen", focus: "Ekran düzlemi", smooth: "Yumuşatma",
  stereo: "Stereo 3D", eye: "Göz açısı", postprocess: "Ağırlıkları yumuşat", capture: "Hareket kaydı (.bvh)", start: "Başlangıç (sn)",
  seconds: "Süre (sn)", speed: "Hız", in_place: "Yerinde", play: "Animasyonu oynat",
};
export const fieldLabel = (name: string) => labels[name] ?? name;

export function FieldInput(props: {
  field: WorkflowField;
  value: unknown;
  onChange(v: unknown): void;
  models: ModelInfo[];
  clips: string[]; // clip names made by earlier steps
  id: string;
}) {
  const { field: f, value, onChange, id } = props;
  const label = (
    <label className="label" htmlFor={id} title={f.doc}>
      {fieldLabel(f.name)} {f.required ? <span className="help">· gerekli</span> : null}
    </label>
  );
  const unset = value === undefined || value === null;
  const placeholder = f.default !== undefined ? `varsayılan: ${JSON.stringify(f.default)}` : "";

  switch (f.type) {
    case "bool":
      return (
        <label className="check" title={f.doc}>
          <input type="checkbox" checked={unset ? Boolean(f.default) : Boolean(value)} onChange={(e) => onChange(e.target.checked)} />
          {fieldLabel(f.name)}
        </label>
      );
    case "int":
    case "float":
      return (
        <div className="field">
          {label}
          <input
            id={id}
            className="input"
            type="number"
            step={f.type === "int" ? 1 : "any"}
            min={f.min}
            max={f.max}
            placeholder={placeholder}
            value={unset ? "" : String(value)}
            onChange={(e) => onChange(e.target.value === "" ? undefined : Number(e.target.value))}
          />
        </div>
      );
    case "enum":
      return (
        <div className="field">
          {label}
          <select id={id} className="input" value={unset ? "" : String(value)} onChange={(e) => onChange(e.target.value || undefined)}>
            <option value="">{placeholder || "—"}</option>
            {f.options?.map((o) => <option key={o} value={o}>{o}</option>)}
          </select>
        </div>
      );
    case "text":
      return (
        <div className="field">
          {label}
          <textarea id={id} className="input" rows={3} value={unset ? "" : String(value)} onChange={(e) => onChange(e.target.value || undefined)} />
        </div>
      );
    case "model": {
      // SD fields also list inpainting and SDXL models (the engine says which steps cannot use them).
      const options = props.models.filter((m) => m.kind === f.model || (f.model === "sd" && (m.kind === "sd-inpaint" || m.kind === "sdxl")));
      const known = options.some((m) => m.path === value);
      return (
        <div className="field">
          {label}
          <select id={id} className="input" value={unset ? "" : String(value)} onChange={(e) => onChange(e.target.value || undefined)}>
            <option value="">{f.required ? "seç…" : "— yok —"}</option>
            {options.map((m) => <option key={m.path} value={m.path}>{m.path}{m.kind === "sdxl" ? " (SDXL)" : ""} · {formatBytes(m.bytes)}</option>)}
            {!unset && !known ? <option value={String(value)}>{String(value)}</option> : null}
          </select>
        </div>
      );
    }
    case "clip":
    case "mesh":
      return (
        <div className="field">
          {label}
          <select id={id} className="input" value={unset ? "" : String(value)} onChange={(e) => onChange(e.target.value || undefined)}>
            <option value="">{f.required ? "seç…" : "— yok —"}</option>
            {props.clips.map((c) => <option key={c} value={c}>{c}</option>)}
            {!unset && !props.clips.includes(String(value)) ? <option value={String(value)}>{String(value)} (bulunamadı)</option> : null}
          </select>
        </div>
      );
    case "clips":
    case "strings":
    case "ints": {
      const list = Array.isArray(value) ? value : unset ? [] : [value];
      return (
        <div className="field">
          {label}
          <textarea
            id={id}
            className="input"
            rows={Math.max(2, list.length)}
            placeholder={f.type === "clips" ? "bir satıra bir klip adı" : f.type === "ints" ? "bir satıra bir sayı" : "bir satıra bir prompt"}
            value={list.join("\n")}
            onChange={(e) => {
              const lines = e.target.value.split("\n").map((s) => s.trim()).filter(Boolean);
              onChange(lines.length ? (f.type === "ints" ? lines.map(Number) : lines) : undefined);
            }}
          />
        </div>
      );
    }
    case "loras": {
      const list = (Array.isArray(value) ? value : []) as string[];
      const loras = props.models.filter((m) => m.kind === "lora");
      const parse = (s: string) => {
        const i = s.lastIndexOf(":");
        return i > 0 && /^[\d.]+$/.test(s.slice(i + 1)) ? { file: s.slice(0, i), scale: Number(s.slice(i + 1)) } : { file: s, scale: 1 };
      };
      const set = (next: { file: string; scale: number }[]) =>
        onChange(next.length ? next.map((l) => (l.scale === 1 ? l.file : `${l.file}:${l.scale}`)) : undefined);
      const items = list.map(parse);
      return (
        <div className="field">
          {label}
          {items.map((l, i) => (
            <div key={i} className="range-row" style={{ marginBottom: 6 }}>
              <select className="input" aria-label="LoRA dosyası" value={l.file} onChange={(e) => set(items.map((x, j) => (j === i ? { ...x, file: e.target.value } : x)))}>
                {loras.map((m) => <option key={m.path} value={m.path}>{m.name}</option>)}
                {!loras.some((m) => m.path === l.file) ? <option value={l.file}>{l.file}</option> : null}
              </select>
              <input className="input" type="number" step={0.05} aria-label="Ölçek" value={l.scale} onChange={(e) => set(items.map((x, j) => (j === i ? { ...x, scale: Number(e.target.value) } : x)))} />
              <button type="button" className="btn small ghost" aria-label="Kaldır" onClick={() => set(items.filter((_, j) => j !== i))}>×</button>
            </div>
          ))}
          <button type="button" className="btn small" disabled={!loras.length} onClick={() => set([...items, { file: loras[0].path, scale: 1 }])}>
            LoRA ekle
          </button>
        </div>
      );
    }
    case "keyframes": {
      const list = (Array.isArray(value) ? value : []) as [number, string][];
      const set = (next: [number, string][]) => onChange(next.length ? next : undefined);
      return (
        <div className="field">
          {label}
          {list.map(([frame, text], i) => (
            <div key={i} className="range-row" style={{ marginBottom: 6, alignItems: "flex-start" }}>
              <input className="input" type="number" min={0} aria-label="Kare" style={{ width: 80 }} value={frame} onChange={(e) => set(list.map((x, j) => (j === i ? [Number(e.target.value), x[1]] : x)))} />
              <input className="input" aria-label="Prompt" value={text} onChange={(e) => set(list.map((x, j) => (j === i ? [x[0], e.target.value] : x)))} />
              <button type="button" className="btn small ghost" aria-label="Kaldır" onClick={() => set(list.filter((_, j) => j !== i))}>×</button>
            </div>
          ))}
          <button type="button" className="btn small" onClick={() => set([...list, [list.length ? list[list.length - 1][0] + 16 : 0, ""]])}>
            Anahtar kare ekle
          </button>
        </div>
      );
    }
    default: // string, file, dir
      return (
        <div className="field">
          {label}
          <input id={id} className="input" placeholder={placeholder} value={unset ? "" : String(value)} onChange={(e) => onChange(e.target.value || undefined)} />
        </div>
      );
  }
}

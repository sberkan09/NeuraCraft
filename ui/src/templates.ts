// Task templates of the simple mode: a short form, turned into an ordinary workflow
// (the same JSON `NeuraCraft workflow` runs), which "Grafikte aç" opens for full control.

import type { ModelInfo, Step, Workflow } from "./api";

export type FormKind = "text" | "string" | "int" | "float" | "enum" | "bool" | "image" | "mask" | "model" | "dir";

export interface FormField {
  key: string;
  label: string;
  kind: FormKind;
  default?: unknown;
  options?: { value: string; label: string; hint?: string }[];
  min?: number;
  max?: number;
  step?: number;
  help?: string;
  model?: string; // model kind for kind "model"
  optional?: boolean;
  wide?: boolean; // full row in the form grid
}

export type Values = Record<string, unknown>;

export interface Template {
  id: string;
  title: string;
  subtitle: string;
  needs: string[]; // model kinds that must exist
  fields: FormField[];
  build(v: Values, models: ModelInfo[], outDir: string): Workflow;
}

const byKind = (models: ModelInfo[], kind: string) => models.filter((m) => m.kind === kind);

// The best default for a model kind (names that usually mean the right one first).
export function pickModel(models: ModelInfo[], kind: string, prefer: string[] = []): string {
  const list = byKind(models, kind);
  for (const p of prefer) {
    const hit = list.find((m) => m.path.toLowerCase().includes(p));
    if (hit) return hit.path;
  }
  return list[0]?.path ?? "";
}

const motionModelFor = (models: ModelInfo[], sd: string) =>
  byKind(models, "motion").find((m) => m.path.startsWith(sd + "/"))?.path ?? pickModel(models, "motion", ["lightning"]);

// Sampling settings that suit a model: SD-Turbo takes 1-4 steps without guidance; other SD 1.5
// models use the LCM-LoRA when there is one (4-8 steps), else 20-40 Euler steps with guidance.
export function recipe(models: ModelInfo[], model: string, quality: number) {
  const q = Math.max(0, Math.min(2, quality));
  if (model.toLowerCase().includes("turbo")) return { steps: [1, 2, 4][q], guidance: 0 } as Record<string, unknown>;
  const lcm = byKind(models, "lora").find((m) => m.name.toLowerCase().includes("lcm"));
  if (lcm) return { steps: [4, 6, 8][q], sampler: "lcm", lora: [lcm.path] } as Record<string, unknown>;
  return { steps: [20, 30, 40][q], guidance: 7 } as Record<string, unknown>;
}

const qualityField: FormField = {
  key: "quality",
  label: "Kalite",
  kind: "enum",
  default: "0",
  options: [
    { value: "0", label: "Hızlı", hint: "en az adım" },
    { value: "1", label: "Dengeli", hint: "biraz daha adım" },
    { value: "2", label: "Kaliteli", hint: "en çok adım" },
  ],
  wide: true,
};
const sizeField = (def = "512x512", extra: string[] = []): FormField => ({
  key: "size",
  label: "Boyut",
  kind: "enum",
  default: def,
  options: ["512x512", "768x512", "512x768", "384x384", ...extra].map((s) => ({ value: s, label: s.replace("x", " × ") })),
});
const size = (v: Values) => {
  const [w, h] = String(v.size ?? "512x512").split("x").map(Number);
  return { width: w || 512, height: h || 512 };
};
const seedField: FormField = { key: "seed", label: "Seed", kind: "int", default: 42, min: 0, max: 4294967295 };
const promptField = (def: string): FormField => ({ key: "prompt", label: "Prompt", kind: "text", default: def, wide: true });
const negativeField: FormField = {
  key: "negative",
  label: "İstemediklerin",
  kind: "string",
  default: "",
  help: "negatif prompt, guidance ile kullanılır",
  optional: true,
  wide: true,
};

export const templates: Template[] = [
  {
    id: "image",
    title: "Resim üret",
    subtitle: "Metinden tek resim · txt2img",
    needs: ["sd"],
    fields: [
      promptField("a red fox in fresh snow, golden hour, photo"),
      negativeField,
      { key: "model", label: "Model", kind: "model", model: "sd" },
      sizeField(),
      seedField,
      qualityField,
    ],
    build(v, models, out) {
      const model = String(v.model);
      const steps: Step[] = [
        {
          op: "txt2img",
          model,
          prompt: v.prompt,
          ...(v.negative ? { negative: v.negative } : {}),
          ...size(v),
          seed: Number(v.seed),
          ...recipe(models, model, Number(v.quality)),
          out: "image",
        },
        { op: "save", in: "image", path: `${out}/image.png` },
      ];
      return { steps };
    },
  },
  {
    id: "video",
    title: "Video üret",
    subtitle: "Metinden kısa klip · AnimateDiff",
    needs: ["sd", "motion"],
    fields: [
      promptField("a red fox walking through fresh snow, golden hour, cinematic"),
      negativeField,
      { key: "model", label: "Model", kind: "model", model: "sd" },
      { key: "frames", label: "Kare sayısı", kind: "int", default: 16, min: 8, max: 256 },
      sizeField("384x384"),
      seedField,
      { key: "init", label: "Başlangıç resmi", kind: "image", optional: true, help: "video bu resimden başlar", wide: true },
      { key: "strength", label: "Başlangıçtan ne kadar uzaklaşsın", kind: "float", default: 0.75, min: 0.1, max: 1, step: 0.05 },
      { key: "smooth", label: "Akıcılaştır (RIFE ×2)", kind: "bool", default: true },
      { key: "upscale", label: "Büyüt (ESRGAN ×4)", kind: "bool", default: false },
    ],
    build(v, models, out) {
      const model = String(v.model);
      const steps: Step[] = [];
      if (v.init) steps.push({ op: "load", path: v.init, out: "start" });
      steps.push({
        op: "t2v",
        model,
        motion: motionModelFor(models, model),
        prompt: v.prompt,
        ...(v.negative ? { negative: v.negative, guidance: 1.5 } : {}),
        frames: Number(v.frames),
        ...size(v),
        seed: Number(v.seed),
        ...(v.init ? { init: "start", strength: Number(v.strength) } : {}),
        out: "clip",
      });
      let last = "clip", fps = 8;
      const rife = pickModel(models, "rife");
      if (v.smooth && rife) {
        steps.push({ op: "interpolate", model: rife, in: last, passes: 1, out: "smooth" });
        last = "smooth";
        fps = 16;
      }
      const esrgan = pickModel(models, "esrgan");
      if (v.upscale && esrgan) {
        steps.push({ op: "upscale", model: esrgan, in: last, out: "big" });
        last = "big";
      }
      steps.push({ op: "save", in: last, path: `${out}/video.mov`, fps });
      steps.push({ op: "save", in: last, path: `${out}/frames/` });
      return { steps };
    },
  },
  {
    id: "parallax",
    title: "Fotoğrafı canlandır",
    subtitle: "Derinlikten 3D parallax",
    needs: ["depth"],
    fields: [
      { key: "image", label: "Fotoğraf", kind: "image", wide: true },
      { key: "frames", label: "Kare sayısı", kind: "int", default: 48, min: 8, max: 240 },
      { key: "amplitude", label: "Kamera salınımı", kind: "float", default: 0.03, min: 0, max: 0.1, step: 0.005 },
      { key: "zoom", label: "Yakınlaşma", kind: "float", default: 0.06, min: 0, max: 0.3, step: 0.01 },
      { key: "loop", label: "Başa dönsün (döngü)", kind: "bool", default: true },
    ],
    build(v, models, out) {
      const depth = pickModel(models, "depth");
      return {
        steps: [
          { op: "load", path: v.image, out: "photo" },
          { op: "depth", model: depth, in: "photo", out: "depth" },
          { op: "parallax", model: depth, in: "photo", frames: Number(v.frames), amplitude: Number(v.amplitude), zoom: Number(v.zoom), loop: Boolean(v.loop), out: "clip" },
          { op: "save", in: "depth", path: `${out}/depth.png` },
          { op: "save", in: "clip", path: `${out}/3d.mov`, fps: 24 },
        ],
      };
    },
  },
  {
    id: "hires",
    title: "Hires portre",
    subtitle: "Üret, ×4 büyüt, yeniden çiz",
    needs: ["sd", "esrgan"],
    fields: [
      promptField("portrait photo of an old fisherman, detailed face, golden hour, 85mm"),
      { key: "model", label: "Model", kind: "model", model: "sd" },
      seedField,
      { key: "final", label: "Son boyut", kind: "enum", default: "1024", options: [{ value: "1024", label: "1024 × 1024" }, { value: "1536", label: "1536 × 1536" }] },
      { key: "strength", label: "Yeniden çizme gücü", kind: "float", default: 0.5, min: 0.2, max: 0.8, step: 0.05, help: "yüksek = daha çok yeni ayrıntı" },
      qualityField,
    ],
    build(v, models, out) {
      const model = String(v.model);
      const r = recipe(models, model, Number(v.quality));
      const side = Number(v.final);
      return {
        steps: [
          { op: "txt2img", model, prompt: v.prompt, seed: Number(v.seed), ...r, out: "base" },
          { op: "upscale", model: pickModel(models, "esrgan"), in: "base", out: "x4" },
          { op: "resize", in: "x4", width: side, height: side, out: "big" },
          { op: "img2img", model, prompt: v.prompt, in: "big", width: side, height: side, ...r, strength: Number(v.strength), seed: Number(v.seed) + 1, out: "detailed" },
          { op: "save", in: "base", path: `${out}/base.png` },
          { op: "save", in: "detailed", path: `${out}/detailed.png` },
        ],
      };
    },
  },
  {
    id: "inpaint",
    title: "Inpaint",
    subtitle: "Maskeli bölgeyi yeniden çiz",
    needs: ["sd-inpaint"],
    fields: [
      { key: "image", label: "Resim ve maske", kind: "mask", wide: true, help: "boyadığın beyaz alan yeniden çizilir" },
      promptField("a small wooden lighthouse model"),
      { key: "model", label: "Model", kind: "model", model: "sd-inpaint" },
      seedField,
      { key: "steps", label: "Adım", kind: "int", default: 20, min: 1, max: 60 },
      { key: "feather", label: "Kenar yumuşatma", kind: "float", default: 8, min: 0, max: 48, step: 1 },
    ],
    build(v, _models, out) {
      const img = v.image as { image: string; mask: string } | undefined;
      return {
        steps: [
          { op: "load", path: img?.image ?? "", out: "picture" },
          { op: "load", path: img?.mask ?? "", out: "mask" },
          { op: "inpaint", model: v.model, in: "picture", mask: "mask", prompt: v.prompt, seed: Number(v.seed), steps: Number(v.steps), feather: Number(v.feather), out: "result" },
          { op: "save", in: "result", path: `${out}/inpaint.png` },
        ],
      };
    },
  },
  {
    id: "upscale",
    title: "Büyüt",
    subtitle: "Real-ESRGAN ×4",
    needs: ["esrgan"],
    fields: [{ key: "image", label: "Resim", kind: "image", wide: true }],
    build(v, models, out) {
      return {
        steps: [
          { op: "load", path: v.image, out: "picture" },
          { op: "upscale", model: pickModel(models, "esrgan"), in: "picture", out: "big" },
          { op: "save", in: "big", path: `${out}/x4.png` },
        ],
      };
    },
  },
  {
    id: "smooth",
    title: "Akıcılaştır",
    subtitle: "RIFE ile ara kare ekle",
    needs: ["rife"],
    fields: [
      { key: "dir", label: "Kare klasörü", kind: "dir", default: "", help: "PNG kareleri (ör. bir videonun outputs/…/frames/ klasörü)", wide: true },
      { key: "passes", label: "Kaç kat", kind: "enum", default: "1", options: [{ value: "1", label: "×2" }, { value: "2", label: "×4" }] },
      { key: "fps", label: "Çıkış fps", kind: "int", default: 16, min: 1, max: 120 },
      { key: "loop", label: "Son kareyle ilk kare arasına da ekle", kind: "bool", default: false },
    ],
    build(v, models, out) {
      return {
        steps: [
          { op: "load", dir: v.dir, out: "frames" },
          { op: "interpolate", model: pickModel(models, "rife"), in: "frames", passes: Number(v.passes), loop: Boolean(v.loop), out: "smooth" },
          { op: "save", in: "smooth", path: `${out}/smooth.mov`, fps: Number(v.fps) },
        ],
      };
    },
  },
];

export function defaults(t: Template, models: ModelInfo[]): Values {
  const v: Values = {};
  for (const f of t.fields) {
    if (f.kind === "model") v[f.key] = pickModel(models, f.model ?? "", f.model === "sd" ? ["turbo", "animatediff"] : []);
    else if (f.default !== undefined) v[f.key] = f.default;
  }
  if (t.id === "video") v.model = pickModel(models, "sd", ["animatediff"]);
  return v;
}

// Values the template cannot run without.
export function missing(t: Template, v: Values): string[] {
  const out: string[] = [];
  for (const f of t.fields) {
    if (f.optional) continue;
    const x = v[f.key];
    if (f.kind === "mask") {
      const m = x as { image?: string; mask?: string } | undefined;
      if (!m?.image) out.push(`${f.label}: resim seç`);
      else if (!m.mask) out.push(`${f.label}: maskeyi boya`);
    } else if (x === undefined || x === "" || x === null) out.push(f.label);
  }
  return out;
}

export const outputFolder = (id: string) => {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return `outputs/${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}-${id}`;
};

import { useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { frameUrl, type Step, type WorkflowOp } from "../api";
import type { LiveJob } from "../engine";
import { stepId } from "../graph";

// Prompt travel editor for a video step: the main prompt sits at frame 0, keyframes ([frame, prompt])
// after it. The engine interpolates the text condition linearly between neighbouring keyframes and
// holds the last one to the end; frames are sampled in overlapping windows of `context` frames.

type Key = [number, string];
const hues = ["#8ab4ff", "#c58af9", "#f28b82", "#fdd663", "#81c995", "#78d9ec"];

const numberDefault = (op: WorkflowOp, name: string, fallback: number) => {
  const f = op.fields.find((x) => x.name === name);
  return typeof f?.default === "number" ? f.default : fallback;
};

function windowStarts(frames: number, context: number, overlap: number): number[] {
  const ctx = Math.min(context, frames);
  const stride = Math.max(1, ctx - Math.min(overlap, ctx - 1));
  const starts: number[] = [];
  for (let s = 0; ; s += stride) {
    if (s + ctx >= frames) {
      starts.push(frames - ctx);
      break;
    }
    starts.push(s);
  }
  return starts;
}

export function PromptTimeline(props: {
  step: Step;
  op: WorkflowOp;
  job?: LiveJob;
  onChange(field: string, value: unknown): void;
  onBack(): void;
}) {
  const { step, op, job } = props;
  const frames = Math.max(2, Number(step.frames ?? numberDefault(op, "frames", 16)));
  const context = Number(step.context ?? numberDefault(op, "context", 16));
  const overlap = Number(step.overlap ?? numberDefault(op, "overlap", 4));
  const keys: Key[] = (Array.isArray(step.prompt_at) ? (step.prompt_at as Key[]) : []).map(([f, t]) => [Number(f), String(t)]);
  const all: Key[] = ([[0, String(step.prompt ?? "")], ...keys] as Key[]).sort((a, b) => a[0] - b[0]);
  const [selected, setSelected] = useState(0); // frame of the selected keyframe
  const track = useRef<HTMLDivElement>(null);
  const drag = useRef<{ from: number } | undefined>(undefined);

  const x = (f: number) => `${(f / (frames - 1)) * 100}%`;
  const frameAt = (clientX: number) => {
    const r = track.current!.getBoundingClientRect();
    return Math.round(Math.min(1, Math.max(0, (clientX - r.left) / r.width)) * (frames - 1));
  };
  const setKeys = (next: Key[]) => {
    const sorted = [...next].sort((a, b) => a[0] - b[0]);
    props.onChange("prompt_at", sorted.length ? sorted : undefined);
  };
  const taken = (f: number) => f === 0 || keys.some((k) => k[0] === f);

  function addAt(f: number) {
    if (f <= 0 || f >= frames) f = Math.min(frames - 1, Math.max(1, f));
    while (taken(f) && f < frames - 1) ++f;
    while (taken(f) && f > 1) --f;
    if (taken(f)) return;
    const before = all.filter((k) => k[0] < f).pop();
    setKeys([...keys, [f, before?.[1] ?? ""]]);
    setSelected(f);
  }
  function moveKey(from: number, to: number) {
    to = Math.min(frames - 1, Math.max(1, to));
    if (to === from || taken(to)) return;
    setKeys(keys.map((k) => (k[0] === from ? [to, k[1]] : k)));
    setSelected(to);
  }

  function onMarkerDown(e: ReactPointerEvent, f: number) {
    e.stopPropagation();
    setSelected(f);
    if (f === 0) return;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    drag.current = { from: f };
  }
  function onMarkerMove(e: ReactPointerEvent) {
    if (!drag.current) return;
    // Stay between the neighbouring keyframes so the order (and the marker under the pointer) is kept.
    const i = all.findIndex((k) => k[0] === drag.current!.from);
    const lo = all[i - 1][0] + 1, hi = (all[i + 1]?.[0] ?? frames) - 1;
    const to = Math.min(hi, Math.max(lo, frameAt(e.clientX)));
    if (to !== drag.current.from && !taken(to)) {
      moveKey(drag.current.from, to);
      drag.current = { from: to };
    }
  }

  // Gaps shorter than a window blend two prompts inside one window.
  const short = all.slice(1).some((k, i) => k[0] - all[i][0] < Math.min(context, frames));
  const tick = [1, 2, 4, 8, 16, 32, 64, 128].find((t) => frames / t <= 16) ?? 256;
  const starts = windowStarts(frames, context, overlap);

  // Frames of the latest run of this step, if any.
  let strip: { clip: string; count: number } | undefined;
  if (job?.workflow) {
    const i = job.workflow.steps.findIndex((s) => stepId(s) && stepId(s) === stepId(step));
    const out = job.outputs.find((o) => o.step === i + 1);
    if (i >= 0 && out?.clip && out.frames) strip = { clip: out.clip, count: out.frames };
  }
  const every = strip ? Math.max(1, Math.ceil(strip.count / 12)) : 1;

  const sel = all.find((k) => k[0] === selected) ?? all[0];
  const invalid = keys.filter((k) => k[0] <= 0 || k[0] >= frames);

  return (
    <div className="timeline">
      <div className="timeline-head">
        <button type="button" className="btn small ghost" onClick={props.onBack}>← Grafik</button>
        <div>
          <strong>Prompt geçişleri</strong>
          <div className="note">{frames} kare · pencere {Math.min(context, frames)}, örtüşme {overlap}</div>
        </div>
        <span className="spacer" />
        <button type="button" className="btn small" onClick={() => addAt(Math.round((selected + (all.find((k) => k[0] > selected)?.[0] ?? frames - 1)) / 2) || 1)}>
          Anahtar kare ekle
        </button>
      </div>

      <div className="tl-body">
        <div className="tl-ruler">
          {Array.from({ length: Math.floor((frames - 1) / tick) + 1 }, (_, k) => k * tick).map((f) => (
            <span key={f} style={{ left: x(f) }}>{f}</span>
          ))}
        </div>

        <div className="tl-track" ref={track} title="Boş bir yere tıklayarak anahtar kare ekle"
          onPointerDown={(e) => addAt(frameAt(e.clientX))}>
          {all.map((k, i) => {
            const next = all[i + 1];
            const c1 = hues[i % hues.length], c2 = next ? hues[(i + 1) % hues.length] : c1;
            const tight = next && next[0] - k[0] < Math.min(context, frames);
            return (
              <div key={`seg${k[0]}`} className={`tl-seg${tight ? " tight" : ""}`}
                style={{ left: x(k[0]), width: `calc(${x((next?.[0] ?? frames - 1))} - ${x(k[0])})`, background: `linear-gradient(90deg, ${c1}55, ${c2}55)` }} />
            );
          })}
          {all.map((k, i) => (
            <button key={`m${i}`} type="button" className={`tl-marker${k[0] === selected ? " on" : ""}`}
              style={{ left: x(k[0]), background: hues[i % hues.length] }}
              aria-label={k[0] === 0 ? "Ana prompt (kare 0)" : `Kare ${k[0]}`}
              onPointerDown={(e) => onMarkerDown(e, k[0])} onPointerMove={onMarkerMove}
              onPointerUp={() => (drag.current = undefined)}
              onKeyDown={(e) => {
                if (k[0] === 0) return;
                if (e.key === "ArrowLeft") moveKey(k[0], k[0] - 1);
                if (e.key === "ArrowRight") moveKey(k[0], k[0] + 1);
                if (e.key === "Delete" || e.key === "Backspace") setKeys(keys.filter((q) => q[0] !== k[0]));
              }}
              title={k[0] === 0 ? "Ana prompt" : `Kare ${k[0]} · sürükle ya da ←/→, Delete ile sil`} />
          ))}
        </div>

        <div className="tl-windows" aria-label="Örnekleme pencereleri">
          {starts.map((s, i) => (
            <div key={s} className="tl-window" style={{ left: x(s), width: `calc(${x(s + Math.min(context, frames) - 1)} - ${x(s)})`, top: (i % 2) * 10 }} />
          ))}
        </div>
        <div className="note">Örnekleme pencereleri: her biri {Math.min(context, frames)} kare; örtüşen kareler yumuşakça karışır.</div>

        {strip ? (
          <div className="tl-strip">
            {Array.from({ length: Math.ceil(strip.count / every) }, (_, k) => k * every).map((f) => (
              <img key={f} src={frameUrl(job!.id, strip!.clip, f, 128)} alt={`Kare ${f}`} style={{ left: x(Math.min(f, frames - 1)) }} loading="lazy" />
            ))}
          </div>
        ) : (
          <div className="note">Çalıştırınca burada üretilen karelerden bir şerit görünür.</div>
        )}
      </div>

      <div className="tl-keys">
        {all.map((k, i) => (
          <div key={`row${i}`} className={`tl-key${k[0] === sel[0] ? " on" : ""}`} onClick={() => setSelected(k[0])}>
            <span className="tl-dot" style={{ background: hues[i % hues.length] }} />
            {k[0] === 0 ? (
              <span className="tl-frame">0 · ana</span>
            ) : (
              <input className="input tl-frame" type="number" min={1} max={frames - 1} aria-label="Kare" value={k[0]}
                onChange={(e) => moveKey(k[0], Number(e.target.value))} />
            )}
            <textarea className="input" rows={2} aria-label={k[0] === 0 ? "Ana prompt" : `Kare ${k[0]} promptu`} value={k[1]}
              onFocus={() => setSelected(k[0])}
              onChange={(e) => (k[0] === 0 ? props.onChange("prompt", e.target.value || undefined) : setKeys(keys.map((q) => (q[0] === k[0] ? [q[0], e.target.value] : q))))} />
            {k[0] === 0 ? <span style={{ width: 32 }} /> : (
              <button type="button" className="btn small ghost danger" aria-label="Kaldır" onClick={(e) => { e.stopPropagation(); setKeys(keys.filter((q) => q[0] !== k[0])); }}>×</button>
            )}
          </div>
        ))}
        {invalid.length ? <div className="error-text">Anahtar kareler 1 ile {frames - 1} arasında olmalı.</div> : null}
        {short ? (
          <div className="note warn">
            İki anahtar kare bir pencereden ({Math.min(context, frames)} kare) yakın olunca promptlar aynı pencerede karışır; temiz geçişler için aralıkları açabilir ya da kare sayısını artırabilirsin.
          </div>
        ) : null}
      </div>
    </div>
  );
}

import { useEffect, useMemo, useState } from "react";
import type { CacheInfo, Step, Workflow, WorkflowOp } from "../api";
import { api } from "../api";
import type { Engine } from "../engine";
import { freeSpot, latestRuns, stepId, topoSort, upstream, withLayout, type StepUi } from "../graph";
import { pickModel } from "../templates";
import { FieldInput, fieldLabel } from "./FieldInput";
import { GraphCanvas } from "./GraphCanvas";
import { PromptTimeline } from "./PromptTimeline";
import { overallFraction } from "./Results";

const groupNames: Record<string, string> = { image: "Resim", video: "Video", analysis: "Analiz", enhance: "İyileştirme", edit: "Düzenle", input: "Giriş / çıkış" };
const groupOrder = ["image", "video", "analysis", "enhance", "edit", "input"];

// Inputs a step reads (clip fields), for the connection summary.
function inputsOf(step: Step, op?: WorkflowOp): string[] {
  if (!op) return [];
  const out: string[] = [];
  for (const f of op.fields) {
    const v = step[f.name];
    if (f.type === "clip" && typeof v === "string") out.push(`${fieldLabel(f.name)}: ${v}`);
    if (f.type === "clips" && Array.isArray(v)) out.push(`${fieldLabel(f.name)}: ${v.join(", ")}`);
  }
  return out;
}

// The editor for any workflow: every operation and field from the engine's schema. Steps run top to
// bottom; any step can read any clip made above it (a graph run in order).
const stateLabel: Record<string, string> = { queued: "sırada", running: "çalışıyor", done: "bitti", failed: "hata", cancelled: "durduruldu" };

export function GraphMode(props: { engine: Engine; workflow: Workflow; title: string; onChange(wf: Workflow): void }) {
  const { engine, workflow } = props;
  const ops = useMemo(() => new Map((engine.schema?.ops ?? []).map((o) => [o.name, o] as [string, WorkflowOp])), [engine.schema]);
  const [selected, setSelected] = useState(0);
  const [view, setView] = useState<"graph" | "steps" | "json" | "timeline">("graph");
  const [message, setMessage] = useState<string>();
  const [jsonText, setJsonText] = useState(JSON.stringify(workflow, null, 2));
  const [jsonError, setJsonError] = useState<string>();
  const [problems, setProblems] = useState<string[]>([]);
  const [jobId, setJobId] = useState<number>();
  const [busy, setBusy] = useState(false);
  const steps = workflow.steps;

  useEffect(() => {
    if (view !== "json") setJsonText(JSON.stringify(workflow, null, 2));
  }, [workflow, view]);

  // Every step needs a node id and a position on the canvas.
  useEffect(() => {
    if (!ops.size) return;
    const laid = withLayout(workflow, ops);
    if (laid !== workflow) props.onChange(laid);
  }, [workflow, ops, props]);
  useEffect(() => {
    if (!message) return;
    const t = setTimeout(() => setMessage(undefined), 4000);
    return () => clearTimeout(t);
  }, [message]);
  const laidSteps = useMemo(() => withLayout(workflow, ops).steps, [workflow, ops]);
  const invalidSteps = useMemo(() => {
    const set = new Set<number>();
    for (const p of problems) {
      const m = /^step (\d+)/.exec(p);
      if (m) set.add(Number(m[1]) - 1);
    }
    return set;
  }, [problems]);

  // Live check against the schema.
  useEffect(() => {
    const t = setTimeout(() => {
      if (!steps.length) return setProblems([]);
      api.check(workflow).then((r) => setProblems(r.problems)).catch(() => undefined);
    }, 300);
    return () => clearTimeout(t);
  }, [workflow, steps.length]);

  const setSteps = (next: Step[]) => props.onChange({ ...workflow, steps: next });
  const update = (i: number, field: string, value: unknown) =>
    setSteps(steps.map((s, j) => {
      if (j !== i) return s;
      const n: Step = { ...s };
      if (value === undefined) delete n[field];
      else n[field] = value;
      return n;
    }));

  function add(opName: string) {
    const op = ops.get(opName);
    if (!op) return;
    const s: Step = { op: opName };
    const used = new Set(steps.map((x) => String(x.out ?? x.op)));
    if (op.output) {
      let name = opName, k = 2;
      while (used.has(name)) name = `${opName}${k++}`;
      s.out = name;
    }
    const lastClip = [...steps].reverse().find((x) => ops.get(x.op)?.output);
    for (const f of op.fields) {
      if (f.type === "model" && f.required) s[f.name] = pickModel(engine.models, f.model ?? "", f.model === "sd" ? ["turbo", "animatediff"] : []);
      if (f.name === "in" && f.type === "clip" && lastClip) s.in = String(lastClip.out ?? lastClip.op);
      if (f.name === "prompt" && f.required) s.prompt = "";
    }
    if (opName === "save") s.path = `outputs/${String(s.in ?? "out")}.png`;
    const at = Math.min(selected + 1, steps.length);
    const anchor = steps[Math.min(selected, steps.length - 1)]?.ui as StepUi | undefined;
    s.ui = freeSpot(steps, anchor ? anchor.x + 300 : 40, anchor ? anchor.y : 40);
    const inserted = [...steps.slice(0, at), s, ...steps.slice(at)];
    const sorted = topoSort(inserted, ops) ?? inserted;  // a new step reading a later clip moves after it
    setSteps(sorted);
    setSelected(sorted.indexOf(s));
  }
  const move = (i: number, d: number) => {
    const j = i + d;
    if (j < 0 || j >= steps.length) return;
    const next = [...steps];
    [next[i], next[j]] = [next[j], next[i]];
    setSteps(next);
    setSelected(j);
  };
  const remove = (i: number) => {
    setSteps(steps.filter((_, j) => j !== i));
    setSelected(Math.max(0, Math.min(i, steps.length - 2)));
  };

  // Runs the whole workflow, or only what step `upTo` needs (it and the steps it reads, without saves).
  // The engine reuses unchanged results, so this is quick for a step late in a long workflow.
  async function run(upTo?: number) {
    setBusy(true);
    try {
      const r = await api.check(workflow);
      setProblems(r.problems);
      if (r.problems.length) return;
      if (upTo === undefined) {
        setJobId(await engine.run(workflow, props.title));
      } else {
        const keep = upstream(steps, ops, upTo).filter((k) => k === upTo || steps[k].op !== "save");
        setJobId(await engine.run({ ...workflow, steps: keep.map((k) => steps[k]) }, `${props.title} · adım ${upTo + 1}`));
      }
    } catch (e) {
      const err = e as Error & { problems?: string[] };
      setProblems(err.problems ?? [err.message]);
    } finally {
      setBusy(false);
    }
  }

  const job = engine.jobs.find((j) => j.id === jobId);
  const runs = useMemo(() => latestRuns(engine.jobs), [engine.jobs]);
  const [cache, setCache] = useState<CacheInfo>();
  const jobState = job?.state;
  useEffect(() => {
    if (jobState === "running" || jobState === "queued") return;
    api.cache().then(setCache).catch(() => setCache(undefined));
  }, [jobState, engine.connected]);
  const sel = steps[selected];
  const selOp = sel ? ops.get(sel.op) : undefined;
  const hasTimeline = !!selOp?.fields.some((f) => f.type === "keyframes");
  useEffect(() => {
    if (view === "timeline" && !hasTimeline) setView("graph");
  }, [view, hasTimeline]);
  const clipsBefore = (i: number) => steps.slice(0, i).filter((s) => ops.get(s.op)?.output).map((s) => String(s.out ?? s.op));
  const groups = groupOrder.map((g) => ({ g, list: [...ops.values()].filter((o) => o.group === g) }));
  const stepState = (i: number) => {
    const r = runs.get(stepId(steps[i]));
    if (!r) return "";
    if (r.job.outputs.some((o) => o.step === r.index + 1)) return "done";
    if (r.job.state === "running" && r.job.step === r.index + 1) return "running";
    return "";
  };

  return (
    <div className="graph">
      <nav className="palette" aria-label="İşlemler">
        <h2 className="section-title">İşlem ekle</h2>
        {groups.map(({ g, list }) =>
          list.length ? (
            <div key={g}>
              <div className="help" style={{ marginBottom: 6 }}>{groupNames[g] ?? g}</div>
              <div className="ops">
                {list.map((o) => <button key={o.name} type="button" title={o.doc} onClick={() => add(o.name)}>{o.name}</button>)}
              </div>
            </div>
          ) : null,
        )}
        <p className="help">Liste motorun işlem şemasından gelir; yeni işlemler burada kendiliğinden görünür.</p>
      </nav>

      <section className="canvas-area" aria-label="Akış">
        <div className="runstrip" style={{ borderTop: 0, borderBottom: "1px solid var(--line)" }}>
          <strong>{props.title}</strong>
          <div className="segmented" role="group" aria-label="Görünüm">
            <button type="button" aria-pressed={view === "graph"} onClick={() => setView("graph")}>Grafik</button>
            <button type="button" aria-pressed={view === "steps"} onClick={() => setView("steps")}>Liste</button>
            <button type="button" aria-pressed={view === "json"} onClick={() => setView("json")}>JSON</button>
            {hasTimeline ? <button type="button" aria-pressed={view === "timeline"} onClick={() => setView("timeline")}>Zaman</button> : null}
          </div>
          <span className="spacer" />
          <span className="note">{steps.length} adım</span>
        </div>
        {message ? <div className="problems" style={{ margin: "10px 16px 0" }} role="alert">{message}</div> : null}
        {view === "timeline" && sel && selOp && hasTimeline ? (
          <PromptTimeline step={sel} op={selOp} run={runs.get(stepId(sel))} onChange={(field, x) => update(selected, field, x)} onBack={() => setView("graph")} />
        ) : view === "graph" ? (
          steps.length === 0 ? (
            <div className="empty">Soldan bir işlem ekleyerek başla. Düğümleri sürükle; bir düğümün sağındaki çıkışı diğerinin solundaki girişe çekerek bağla.</div>
          ) : (
            <GraphCanvas steps={laidSteps} ops={ops} models={engine.models} runs={runs} invalidSteps={invalidSteps} selected={selected}
              onSelect={(i) => i >= 0 && setSelected(i)} onChange={setSteps} onMessage={setMessage} />
          )
        ) : view === "json" ? (
          <>
            <textarea className="json-editor" aria-label="Workflow JSON" spellCheck={false} value={jsonText}
              onChange={(e) => {
                setJsonText(e.target.value);
                try {
                  const wf = JSON.parse(e.target.value) as Workflow;
                  if (!Array.isArray(wf.steps)) throw new Error("\"steps\" listesi yok");
                  setJsonError(undefined);
                  props.onChange(wf);
                } catch (err) {
                  setJsonError((err as Error).message);
                }
              }} />
            {jsonError ? <div className="error-text" style={{ padding: "8px 16px" }}>{jsonError}</div> : null}
          </>
        ) : (
          <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 10, overflowY: "auto" }}>
            {steps.length === 0 ? <div className="empty">Soldan bir işlem ekleyerek başla.</div> : null}
            {steps.map((s, i) => {
              const op = ops.get(s.op);
              const st = stepState(i);
              return (
                <div key={i} role="button" tabIndex={0} className="task" aria-pressed={i === selected}
                  onClick={() => setSelected(i)} onKeyDown={(e) => e.key === "Enter" && setSelected(i)}
                  style={{ cursor: "pointer", borderColor: st === "running" ? "var(--run)" : undefined }}>
                  <div className="row">
                    <span className="t">{i + 1}. {op?.title ?? s.op} <span className="help">· {s.op}{s.out ? ` → ${s.out}` : ""}</span></span>
                    <span className="toolbar">
                      {st === "done" ? <span className="state done">bitti</span> : st === "running" ? <span className="state running">{runs.get(stepId(s))?.job.stage}</span> : null}
                      <button type="button" className="btn small ghost" aria-label="Yukarı taşı" onClick={(e) => { e.stopPropagation(); move(i, -1); }}>↑</button>
                      <button type="button" className="btn small ghost" aria-label="Aşağı taşı" onClick={(e) => { e.stopPropagation(); move(i, 1); }}>↓</button>
                      <button type="button" className="btn small ghost danger" aria-label="Sil" onClick={(e) => { e.stopPropagation(); remove(i); }}>×</button>
                    </span>
                  </div>
                  <span className="s">{inputsOf(s, op).join(" · ") || (op?.doc ?? "")}</span>
                </div>
              );
            })}
          </div>
        )}
        <div className="runstrip" style={{ marginTop: "auto" }}>
          {problems.length ? (
            <ul className="problems" style={{ flex: "1 1 100%" }}>{problems.slice(0, 6).map((p) => <li key={p}>{p}</li>)}</ul>
          ) : null}
          {job ? (
            <div style={{ flex: "1 1 260px", display: "flex", flexDirection: "column", gap: 6 }}>
              <span className="note">
                İş {job.id} · {stateLabel[job.state] ?? job.state}
                {job.state === "running" ? ` · ${job.op} ${job.stage}` : job.error ? ` · ${job.error}` : ""}
              </span>
              <div className={`bar ${job.state === "done" ? "done" : ""}`}>
                <div style={{ width: `${job.state === "done" ? 100 : Math.round(overallFraction(job) * 100)}%` }} />
              </div>
            </div>
          ) : <span className="spacer" />}
          {job && (job.state === "running" || job.state === "queued") ? (
            <button type="button" className="btn danger" onClick={() => void engine.cancel(job.id)}>Durdur</button>
          ) : null}
          {cache?.enabled && cache.entries > 0 ? (
            <button type="button" className="btn small ghost" title="Değişmeyen adımlar yeniden hesaplanmaz; temizlersen her şey baştan çalışır"
              onClick={() => void api.clearCache().then(setCache)}>
              Önbellek: {cache.entries} sonuç · {Math.max(1, Math.round(cache.bytes / 1048576))} MB · temizle
            </button>
          ) : null}
          <button type="button" className="btn primary" disabled={busy || !steps.length || problems.length > 0 || !engine.connected} onClick={() => void run()}>▶ Çalıştır</button>
        </div>
      </section>

      <aside className="inspector" aria-label="Seçili adım">
        {sel && selOp ? (
          <>
            <div>
              <span className="help">Seçili adım {selected + 1} · {sel.op}</span>
              <h2 style={{ margin: "2px 0 0", fontSize: 20 }}>{selOp.title}</h2>
              <p className="note" style={{ margin: "4px 0 0" }}>{selOp.doc}</p>
            </div>
            <button type="button" className="btn small" title="Yalnızca bu adımı ve okuduğu adımları çalıştırır; değişmeyen sonuçlar yeniden kullanılır"
              disabled={busy || problems.length > 0 || !engine.connected || (job?.state === "running")} onClick={() => void run(selected)}>
              ▶ Bu adıma kadar çalıştır
            </button>
            {selOp.fields.map((f) => f.type === "keyframes" ? (
              <div key={f.name} className="field">
                <span className="label">{fieldLabel(f.name)}</span>
                <span className="note">
                  {Array.isArray(sel[f.name]) && (sel[f.name] as unknown[]).length ? `${(sel[f.name] as unknown[]).length} anahtar kare` : "Tek prompt (geçiş yok)"}
                </span>
                <button type="button" className="btn small" onClick={() => setView("timeline")}>Zaman çizelgesinde düzenle</button>
              </div>
            ) : (
              <FieldInput key={f.name} id={`insp-${f.name}`} field={f} value={sel[f.name]} models={engine.models} clips={clipsBefore(selected)}
                onChange={(x) => update(selected, f.name, x)} />
            ))}
          </>
        ) : (
          <p className="note">Bir adım seç; ayarları burada görünür.</p>
        )}
      </aside>
    </div>
  );
}

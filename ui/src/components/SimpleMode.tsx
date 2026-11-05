import { useEffect, useMemo, useState } from "react";
import type { Step, Workflow, WorkflowOp } from "../api";
import { api } from "../api";
import type { Engine } from "../engine";
import { defaults, missing, outputFolder, templates, type FormField, type Template, type Values } from "../templates";
import { FieldInput } from "./FieldInput";
import { ImageInput } from "./ImageInput";
import { MaskPainter, type MaskValue } from "./MaskPainter";
import { Results } from "./Results";

type Overrides = Record<string, Record<string, unknown>>; // step key -> field -> value
const stepKey = (s: Step, i: number) => `${s.op}#${s.out ?? i}`;

function applyOverrides(wf: Workflow, o: Overrides): Workflow {
  return {
    ...wf,
    steps: wf.steps.map((s, i) => {
      const patch = o[stepKey(s, i)];
      if (!patch) return s;
      const next: Step = { ...s };
      for (const [k, v] of Object.entries(patch)) {
        if (v === undefined) delete next[k];
        else next[k] = v;
      }
      return next;
    }),
  };
}

function FormInput(props: { f: FormField; value: unknown; onChange(v: unknown): void; engine: Engine }) {
  const { f, value, onChange } = props;
  const id = `f-${f.key}`;
  const label = (
    <label className="label" htmlFor={id}>
      {f.label} {f.help ? <span className="help">({f.help})</span> : null}
    </label>
  );
  switch (f.kind) {
    case "text":
      return <div className="field">{label}<textarea id={id} className="input" rows={3} value={String(value ?? "")} onChange={(e) => onChange(e.target.value)} /></div>;
    case "string":
    case "dir":
      return <div className="field">{label}<input id={id} className="input" value={String(value ?? "")} onChange={(e) => onChange(e.target.value)} /></div>;
    case "int":
    case "float":
      if (f.kind === "float" && f.step) {
        return (
          <div className="field">
            {label}
            <div className="range-row">
              <input type="range" aria-label={f.label} min={f.min} max={f.max} step={f.step} value={Number(value ?? f.default)} onChange={(e) => onChange(Number(e.target.value))} />
              <input id={id} className="input" type="number" min={f.min} max={f.max} step={f.step} value={Number(value ?? f.default)} onChange={(e) => onChange(Number(e.target.value))} />
            </div>
          </div>
        );
      }
      return <div className="field">{label}<input id={id} className="input" type="number" min={f.min} max={f.max} value={String(value ?? "")} onChange={(e) => onChange(e.target.value === "" ? "" : Number(e.target.value))} /></div>;
    case "bool":
      return <label className="check"><input type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(e.target.checked)} />{f.label}</label>;
    case "enum":
      if (f.wide) {
        return (
          <fieldset className="field" style={{ border: 0, padding: 0, margin: 0 }}>
            <legend className="label">{f.label}</legend>
            <div className="options">
              {f.options?.map((o) => (
                <button key={o.value} type="button" className="option" aria-pressed={String(value) === o.value} onClick={() => onChange(o.value)}>
                  <strong>{o.label}</strong>
                  {o.hint ? <span className="h">{o.hint}</span> : null}
                </button>
              ))}
            </div>
          </fieldset>
        );
      }
      return (
        <div className="field">
          {label}
          <select id={id} className="input" value={String(value ?? "")} onChange={(e) => onChange(e.target.value)}>
            {f.options?.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
        </div>
      );
    case "model": {
      const list = props.engine.models.filter((m) => m.kind === f.model || (f.alsoModels ?? []).includes(m.kind));
      return (
        <div className="field">
          {label}
          <select id={id} className="input" value={String(value ?? "")} onChange={(e) => onChange(e.target.value)}>
            {list.map((m) => <option key={m.path} value={m.path}>{m.path}</option>)}
          </select>
        </div>
      );
    }
    case "image":
      return <div className="field"><span className="label">{f.label} {f.help ? <span className="help">({f.help})</span> : null}</span><ImageInput label={f.label} value={value as string | undefined} onChange={(p) => onChange(p)} /></div>;
    case "mask":
      return <div className="field"><span className="label">{f.label} {f.help ? <span className="help">({f.help})</span> : null}</span><MaskPainter value={value as MaskValue | undefined} onChange={onChange} /></div>;
  }
}

export function SimpleMode(props: { engine: Engine; openGraph(wf: Workflow, title: string): void }) {
  const { engine } = props;
  const available = (t: Template) => t.needs.every((k) => engine.models.some((m) => m.kind === k));
  const [taskId, setTaskId] = useState(templates[0].id);
  const template = templates.find((t) => t.id === taskId)!;
  const [values, setValues] = useState<Record<string, Values>>({});
  const [overrides, setOverrides] = useState<Record<string, Overrides>>({});
  const [problems, setProblems] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);

  // Defaults once the model list is known (per task, kept while switching).
  useEffect(() => {
    if (!engine.models.length || values[taskId]) return;
    setValues((v) => ({ ...v, [taskId]: defaults(template, engine.models) }));
  }, [engine.models, taskId, template, values]);

  const v = values[taskId] ?? {};
  const set = (key: string, value: unknown) => setValues((all) => ({ ...all, [taskId]: { ...(all[taskId] ?? {}), [key]: value } }));
  const o = overrides[taskId] ?? {};
  const outDir = useMemo(() => outputFolder(taskId), [taskId]);
  const workflow = useMemo(() => applyOverrides(template.build(v, engine.models, outDir), o), [template, v, engine.models, outDir, o]);
  const lacking = missing(template, v);
  const opsByName = useMemo(() => new Map((engine.schema?.ops ?? []).map((x) => [x.name, x] as [string, WorkflowOp])), [engine.schema]);

  async function run() {
    setBusy(true);
    setProblems([]);
    try {
      const wf = applyOverrides(template.build(v, engine.models, outputFolder(taskId)), o);
      const { problems: p } = await api.check(wf);
      if (p.length) {
        setProblems(p);
        return;
      }
      await engine.run(wf, template.title);
    } catch (e) {
      const err = e as Error & { problems?: string[] };
      setProblems(err.problems ?? [err.message]);
    } finally {
      setBusy(false);
    }
  }

  const clipsBefore = (i: number) => workflow.steps.slice(0, i).map((s) => String(s.out ?? s.op));

  return (
    <div className="simple">
      <nav className="tasks" aria-label="Görevler">
        <h2 className="section-title">Ne yapmak istiyorsun?</h2>
        {templates.map((t) => {
          const ok = available(t);
          return (
            <button key={t.id} type="button" className="task" aria-pressed={t.id === taskId} disabled={!ok} onClick={() => setTaskId(t.id)}
              title={ok ? t.subtitle : `Gereken model yok: ${t.needs.join(", ")}`}>
              <span className="t">{t.title}</span>
              <span className="s">{ok ? t.subtitle : `model yok: ${t.needs.join(", ")}`}</span>
            </button>
          );
        })}
        <button type="button" className="btn ghost" onClick={() => props.openGraph({ steps: [] }, "Boş grafik")}>+ Boş grafikle başla</button>
      </nav>

      <section className="panel form" aria-label={`${template.title} ayarları`}>
        <div className="row" style={{ justifyContent: "flex-start", flexWrap: "wrap" }}>
          <h1>{template.title}</h1>
          <span className="note">{template.subtitle}</span>
        </div>
        {!available(template) ? <p className="problems">Bu görev için gereken model bulunamadı: {template.needs.join(", ")}. Modelleri data/ klasörüne indirip sayfayı yenile.</p> : null}
        <div className="grid">
          {template.fields.map((f) => (
            <div key={f.key} className={f.wide || f.kind === "image" || f.kind === "mask" || f.kind === "text" ? "wide" : ""}>
              <FormInput f={f} value={v[f.key]} onChange={(x) => set(f.key, x)} engine={engine} />
            </div>
          ))}
        </div>

        <details className="advanced">
          <summary>Gelişmiş · tüm ayarlar</summary>
          {workflow.steps.map((s, i) => {
            const op = opsByName.get(s.op);
            if (!op || s.op === "save" || s.op === "load") return null;
            const key = stepKey(s, i);
            return (
              <div key={key} className="adv-step">
                <h3>{i + 1}. {op.title} <span className="help">· {s.op}</span></h3>
                <div className="grid">
                  {op.fields.filter((f) => f.name !== "out").map((f) => (
                    <div key={f.name} className={f.type === "text" || f.type === "keyframes" || f.type === "loras" ? "wide" : ""}>
                      <FieldInput id={`adv-${i}-${f.name}`} field={f} value={s[f.name]} models={engine.models} clips={clipsBefore(i)}
                        onChange={(x) => setOverrides((all) => ({ ...all, [taskId]: { ...(all[taskId] ?? {}), [key]: { ...((all[taskId] ?? {})[key] ?? {}), [f.name]: x } } }))} />
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </details>

        {problems.length ? <ul className="problems">{problems.map((p) => <li key={p}>{p}</li>)}</ul> : null}
        <div className="actions">
          <span className="note" style={{ flex: 1 }}>
            {lacking.length ? `Eksik: ${lacking.join(", ")}` : `Çıktılar: ${outDir}/`}
          </span>
          <button type="button" className="btn ghost" onClick={() => {
            void navigator.clipboard?.writeText(JSON.stringify(workflow, null, 2));
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          }}>{copied ? "Kopyalandı" : "JSON'u kopyala"}</button>
          <button type="button" className="btn" onClick={() => props.openGraph(workflow, template.title)}>Grafikte aç</button>
          <button type="button" className="btn primary" disabled={busy || lacking.length > 0 || !available(template) || !engine.connected} onClick={() => void run()}>
            ▶ Çalıştır
          </button>
        </div>
      </section>

      <Results engine={engine} onOpenWorkflow={(j) => j.workflow && props.openGraph(j.workflow, j.title ?? `İş ${j.id}`)} />
    </div>
  );
}

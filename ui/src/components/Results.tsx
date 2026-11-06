import { useEffect, useState } from "react";
import { frameUrl, type JobOutput } from "../api";
import type { Engine, LiveJob } from "../engine";

const opNames: Record<string, string> = {
  txt2img: "resim", img2img: "yeniden çizim", inpaint: "inpaint", t2v: "video", morph: "morph", zoom: "zoom", outpaint: "outpaint",
  canny: "kenarlar", depth: "derinlik", parallax: "3D", upscale: "büyütme", interpolate: "akıcılaştırma", load: "yükleme",
  resize: "boyut", crop: "kırpma", concat: "birleştirme", blend: "harmanlama", save: "kaydet", depth_mesh: "3D model", render: "3D görüntü",
};
const stateNames: Record<string, string> = { queued: "sırada", running: "çalışıyor", done: "bitti", failed: "hata", cancelled: "iptal" };

export function overallFraction(j: LiveJob) {
  if (j.state === "done") return 1;
  if (!j.steps) return 0;
  return Math.min(1, (Math.max(0, j.step - 1) + Math.max(0, j.fraction)) / j.steps);
}

function elapsed(j: LiveJob) {
  const end = j.finished || Date.now() / 1000;
  return j.started ? `${Math.max(0, end - j.started).toFixed(1)} s` : "";
}

export function ClipViewer(props: { job: number; output: JobOutput; onClose(): void }) {
  const { job, output } = props;
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(output.frames > 1);
  useEffect(() => {
    if (!playing || output.frames < 2) return;
    const t = setInterval(() => setFrame((f) => (f + 1) % output.frames), 1000 / 12);
    return () => clearInterval(t);
  }, [playing, output.frames]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && props.onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [props]);
  return (
    <div className="viewer" role="dialog" aria-modal="true" aria-label={`${output.clip} klibi`} onClick={props.onClose}>
      <div className="viewer-box" onClick={(e) => e.stopPropagation()}>
        <img src={frameUrl(job, output.clip, frame)} alt={`${output.clip}, kare ${frame + 1}`} />
        <div className="row">
          <span className="note">
            {output.clip} · {opNames[output.op] ?? output.op} · {output.frames > 1 ? `kare ${frame + 1} / ${output.frames}` : "tek kare"}
          </span>
          <div className="toolbar">
            {output.frames > 1 ? (
              <>
                <button type="button" className="btn small" onClick={() => setPlaying((p) => !p)}>{playing ? "Durdur" : "Oynat"}</button>
                <input type="range" aria-label="Kare" min={0} max={output.frames - 1} value={frame} onChange={(e) => { setPlaying(false); setFrame(Number(e.target.value)); }} />
              </>
            ) : null}
            <a className="btn small" href={frameUrl(job, output.clip, frame)} download={`${output.clip}_${frame}.jpg`}>İndir</a>
            <button type="button" className="btn small" onClick={props.onClose}>Kapat</button>
          </div>
        </div>
      </div>
    </div>
  );
}

export function Results(props: { engine: Engine; onOpenWorkflow?(job: LiveJob): void }) {
  const { jobs, cancel } = props.engine;
  const [viewing, setViewing] = useState<{ job: number; output: JobOutput }>();
  const active = jobs.find((j) => j.state === "running") ?? jobs.find((j) => j.state === "queued");
  const queued = jobs.filter((j) => j.state === "queued").length;

  return (
    <aside className="results" aria-label="Sonuçlar">
      <section className="panel card" aria-live="polite">
        <div className="row">
          <h2>{active ? (active.state === "running" ? "Çalışıyor" : "Sırada") : "Boşta"}</h2>
          {active ? <span className="state running">{opNames[active.op] ?? active.op} · {active.stage || "başlıyor"}</span> : null}
        </div>
        <div className="preview">
          {active?.preview ? <img src={active.preview} alt="Canlı önizleme" /> : <span>{active ? "Canlı önizleme ilk adımdan sonra gelir" : "Bir görev çalıştırınca önizleme burada görünür"}</span>}
        </div>
        {active ? (
          <>
            <div className="bar" aria-label="İlerleme"><div style={{ width: `${Math.round(overallFraction(active) * 100)}%` }} /></div>
            <div className="row note">
              <span>adım {active.step || 0} / {active.steps || "?"} · {elapsed(active)}{queued > (active.state === "queued" ? 1 : 0) ? ` · sırada ${queued}` : ""}</span>
              <button type="button" className="btn small danger" onClick={() => void cancel(active.id)}>İptal</button>
            </div>
          </>
        ) : null}
      </section>

      <section className="history" aria-label="Geçmiş">
        <h2 className="section-title">Geçmiş</h2>
        {jobs.length === 0 ? <p className="note">Henüz çalıştırma yok.</p> : null}
        {jobs.map((j) => {
          const outs = j.outputs.filter((o) => o.clip && o.frames > 0);
          return (
            <article key={j.id} className="job">
              <div className="row">
                <strong>{j.title ?? `İş ${j.id}`}</strong>
                <span className={`state ${j.state}`}>{stateNames[j.state] ?? j.state}</span>
              </div>
              {j.state === "failed" ? <div className="error-text">{j.error}</div> : null}
              {outs.length && j.clipsKept === false && j.state !== "running" && j.state !== "queued" ? (
                <div className="note">{outs.length} çıktı · kareler artık bellekte değil (kaydedilen dosyalar duruyor)</div>
              ) : outs.length ? (
                <div className="thumbs">
                  {outs.slice(-6).map((o) => (
                    <button key={o.step} type="button" className="thumb" onClick={() => setViewing({ job: j.id, output: o })} aria-label={`${o.clip} çıktısını aç`}>
                      <img src={frameUrl(j.id, o.clip, 0, 256)} alt="" loading="lazy" />
                      <span>{o.clip}{o.frames > 1 ? ` · ${o.frames}` : ""}</span>
                    </button>
                  ))}
                </div>
              ) : null}
              <div className="row note">
                <span>{elapsed(j)}</span>
                {j.workflow && props.onOpenWorkflow ? (
                  <button type="button" className="btn small ghost" onClick={() => props.onOpenWorkflow?.(j)}>Grafikte aç</button>
                ) : null}
              </div>
            </article>
          );
        })}
      </section>
      {viewing ? <ClipViewer job={viewing.job} output={viewing.output} onClose={() => setViewing(undefined)} /> : null}
    </aside>
  );
}

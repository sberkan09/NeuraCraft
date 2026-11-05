import { useState } from "react";
import type { Workflow } from "./api";
import { GraphMode } from "./components/GraphMode";
import { SimpleMode } from "./components/SimpleMode";
import { useEngine } from "./engine";

function Logo() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="#8AB4FF" strokeWidth="1.8" strokeLinejoin="round">
      <path d="M12 2 L21 7 L21 17 L12 22 L3 17 L3 7 Z" />
      <path d="M8 9.5 L12 7 L16 9.5 L16 14.5 L12 17 L8 14.5 Z" />
    </svg>
  );
}

export default function App() {
  const engine = useEngine();
  const [mode, setMode] = useState<"simple" | "graph">("simple");
  const [graph, setGraph] = useState<{ workflow: Workflow; title: string }>({ workflow: { steps: [] }, title: "Yeni akış" });
  const running = engine.jobs.filter((j) => j.state === "running" || j.state === "queued").length;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand"><Logo /> NeuraCraft Studio</div>
        <div className="segmented" role="group" aria-label="Mod">
          <button type="button" aria-pressed={mode === "simple"} onClick={() => setMode("simple")}>Basit</button>
          <button type="button" aria-pressed={mode === "graph"} onClick={() => setMode("graph")}>Grafik</button>
        </div>
        <span className="spacer" />
        <span className="chip" title={engine.info?.baseDir}>
          <span className={`dot ${engine.connected ? "on" : "off"}`} />
          {engine.connected ? engine.info?.device ?? "bağlanıyor…" : "motora bağlı değil"}
          {engine.info?.gpuAttention ? " + GPU attention" : ""}
        </span>
        {running ? <span className="chip">Kuyruk {running}</span> : null}
      </header>

      {engine.loadError && !engine.connected ? (
        <div className="empty">
          <p>Motora ulaşılamadı: {engine.loadError}</p>
          <p className="note">Önce motoru başlat: <span className="mono">./bin/NeuraCraft studio</span> (ya da <span className="mono">serve</span>), sonra bu sayfayı yenile.</p>
        </div>
      ) : mode === "simple" ? (
        <SimpleMode engine={engine} openGraph={(workflow, title) => { setGraph({ workflow, title }); setMode("graph"); }} />
      ) : (
        <GraphMode engine={engine} workflow={graph.workflow} title={graph.title} onChange={(workflow) => setGraph((g) => ({ ...g, workflow }))} />
      )}
    </div>
  );
}

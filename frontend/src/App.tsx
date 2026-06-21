import {
  Archive,
  BarChart3,
  Boxes,
  Check,
  ChevronDown,
  CircleAlert,
  Cloud,
  Copy,
  Cpu,
  Database,
  Download,
  FileText,
  FlaskConical,
  Gauge,
  GitCompare,
  LayoutGrid,
  Loader2,
  Lock,
  type LucideIcon,
  MessageSquare,
  Moon,
  Plus,
  RefreshCw,
  Send,
  ShieldCheck,
  Sun,
  Terminal,
  Upload,
  X
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type * as React from "react";
import { archiveRuns, cancelRun, compareRuns, evalRun, exportHf, generateEvalStarter, generatePreferences, generateSftStarter, generateText, importHf, importRun, initDatasetPack, inspectTuning, listRuns, loadLeaderboard, loadPackEditor, loadPresets, loadReport, loadRun, loadStatus, remoteModalPull, remoteModalStart, remoteStatus, runLogStreamUrl, savePackEditor, serveStart, serveStatus, serveStop, startRun, trainHfSft } from "./api";
import type { GenerateResult, JobStatus, ModelConfig, RunDetail, RunLog, RunSummary, Tone } from "./types";
import { compactNumber, fixed, latestRun, lossPoints, parseEvalScore, percent, releaseTone, runTone, statusLabel } from "./utils";

type SectionId = "overview" | "runs" | "compare" | "playground" | "training" | "eval" | "release" | "dataset" | "cloud" | "settings";

const NAV: Array<{ id: SectionId; label: string; icon: LucideIcon; group: "main" | "model" | "system" }> = [
  { id: "overview", label: "Overview", icon: LayoutGrid, group: "main" },
  { id: "runs", label: "Runs", icon: Boxes, group: "main" },
  { id: "compare", label: "Compare", icon: GitCompare, group: "main" },
  { id: "playground", label: "Playground", icon: MessageSquare, group: "model" },
  { id: "training", label: "Training", icon: Gauge, group: "model" },
  { id: "eval", label: "Evaluation", icon: BarChart3, group: "model" },
  { id: "release", label: "Release gate", icon: Lock, group: "model" },
  { id: "dataset", label: "Dataset", icon: Database, group: "system" },
  { id: "cloud", label: "Cloud", icon: Cloud, group: "system" },
  { id: "settings", label: "Settings", icon: ShieldCheck, group: "system" }
];

const DEFAULT_SETTINGS = {
  theme: "dark",
  autoRefresh: true,
  refreshSeconds: 5,
  hfToken: "",
  defaultDataset: "HuggingFaceTB/smollm-corpus"
};
type SettingsState = typeof DEFAULT_SETTINGS;

function readSettings(): SettingsState {
  try {
    return { ...DEFAULT_SETTINGS, ...JSON.parse(localStorage.getItem("picochat.forge.settings") || "{}") };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export default function App() {
  const [section, setSection] = useState<SectionId>("overview");
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedRunName, setSelectedRunName] = useState("");
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [jobs, setJobs] = useState<JobStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [settings, setSettings] = useState<SettingsState>(() => readSettings());
  const [logTarget, setLogTarget] = useState<{ run?: string; job?: string } | null>(null);
  const [newRun, setNewRun] = useState<{ pack?: string } | null>(null);
  const [reportTarget, setReportTarget] = useState<{ run: string; report: string } | null>(null);
  const [navOpen, setNavOpen] = useState(false);
  const lastRefresh = useRef<Date | null>(null);

  const selectedRun = useMemo(() => runs.find((r) => r.name === selectedRunName) || latestRun(runs), [runs, selectedRunName]);
  const activeJob = useMemo(
    () => jobs.find((j) => j.run_name === selectedRun?.name) || jobs.find((j) => j.state === "running") || null,
    [jobs, selectedRun?.name]
  );

  const refresh = async (quiet = false) => {
    if (document.activeElement?.closest("input, textarea, select, [contenteditable='true']")) return;
    try {
      if (!quiet) setLoading(true);
      const [runsPayload, statusPayload] = await Promise.all([listRuns(), loadStatus().catch(() => ({ jobs: [], job: null }))]);
      setRuns(runsPayload.runs || []);
      setJobs(statusPayload.jobs || []);
      const nextName = selectedRunName || latestRun(runsPayload.runs || [])?.name || "";
      if (nextName) {
        setSelectedRunName(nextName);
        setDetail(await loadRun(nextName));
      } else {
        setDetail(null);
      }
      setError("");
      lastRefresh.current = new Date();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { refresh(false); /* eslint-disable-next-line */ }, []);
  useEffect(() => {
    const fromHash = window.location.hash.replace("#", "");
    if (fromHash && NAV.some((n) => n.id === fromHash)) setSection(fromHash as SectionId);
  }, []);
  useEffect(() => { if (window.location.hash.replace("#", "") !== section) window.location.hash = section; }, [section]);
  useEffect(() => { if (selectedRunName) loadRun(selectedRunName).then(setDetail).catch((e) => setError(e.message)); }, [selectedRunName]);
  useEffect(() => {
    document.documentElement.dataset.theme = settings.theme;
    localStorage.setItem("picochat.forge.settings", JSON.stringify(settings));
  }, [settings]);
  useEffect(() => {
    if (!settings.autoRefresh) return;
    const ms = Math.max(2, Number(settings.refreshSeconds) || 5) * 1000;
    const handle = window.setInterval(() => refresh(true), ms);
    return () => window.clearInterval(handle);
    // eslint-disable-next-line
  }, [settings.autoRefresh, settings.refreshSeconds, selectedRunName]);

  const tone = releaseTone(detail);
  const update = <K extends keyof SettingsState>(k: K, v: SettingsState[K]) => setSettings((s) => ({ ...s, [k]: v }));

  const launchSmoke = async (preflightOnly = false) => {
    const datasetPack = (detail as any)?.summary?.artifacts?.dataset_pack || (detail as any)?.summary?.config?.dataset_pack || "examples/tiny_dataset_pack.json";
    setBusy(preflightOnly ? "Running preflight" : "Launching smoke run");
    try {
      const started = await startRun({
        run_name: `${preflightOnly ? "preflight" : "ui-smoke"}-${Date.now()}`,
        dataset_pack: datasetPack, preset: "smoke", preflight_only: preflightOnly, sft_peft: "none", sft_steps: 1, base_steps: 1
      });
      setLogTarget({ job: started.job?.id, run: started.job?.run_name });
      await refresh(true);
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); } finally { setBusy(""); }
  };

  const archiveSelected = async () => {
    if (!selectedRun?.name) return;
    setBusy("Archiving run");
    try { await archiveRuns([selectedRun.name]); await refresh(true); }
    catch (err) { setError(err instanceof Error ? err.message : String(err)); } finally { setBusy(""); }
  };

  const ctx: SectionProps = {
    section, runs, selectedRun: selectedRun ?? null, detail, jobs, settings, tone,
    update, launchSmoke, archiveSelected, selectRun: setSelectedRunName,
    openLogs: () => selectedRun && setLogTarget({ run: selectedRun.name, job: activeJob?.id }),
    openLogsFor: (t) => setLogTarget(t),
    openNew: (pack?: string) => setNewRun({ pack }),
    openReport: (report: string) => selectedRun && setReportTarget({ run: selectedRun.name, report }),
    setSection
  };

  return (
    <div className="pc">
      <aside className={`pc-rail ${navOpen ? "open" : ""}`}>
        <div className="pc-brand">
          <span className="pc-logo"><img src="/assets/picochat-symbol.svg" alt="" /></span>
          <div><strong>Picochat</strong><span>SLM factory</span></div>
        </div>
        <nav className="pc-nav">
          <NavGroup items={NAV.filter((i) => i.group === "main")} {...ctx} close={() => setNavOpen(false)} />
          <NavGroup title="Model" items={NAV.filter((i) => i.group === "model")} {...ctx} close={() => setNavOpen(false)} />
          <NavGroup title="Pipeline" items={NAV.filter((i) => i.group === "system")} {...ctx} close={() => setNavOpen(false)} />
        </nav>
        <div className="pc-rail-foot">
          <button className="pc-theme" onClick={() => update("theme", settings.theme === "dark" ? "light" : "dark")}>
            {settings.theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}
            <span>{settings.theme === "dark" ? "Light" : "Dark"}</span>
          </button>
          <div className="pc-live"><span className={activeJob?.state === "running" ? "run" : ""} />{activeJob?.state === "running" ? "Training" : "Idle"} · {settings.autoRefresh ? `${settings.refreshSeconds}s` : "paused"}</div>
        </div>
      </aside>
      {navOpen ? <button className="pc-scrim" aria-label="Close" onClick={() => setNavOpen(false)} /> : null}

      <main className="pc-main">
        <header className="pc-top">
          <div className="pc-top-l">
            <button className="pc-burger" onClick={() => setNavOpen(true)} aria-label="Menu"><LayoutGrid size={18} /></button>
            <span className="pc-bc">{NAV.find((n) => n.id === section)?.label}</span>
            {selectedRun ? <StatusDot tone={tone} label={statusLabel(tone)} /> : null}
          </div>
          <div className="pc-top-r">
            <RunPicker runs={runs} value={selectedRun?.name || ""} onChange={setSelectedRunName} />
            <button className="pc-btn ghost" onClick={() => refresh(false)} disabled={loading || !!busy}><RefreshCw size={15} className={loading ? "spin" : ""} /> Sync</button>
            <button className="pc-btn primary" onClick={() => setNewRun({})}><Plus size={15} /> New model</button>
          </div>
        </header>

        <section className="pc-content">
          {error ? <Banner tone="block" title="Action failed" body={error} onClose={() => setError("")} /> : null}
          {busy ? <Banner tone="info" title={busy} body="Running a backend action — live status and logs update automatically." /> : null}
          {!loading && runs.length === 0 && section === "overview" ? <Welcome {...ctx} /> : render(ctx)}
        </section>
      </main>

      {logTarget ? <LogModal target={logTarget} onClose={() => setLogTarget(null)} onCancel={cancelRun} /> : null}
      {newRun ? <NewRunModal initialPack={newRun.pack} onClose={() => setNewRun(null)} onLaunched={(t) => { setNewRun(null); setLogTarget(t); refresh(true); }} /> : null}
      {reportTarget ? <ReportModal target={reportTarget} onClose={() => setReportTarget(null)} /> : null}
    </div>
  );
}

type SectionProps = {
  section: SectionId;
  runs: RunSummary[];
  selectedRun: RunSummary | null;
  detail: RunDetail | null;
  jobs: JobStatus[];
  settings: SettingsState;
  tone: Tone;
  update: <K extends keyof SettingsState>(k: K, v: SettingsState[K]) => void;
  launchSmoke: (preflightOnly?: boolean) => void;
  archiveSelected: () => void;
  selectRun: (name: string) => void;
  openLogs: () => void;
  openLogsFor: (t: { run?: string; job?: string }) => void;
  openNew: (pack?: string) => void;
  openReport: (report: string) => void;
  setSection: (s: SectionId) => void;
};

function render(p: SectionProps) {
  switch (p.section) {
    case "runs": return <RunsView {...p} />;
    case "compare": return <CompareView {...p} />;
    case "playground": return <PlaygroundView {...p} />;
    case "training": return <TrainingView {...p} />;
    case "eval": return <EvalView {...p} />;
    case "release": return <ReleaseView {...p} />;
    case "dataset": return <DatasetView {...p} />;
    case "cloud": return <RemoteView {...p} />;
    case "settings": return <SettingsView {...p} />;
    default: return <OverviewView {...p} />;
  }
}

/* --------------------------------------------------------------- overview */

function modelConfig(detail: RunDetail | null, run: RunSummary | null): ModelConfig {
  const c = (detail as any)?.summary?.config || {};
  return {
    n_layer: Number(c.n_layer) || 1,
    n_head: Number(c.n_head) || 1,
    n_embd: Number(c.n_embd) || 0,
    context_size: Number(c.context_size) || run?.context_size || 0,
    params: (detail as any)?.summary?.base?.num_parameters || run?.num_parameters || 0
  };
}

function OverviewView(p: SectionProps) {
  const { detail, selectedRun, tone, setSection } = p;
  const s = (detail as any)?.summary || {};
  const cfg = modelConfig(detail, selectedRun);
  const valBpb = s.sft?.final_val_bpb ?? s.base?.final_val_bpb;
  const tps = s.sft?.throughput?.tokens_per_second ?? s.base?.throughput?.tokens_per_second;

  return (
    <div className="pc-stack">
      <section className="pc-overview-top">
        <div className="pc-overview-id">
          <span className="pc-eyebrow">{s.config?.scale || "model"}</span>
          <strong>{selectedRun?.name || "No run selected"}</strong>
          <span>{compactNumber(cfg.params)} parameters · {cfg.n_layer} blocks · {cfg.n_head} heads · d{cfg.n_embd || "--"}</span>
        </div>
        <button className="pc-btn" onClick={() => setSection("playground")}><MessageSquare size={15} /> Open playground</button>
      </section>

      <div className="pc-grid two">
        <VerdictCard detail={detail} tone={tone} setSection={setSection} />
        <div className="pc-kpis">
          <Kpi label="Eval pass" value={percent(selectedRun?.pass_rate)} sub={selectedRun?.eval_score || "--"} tone={runTone(selectedRun)} />
          <Kpi label="SFT val loss" value={fixed(selectedRun?.sft_val_loss, 3)} sub="lower is better" />
          <Kpi label="Val BPB" value={fixed(valBpb, 3)} sub="bits / byte" />
          <Kpi label="Throughput" value={tps ? compactNumber(tps) : "--"} sub="tokens / s" />
        </div>
      </div>

      <div className="pc-grid two">
        <Panel title="Training loss" sub="base → SFT" >
          <LossChart points={lossPoints(detail)} />
        </Panel>
        <Panel title="Architecture">
          <Spec rows={{
            Layers: cfg.n_layer, Heads: cfg.n_head, "Embedding dim": cfg.n_embd || "--",
            Context: cfg.context_size || "--", Norm: s.config?.norm_type || "--",
            Position: s.config?.position_encoding || "--", Activation: s.config?.activation || "--",
            Precision: s.config?.precision || "--", Optimizer: s.config?.base_optimizer || "--",
            "Vocab size": (detail as any)?.summary?.tokenizer?.vocab_size || "--"
          }} />
        </Panel>
      </div>

      <Panel title="Recent runs" action={<button className="pc-link" onClick={() => setSection("runs")}>All runs →</button>}>
        <RunTable runs={p.runs.slice(-6).reverse()} selected={selectedRun?.name} onSelect={p.selectRun} />
      </Panel>
    </div>
  );
}

function VerdictCard({ detail, tone, setSection }: { detail: RunDetail | null; tone: Tone; setSection: (s: SectionId) => void }) {
  const reasons = gateReasons(detail);
  const title = tone === "pass" ? "Ready for handoff" : tone === "block" ? "Release blocked" : tone === "neutral" ? "No evidence yet" : "Needs review";
  return (
    <button className={`pc-verdict ${tone}`} onClick={() => setSection("release")}>
      <div className="pc-verdict-head">
        <StatusGlyph tone={tone} />
        <div><span className="pc-eyebrow">Release gate</span><strong>{title}</strong></div>
        <ChevronDown size={16} className="pc-verdict-arrow" />
      </div>
      <ul className="pc-verdict-list">
        {reasons.slice(0, 3).map((r, i) => (
          <li key={i} className={r.tone}><StatusGlyph tone={r.tone} small /> {r.label}</li>
        ))}
      </ul>
    </button>
  );
}

/* ------------------------------------------------------------------ runs */

function RunsView(p: SectionProps) {
  const { runs, selectedRun, detail, launchSmoke, openLogs, archiveSelected } = p;
  const [importing, setImporting] = useState(false);
  return (
    <div className="pc-stack">
      {importing ? <ImportRunModal onClose={() => setImporting(false)} /> : null}
      <div className="pc-toolbar">
        <div className="pc-toolbar-meta"><strong>{runs.length}</strong> runs in the local bank</div>
        <div className="pc-row">
          <button className="pc-btn" onClick={() => launchSmoke(true)}><ShieldCheck size={15} /> Preflight</button>
          <button className="pc-btn" onClick={() => launchSmoke(false)}><Terminal size={15} /> Smoke run</button>
          <button className="pc-btn ghost" onClick={() => setImporting(true)}><Upload size={15} /> Import</button>
          <button className="pc-btn ghost" onClick={openLogs}>Logs</button>
          {selectedRun ? <button className="pc-btn ghost danger" onClick={archiveSelected}><Archive size={15} /> Archive</button> : null}
        </div>
      </div>
      <Panel flush>
        <RunTable runs={[...runs].reverse()} selected={selectedRun?.name} onSelect={p.selectRun} detailed />
      </Panel>
      {selectedRun ? (
        <div className="pc-grid two">
          <Spec title="Selected run" rows={{
            Name: selectedRun.name,
            Parameters: compactNumber(selectedRun.num_parameters),
            "Eval pass": `${percent(selectedRun.pass_rate)} (${selectedRun.eval_score})`,
            "Base val loss": fixed(selectedRun.base_val_loss, 3),
            "SFT val loss": fixed(selectedRun.sft_val_loss, 3),
            Context: selectedRun.context_size || "--",
            "Truncated SFT": selectedRun.truncated_examples ?? "--"
          }} />
          <Spec title="Run config" rows={{
            Scale: (detail as any)?.summary?.config?.scale || "--",
            Device: (detail as any)?.summary?.config?.device || "--",
            Precision: (detail as any)?.summary?.config?.precision || "--",
            "torch.compile": (detail as any)?.summary?.config?.torch_compile ? "on" : "off",
            DDP: (detail as any)?.summary?.config?.ddp ? "on" : "off",
            Seed: (detail as any)?.summary?.config?.seed ?? "--"
          }} />
        </div>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------- playground */

type ChatMsg = { role: "user" | "assistant"; content: string; meta?: string };

function PlaygroundView({ selectedRun, detail }: SectionProps) {
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState("");
  const [checkpoint, setCheckpoint] = useState<"base" | "sft">("sft");
  const [temperature, setTemperature] = useState(0.8);
  const [maxTokens, setMaxTokens] = useState(80);
  const [sending, setSending] = useState(false);
  const [genError, setGenError] = useState("");
  const threadRef = useRef<HTMLDivElement | null>(null);
  const runName = selectedRun?.name || "";

  useEffect(() => { threadRef.current?.scrollTo({ top: threadRef.current.scrollHeight, behavior: "smooth" }); }, [messages, sending]);

  const send = async () => {
    const prompt = input.trim();
    if (!prompt || !runName || sending) return;
    setMessages((m) => [...m, { role: "user", content: prompt }]);
    setInput("");
    setSending(true);
    setGenError("");
    try {
      const res: GenerateResult = await generateText({ run: runName, prompt, checkpoint, max_new_tokens: maxTokens, temperature });
      const text = (res.completion || res.text || "").trim() || "(empty completion)";
      setMessages((m) => [...m, { role: "assistant", content: text, meta: `${res.completion_tokens} tok · ${res.finish_reason} · ${checkpoint}` }]);
    } catch (err) {
      setGenError(err instanceof Error ? err.message : String(err));
    } finally { setSending(false); }
  };

  return (
    <div className="pc-play">
      <div className="pc-play-main">
        <div className="pc-play-head">
          <div><strong>{runName || "No run selected"}</strong><span>Native PyTorch inference · CPU</span></div>
          <Segmented value={checkpoint} options={["base", "sft"]} onChange={(v) => setCheckpoint(v as "base" | "sft")} />
        </div>
        <div className="pc-thread" ref={threadRef}>
          {messages.length === 0 && !sending ? (
            <div className="pc-thread-empty">
              <FlaskConical size={26} />
              <p>Talk to the model you trained.</p>
              <div className="pc-suggest">
                {["What is Picochat?", "Write a haiku about GPUs.", "2 + 2 ="].map((q) => (
                  <button key={q} onClick={() => setInput(q)}>{q}</button>
                ))}
              </div>
            </div>
          ) : null}
          {messages.map((m, i) => (
            <div key={i} className={`pc-msg ${m.role}`}>
              <div className="pc-msg-role">{m.role === "user" ? "You" : "picochat"}</div>
              <div className="pc-msg-body">{m.content}{m.meta ? <span className="pc-msg-meta">{m.meta}</span> : null}</div>
            </div>
          ))}
          {sending ? <div className="pc-msg assistant"><div className="pc-msg-role">picochat</div><div className="pc-msg-body"><span className="pc-typing"><i /><i /><i /></span></div></div> : null}
          {genError ? <Banner tone="block" title="Generation failed" body={genError} onClose={() => setGenError("")} /> : null}
        </div>
        <div className="pc-composer">
          <textarea
            value={input}
            placeholder={runName ? "Message the model…" : "Select a trained run first"}
            disabled={!runName}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
          />
          <button className="pc-btn primary" onClick={send} disabled={!input.trim() || !runName || sending}>
            {sending ? <Loader2 size={15} className="spin" /> : <Send size={15} />}
          </button>
        </div>
      </div>
      <aside className="pc-play-side">
        <Panel title="Sampling">
          <Slider label="Temperature" value={temperature} min={0} max={2} step={0.05} onChange={setTemperature} />
          <Slider label="Max new tokens" value={maxTokens} min={8} max={240} step={8} onChange={(v) => setMaxTokens(Math.round(v))} fmt={(v) => String(Math.round(v))} />
          <div className="pc-hint">Checkpoint: <code>{checkpoint}</code>. The <code>sft</code> checkpoint is chat-tuned; <code>base</code> is raw pretraining.</div>
        </Panel>
        <Panel title="Model">
          <Spec rows={{
            Run: runName || "--",
            Eval: selectedRun ? percent(selectedRun.pass_rate) : "--",
            Params: compactNumber(selectedRun?.num_parameters),
            Context: (detail as any)?.summary?.config?.context_size || selectedRun?.context_size || "--"
          }} />
        </Panel>
        <Panel title="Share / API access">
          <ServePanel run={runName} />
        </Panel>
        <Panel title="Export">
          <ExportPanel run={runName} />
        </Panel>
      </aside>
    </div>
  );
}

function ExportPanel({ run }: { run: string }) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [result, setResult] = useState<{ out_dir: string } | null>(null);
  const go = async () => {
    setBusy(true); setErr("");
    try { setResult(await exportHf(run)); } catch (e) { setErr(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); }
  };
  if (!run) return <div className="pc-hint">Select a run to export it.</div>;
  return (
    <div className="pc-serve">
      {result ? (
        <div className="pc-import-result">
          <div><strong>Exported to Hugging Face format.</strong><code>{result.out_dir}</code></div>
        </div>
      ) : (
        <>
          <button className="pc-btn" onClick={go} disabled={busy}>{busy ? <Loader2 size={15} className="spin" /> : <Download size={15} />} Export to Hugging Face</button>
          <div className="pc-hint">Converts this run to a transformers model + model card you can load or upload anywhere.</div>
        </>
      )}
      {err ? <Banner tone="block" title="Export failed" body={err} onClose={() => setErr("")} /> : null}
    </div>
  );
}

function ServePanel({ run }: { run: string }) {
  const [server, setServer] = useState<Record<string, any> | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [copied, setCopied] = useState(false);

  const pick = (payload: { servers?: Array<Record<string, any>>; server?: Record<string, any> | null }) =>
    (payload.servers || []).find((s) => s.run === run) || payload.server || null;

  useEffect(() => {
    setServer(null);
    if (!run) return;
    serveStatus().then((s) => setServer(pick(s))).catch(() => {});
    // eslint-disable-next-line
  }, [run]);

  const start = async () => {
    setBusy(true); setErr("");
    try { setServer(pick(await serveStart(run))); }
    catch (e) { setErr(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); }
  };
  const stop = async () => {
    setBusy(true); setErr("");
    try { await serveStop(run); setServer(null); }
    catch (e) { setErr(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); }
  };

  if (!run) return <div className="pc-hint">Select a run to share it as an API.</div>;

  const host = (typeof window !== "undefined" && window.location.hostname) || "127.0.0.1";
  const base = server ? `http://${host}:${server.port}/v1` : "";
  const auth = server?.api_key ? ` -H "authorization: Bearer ${server.api_key}"` : "";
  const snippet = server
    ? `curl ${base}/chat/completions -H "content-type: application/json"${auth} -d '{"model":"${run}","messages":[{"role":"user","content":"hello"}]}'`
    : "";
  const copy = () => { navigator.clipboard?.writeText(snippet); setCopied(true); window.setTimeout(() => setCopied(false), 1500); };

  return (
    <div className="pc-serve">
      {server ? (
        <>
          <div className="pc-serve-on"><StatusDot tone="running" label="Serving" /><button className="pc-btn ghost danger" onClick={stop} disabled={busy}>Stop</button></div>
          <label className="pc-field">Endpoint<input readOnly value={base} onFocus={(e) => e.target.select()} /></label>
          {server.api_key ? <label className="pc-field">API key<input readOnly value={server.api_key} onFocus={(e) => e.target.select()} /></label> : null}
          <div className="pc-serve-snip">
            <pre>{snippet}</pre>
            <button className="pc-btn ghost" onClick={copy}><Copy size={14} /> {copied ? "Copied" : "Copy"}</button>
          </div>
          <div className="pc-hint">OpenAI-compatible — point any client at <code>{base}</code> using model <code>{run}</code>.</div>
        </>
      ) : (
        <>
          <button className="pc-btn primary" onClick={start} disabled={busy}>{busy ? <Loader2 size={15} className="spin" /> : <Send size={15} />} Serve for your team</button>
          <div className="pc-hint">Starts an OpenAI-compatible endpoint your team can call from code.</div>
        </>
      )}
      {err ? <Banner tone="block" title="Serve failed" body={err} onClose={() => setErr("")} /> : null}
    </div>
  );
}

/* -------------------------------------------------------------- training */

function TrainingView({ detail, selectedRun, openLogs, openReport }: SectionProps) {
  const s = (detail as any)?.summary || {};
  const tps = s.sft?.throughput?.tokens_per_second ?? s.base?.throughput?.tokens_per_second;
  return (
    <div className="pc-stack">
      <ReportLinks reports={detail?.reports} openReport={openReport} only={["base", "sft"]} />
      <div className="pc-kpis four">
        <Kpi label="Base val loss" value={fixed(s.base?.final_val_loss ?? selectedRun?.base_val_loss, 3)} sub="final" />
        <Kpi label="SFT val loss" value={fixed(s.sft?.final_val_loss ?? selectedRun?.sft_val_loss, 3)} sub="final" />
        <Kpi label="Val BPB" value={fixed(s.sft?.final_val_bpb ?? s.base?.final_val_bpb, 3)} sub="bits/byte" />
        <Kpi label="Throughput" value={tps ? compactNumber(tps) : "--"} sub="tokens/s" />
      </div>
      <Panel title="Loss curve" action={<button className="pc-link" onClick={openLogs}>Live log →</button>}>
        <LossChart points={lossPoints(detail)} />
      </Panel>
      <div className="pc-grid two">
        <Spec title="Hyperparameters" rows={{
          "Base LR": s.config?.base_learning_rate ?? "--",
          "SFT LR": s.config?.sft_learning_rate ?? "--",
          "Base steps": s.config?.base_steps ?? "--",
          "SFT steps": s.config?.sft_steps ?? "--",
          "Batch (base/sft)": `${s.config?.base_batch_size ?? "--"} / ${s.config?.sft_batch_size ?? "--"}`,
          "Stop reason": s.base?.stop_reason || "--"
        }} />
        <Spec title="Hardware & runtime" rows={{
          Device: s.config?.device || "--",
          Precision: s.config?.precision || "--",
          "torch.compile": s.config?.torch_compile ? "on" : "off",
          "Grad checkpointing": s.config?.gradient_checkpointing ? "on" : "off",
          DDP: s.config?.ddp ? "on" : "off",
          "Packing efficiency": s.sft?.packing_efficiency != null ? percent(s.sft.packing_efficiency) : "--"
        }} />
      </div>
    </div>
  );
}

/* ----------------------------------------------------------------- eval */

function EvalView({ detail, selectedRun, openReport, openLogsFor }: SectionProps) {
  const s = (detail as any)?.summary || {};
  const ev = s.eval || {};
  const score = parseEvalScore(selectedRun?.eval_score);
  const cats: Record<string, any> = ev.category_breakdown || {};
  const catRows = Object.entries(cats).slice(0, 8);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const reEval = async () => {
    if (!selectedRun?.name) return;
    setBusy(true); setErr("");
    try { const r = await evalRun(selectedRun.name); openLogsFor({ job: r.job?.id, run: r.job?.run_name }); }
    catch (e) { setErr(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); }
  };
  return (
    <div className="pc-stack">
      <div className="pc-row">
        <ReportLinks reports={detail?.reports} openReport={openReport} only={["eval", "honesty"]} />
        {selectedRun ? <button className="pc-btn ghost" onClick={reEval} disabled={busy}>{busy ? <Loader2 size={14} className="spin" /> : <RefreshCw size={14} />} Re-run eval</button> : null}
      </div>
      {err ? <Banner tone="block" title="Could not start eval" body={err} onClose={() => setErr("")} /> : null}
      <div className="pc-kpis four">
        <Kpi label="Overall" value={percent(selectedRun?.pass_rate)} sub={selectedRun?.eval_score || "--"} tone={runTone(selectedRun)} />
        <Kpi label="Domain" value={percent(ev.domain_pass_rate)} sub="answerable" />
        <Kpi label="Refusal" value={percent(ev.refusal_pass_rate)} sub="should refuse" />
        <Kpi label="Prompt echo" value={percent(ev.prompt_echo_rate || 0)} sub="keep low" tone={(ev.prompt_echo_rate || 0) > 0.1 ? "warn" : "pass"} />
      </div>
      {catRows.length ? (
        <Panel title="By category">
          <div className="pc-bars">
            {catRows.map(([name, v]) => {
              const rate = typeof v === "object" ? (v.pass_rate ?? v.accuracy ?? 0) : Number(v) || 0;
              return (
                <div className="pc-bar-row" key={name}>
                  <span className="pc-bar-label">{name}</span>
                  <div className="pc-bar-track"><i className={barTone(rate)} style={{ width: `${Math.round(rate * 100)}%` }} /></div>
                  <span className="pc-bar-val">{percent(rate)}</span>
                </div>
              );
            })}
          </div>
        </Panel>
      ) : null}
      <div className="pc-grid two">
        <Spec title="Scoreboard" rows={{
          Passed: score.passed, Failed: Math.max(0, score.total - score.passed), Total: score.total,
          "Choice accuracy": ev.choice_accuracy != null ? percent(ev.choice_accuracy) : "--",
          "Unsupported claims": ev.unsupported_claims ?? "--",
          "Length violations": ev.length_violations ?? "--"
        }} />
        <Panel title="Honesty">
          <Spec rows={{
            Status: s.honesty?.status || "--",
            "Exact leaks": s.honesty?.exact_prompt_leaks ?? "0",
            "Near leaks": s.honesty?.near_prompt_leaks ?? "0",
            "Corpus hits": s.honesty?.corpus_prompt_hits ?? "0",
            "Duplicate eval": s.honesty?.duplicate_eval_prompts ?? "0"
          }} />
        </Panel>
      </div>
    </div>
  );
}

/* --------------------------------------------------------------- release */

function ReleaseView({ detail, tone, openReport }: SectionProps) {
  const reasons = gateReasons(detail);
  return (
    <div className="pc-stack">
      <Banner
        tone={tone}
        title={tone === "pass" ? "Release gate passed" : tone === "block" ? "Release gate blocked" : tone === "neutral" ? "No release evidence yet" : "Release gate needs review"}
        body={tone === "pass" ? "Every checked gate is satisfied. This run can move to handoff." : "Resolve the failing checks below before publishing or handing off this model."}
      />
      <ReportLinks reports={detail?.reports} openReport={openReport} only={["honesty", "summary"]} />
      <Panel title="Gate checks" flush>
        <div className="pc-checks">
          {reasons.map((r, i) => (
            <div className={`pc-check ${r.tone}`} key={i}>
              <StatusGlyph tone={r.tone} />
              <div><strong>{r.label}</strong>{r.detail ? <span>{r.detail}</span> : null}</div>
              <span className="pc-chip2">{statusLabel(r.tone)}</span>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}

/* --------------------------------------------------------------- dataset */

function DatasetView({ detail, settings, openNew }: SectionProps) {
  const s = (detail as any)?.summary || {};
  const corpus = s.corpus || {};
  const preview = (detail as any)?.corpus_preview || "";
  return (
    <div className="pc-stack">
      <Panel title="Import a Hugging Face dataset" sub="Turn a public dataset into a local pack you can train on">
        <HfImport defaultDataset={settings.defaultDataset} token={settings.hfToken} openNew={openNew} />
      </Panel>
      {s.corpus || s.tokenizer ? (
        <>
          <div className="pc-kpis four">
            <Kpi label="Documents" value={corpus.num_documents != null ? compactNumber(corpus.num_documents) : "--"} sub="in corpus" />
            <Kpi label="Characters" value={corpus.num_characters != null ? compactNumber(corpus.num_characters) : "--"} sub="total" />
            <Kpi label="Dup docs" value={corpus.duplicate_document_rate != null ? percent(corpus.duplicate_document_rate) : "--"} sub="lower is better" tone={(corpus.duplicate_document_rate || 0) > 0.2 ? "warn" : "pass"} />
            <Kpi label="Non-ASCII" value={corpus.non_ascii_rate != null ? percent(corpus.non_ascii_rate) : "--"} sub="rate" />
          </div>
          <Spec title="Tokenizer" rows={{
            Type: s.tokenizer?.tokenizer_type || "--",
            "Vocab size": s.tokenizer?.vocab_size || "--",
            "Special tokens": s.tokenizer?.num_special_tokens ?? "--",
            "Text tokens": s.tokenizer?.num_text_tokens ?? "--"
          }} />
        </>
      ) : null}
      {preview ? (
        <Panel title="Corpus preview">
          <pre className="pc-pre">{String(preview).slice(0, 1600)}</pre>
        </Panel>
      ) : null}
    </div>
  );
}

function HfImport({ defaultDataset, token, openNew }: { defaultDataset: string; token: string; openNew: (pack?: string) => void }) {
  const [dataset, setDataset] = useState(defaultDataset || "HuggingFaceTB/smollm-corpus");
  const [maxRows, setMaxRows] = useState(2000);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [result, setResult] = useState<Record<string, any> | null>(null);
  const run = async () => {
    if (!dataset.trim()) { setErr("Enter a dataset id, e.g. org/name."); return; }
    setBusy(true); setErr(""); setResult(null);
    try {
      const payload: Record<string, unknown> = { dataset: dataset.trim(), max_rows: maxRows, force: true };
      if (token.trim()) payload.token = token.trim();
      setResult(await importHf(payload));
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); }
  };
  return (
    <div className="pc-form">
      <div className="pc-grid two">
        <label className="pc-field">Dataset
          <input value={dataset} onChange={(e) => setDataset(e.target.value)} placeholder="org/dataset" />
        </label>
        <label className="pc-field">Max rows
          <input type="number" value={maxRows} onChange={(e) => setMaxRows(Math.max(1, Number(e.target.value) || 1000))} />
        </label>
      </div>
      <div className="pc-row">
        <button className="pc-btn primary" onClick={run} disabled={busy}>
          {busy ? <Loader2 size={15} className="spin" /> : <Database size={15} />} Import dataset
        </button>
        <span className="pc-hint">Downloads to a local pack with starter chat/eval files. Token is optional and stays local.</span>
      </div>
      {err ? <Banner tone="block" title="Import failed" body={err} onClose={() => setErr("")} /> : null}
      {result ? (
        <div className="pc-import-result">
          <div>
            <strong>Imported {compactNumber(result.rows_written)} rows.</strong>
            <code>{result.dataset_pack}</code>
          </div>
          <button className="pc-btn primary" onClick={() => openNew(result.dataset_pack)}>Train on this pack →</button>
        </div>
      ) : null}
    </div>
  );
}

/* ----------------------------------------------------------------- cloud */

const REMOTE_REPO = "https://github.com/gowtham0992/picochat.git";
const REMOTE_SCALES = ["smoke", "tiny", "small", "medium", "h100-100m", "h200-1b-ddp8"];
const REMOTE_GPUS = ["A100", "H100", "A10G", "L4", "T4"];

function RemoteView({ openLogsFor }: SectionProps) {
  const [provider, setProvider] = useState<"modal" | "colab" | "lambda">("modal");
  const [branch, setBranch] = useState("develop");
  const [runName, setRunName] = useState("picochat-cloud-v1");
  const [scale, setScale] = useState("h100-100m");
  const [hfDataset, setHfDataset] = useState("karpathy/climbmix-400b-shuffle");
  const [hfMaxRows, setHfMaxRows] = useState(800000);
  const [gpu, setGpu] = useState("A100");
  const [secretName, setSecretName] = useState("");
  const [modal, setModal] = useState<{ modal_available: boolean; modal_script: boolean } | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [pullRun, setPullRun] = useState("");
  const [pullBusy, setPullBusy] = useState(false);

  useEffect(() => { remoteStatus().then(setModal).catch(() => setModal({ modal_available: false, modal_script: true })); }, []);

  const modalDisabled = !!(modal && !modal.modal_available);

  const launchModal = async () => {
    setBusy(true); setErr("");
    try {
      const started = await remoteModalStart({ repo_url: REMOTE_REPO, branch, run_name: runName, scale, gpu, hf_dataset: hfDataset, hf_max_rows: hfMaxRows, secret_name: secretName || undefined });
      openLogsFor({ job: started.job?.id, run: started.job?.run_name });
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); }
  };

  const pull = async () => {
    if (!pullRun.trim()) { setErr("Enter the cloud run name to pull."); return; }
    setPullBusy(true); setErr("");
    try {
      const started = await remoteModalPull(pullRun.trim());
      openLogsFor({ job: started.job?.id, run: started.job?.run_name });
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)); } finally { setPullBusy(false); }
  };

  const colabSnippet = `!pip install -q "picochat[hf] @ git+${REMOTE_REPO}@${branch}"\n!picochat data hf-import --dataset ${hfDataset} --pack-out my_pack --max-rows ${Math.min(hfMaxRows, 20000)}\n!picochat run ${scale} --dataset-pack my_pack/dataset_pack.json --device cuda --out-dir runs/${runName}`;
  const lambdaSnippet = `# On a fresh Lambda GPU instance (cloud.lambda.ai):\ngit clone -b ${branch} ${REMOTE_REPO} && cd picochat\npip install -e ".[hf]"\npicochat data hf-import --dataset ${hfDataset} --pack-out my_pack --max-rows ${Math.min(hfMaxRows, 100000)}\npicochat run ${scale} --dataset-pack my_pack/dataset_pack.json --device cuda --out-dir runs/${runName}`;

  return (
    <div className="pc-stack">
      <div className="pc-toolbar">
        <div className="pc-toolbar-meta"><strong>Cloud training</strong> — train on remote GPUs with this dashboard as the control plane.</div>
        <Segmented value={provider} options={["modal", "colab", "lambda"]} onChange={(v) => setProvider(v as "modal" | "colab" | "lambda")} />
      </div>

      <div className="pc-grid two">
        <Panel title="Recipe" sub="Shared across providers">
          <label className="pc-field">Run name<input value={runName} onChange={(e) => setRunName(e.target.value)} /></label>
          <div className="pc-grid two">
            <label className="pc-field">Scale<select value={scale} onChange={(e) => setScale(e.target.value)}>{REMOTE_SCALES.map((s) => <option key={s}>{s}</option>)}</select></label>
            <label className="pc-field">Branch<input value={branch} onChange={(e) => setBranch(e.target.value)} /></label>
          </div>
          <label className="pc-field">Hugging Face dataset<input value={hfDataset} onChange={(e) => setHfDataset(e.target.value)} /></label>
          <label className="pc-field">Max rows<input type="number" value={hfMaxRows} onChange={(e) => setHfMaxRows(Math.max(1, Number(e.target.value) || 1000))} /></label>
        </Panel>

        {provider === "modal" ? (
          <Panel title="Modal" sub="Serverless GPUs, launched from here">
            {modal && !modal.modal_available ? (
              <Banner tone="warn" title="Modal CLI not detected" body="Install with `pip install modal`, then `modal token new` to authenticate. Re-open this tab afterward." />
            ) : null}
            <div className="pc-grid two">
              <label className="pc-field">GPU<select value={gpu} onChange={(e) => setGpu(e.target.value)}>{REMOTE_GPUS.map((g) => <option key={g}>{g}</option>)}</select></label>
              <label className="pc-field">Modal secret (HF token)<input value={secretName} onChange={(e) => setSecretName(e.target.value)} placeholder="optional, e.g. hf-secret" /></label>
            </div>
            <div className="pc-hint">Runs <code>modal run scripts/modal_picochat_train.py</code> and streams logs here. Artifacts land on the Modal <code>picochat-runs</code> volume — pull with <code>modal volume get picochat-runs {runName}</code>.</div>
            {err ? <Banner tone="block" title="Modal action failed" body={err} onClose={() => setErr("")} /> : null}
            <button className="pc-btn primary" onClick={launchModal} disabled={busy || modalDisabled}>{busy ? <Loader2 size={15} className="spin" /> : <Cloud size={15} />} Launch on Modal</button>
            <div className="pc-pull">
              <label className="pc-field">Pull a finished cloud run into local runs/
                <input value={pullRun} onChange={(e) => setPullRun(e.target.value)} placeholder="run name on the Modal volume" />
              </label>
              <button className="pc-btn" onClick={pull} disabled={pullBusy || modalDisabled}>{pullBusy ? <Loader2 size={15} className="spin" /> : <Boxes size={15} />} Pull from Modal</button>
              <div className="pc-hint">Downloads it locally so it appears in the run picker for chat and serving.</div>
            </div>
          </Panel>
        ) : null}

        {provider === "colab" ? (
          <Panel title="Google Colab" action={<a className="pc-link" href="https://colab.research.google.com/#create=true" target="_blank" rel="noreferrer">Open Colab →</a>}>
            <div className="pc-hint">Open a GPU notebook (Runtime → Change runtime type → GPU), then paste:</div>
            <CodeBlock code={colabSnippet} />
          </Panel>
        ) : null}

        {provider === "lambda" ? (
          <Panel title="Lambda Cloud" action={<a className="pc-link" href="https://cloud.lambda.ai" target="_blank" rel="noreferrer">Open Lambda →</a>}>
            <div className="pc-hint">Launch a GPU instance, SSH in, then run:</div>
            <CodeBlock code={lambdaSnippet} />
          </Panel>
        ) : null}
      </div>
    </div>
  );
}

function CodeBlock({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);
  const copy = () => { navigator.clipboard?.writeText(code); setCopied(true); window.setTimeout(() => setCopied(false), 1500); };
  return (
    <div className="pc-serve-snip">
      <pre>{code}</pre>
      <button className="pc-btn ghost" onClick={copy}><Copy size={14} /> {copied ? "Copied" : "Copy"}</button>
    </div>
  );
}

/* -------------------------------------------------------------- settings */

function SettingsView({ settings, update }: SectionProps) {
  return (
    <div className="pc-grid two">
      <Panel title="Workbench">
        <div className="pc-set-row"><span>Theme</span><Segmented value={settings.theme} options={["light", "dark"]} onChange={(v) => update("theme", v)} /></div>
        <div className="pc-set-row"><span>Auto refresh</span><input type="checkbox" checked={settings.autoRefresh} onChange={(e) => update("autoRefresh", e.target.checked)} /></div>
        <label className="pc-field">Refresh interval (s)<input value={settings.refreshSeconds} onChange={(e) => update("refreshSeconds", Number(e.target.value) || 5)} /></label>
      </Panel>
      <Panel title="Hugging Face">
        <label className="pc-field">Access token<input value={settings.hfToken} onChange={(e) => update("hfToken", e.target.value)} placeholder="stored locally" /></label>
        <label className="pc-field">Default dataset<input value={settings.defaultDataset} onChange={(e) => update("defaultDataset", e.target.value)} /></label>
        <div className="pc-hint">Secrets are kept in your browser only and never committed.</div>
      </Panel>
    </div>
  );
}

/* ------------------------------------------------------------ primitives */

function NavGroup(p: { title?: string; items: typeof NAV; section: SectionId; setSection: (s: SectionId) => void; close: () => void; runs: RunSummary[] }) {
  return (
    <div className="pc-nav-group">
      {p.title ? <div className="pc-nav-title">{p.title}</div> : null}
      {p.items.map((item) => {
        const Icon = item.icon;
        return (
          <button key={item.id} className={p.section === item.id ? "active" : ""} onClick={() => { p.setSection(item.id); p.close(); }}>
            <Icon size={16} /><span>{item.label}</span>
          </button>
        );
      })}
    </div>
  );
}

function RunPicker({ runs, value, onChange }: { runs: RunSummary[]; value: string; onChange: (v: string) => void }) {
  return (
    <div className="pc-picker">
      <Cpu size={14} />
      <select value={value} onChange={(e) => onChange(e.target.value)} aria-label="Select run">
        {runs.length ? runs.map((r) => <option key={r.name} value={r.name}>{r.name}</option>) : <option>No runs</option>}
      </select>
      <ChevronDown size={14} />
    </div>
  );
}

function Kpi({ label, value, sub, tone = "neutral" }: { label: string; value: React.ReactNode; sub?: string; tone?: Tone }) {
  return <div className={`pc-kpi ${tone}`}><span>{label}</span><strong>{value}</strong>{sub ? <em>{sub}</em> : null}</div>;
}

function Panel({ title, sub, action, children, flush }: { title?: string; sub?: string; action?: React.ReactNode; children: React.ReactNode; flush?: boolean }) {
  return (
    <section className="pc-panel">
      {title ? <header><div><h2>{title}</h2>{sub ? <span>{sub}</span> : null}</div>{action}</header> : null}
      <div className={flush ? "flush" : ""}>{children}</div>
    </section>
  );
}

function Spec({ title, rows }: { title?: string; rows: Record<string, React.ReactNode> }) {
  const body = <dl className="pc-spec">{Object.entries(rows).map(([k, v]) => <div key={k}><dt>{k}</dt><dd title={String(v)}>{v}</dd></div>)}</dl>;
  return title ? <Panel title={title}>{body}</Panel> : body;
}

function Banner({ tone, title, body, onClose }: { tone: Tone; title: string; body: string; onClose?: () => void }) {
  const Icon = tone === "pass" ? Check : tone === "info" ? Loader2 : CircleAlert;
  return <div className={`pc-banner ${tone}`}><Icon size={17} className={tone === "info" ? "spin" : ""} /><div><strong>{title}</strong><p>{body}</p></div>{onClose ? <button onClick={onClose} aria-label="Dismiss"><X size={15} /></button> : null}</div>;
}

function StatusDot({ tone, label }: { tone: Tone; label: string }) {
  return <span className={`pc-status ${tone}`}><i />{label}</span>;
}

function StatusGlyph({ tone, small }: { tone: Tone; small?: boolean }) {
  const size = small ? 13 : 16;
  if (tone === "pass") return <Check size={size} className="pc-g pass" />;
  if (tone === "block") return <CircleAlert size={size} className="pc-g block" />;
  if (tone === "warn") return <CircleAlert size={size} className="pc-g warn" />;
  return <span className={`pc-g-dot ${small ? "sm" : ""}`} />;
}

function RunTable({ runs, selected, onSelect, detailed }: { runs: RunSummary[]; selected?: string; onSelect?: (name: string) => void; detailed?: boolean }) {
  if (!runs.length) return <Empty label="No runs yet — launch a smoke run to begin." />;
  return (
    <div className="pc-table">
      <div className="pc-tr pc-th">
        <span>Run</span><span>Params</span><span>Eval</span><span>SFT loss</span>{detailed ? <span>Context</span> : null}<span>Status</span>
      </div>
      {runs.map((r) => {
        const t = runTone(r);
        return (
          <button key={r.name} className={`pc-tr ${selected === r.name ? "sel" : ""}`} onClick={() => onSelect?.(r.name)}>
            <span className="pc-tname">{r.name}</span>
            <span className="pc-mono">{compactNumber(r.num_parameters)}</span>
            <span className="pc-mono">{percent(r.pass_rate)}</span>
            <span className="pc-mono">{fixed(r.sft_val_loss, 3)}</span>
            {detailed ? <span className="pc-mono">{r.context_size || "--"}</span> : null}
            <span><StatusDot tone={t} label={statusLabel(t)} /></span>
          </button>
        );
      })}
    </div>
  );
}

function LossChart({ points }: { points: Array<{ step: number; train?: number; val?: number }> }) {
  const W = 940, H = 280;
  const vals = points.flatMap((p) => [p.train, p.val]).filter((v): v is number => v != null && Number.isFinite(v));
  if (!vals.length) return <Empty label="No loss points recorded yet." />;
  const min = Math.min(...vals), max = Math.max(...vals), span = max - min || 1;
  const x = (i: number) => 50 + (i / Math.max(points.length - 1, 1)) * (W - 90);
  const y = (v: number) => 230 - ((v - min) / span) * 170;
  const line = (k: "train" | "val") => points.map((p, i) => (p[k] == null ? "" : `${i ? "L" : "M"} ${x(i).toFixed(1)} ${y(p[k] as number).toFixed(1)}`)).join(" ");
  const area = () => {
    const segs = points.map((p, i) => (p.train == null ? "" : `${i ? "L" : "M"} ${x(i).toFixed(1)} ${y(p.train).toFixed(1)}`)).filter(Boolean);
    return segs.length ? `${segs.join(" ")} L ${x(points.length - 1).toFixed(1)} 230 L ${x(0).toFixed(1)} 230 Z` : "";
  };
  return (
    <svg className="pc-loss" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Training and validation loss">
      <defs><linearGradient id="lg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="var(--ac)" stopOpacity="0.22" /><stop offset="100%" stopColor="var(--ac)" stopOpacity="0" /></linearGradient></defs>
      {[0, 0.5, 1].map((t) => <line key={t} x1="50" x2={W - 40} y1={60 + t * 170} y2={60 + t * 170} />)}
      <path className="area" d={area()} fill="url(#lg)" stroke="none" />
      <path className="train" d={line("train")} />
      <path className="val" d={line("val")} />
      <text x={W - 96} y="52">train</text>
      <text x={W - 96} y="72" className="vl">val</text>
    </svg>
  );
}

function Segmented({ value, options, onChange }: { value: string; options: string[]; onChange: (v: string) => void }) {
  return <div className="pc-seg">{options.map((o) => <button key={o} className={value === o ? "on" : ""} onClick={() => onChange(o)}>{o}</button>)}</div>;
}

function Slider({ label, value, min, max, step, onChange, fmt }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; fmt?: (v: number) => string }) {
  return (
    <label className="pc-slider">
      <div><span>{label}</span><strong>{fmt ? fmt(value) : value.toFixed(2)}</strong></div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

function Empty({ label }: { label: string }) {
  return <div className="pc-empty">{label}</div>;
}

/* --------------------------------------------------------- getting started */

function Welcome({ launchSmoke, openNew, setSection }: SectionProps) {
  const steps: Array<{ icon: LucideIcon; t: string; d: string }> = [
    { icon: Database, t: "Bring your data", d: "Point Picochat at your domain text — a folder, a file, or a public Hugging Face dataset." },
    { icon: Gauge, t: "Train a model", d: "Train a small model from scratch, or fine-tune an existing Hugging Face model on your data — all locally." },
    { icon: MessageSquare, t: "Test it", d: "Chat with the model you trained in the Playground and inspect what it actually learned." },
    { icon: ShieldCheck, t: "Share with your team", d: "Pass the release gate, then serve an OpenAI-compatible endpoint your team can call." }
  ];
  return (
    <div className="pc-welcome">
      <div className="pc-welcome-hero">
        <span className="pc-eyebrow">Get started</span>
        <h1>Train a small language model on your domain.</h1>
        <p>
          Picochat turns your own text into a compact, specialized model you can run locally, evaluate
          honestly, and share with your team — no giant GPU bill, no black box. These are small focused
          models, not a general chatbot: best when the domain is narrow and the data is yours.
        </p>
        <div className="pc-welcome-cta">
          <button className="pc-btn primary" onClick={() => openNew()}><Gauge size={15} /> Train your first model</button>
          <button className="pc-btn" onClick={() => launchSmoke(false)}><FlaskConical size={15} /> Run the demo</button>
          <button className="pc-btn ghost" onClick={() => setSection("dataset")}><Database size={15} /> Bring your own data</button>
        </div>
      </div>
      <div className="pc-welcome-steps">
        {steps.map((step, i) => {
          const Icon = step.icon;
          return (
            <div className="pc-welcome-step" key={i}>
              <span className="pc-welcome-num">{i + 1}</span>
              <Icon size={18} />
              <strong>{step.t}</strong>
              <p>{step.d}</p>
            </div>
          );
        })}
      </div>
      <div className="pc-welcome-foot">
        <span className="pc-eyebrow">Prefer the terminal?</span>
        <code>picochat data hf-import --dataset &lt;hf/dataset&gt; --pack-out my_pack</code>
        <code>picochat run tiny --dataset-pack my_pack/dataset_pack.json</code>
      </div>
    </div>
  );
}

const PRESET_FALLBACK = ["smoke", "tiny", "small-local", "small", "medium"];
const EXAMPLE_PACKS = ["examples/tiny_dataset_pack.json", "examples/tinystories_dataset_pack.json"];

type WizStep = 1 | 2 | 3;
type DataSource = "example" | "folder" | "pack" | "hf";

function NewRunModal({ initialPack, onClose, onLaunched }: { initialPack?: string; onClose: () => void; onLaunched: (t: { run?: string; job?: string }) => void }) {
  const [step, setStep] = useState<WizStep>(initialPack ? 2 : 1);
  const [source, setSource] = useState<DataSource>("example");
  const [pack, setPack] = useState(initialPack || "");

  const [dataset, setDataset] = useState("HuggingFaceTB/smollm-corpus");
  const [maxRows, setMaxRows] = useState(2000);
  const [importBusy, setImportBusy] = useState(false);
  const [importErr, setImportErr] = useState("");

  const [folderPath, setFolderPath] = useState("");
  const [packName, setPackName] = useState("my-pack");

  const [insp, setInsp] = useState<Record<string, any> | null>(null);
  const [inspBusy, setInspBusy] = useState(false);
  const [inspErr, setInspErr] = useState("");
  const [genBusy, setGenBusy] = useState(false);
  const [editing, setEditing] = useState(false);

  const [adv, setAdv] = useState(false);
  const [advLayers, setAdvLayers] = useState("");
  const [advHeads, setAdvHeads] = useState("");
  const [advEmbd, setAdvEmbd] = useState("");
  const [advCtx, setAdvCtx] = useState("");
  const [advOptimizer, setAdvOptimizer] = useState("");
  const [advPrecision, setAdvPrecision] = useState("");
  const [advLora, setAdvLora] = useState(false);
  const [advLoraRank, setAdvLoraRank] = useState(8);
  const [advDpoInput, setAdvDpoInput] = useState("");
  const [advDpoSteps, setAdvDpoSteps] = useState(50);
  const [prefBusy, setPrefBusy] = useState(false);
  const [prefMsg, setPrefMsg] = useState("");

  const [name, setName] = useState("");
  const [preset, setPreset] = useState("tiny");
  const [presets, setPresets] = useState<Record<string, any>>({});
  const [launching, setLaunching] = useState(false);
  const [err, setErr] = useState("");

  const [mode, setMode] = useState<"scratch" | "finetune">("scratch");
  const [baseModel, setBaseModel] = useState("HuggingFaceTB/SmolLM2-135M-Instruct");
  const [ftPack, setFtPack] = useState(initialPack || EXAMPLE_PACKS[0]);
  const [ftName, setFtName] = useState("");
  const [ftSteps, setFtSteps] = useState(100);
  const [ftLora, setFtLora] = useState(true);
  const [ftDevice, setFtDevice] = useState("auto");
  const [ftBusy, setFtBusy] = useState(false);
  const [ftErr, setFtErr] = useState("");

  useEffect(() => { loadPresets().then((p) => setPresets(p.presets || {})).catch(() => {}); }, []);
  const keys = Object.keys(presets).length ? Object.keys(presets) : PRESET_FALLBACK;

  const inspectPack = async (p: string) => {
    setInspBusy(true); setInspErr(""); setInsp(null);
    try { setInsp(await inspectTuning({ dataset_pack: p })); }
    catch (e) { setInspErr(e instanceof Error ? e.message : String(e)); }
    finally { setInspBusy(false); }
  };
  const goPrepare = (p: string) => { setPack(p); setStep(2); inspectPack(p); };
  useEffect(() => { if (initialPack) inspectPack(initialPack); /* eslint-disable-next-line */ }, []);

  const runImport = async () => {
    if (!dataset.trim()) { setImportErr("Enter a dataset id, e.g. org/name."); return; }
    setImportBusy(true); setImportErr("");
    try {
      const res = await importHf({ dataset: dataset.trim(), max_rows: maxRows, force: true });
      goPrepare(res.dataset_pack);
    } catch (e) { setImportErr(e instanceof Error ? e.message : String(e)); }
    finally { setImportBusy(false); }
  };

  const createFromFolder = async () => {
    if (!folderPath.trim()) { setImportErr("Enter a path to a folder or file of documents."); return; }
    const name = packName.trim() || "my-pack";
    setImportBusy(true); setImportErr("");
    try {
      const res = await initDatasetPack({ name, corpus_path: folderPath.trim(), out_dir: `packs/${name}`, force: true });
      goPrepare(res.dataset_pack);
    } catch (e) { setImportErr(e instanceof Error ? e.message : String(e)); }
    finally { setImportBusy(false); }
  };

  const generateData = async () => {
    if (!insp?.chat_input || !insp?.eval_input) return;
    setGenBusy(true); setInspErr("");
    try {
      const chatOut = String(insp.chat_input).replace(/\.jsonl$/, "_generated.jsonl");
      const evalOut = String(insp.eval_input).replace(/\.jsonl$/, "_generated.jsonl");
      await generateSftStarter({ dataset_pack: pack, out_path: chatOut, max_items: 48, force: true, promote_to_pack: true });
      await generateEvalStarter({ dataset_pack: pack, out_path: evalOut, max_items: 24, force: true, promote_to_pack: true });
      await inspectPack(pack);
    } catch (e) { setInspErr(e instanceof Error ? e.message : String(e)); }
    finally { setGenBusy(false); }
  };

  const genPreferences = async () => {
    const chatIn = insp?.chat_input;
    if (!chatIn) { setErr("Open the 'Training data' step first so the chat file is known."); return; }
    setPrefBusy(true); setErr(""); setPrefMsg("");
    try {
      const out = String(chatIn).replace(/\.jsonl$/, "_preferences.jsonl");
      const res = await generatePreferences({ input_path: chatIn, out_path: out, force: true });
      setAdvDpoInput(res.output_path || out);
      setPrefMsg(`Generated ${res.num_examples ?? "?"} pairs (synthetic — for DPO plumbing; review before release).`);
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)); } finally { setPrefBusy(false); }
  };

  const launch = async () => {
    if (!pack.trim()) { setErr("Choose a dataset pack to train on."); return; }
    setLaunching(true); setErr("");
    const num = (s: string) => { const n = Number(s); return s.trim() && Number.isFinite(n) ? n : undefined; };
    const payload: Record<string, unknown> = { dataset_pack: pack.trim(), run_name: name.trim() || undefined, preset };
    if (adv) {
      const set = (k: string, v: unknown) => { if (v !== undefined) payload[k] = v; };
      set("n_layer", num(advLayers)); set("n_head", num(advHeads)); set("n_embd", num(advEmbd)); set("context_size", num(advCtx));
      if (advOptimizer) { payload.base_optimizer = advOptimizer; payload.sft_optimizer = advOptimizer; }
      if (advPrecision) payload.precision = advPrecision;
      if (advLora) { payload.sft_peft = "lora"; payload.sft_lora_rank = advLoraRank; }
      if (advDpoInput.trim()) { payload.dpo_input = advDpoInput.trim(); payload.dpo_steps = Math.max(1, advDpoSteps); }
    }
    try {
      const started = await startRun(payload);
      onLaunched({ job: started.job?.id, run: started.job?.run_name });
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)); } finally { setLaunching(false); }
  };

  const launchFinetune = async () => {
    if (!baseModel.trim()) { setFtErr("Enter a base model id, e.g. org/name."); return; }
    if (!ftPack.trim()) { setFtErr("Choose a dataset pack with chat data."); return; }
    setFtBusy(true); setFtErr("");
    try {
      const started = await trainHfSft({ model: baseModel.trim(), dataset_pack: ftPack.trim(), run_name: ftName.trim() || undefined, max_steps: ftSteps, peft: ftLora ? "lora" : "none", device: ftDevice });
      onLaunched({ job: started.job?.id, run: started.job?.run_name });
    } catch (e) { setFtErr(e instanceof Error ? e.message : String(e)); } finally { setFtBusy(false); }
  };

  const stepLabel = step === 1 ? "Choose your data" : step === 2 ? "Training data" : "Train";
  const statTone = (st?: string): Tone => st === "ready" ? "pass" : st === "caution" ? "warn" : "block";

  return (
    <>
    <div className="pc-modal-back" onClick={onClose}>
      <section className="pc-modal pc-new" onClick={(e) => e.stopPropagation()}>
        <header>
          <div><span className="pc-eyebrow">{mode === "finetune" ? "Fine-tune an existing model" : `Create a domain model · step ${step} of 3`}</span><h2>{mode === "finetune" ? "Fine-tune" : stepLabel}</h2></div>
          <button className="pc-btn ghost" onClick={onClose}><X size={15} /></button>
        </header>
        <div className="pc-new-body">
          {(step === 1 || mode === "finetune") ? (
            <div className="pc-mode">
              <button className={`pc-source-opt ${mode === "scratch" ? "on" : ""}`} onClick={() => setMode("scratch")}>Train from scratch</button>
              <button className={`pc-source-opt ${mode === "finetune" ? "on" : ""}`} onClick={() => setMode("finetune")}>Fine-tune existing model</button>
            </div>
          ) : null}
          {mode === "finetune" ? (
            <>
              <label className="pc-field">Base model
                <input value={baseModel} onChange={(e) => setBaseModel(e.target.value)} placeholder="HuggingFaceTB/SmolLM2-135M-Instruct" />
              </label>
              <label className="pc-field">Training data (dataset pack)
                <input value={ftPack} onChange={(e) => setFtPack(e.target.value)} placeholder="path/to/dataset_pack.json" />
              </label>
              <div className="pc-new-examples">
                <span>Packs</span>
                {EXAMPLE_PACKS.map((ex) => <button key={ex} className={`pc-chip ${ftPack === ex ? "on" : ""}`} onClick={() => setFtPack(ex)}>{ex.split("/").pop()}</button>)}
              </div>
              <div className="pc-grid two">
                <label className="pc-field">Run name (optional)<input value={ftName} onChange={(e) => setFtName(e.target.value)} placeholder="my-domain-ft-v1" /></label>
                <label className="pc-field">Max steps<input type="number" value={ftSteps} onChange={(e) => setFtSteps(Math.max(1, Number(e.target.value) || 100))} /></label>
              </div>
              <div className="pc-grid two">
                <label className="pc-field">Device<select value={ftDevice} onChange={(e) => setFtDevice(e.target.value)}>{["auto", "cpu", "cuda", "mps"].map((d) => <option key={d} value={d}>{d}</option>)}</select></label>
                <label className="pc-check">LoRA adapter (faster, lighter)<input type="checkbox" checked={ftLora} onChange={(e) => setFtLora(e.target.checked)} /></label>
              </div>
              <div className="pc-hint">Downloads the base model and fine-tunes it on the pack's chat data. Needs the <code>hf</code> extras installed; watch the live log for progress.</div>
              {ftErr ? <Banner tone="block" title="Could not start" body={ftErr} onClose={() => setFtErr("")} /> : null}
            </>
          ) : (
            <>
          {step === 1 ? (
            <>
              <div className="pc-source four">
                {([["example", "Use an example"], ["folder", "My folder of docs"], ["pack", "I have a pack"], ["hf", "Hugging Face"]] as [DataSource, string][]).map(([k, l]) => (
                  <button key={k} className={`pc-source-opt ${source === k ? "on" : ""}`} onClick={() => setSource(k)}>{l}</button>
                ))}
              </div>
              {source === "example" ? (
                <div className="pc-new-examples">
                  <span>Packs</span>
                  {EXAMPLE_PACKS.map((ex) => (
                    <button key={ex} className={`pc-chip ${pack === ex ? "on" : ""}`} onClick={() => setPack(ex)}>{ex.split("/").pop()}</button>
                  ))}
                </div>
              ) : null}
              {source === "folder" ? (
                <>
                  <label className="pc-field">Folder or file of documents
                    <input value={folderPath} onChange={(e) => setFolderPath(e.target.value)} placeholder="path/to/your/docs" />
                  </label>
                  <label className="pc-field">Pack name
                    <input value={packName} onChange={(e) => setPackName(e.target.value)} placeholder="my-pack" />
                  </label>
                  <div className="pc-hint">Builds a dataset pack at <code>packs/{packName || "my-pack"}</code> with starter chat/eval you can refine next.</div>
                  {importErr ? <Banner tone="block" title="Could not create pack" body={importErr} onClose={() => setImportErr("")} /> : null}
                </>
              ) : null}
              {source === "pack" ? (
                <label className="pc-field">Dataset pack path
                  <input value={pack} onChange={(e) => setPack(e.target.value)} placeholder="path/to/dataset_pack.json" />
                </label>
              ) : null}
              {source === "hf" ? (
                <>
                  <div className="pc-grid two">
                    <label className="pc-field">Dataset<input value={dataset} onChange={(e) => setDataset(e.target.value)} placeholder="org/dataset" /></label>
                    <label className="pc-field">Max rows<input type="number" value={maxRows} onChange={(e) => setMaxRows(Math.max(1, Number(e.target.value) || 1000))} /></label>
                  </div>
                  <div className="pc-hint">Imports to a local pack with starter chat/eval. This can take a minute.</div>
                  {importErr ? <Banner tone="block" title="Import failed" body={importErr} onClose={() => setImportErr("")} /> : null}
                </>
              ) : null}
            </>
          ) : null}

          {step === 2 ? (
            <>
              <div className="pc-prep-pack"><span>Pack</span><code>{pack}</code></div>
              <button className="pc-btn ghost pc-edit-btn" onClick={() => setEditing(true)}><FileText size={14} /> Edit chat &amp; eval</button>
              {inspBusy ? <div className="pc-hint"><Loader2 size={14} className="spin" /> Checking training data…</div> : null}
              {inspErr ? <Banner tone="block" title="Could not inspect" body={inspErr} onClose={() => setInspErr("")} /> : null}
              {insp ? (
                <>
                  <div className="pc-prep-stats">
                    <Stat label="Readiness" value={insp.status} tone={statTone(insp.status)} />
                    <Stat label="Chat examples" value={insp.chat_data?.num_examples ?? "--"} tone={statTone(insp.chat_data?.status)} />
                    <Stat label="Eval items" value={insp.eval_data?.num_items ?? "--"} tone={statTone(insp.eval_data?.status)} />
                  </div>
                  {insp.status !== "ready" ? (
                    <div className="pc-prep-gen">
                      <div>
                        <strong>Build starter training data</strong>
                        <p>Generate chat + eval examples from this pack's text. You can refine them later for a stronger model.</p>
                      </div>
                      <button className="pc-btn primary" onClick={generateData} disabled={genBusy}>{genBusy ? <Loader2 size={15} className="spin" /> : <FlaskConical size={15} />} Generate</button>
                    </div>
                  ) : null}
                  {Array.isArray(insp.next_actions) && insp.next_actions.length ? (
                    <ul className="pc-prep-actions">{insp.next_actions.slice(0, 3).map((a: string, i: number) => <li key={i}>{a}</li>)}</ul>
                  ) : null}
                </>
              ) : null}
            </>
          ) : null}

          {step === 3 ? (
            <>
              <div className="pc-grid two">
                <label className="pc-field">Run name (optional)<input value={name} onChange={(e) => setName(e.target.value)} placeholder="my-domain-v1" /></label>
                <label className="pc-field">Preset<select value={preset} onChange={(e) => setPreset(e.target.value)}>{keys.map((k) => <option key={k} value={k}>{presets[k]?.label || k}</option>)}</select></label>
              </div>
              <div className="pc-hint">{presets[preset]?.description || "Builds tokenizer → base pretraining → SFT → eval, then opens the live training log."}</div>
              <button type="button" className="pc-adv-toggle" onClick={() => setAdv(!adv)}>{adv ? "▾" : "▸"} Advanced (optional)</button>
              {adv ? (
                <div className="pc-adv">
                  <div className="pc-grid two">
                    <label className="pc-field">Layers<input value={advLayers} onChange={(e) => setAdvLayers(e.target.value)} placeholder="preset" /></label>
                    <label className="pc-field">Heads<input value={advHeads} onChange={(e) => setAdvHeads(e.target.value)} placeholder="preset" /></label>
                  </div>
                  <div className="pc-grid two">
                    <label className="pc-field">Embedding dim<input value={advEmbd} onChange={(e) => setAdvEmbd(e.target.value)} placeholder="preset" /></label>
                    <label className="pc-field">Context<input value={advCtx} onChange={(e) => setAdvCtx(e.target.value)} placeholder="preset" /></label>
                  </div>
                  <div className="pc-grid two">
                    <label className="pc-field">Optimizer<select value={advOptimizer} onChange={(e) => setAdvOptimizer(e.target.value)}><option value="">preset</option><option value="adamw">adamw</option><option value="muon">muon</option></select></label>
                    <label className="pc-field">Precision<select value={advPrecision} onChange={(e) => setAdvPrecision(e.target.value)}><option value="">preset</option><option value="float32">float32</option><option value="bfloat16">bfloat16</option><option value="float16">float16</option></select></label>
                  </div>
                  <div className="pc-grid two">
                    <label className="pc-check">LoRA fine-tune<input type="checkbox" checked={advLora} onChange={(e) => setAdvLora(e.target.checked)} /></label>
                    {advLora ? <label className="pc-field">LoRA rank<input type="number" value={advLoraRank} onChange={(e) => setAdvLoraRank(Math.max(1, Number(e.target.value) || 8))} /></label> : <span />}
                  </div>
                  <label className="pc-field">DPO preferences (optional JSONL)<input value={advDpoInput} onChange={(e) => setAdvDpoInput(e.target.value)} placeholder="path/to/preferences.jsonl" /></label>
                  <div className="pc-row">
                    <button type="button" className="pc-btn ghost" onClick={genPreferences} disabled={prefBusy}>{prefBusy ? <Loader2 size={14} className="spin" /> : <FlaskConical size={14} />} Generate from chat data</button>
                    {prefMsg ? <span className="pc-hint">{prefMsg}</span> : null}
                  </div>
                  {advDpoInput.trim() ? <label className="pc-field">DPO steps<input type="number" value={advDpoSteps} onChange={(e) => setAdvDpoSteps(Math.max(1, Number(e.target.value) || 50))} /></label> : null}
                </div>
              ) : null}
              {err ? <Banner tone="block" title="Could not launch" body={err} onClose={() => setErr("")} /> : null}
            </>
          ) : null}
            </>
          )}
        </div>
        <footer className="pc-new-foot">
          {mode === "finetune" ? (
            <>
              <button className="pc-btn ghost" onClick={onClose}>Cancel</button>
              <button className="pc-btn primary" onClick={launchFinetune} disabled={ftBusy}>{ftBusy ? <Loader2 size={15} className="spin" /> : <Gauge size={15} />} Start fine-tuning</button>
            </>
          ) : (
            <>
              {step > 1
                ? <button className="pc-btn ghost" onClick={() => setStep((s) => (s - 1) as WizStep)}>Back</button>
                : <button className="pc-btn ghost" onClick={onClose}>Cancel</button>}
              {step === 1 ? (
                source === "hf"
                  ? <button className="pc-btn primary" onClick={runImport} disabled={importBusy}>{importBusy ? <Loader2 size={15} className="spin" /> : <Database size={15} />} Import & continue</button>
                  : source === "folder"
                    ? <button className="pc-btn primary" onClick={createFromFolder} disabled={importBusy}>{importBusy ? <Loader2 size={15} className="spin" /> : <Database size={15} />} Create pack & continue</button>
                    : <button className="pc-btn primary" onClick={() => pack.trim() && goPrepare(pack.trim())} disabled={!pack.trim()}>Next: training data →</button>
              ) : null}
              {step === 2 ? <button className="pc-btn primary" onClick={() => setStep(3)} disabled={inspBusy}>Next: train →</button> : null}
              {step === 3 ? <button className="pc-btn primary" onClick={launch} disabled={launching}>{launching ? <Loader2 size={15} className="spin" /> : <Gauge size={15} />} Start training</button> : null}
            </>
          )}
        </footer>
      </section>
    </div>
    {editing ? <PackEditorModal pack={pack} onClose={() => setEditing(false)} onSaved={() => inspectPack(pack)} /> : null}
    </>
  );
}

function PackEditorModal({ pack, onClose, onSaved }: { pack: string; onClose: () => void; onSaved: () => void }) {
  const [chatText, setChatText] = useState("");
  const [evalText, setEvalText] = useState("");
  const [status, setStatus] = useState<Record<string, any> | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState("");
  useEffect(() => {
    setLoading(true); setErr("");
    loadPackEditor({ dataset_pack: pack })
      .then((d) => { setChatText(d.chat_text || ""); setEvalText(d.eval_text || ""); setStatus(d); })
      .catch((e) => setErr(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, [pack]);
  const save = async () => {
    setSaving(true); setErr("");
    try { setStatus(await savePackEditor({ dataset_pack: pack, chat_text: chatText, eval_text: evalText })); onSaved(); }
    catch (e) { setErr(e instanceof Error ? e.message : String(e)); } finally { setSaving(false); }
  };
  return (
    <div className="pc-modal-back" onClick={onClose}>
      <section className="pc-modal pc-editor" onClick={(e) => e.stopPropagation()}>
        <header>
          <div><span className="pc-eyebrow">Edit training data</span><h2>Chat &amp; eval</h2></div>
          <button className="pc-btn ghost" onClick={onClose}><X size={15} /></button>
        </header>
        <div className="pc-editor-body">
          {loading ? <div className="pc-hint"><Loader2 size={14} className="spin" /> Loading…</div> : (
            <div className="pc-grid two">
              <label className="pc-field">Chat SFT — one JSON object per line
                <textarea className="pc-code" value={chatText} onChange={(e) => setChatText(e.target.value)} spellCheck={false} />
              </label>
              <label className="pc-field">Eval — one JSON object per line
                <textarea className="pc-code" value={evalText} onChange={(e) => setEvalText(e.target.value)} spellCheck={false} />
              </label>
            </div>
          )}
          {err ? <Banner tone="block" title="Could not save" body={err} onClose={() => setErr("")} /> : null}
          {status ? (
            <div className="pc-hint">
              Chat: {status.chat_data?.num_examples ?? "--"} examples · Eval: {status.eval_data?.num_items ?? "--"} items
              {status.saved ? " · saved ✓" : ""}
            </div>
          ) : null}
        </div>
        <footer className="pc-new-foot">
          <button className="pc-btn ghost" onClick={onClose}>Close</button>
          <button className="pc-btn primary" onClick={save} disabled={saving || loading}>{saving ? <Loader2 size={15} className="spin" /> : <Check size={15} />} Save</button>
        </footer>
      </section>
    </div>
  );
}

function Stat({ label, value, tone }: { label: string; value: React.ReactNode; tone: Tone }) {
  return <div className={`pc-stat ${tone}`}><span>{label}</span><strong>{value}</strong></div>;
}

/* --------------------------------------------------------------- reports */

const REPORT_LABELS: Record<string, string> = {
  summary: "Model card",
  honesty: "Honesty report",
  base: "Base report",
  sft: "SFT report",
  eval: "Eval report"
};

function ReportLinks({ reports, openReport, only }: { reports?: Record<string, { exists?: boolean }>; openReport: (r: string) => void; only?: string[] }) {
  const keys = (only || Object.keys(REPORT_LABELS)).filter((k) => reports?.[k]?.exists);
  if (!keys.length) return null;
  return (
    <div className="pc-row">
      {keys.map((k) => (
        <button key={k} className="pc-btn ghost" onClick={() => openReport(k)}><FileText size={14} /> {REPORT_LABELS[k]}</button>
      ))}
    </div>
  );
}

function ReportModal({ target, onClose }: { target: { run: string; report: string }; onClose: () => void }) {
  const [markdown, setMarkdown] = useState<string | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    setMarkdown(null); setErr("");
    loadReport(target.run, target.report).then((d) => setMarkdown(d.markdown || "")).catch((e) => setErr(e instanceof Error ? e.message : String(e)));
  }, [target.run, target.report]);
  return (
    <div className="pc-modal-back" onClick={onClose}>
      <section className="pc-modal pc-report" onClick={(e) => e.stopPropagation()}>
        <header>
          <div><span className="pc-eyebrow">{target.run}</span><h2>{REPORT_LABELS[target.report] || target.report}</h2></div>
          <button className="pc-btn ghost" onClick={onClose}><X size={15} /></button>
        </header>
        <div className="pc-report-body">
          {err ? <Banner tone="block" title="Could not load report" body={err} /> : markdown != null ? <Markdown source={markdown} /> : <div className="pc-hint"><Loader2 size={14} className="spin" /> Loading…</div>}
        </div>
      </section>
    </div>
  );
}

function inlineMd(text: string, keyBase: string): React.ReactNode[] {
  const out: React.ReactNode[] = [];
  text.split(/(`[^`]+`)/g).forEach((chunk, i) => {
    if (chunk.startsWith("`") && chunk.endsWith("`")) {
      out.push(<code key={`${keyBase}-c${i}`}>{chunk.slice(1, -1)}</code>);
    } else {
      chunk.split(/(\*\*[^*]+\*\*)/g).forEach((part, j) => {
        if (part.startsWith("**") && part.endsWith("**")) out.push(<strong key={`${keyBase}-b${i}-${j}`}>{part.slice(2, -2)}</strong>);
        else if (part) out.push(<span key={`${keyBase}-t${i}-${j}`}>{part}</span>);
      });
    }
  });
  return out;
}

function Markdown({ source }: { source: string }) {
  const lines = source.replace(/\r/g, "").split("\n");
  const blocks: React.ReactNode[] = [];
  let i = 0;
  const isTableSep = (s: string) => /^\s*\|?[\s:|-]+\|?\s*$/.test(s) && s.includes("-");
  const cells = (row: string) => row.replace(/^\s*\|/, "").replace(/\|\s*$/, "").split("|").map((c) => c.trim());
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i += 1; continue; }
    const heading = line.match(/^(#{1,4})\s+(.*)/);
    if (heading) {
      const level = heading[1].length;
      const Tag = (`h${Math.min(level + 1, 4)}`) as "h2" | "h3" | "h4";
      blocks.push(<Tag key={i}>{inlineMd(heading[2], `h${i}`)}</Tag>);
      i += 1; continue;
    }
    if (line.trim().startsWith("|") && i + 1 < lines.length && isTableSep(lines[i + 1])) {
      const header = cells(line);
      const body: string[][] = [];
      i += 2;
      while (i < lines.length && lines[i].trim().startsWith("|")) { body.push(cells(lines[i])); i += 1; }
      blocks.push(
        <div className="pc-md-table" key={`tbl${i}`}>
          <table>
            <thead><tr>{header.map((h, c) => <th key={c}>{inlineMd(h, `th${i}-${c}`)}</th>)}</tr></thead>
            <tbody>{body.map((r, ri) => <tr key={ri}>{r.map((cell, ci) => <td key={ci}>{inlineMd(cell, `td${ri}-${ci}`)}</td>)}</tr>)}</tbody>
          </table>
        </div>
      );
      continue;
    }
    if (/^\s*[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) { items.push(lines[i].replace(/^\s*[-*]\s+/, "")); i += 1; }
      blocks.push(<ul key={`ul${i}`}>{items.map((it, k) => <li key={k}>{inlineMd(it, `li${i}-${k}`)}</li>)}</ul>);
      continue;
    }
    const para: string[] = [];
    while (i < lines.length && lines[i].trim() && !/^(#{1,4}\s|\s*[-*]\s|\s*\|)/.test(lines[i])) { para.push(lines[i]); i += 1; }
    blocks.push(<p key={`p${i}`}>{inlineMd(para.join(" "), `p${i}`)}</p>);
  }
  return <div className="pc-md">{blocks}</div>;
}

/* --------------------------------------------------------------- compare */

function CompareView({ runs, selectRun }: SectionProps) {
  const [selected, setSelected] = useState<string[]>([]);
  const [data, setData] = useState<Record<string, any> | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [board, setBoard] = useState<{ rows: Array<Record<string, any>>; best_run: string | null } | null>(null);
  const toggle = (name: string) => setSelected((s) => s.includes(name) ? s.filter((n) => n !== name) : [...s, name]);
  useEffect(() => { loadLeaderboard().then(setBoard).catch(() => {}); }, []);
  const run = async () => {
    setBusy(true); setErr("");
    try { setData(await compareRuns(selected)); } catch (e) { setErr(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); }
  };
  const rows = (data?.rows || []) as Array<Record<string, any>>;
  const boardRows = (board?.rows || []).filter((r) => r.suite === "overall").sort((a, b) => (b.pass_rate ?? 0) - (a.pass_rate ?? 0));
  return (
    <div className="pc-stack">
      <div className="pc-toolbar">
        <div className="pc-toolbar-meta">Pick two or more runs to compare side by side.</div>
        <button className="pc-btn primary" onClick={run} disabled={busy || selected.length < 2}>{busy ? <Loader2 size={15} className="spin" /> : <GitCompare size={15} />} Compare {selected.length || ""}</button>
      </div>
      {boardRows.length ? (
        <Panel title="Leaderboard" sub="All runs ranked by visible eval" flush>
          <div className="pc-lb">
            <div className="pc-lb-tr head"><span>#</span><span>Run</span><span>Score</span><span>Pass</span><span>Prompt echo</span></div>
            {boardRows.map((r, i) => (
              <button key={r.run} className={`pc-lb-tr ${r.run === board?.best_run ? "best" : ""}`} onClick={() => selectRun(r.run)}>
                <span className="pc-mono">{i + 1}</span>
                <span className="pc-lb-name">{r.run}{r.run === board?.best_run ? " ★" : ""}</span>
                <span className="pc-mono">{r.score ?? "--"}</span>
                <span className="pc-mono">{percent(r.pass_rate)}</span>
                <span className="pc-mono">{r.prompt_echo_rate != null ? percent(r.prompt_echo_rate) : "--"}</span>
              </button>
            ))}
          </div>
        </Panel>
      ) : null}
      <Panel title="Runs" sub="Select runs to compare">
        {runs.length ? (
          <div className="pc-pick-grid">
            {runs.map((r) => (
              <label key={r.name} className={`pc-pick ${selected.includes(r.name) ? "on" : ""}`}>
                <input type="checkbox" checked={selected.includes(r.name)} onChange={() => toggle(r.name)} />
                <span>{r.name}</span><em>{percent(r.pass_rate)}</em>
              </label>
            ))}
          </div>
        ) : <Empty label="No runs to compare yet." />}
      </Panel>
      {err ? <Banner tone="block" title="Compare failed" body={err} onClose={() => setErr("")} /> : null}
      {rows.length ? (
        <Panel title="Comparison" sub={data?.best_run ? `Best eval: ${data.best_run}` : undefined}>
          <CompareTable rows={rows} best={data?.best_run} />
        </Panel>
      ) : null}
    </div>
  );
}

function ImportRunModal({ onClose }: { onClose: () => void }) {
  const [sourcePath, setSourcePath] = useState("");
  const [runName, setRunName] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [done, setDone] = useState("");
  const go = async () => {
    if (!sourcePath.trim()) { setErr("Enter the path to a run folder (must contain summary.json)."); return; }
    setBusy(true); setErr(""); setDone("");
    try {
      const res = await importRun({ source_path: sourcePath.trim(), run_name: runName.trim() || undefined });
      setDone(`Imported as ${res.run_name || "run"} — it will appear in the bank shortly.`);
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); }
  };
  return (
    <div className="pc-modal-back" onClick={onClose}>
      <section className="pc-modal pc-new" onClick={(e) => e.stopPropagation()}>
        <header><div><span className="pc-eyebrow">Import a run</span><h2>Import run</h2></div><button className="pc-btn ghost" onClick={onClose}><X size={15} /></button></header>
        <div className="pc-new-body">
          <label className="pc-field">Run folder path<input value={sourcePath} onChange={(e) => setSourcePath(e.target.value)} placeholder="path/to/run-dir (contains summary.json)" /></label>
          <label className="pc-field">Rename (optional)<input value={runName} onChange={(e) => setRunName(e.target.value)} placeholder="keep original name" /></label>
          <div className="pc-hint">Copies a completed run into the local bank so it shows up for compare, reports, chat, and serve.</div>
          {err ? <Banner tone="block" title="Import failed" body={err} onClose={() => setErr("")} /> : null}
          {done ? <Banner tone="pass" title="Imported" body={done} /> : null}
        </div>
        <footer className="pc-new-foot">
          <button className="pc-btn ghost" onClick={onClose}>Close</button>
          <button className="pc-btn primary" onClick={go} disabled={busy}>{busy ? <Loader2 size={15} className="spin" /> : <Upload size={15} />} Import</button>
        </footer>
      </section>
    </div>
  );
}

function CompareTable({ rows, best }: { rows: Array<Record<string, any>>; best?: string }) {
  const metrics: Array<[string, (r: Record<string, any>) => React.ReactNode]> = [
    ["Eval pass", (r) => percent(r.pass_rate)],
    ["Eval score", (r) => r.eval_score ?? "--"],
    ["Refusal", (r) => r.refusal_pass_rate != null ? percent(r.refusal_pass_rate) : "--"],
    ["Prompt echo", (r) => r.prompt_echo_rate != null ? percent(r.prompt_echo_rate) : "--"],
    ["SFT val loss", (r) => fixed(r.sft_val_loss, 3)],
    ["Base val loss", (r) => fixed(r.base_val_loss, 3)],
    ["SFT BPB", (r) => fixed(r.sft_val_bpb, 3)],
    ["Params", (r) => compactNumber(r.num_parameters)],
    ["Context", (r) => r.context_size ?? "--"],
    ["Device", (r) => r.device ?? "--"]
  ];
  const cols = `170px repeat(${rows.length}, minmax(0, 1fr))`;
  return (
    <div className="pc-ctable">
      <div className="pc-ctable-row head" style={{ gridTemplateColumns: cols }}>
        <span>Metric</span>
        {rows.map((r) => <span key={r.run} className={r.run === best ? "best" : ""}>{r.run}{r.run === best ? " ★" : ""}</span>)}
      </div>
      {metrics.map(([label, fn]) => (
        <div className="pc-ctable-row" key={label} style={{ gridTemplateColumns: cols }}>
          <span className="pc-ctable-k">{label}</span>
          {rows.map((r) => <span key={r.run} className={`pc-mono ${r.run === best ? "best" : ""}`}>{fn(r)}</span>)}
        </div>
      ))}
    </div>
  );
}

function LogModal({ target, onClose, onCancel }: { target: { run?: string; job?: string }; onClose: () => void; onCancel: (id: string) => Promise<any> }) {
  const [log, setLog] = useState<RunLog | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    const source = new EventSource(runLogStreamUrl({ ...target, limit: 80_000 }));
    source.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as RunLog & { error?: string };
        if (payload.error) { setErr(payload.error); source.close(); return; }
        setLog(payload);
        setErr("");
        if (payload.running === false) source.close(); // run ended; stop reconnecting
      } catch { /* ignore malformed frame */ }
    };
    // EventSource auto-reconnects on transient drops; nothing to do here.
    source.onerror = () => {};
    return () => source.close();
  }, [target.run, target.job]);
  return (
    <div className="pc-modal-back" onClick={onClose}>
      <section className="pc-modal" onClick={(e) => e.stopPropagation()}>
        <header>
          <div><span className="pc-eyebrow">Live run log</span><h2>{log?.run_name || target.run || target.job || "Run"}</h2></div>
          <div className="pc-row">
            <StatusDot tone={log?.running ? "running" : log?.state === "succeeded" ? "pass" : "neutral"} label={log?.state || "loading"} />
            {log?.running && log.job_id ? <button className="pc-btn ghost danger" onClick={() => onCancel(log.job_id!)}>Cancel</button> : null}
            <button className="pc-btn ghost" onClick={() => navigator.clipboard?.writeText(log?.log_tail || "")}><Copy size={14} /> Copy</button>
            <button className="pc-btn ghost" onClick={onClose}>Close</button>
          </div>
        </header>
        <pre>{err || log?.log_tail || "Waiting for log output…"}</pre>
      </section>
    </div>
  );
}

/* --------------------------------------------------------------- helpers */

function barTone(rate: number): Tone {
  if (rate >= 0.7) return "pass";
  if (rate >= 0.4) return "warn";
  return "block";
}

function gateReasons(detail: RunDetail | null): Array<{ label: string; detail?: string; tone: Tone }> {
  const s = (detail as any)?.summary || {};
  if (!detail || !s.config) return [{ label: "No run evidence yet", detail: "Train a run to populate the gate.", tone: "neutral" }];
  const out: Array<{ label: string; detail?: string; tone: Tone }> = [];

  const pf = s.preflight || (detail as any).preflight || {};
  const pfStatus = String(pf.status || "").toLowerCase();
  const blocking = (pf.blocking_checks || []).length;
  const warning = (pf.warning_checks || []).length;
  out.push({
    label: "Preflight",
    detail: blocking ? `${blocking} blocking check(s)` : warning ? `${warning} warning(s)` : pf.summary || "checks recorded",
    tone: blocking || pfStatus.includes("block") ? "block" : warning || pfStatus.includes("warn") ? "warn" : pf.status ? "pass" : "warn"
  });

  const ev = s.eval || {};
  const rate = ev.pass_rate ?? 0;
  out.push({ label: "Visible eval", detail: `${ev.num_passed ?? 0}/${ev.num_examples ?? 0} passed`, tone: rate >= 0.5 ? "pass" : rate > 0 ? "warn" : "block" });

  const echo = ev.prompt_echo_rate ?? 0;
  out.push({ label: "Prompt echo", detail: percent(echo), tone: echo > 0.1 ? "warn" : "pass" });

  const hon = s.honesty || {};
  const leaks = (hon.exact_prompt_leaks ?? 0) + (hon.near_prompt_leaks ?? 0);
  out.push({ label: "Data honesty", detail: leaks ? `${leaks} prompt leak(s)` : hon.summary || "no leaks detected", tone: leaks ? "block" : hon.status ? "pass" : "warn" });

  const gate = s.long_run_gate || s.release_gate || {};
  const gs = String(gate.status || gate.verdict || "").toLowerCase();
  if (gate.status || gate.verdict) {
    out.push({ label: "Release gate", detail: gate.summary || (gate.issues || []).slice(0, 1).join("; ") || gs, tone: gs.includes("pass") || gs.includes("approv") ? "pass" : gs.includes("block") || gs.includes("fail") ? "block" : "warn" });
  }
  return out;
}

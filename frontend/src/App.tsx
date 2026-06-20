import {
  Archive,
  BarChart3,
  Boxes,
  Check,
  ChevronDown,
  CircleAlert,
  Copy,
  Cpu,
  Database,
  FlaskConical,
  Gauge,
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
  X
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type * as React from "react";
import { archiveRuns, cancelRun, generateEvalStarter, generateSftStarter, generateText, importHf, inspectTuning, listRuns, loadPresets, loadRun, loadStatus, runLogStreamUrl, startRun } from "./api";
import type { GenerateResult, JobStatus, ModelConfig, RunDetail, RunLog, RunSummary, Tone } from "./types";
import { compactNumber, fixed, latestRun, lossPoints, parseEvalScore, percent, releaseTone, runTone, statusLabel } from "./utils";

type SectionId = "overview" | "runs" | "playground" | "training" | "eval" | "release" | "dataset" | "settings";

const NAV: Array<{ id: SectionId; label: string; icon: LucideIcon; group: "main" | "model" | "system" }> = [
  { id: "overview", label: "Overview", icon: LayoutGrid, group: "main" },
  { id: "runs", label: "Runs", icon: Boxes, group: "main" },
  { id: "playground", label: "Playground", icon: MessageSquare, group: "model" },
  { id: "training", label: "Training", icon: Gauge, group: "model" },
  { id: "eval", label: "Evaluation", icon: BarChart3, group: "model" },
  { id: "release", label: "Release gate", icon: Lock, group: "model" },
  { id: "dataset", label: "Dataset", icon: Database, group: "system" },
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
    openNew: (pack?: string) => setNewRun({ pack }),
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
          {!loading && runs.length === 0 ? <Welcome {...ctx} /> : render(ctx)}
        </section>
      </main>

      {logTarget ? <LogModal target={logTarget} onClose={() => setLogTarget(null)} onCancel={cancelRun} /> : null}
      {newRun ? <NewRunModal initialPack={newRun.pack} onClose={() => setNewRun(null)} onLaunched={(t) => { setNewRun(null); setLogTarget(t); refresh(true); }} /> : null}
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
  openNew: (pack?: string) => void;
  setSection: (s: SectionId) => void;
};

function render(p: SectionProps) {
  switch (p.section) {
    case "runs": return <RunsView {...p} />;
    case "playground": return <PlaygroundView {...p} />;
    case "training": return <TrainingView {...p} />;
    case "eval": return <EvalView {...p} />;
    case "release": return <ReleaseView {...p} />;
    case "dataset": return <DatasetView {...p} />;
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
  return (
    <div className="pc-stack">
      <div className="pc-toolbar">
        <div className="pc-toolbar-meta"><strong>{runs.length}</strong> runs in the local bank</div>
        <div className="pc-row">
          <button className="pc-btn" onClick={() => launchSmoke(true)}><ShieldCheck size={15} /> Preflight</button>
          <button className="pc-btn" onClick={() => launchSmoke(false)}><Terminal size={15} /> Smoke run</button>
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
      </aside>
    </div>
  );
}

/* -------------------------------------------------------------- training */

function TrainingView({ detail, selectedRun, openLogs }: SectionProps) {
  const s = (detail as any)?.summary || {};
  const tps = s.sft?.throughput?.tokens_per_second ?? s.base?.throughput?.tokens_per_second;
  return (
    <div className="pc-stack">
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

function EvalView({ detail, selectedRun }: SectionProps) {
  const s = (detail as any)?.summary || {};
  const ev = s.eval || {};
  const score = parseEvalScore(selectedRun?.eval_score);
  const cats: Record<string, any> = ev.category_breakdown || {};
  const catRows = Object.entries(cats).slice(0, 8);
  return (
    <div className="pc-stack">
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

function ReleaseView({ detail, tone }: SectionProps) {
  const reasons = gateReasons(detail);
  return (
    <div className="pc-stack">
      <Banner
        tone={tone}
        title={tone === "pass" ? "Release gate passed" : tone === "block" ? "Release gate blocked" : tone === "neutral" ? "No release evidence yet" : "Release gate needs review"}
        body={tone === "pass" ? "Every checked gate is satisfied. This run can move to handoff." : "Resolve the failing checks below before publishing or handing off this model."}
      />
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
    { icon: Gauge, t: "Train a model", d: "Pick a preset. It builds a tokenizer, pretrains, fine-tunes on chat, and evaluates — all locally." },
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
type DataSource = "example" | "pack" | "hf";

function NewRunModal({ initialPack, onClose, onLaunched }: { initialPack?: string; onClose: () => void; onLaunched: (t: { run?: string; job?: string }) => void }) {
  const [step, setStep] = useState<WizStep>(initialPack ? 2 : 1);
  const [source, setSource] = useState<DataSource>("example");
  const [pack, setPack] = useState(initialPack || "");

  const [dataset, setDataset] = useState("HuggingFaceTB/smollm-corpus");
  const [maxRows, setMaxRows] = useState(2000);
  const [importBusy, setImportBusy] = useState(false);
  const [importErr, setImportErr] = useState("");

  const [insp, setInsp] = useState<Record<string, any> | null>(null);
  const [inspBusy, setInspBusy] = useState(false);
  const [inspErr, setInspErr] = useState("");
  const [genBusy, setGenBusy] = useState(false);

  const [name, setName] = useState("");
  const [preset, setPreset] = useState("tiny");
  const [presets, setPresets] = useState<Record<string, any>>({});
  const [launching, setLaunching] = useState(false);
  const [err, setErr] = useState("");

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

  const launch = async () => {
    if (!pack.trim()) { setErr("Choose a dataset pack to train on."); return; }
    setLaunching(true); setErr("");
    try {
      const started = await startRun({ dataset_pack: pack.trim(), run_name: name.trim() || undefined, preset });
      onLaunched({ job: started.job?.id, run: started.job?.run_name });
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)); } finally { setLaunching(false); }
  };

  const stepLabel = step === 1 ? "Choose your data" : step === 2 ? "Training data" : "Train";
  const statTone = (st?: string): Tone => st === "ready" ? "pass" : st === "caution" ? "warn" : "block";

  return (
    <div className="pc-modal-back" onClick={onClose}>
      <section className="pc-modal pc-new" onClick={(e) => e.stopPropagation()}>
        <header>
          <div><span className="pc-eyebrow">Create a domain model · step {step} of 3</span><h2>{stepLabel}</h2></div>
          <button className="pc-btn ghost" onClick={onClose}><X size={15} /></button>
        </header>
        <div className="pc-new-body">
          {step === 1 ? (
            <>
              <div className="pc-source">
                {([["example", "Use an example"], ["pack", "I have a dataset pack"], ["hf", "Import from Hugging Face"]] as [DataSource, string][]).map(([k, l]) => (
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
              {err ? <Banner tone="block" title="Could not launch" body={err} onClose={() => setErr("")} /> : null}
            </>
          ) : null}
        </div>
        <footer className="pc-new-foot">
          {step > 1
            ? <button className="pc-btn ghost" onClick={() => setStep((s) => (s - 1) as WizStep)}>Back</button>
            : <button className="pc-btn ghost" onClick={onClose}>Cancel</button>}
          {step === 1 ? (
            source === "hf"
              ? <button className="pc-btn primary" onClick={runImport} disabled={importBusy}>{importBusy ? <Loader2 size={15} className="spin" /> : <Database size={15} />} Import & continue</button>
              : <button className="pc-btn primary" onClick={() => pack.trim() && goPrepare(pack.trim())} disabled={!pack.trim()}>Next: training data →</button>
          ) : null}
          {step === 2 ? <button className="pc-btn primary" onClick={() => setStep(3)} disabled={inspBusy}>Next: train →</button> : null}
          {step === 3 ? <button className="pc-btn primary" onClick={launch} disabled={launching}>{launching ? <Loader2 size={15} className="spin" /> : <Gauge size={15} />} Start training</button> : null}
        </footer>
      </section>
    </div>
  );
}

function Stat({ label, value, tone }: { label: string; value: React.ReactNode; tone: Tone }) {
  return <div className={`pc-stat ${tone}`}><span>{label}</span><strong>{value}</strong></div>;
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

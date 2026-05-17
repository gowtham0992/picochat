const state = {
  runs: [],
  selectedRun: null,
  pendingArchiveRun: null,
  archiveSelection: new Set(),
  detail: null,
  workflowRunName: null,
  activeView: readInitialAppView(),
  guideStep: 0,
  guideRunName: null,
  activePanel: "dataset",
  activeStage: "dataset",
  viewMode: readInitialViewMode(),
  theme: readInitialTheme(),
  activeReport: "summary",
  compareRuns: [],
  compareDetails: {},
  comparison: null,
  corpusSourcePreview: null,
  datasetFlightPlan: null,
  hfImport: null,
  sftStarter: null,
  evalStarter: null,
  benchmarkPack: null,
  datasetPackInit: null,
  tuningInspection: null,
  packEditor: null,
  runJob: null,
  runJobs: [],
  runPresets: {},
  runJobLoaded: false,
  runPollTimer: null,
  tokenTimer: null,
  generationTimer: null,
  statusTimer: null,
};

const SAMPLE_DATASET_PACK = "examples/tiny_dataset_pack.json";
const SAMPLE_CHAT_INPUT = "examples/tiny_chat.jsonl";
const SAMPLE_EVAL_INPUT = "examples/tiny_eval.jsonl";
const STARTER_ROW_TARGETS = {
  sft: 300,
  eval: 80,
};
const BENCHMARK_SFT_DEFAULTS = {
  sftSteps: 300,
  sftLearningRate: 0.0001,
  sftPatience: 2,
};
const MUON_EMA_TRIAL_DEFAULTS = {
  baseOptimizer: "muon",
  sftOptimizer: "adamw",
  baseMuonLearningRate: 0.02,
  sftMuonLearningRate: 0.02,
  baseEmaDecay: 0.995,
  sftEmaDecay: 0.995,
};
const APP_VIEWS = ["home", "guide", "workbench", "scale"];
const PICOCHAT_REPO_URL = "https://github.com/gowtham0992/picochat.git";
const SCALE_PRESETS = ["h100-100m-ddp8", "h100-100m", "h100-pilot", "climbmix-pilot", "mps-local", "medium", "small"];
const H100_SCALE_PRESETS = new Set(["h100-100m-ddp8", "h100-100m", "h100-pilot"]);
const DDP_SCALE_PRESETS = new Set(["h100-100m-ddp8"]);
const SCALE_IMPORT_DEFAULTS = {
  "h100-100m-ddp8": { shards: 170, maxRows: 800000 },
  "h100-100m": { shards: 170, maxRows: 800000 },
  "h100-pilot": { shards: 16, maxRows: 80000 },
  "climbmix-pilot": { shards: 1, maxRows: 5000 },
  "mps-local": { shards: 1, maxRows: 5000 },
  medium: { shards: 1, maxRows: 5000 },
  small: { shards: 1, maxRows: 1000 },
};

const LAUNCH_CONTROL_IDS = [
  "launch-pack-path",
  "launch-run-name",
  "launch-preset",
  "launch-tokenizer-type",
  "launch-bpe-pretokenizer",
  "launch-tokenizer-vocab-size",
  "launch-context-size",
  "launch-base-steps",
  "launch-sft-steps",
  "launch-n-embd",
  "launch-n-head",
  "launch-n-kv-head",
  "launch-n-layer",
  "launch-norm-type",
  "launch-position-encoding",
  "launch-activation",
  "launch-tie-embeddings",
  "launch-qk-norm",
  "launch-parallel-residual",
  "launch-base-batch-size",
  "launch-sft-batch-size",
  "launch-base-learning-rate",
  "launch-sft-learning-rate",
  "launch-base-lr-decay",
  "launch-sft-lr-decay",
  "launch-base-lr-warmup-steps",
  "launch-sft-lr-warmup-steps",
  "launch-base-grad-clip",
  "launch-sft-grad-clip",
  "launch-base-grad-accum-steps",
  "launch-base-dataset-mode",
  "launch-base-shard-token-size",
  "launch-base-shard-cache-size",
  "launch-sft-grad-accum-steps",
  "launch-base-optimizer",
  "launch-sft-optimizer",
  "launch-base-muon-learning-rate",
  "launch-sft-muon-learning-rate",
  "launch-base-ema-decay",
  "launch-sft-ema-decay",
  "launch-device",
  "launch-precision",
  "launch-matmul-precision",
  "launch-attn-backend",
  "launch-torch-compile",
  "launch-torch-compile-mode",
  "launch-gradient-checkpointing",
  "launch-auto-lr-scaling",
  "launch-loss-spike-rollback",
  "launch-base-early-stop-patience",
  "launch-sft-early-stop-patience",
  "launch-sft-sampling",
  "launch-sft-packing",
  "launch-eval-max-new-tokens",
  "launch-target-param-data-ratio",
  "launch-seed",
  "launch-min-score",
];

const $ = (id) => document.getElementById(id);

const PANEL_GUIDES = {
  dataset: {
    station: "DATASET BAY",
    question: "Is this source worth training on?",
    signal: "Low duplication, enough documents, clear held-out split.",
    caution: "Tiny or repeated text mostly teaches memorization.",
  },
  tokenizer: {
    station: "TOKENIZER LAB",
    question: "How does text become model IDs?",
    signal: "BPE should lower BPB versus char/byte on the same corpus.",
    caution: "Bigger vocab can help compression but raises model parameters.",
  },
  training: {
    station: "TRAINING DASH",
    question: "Is loss improving without memorizing?",
    signal: "Validation loss falls, gap stays controlled, best checkpoint is clear.",
    caution: "Train loss falling while val stalls means more steps may hurt.",
  },
  generation: {
    station: "GENERATION DECK",
    question: "What does the checkpoint actually say?",
    signal: "Prompt constraints appear without role echo or loops.",
    caution: "Sampling settings change style; they do not add knowledge.",
  },
  eval: {
    station: "EVAL SCOREBOARD",
    question: "Which behavior failed, not just how many?",
    signal: "Ladder levels separate heldout, transfer, adversarial, and memorization.",
    caution: "Treat weak levels as the next curriculum target.",
  },
  report: {
    station: "REPORT VAULT",
    question: "Can the score be traced back to evidence?",
    signal: "Reports tie data, config, checkpoint, loss, eval, and warnings together.",
    caution: "A number without artifacts is not a useful experiment.",
  },
  compare: {
    station: "COMPARE RUNS",
    question: "Did the change actually help?",
    signal: "Compare pass rate with BPB and memorization, not raw loss alone.",
    caution: "Raw loss is only comparable when tokenizer and eval are the same.",
  },
};

const PANEL_METRICS = {
  dataset: [
    ["Documents", "How many separate text units the corpus builder found."],
    ["Duplicate rate", "Repeated data can make a tiny model look smarter than it is."],
    ["Held-out split", "Validation text should be separate enough to catch memorization."],
  ],
  tokenizer: [
    ["Vocab", "How many token IDs the model can choose from."],
    ["BPB", "Bits per byte; lower means the tokenizer/model compresses text better."],
    ["Special tokens", "Control markers like BOS, EOS, PAD, and UNK."],
  ],
  training: [
    ["Train loss", "How well the model predicts text it trains on."],
    ["Val loss", "How well it predicts held-out text it did not directly train on."],
    ["Gap", "Val loss minus train loss; a growing gap is an overfitting warning."],
  ],
  generation: [
    ["Temperature", "Higher values make sampling more random."],
    ["Top K / Top P", "Limits the token choices during generation."],
    ["Logprob", "Model confidence for each generated token; less negative is more confident."],
  ],
  eval: [
    ["Pass rate", "How many checks passed; useful only with a good eval set."],
    ["Ladder level", "Heldout, transfer, adversarial, and memorization checks separate failure types."],
    ["Unsupported claim", "The model said something the eval says it should not say."],
  ],
  report: [
    ["Artifacts", "Files that let us reproduce and inspect the run later."],
    ["Honesty report", "Leakage and overlap checks for the dataset/eval split."],
    ["Summary", "The run's compact ledger of config, scores, and warnings."],
  ],
  compare: [
    ["Delta", "Change versus another run; this is how we learn what helped."],
    ["Best eval", "Highest pass rate on the selected eval set."],
    ["Comparable", "Runs are most comparable when tokenizer, eval, and dataset are controlled."],
  ],
};

const STAGE_LESSONS = {
  dataset: "This is the raw reading material. Bad data makes every later metric suspicious.",
  tokenizer: "This is the text-to-number bridge. Better compression usually lowers BPB.",
  base: "This is the actual language-model training loop: predict the next token.",
  sft: "This teaches the base model the chat format and preferred behavior.",
  eval: "This checks behavior on prompts the model should answer, refuse, or avoid memorizing.",
  chat: "This samples from the checkpoint; settings change style, not learned knowledge.",
  report: "This keeps receipts so the experiment can be reviewed later.",
};

const GUIDE_STEPS = [
  {
    label: "Choose dataset",
    note: "Pick Hugging Face, local docs, an existing pack, or the sample.",
  },
  {
    label: "Check corpus",
    note: "Make sure the source can actually become training text.",
  },
  {
    label: "Create SFT starter",
    note: "Draft chat examples from the corpus.",
  },
  {
    label: "Create eval starter",
    note: "Draft held-out checks before trusting a score.",
  },
  {
    label: "Edit and validate",
    note: "Replace scaffolds with real domain examples.",
  },
  {
    label: "Smoke train",
    note: "Launch a tiny run only after the files are wired.",
  },
  {
    label: "Read result",
    note: "Evaluate, compare, and decide whether to scale.",
  },
];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function fmtPercent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(2)}%` : "--";
}

function fmtLoss(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(4) : "--";
}

function fmtInt(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString() : "--";
}

function fmtBytes(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  if (number < 1024) return `${number} B`;
  if (number < 1024 * 1024) return `${(number / 1024).toFixed(1)} KB`;
  return `${(number / 1024 / 1024).toFixed(1)} MB`;
}

function shellToken(value) {
  const text = String(value ?? "");
  if (!text) return "''";
  if (/^[A-Za-z0-9_./:=+-]+$/.test(text)) return text;
  return `'${text.replaceAll("'", "'\\''")}'`;
}

function shellCommand(parts) {
  return parts.map(shellToken).join(" ");
}

function padSeed(value) {
  return String(value ?? 42).padStart(4, "0").slice(-4);
}

async function requestJson(url, options = {}) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (error) {
    throw new Error(apiTransportMessage(error));
  }
  const text = await response.text();
  let payload = {};
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      throw new Error(apiNonJsonMessage(url, response, text));
    }
  }
  if (!response.ok) throw apiError(payload, response.statusText);
  return payload;
}

async function fetchJson(url) {
  return requestJson(url);
}

async function postJson(url, body) {
  return requestJson(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function apiTransportMessage(error) {
  const hint = location.protocol === "file:"
    ? " Open Picochat through the web server, not file://. Run: PYTHONPATH=src python -m picochat.cli web"
    : "";
  return `API request failed: ${error.message}.${hint}`;
}

function apiNonJsonMessage(url, response, text) {
  const preview = String(text || "").replace(/\s+/g, " ").trim().slice(0, 160);
  const restartHint = " Stop the terminal running Picochat web, then restart it: PYTHONPATH=src python -m picochat.cli web --port 8765";
  const hint = location.protocol === "file:"
    ? " You are on file://, so /api routes are not available. Start Picochat with: PYTHONPATH=src python -m picochat.cli web"
    : response.status === 404 && String(url).startsWith("/api/")
      ? ` The browser has newer UI code, but the running Python server does not know this API route yet.${restartHint}`
      : ` The Picochat server returned a page instead of JSON.${restartHint}`;
  return `API returned non-JSON for ${url} (${response.status} ${response.statusText}).${hint}${preview ? ` Response starts: ${preview}` : ""}`;
}

function apiError(payload, fallback) {
  const error = new Error(payload?.error || fallback);
  error.payload = payload || {};
  return error;
}

async function boot() {
  bindControls();
  setViewMode(state.viewMode, { persist: false, render: false });
  setTheme(state.theme, { persist: false });
  setAppView(state.activeView, { persist: false, render: false });
  await loadRunPresets();
  await loadRuns();
  await loadRunJobs();
  renderScalePlan();
  renderGuide();
}

function bindControls() {
  document.querySelectorAll("[data-app-view]").forEach((button) => {
    button.addEventListener("click", () => {
      if (button.dataset.workbenchMode) setViewMode(button.dataset.workbenchMode);
      setAppView(button.dataset.appView);
    });
  });
  $("guide-back-button").addEventListener("click", () => setGuideStep(state.guideStep - 1));
  $("guide-next-button").addEventListener("click", () => setGuideStep(state.guideStep + 1));
  $("guide-workbench-button").addEventListener("click", () => {
    setAppView("workbench");
    setViewMode("inspect");
    setPanel("dataset", { focus: true, focusTarget: "panel-dataset" });
  });
  $("guide-step-list").addEventListener("click", (event) => {
    const button = event.target.closest("[data-guide-step]");
    if (!button) return;
    setGuideStep(Number(button.dataset.guideStep));
  });
  $("guide-content").addEventListener("input", syncGuideInputs);
  $("guide-content").addEventListener("click", (event) => {
    const button = event.target.closest("[data-guide-action]");
    if (!button) return;
    handleGuideAction(button.dataset.guideAction).catch((error) => renderGuideError(error));
  });
  $("learn-mode-button").addEventListener("click", () => setViewMode("learn"));
  $("inspect-mode-button").addEventListener("click", () => setViewMode("inspect"));
  $("classic-theme-button").addEventListener("click", () => setTheme("classic"));
  $("paper-theme-button").addEventListener("click", () => setTheme("paper"));
  $("refresh-button").addEventListener("click", () => {
    refreshDashboard().catch((error) => flashStatus(`REFRESH FAULT. | ${error.message}`));
  });
  $("archive-selected-button").addEventListener("click", () => {
    archiveSelectedRun().catch((error) => flashStatus(`ARCHIVE FAULT. | ${error.message}`));
  });
  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-copy-command]");
    if (!button) return;
    copyCommand(button.dataset.copyCommand || "", button);
  });
  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-guide-panel]");
    if (!button) return;
    if (button.dataset.sourceMode) prepareDatasetSourceMode(button.dataset.sourceMode);
    const mode = button.dataset.guideMode;
    if (mode) setViewMode(mode);
    setPanel(button.dataset.guidePanel);
    window.setTimeout(() => focusGuideTarget(button.dataset.guideTarget), 80);
  });
  document.querySelectorAll("[data-panel]").forEach((button) => {
    button.addEventListener("click", () => setPanel(button.dataset.panel, { focus: true }));
  });
  $("pipeline-strip").addEventListener("click", (event) => {
    const button = event.target.closest("[data-stage]");
    if (!button) return;
    setStage(button.dataset.stage, { focus: true });
  });
  $("run-storyline").addEventListener("click", (event) => {
    const button = event.target.closest("[data-stage]");
    if (!button) return;
    setStage(button.dataset.stage, { focus: true });
  });
  $("run-doctor").addEventListener("click", (event) => {
    const button = event.target.closest("[data-stage]");
    if (!button) return;
    setStage(button.dataset.stage, { focus: true });
  });
  $("report-select").addEventListener("change", () => {
    state.activeReport = $("report-select").value;
    loadReport().catch((error) => renderReportError(error));
  });
  $("compare-run-list").addEventListener("change", (event) => {
    if (event.target.matches("[data-compare-run]")) {
      const run = event.target.dataset.compareRun;
      if (event.target.checked) {
        state.compareRuns = [...new Set([...state.compareRuns, run])];
      } else {
        state.compareRuns = state.compareRuns.filter((name) => name !== run);
      }
      renderCompareControls();
      resetCompareLearningPanels();
      renderStartHere();
    }
  });
  $("compare-button").addEventListener("click", () => {
    loadComparison().catch((error) => renderCompareError(error));
  });
  $("preview-corpus-button").addEventListener("click", () => {
    previewCorpusSources().catch((error) => renderCorpusSourcePreviewError(error));
  });
  $("sample-dataset-button").addEventListener("click", () => {
    useSampleDataset().catch((error) => renderDatasetFlightPlanError(error));
  });
  $("flight-check-button").addEventListener("click", () => {
    checkDatasetFlightPlan().catch((error) => renderDatasetFlightPlanError(error));
  });
  $("flight-sft-button").addEventListener("click", () => {
    createSftStarter().catch((error) => renderSftStarterError(error));
  });
  $("flight-eval-button").addEventListener("click", () => {
    createEvalStarter().catch((error) => renderEvalStarterError(error));
  });
  $("flight-benchmark-button").addEventListener("click", () => {
    createBenchmarkTuningPack().catch((error) => renderBenchmarkPackError(error));
  });
  $("flight-apply-button").addEventListener("click", () => {
    applyFlightPlanToLauncher();
  });
  [
    "flight-pack-path",
    "flight-input-path",
    "flight-chat-path",
    "flight-eval-path",
    "flight-sft-max-items",
    "flight-eval-max-items",
  ].forEach((id) => {
    $(id).addEventListener("input", () => {
      syncFlightStarterDefaults();
      renderStartHere();
    });
  });
  $("hf-dataset-input").addEventListener("input", seedHfOutDirFromDataset);
  $("hf-import-button").addEventListener("click", () => {
    importHfDataset().catch((error) => renderHfImportError(error));
  });
  $("hf-climbmix-button")?.addEventListener("click", fillClimbMixImport);
  [
    "scale-dataset-pack",
    "scale-run-name",
    "scale-preset",
    "scale-device",
    "scale-climbmix-shards",
    "scale-max-rows",
    "scale-import-source",
    "scale-import-name",
  ].forEach((id) => {
    $(id)?.addEventListener("input", renderScalePlan);
    $(id)?.addEventListener("change", () => {
      if (id === "scale-preset") applyScalePresetDefaults();
      renderScalePlan();
    });
  });
  $("scale-use-launcher-button")?.addEventListener("click", seedScaleFromLauncher);
  $("scale-refresh-button")?.addEventListener("click", renderScalePlan);
  $("scale-import-button")?.addEventListener("click", () => {
    importCompletedRun().catch((error) => renderScaleImportError(error));
  });
  $("init-pack-button").addEventListener("click", () => {
    initDatasetPack().catch((error) => renderDatasetPackInitError(error));
  });
  $("inspect-tuning-button").addEventListener("click", () => {
    inspectTuningData().catch((error) => renderTuningInspectionError(error));
  });
  $("load-editor-button").addEventListener("click", () => {
    loadPackEditor().catch((error) => renderPackEditorError(error));
  });
  $("save-editor-button").addEventListener("click", () => {
    savePackEditor().catch((error) => renderPackEditorError(error));
  });
  $("add-chat-row-button").addEventListener("click", addChatEditorRow);
  $("add-eval-row-button").addEventListener("click", addEvalEditorRow);
  $("launch-preset").addEventListener("change", applyLaunchPreset);
  LAUNCH_CONTROL_IDS.forEach((id) => {
    $(id)?.addEventListener("input", renderLaunchReadiness);
    $(id)?.addEventListener("change", renderLaunchReadiness);
  });
  $("launch-run-button").addEventListener("click", () => {
    launchRun().catch((error) => renderRunJobError(error));
  });
  $("refresh-run-job-button").addEventListener("click", () => {
    loadRunJobs().catch((error) => renderRunJobError(error));
  });
  $("cancel-run-job-button").addEventListener("click", () => {
    cancelRunJob().catch((error) => renderRunJobError(error));
  });
  $("apply-budget-button").addEventListener("click", () => {
    applyPreviewBudgetToLauncher();
  });
  $("apply-muon-ema-button")?.addEventListener("click", () => {
    applyMuonEmaTrial();
  });
  $("tokenize-button").addEventListener("click", () => animateTokenizer());
  $("tokenize-sample-button").addEventListener("click", () => {
    $("tokenizer-input").value = "Picochat";
    animateTokenizer();
  });
  $("generate-button").addEventListener("click", () => {
    animateGeneration().catch((error) => {
      $("generation-output").textContent = `FAULT: ${error.message}`;
      $("logprob-bars").innerHTML = "";
      $("generate-button").disabled = false;
    });
  });
  $("temperature-slider").addEventListener("input", () => {
    $("temperature-value").textContent = Number($("temperature-slider").value).toFixed(2);
  });
  $("topk-slider").addEventListener("input", () => {
    $("topk-value").textContent = $("topk-slider").value;
  });
  $("topp-slider").addEventListener("input", () => {
    $("topp-value").textContent = Number($("topp-slider").value).toFixed(2);
  });
  $("repeat-slider").addEventListener("input", () => {
    $("repeat-value").textContent = Number($("repeat-slider").value).toFixed(2);
  });
  $("max-tokens-slider").addEventListener("input", () => {
    $("max-tokens-value").textContent = $("max-tokens-slider").value;
  });
}

async function loadRuns() {
  const payload = await fetchJson("/api/runs");
  state.runs = payload.runs;
  $("run-count").textContent = `${state.runs.length} RUNS`;
  const runNames = state.runs.map((run) => run.name);
  state.archiveSelection = new Set([...state.archiveSelection].filter((name) => runNames.includes(name)));
  if (state.selectedRun && !runNames.includes(state.selectedRun)) {
    state.selectedRun = null;
    state.pendingArchiveRun = null;
  }
  if (!state.selectedRun && state.runs.length) {
    state.selectedRun = state.runs[state.runs.length - 1].name;
  }
  state.compareRuns = state.compareRuns.filter((name) => runNames.includes(name));
  if (!state.compareRuns.length) {
    state.compareRuns = state.runs.slice(-2).map((run) => run.name);
  }
  renderRuns();
  renderCompareControls();
  if (state.selectedRun) {
    await loadRun(state.selectedRun);
  } else {
    state.detail = null;
    renderAll();
  }
}

function renderRuns() {
  const list = $("run-list");
  renderRunArchiveAction();
  if (!state.runs.length) {
    list.innerHTML = '<div class="empty">NO RUN ARTIFACTS FOUND.</div>';
    return;
  }
  list.innerHTML = state.runs.map((run) => `
    <div class="run-row ${run.name === state.selectedRun ? "active" : ""} ${state.archiveSelection.has(run.name) ? "marked" : ""}">
      <label class="run-archive-toggle">
        <input type="checkbox" data-archive-run="${escapeHtml(run.name)}" ${state.archiveSelection.has(run.name) ? "checked" : ""}>
        <span>${state.archiveSelection.has(run.name) ? "SELECTED" : "SELECT"}</span>
      </label>
      <button class="run-button ${run.name === state.selectedRun ? "active" : ""}" type="button" data-run="${escapeHtml(run.name)}">
        <span>${escapeHtml(run.name)}</span>
        <small>${escapeHtml(run.eval_score)} | ${fmtPercent(run.pass_rate)} | CTX ${escapeHtml(run.context_size)}</small>
      </button>
    </div>
  `).join("");
  list.querySelectorAll("[data-archive-run]").forEach((checkbox) => {
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) {
        state.archiveSelection.add(checkbox.dataset.archiveRun);
      } else {
        state.archiveSelection.delete(checkbox.dataset.archiveRun);
      }
      state.pendingArchiveRun = null;
      renderRuns();
    });
  });
  list.querySelectorAll("[data-run]").forEach((button) => {
    button.addEventListener("click", async () => {
      state.selectedRun = button.dataset.run;
      state.pendingArchiveRun = null;
      renderRuns();
      await loadRun(state.selectedRun);
      await loadRunJobs();
    });
  });
}

function renderRunArchiveAction() {
  const button = $("archive-selected-button");
  const runNames = selectedArchiveRuns();
  const key = archiveSelectionKey(runNames);
  const armed = Boolean(runNames.length && state.pendingArchiveRun === key);
  button.disabled = runNames.length === 0;
  button.textContent = armed
    ? `CONFIRM ARCHIVE ${runNames.length} RUN${runNames.length === 1 ? "" : "S"}`
    : runNames.length
      ? `ARCHIVE ${runNames.length} RUN${runNames.length === 1 ? "" : "S"}`
      : "ARCHIVE SELECTED";
  button.classList.toggle("armed", armed);
}

async function archiveSelectedRun() {
  const runNames = selectedArchiveRuns();
  if (!runNames.length) throw new Error("mark one or more runs first");
  const key = archiveSelectionKey(runNames);
  if (state.pendingArchiveRun !== key) {
    state.pendingArchiveRun = key;
    renderRunArchiveAction();
    flashStatus(`ARCHIVE ARMED FOR ${runNames.length} RUN${runNames.length === 1 ? "" : "S"}. | CLICK AGAIN TO MOVE OUT OF RUN BANK.`);
    return;
  }
  const button = $("archive-selected-button");
  button.disabled = true;
  button.textContent = "ARCHIVING";
  try {
    const payload = await postJson("/api/run/archive", { run_names: runNames });
    const archivedNames = new Set((payload.archived_runs || []).map((run) => run.run_name));
    if (archivedNames.has(state.selectedRun)) state.selectedRun = null;
    state.pendingArchiveRun = null;
    state.archiveSelection.clear();
    state.detail = null;
    state.runJobs = (state.runJobs || []).filter((job) => !archivedNames.has(job.run_name));
    if (state.runJob && archivedNames.has(state.runJob.run_name)) state.runJob = null;
    renderRunJob(state.runJob);
    renderRunJobList();
    flashStatus(`ARCHIVED ${archivedNames.size} RUN${archivedNames.size === 1 ? "" : "S"}. | ${payload.archive_root || "ARCHIVE READY"}`);
    await loadRuns();
  } finally {
    button.textContent = "ARCHIVE SELECTED";
    renderRunArchiveAction();
  }
}

function selectedArchiveRuns() {
  return state.runs
    .filter((run) => state.archiveSelection.has(run.name))
    .map((run) => run.name);
}

function archiveSelectionKey(runNames = selectedArchiveRuns()) {
  return runNames.join("\n");
}

async function refreshDashboard() {
  const button = $("refresh-button");
  button.disabled = true;
  button.textContent = "REFRESHING";
  try {
    await loadRuns();
    await loadRunJobs();
    flashStatus("REFRESHED. | RUN BANK, SELECTED RUN, AND JOB STATUS UPDATED.");
  } finally {
    button.disabled = false;
    button.textContent = "REFRESHED";
    window.setTimeout(() => {
      if (!button.disabled && button.textContent === "REFRESHED") {
        button.textContent = "REFRESH";
      }
    }, 2200);
  }
}

async function loadRun(name) {
  state.detail = await fetchJson(`/api/run?name=${encodeURIComponent(name)}`);
  if (state.workflowRunName !== name) {
    state.workflowRunName = name;
    resetWorkflowFromRunConfig(state.detail?.summary?.config || {});
  }
  renderAll();
}

function resetWorkflowFromRunConfig(config = {}) {
  state.datasetFlightPlan = null;
  state.corpusSourcePreview = null;
  state.hfImport = null;
  state.sftStarter = null;
  state.evalStarter = null;
  state.datasetPackInit = null;
  state.tuningInspection = null;
  state.packEditor = null;

  const datasetPack = config.dataset_pack || "";
  const corpusInput = datasetPack ? "" : config.corpus_input || "";
  const chatInput = config.chat_input || "";
  const evalInput = config.eval_input || "";
  const minScore = config.min_quality_score || 0;

  $("flight-pack-path").value = datasetPack;
  $("flight-input-path").value = corpusInput;
  $("flight-chat-path").value = chatInput;
  $("flight-sft-out-path").value = suggestedSftStarterPath(chatInput || datasetPack || corpusInput || "my_pack/chat.jsonl");
  $("flight-eval-path").value = evalInput;
  $("flight-eval-out-path").value = suggestedEvalStarterPath(evalInput || datasetPack || corpusInput || "my_pack/eval.jsonl");
  $("flight-min-score").value = minScore;

  $("preview-pack-path").value = datasetPack;
  $("preview-recipe-path").value = config.corpus_recipe || "";
  $("preview-input-path").value = corpusInput;
  $("preview-chat-path").value = chatInput;
  $("preview-eval-path").value = evalInput;
  $("preview-min-score").value = minScore;

  $("tuning-pack-path").value = datasetPack;
  $("tuning-chat-path").value = datasetPack ? "" : chatInput;
  $("tuning-eval-path").value = datasetPack ? "" : evalInput;

  $("editor-pack-path").value = datasetPack;
  $("editor-chat-path").value = datasetPack ? "" : chatInput;
  $("editor-eval-path").value = datasetPack ? "" : evalInput;

  $("launch-pack-path").value = datasetPack;
  $("launch-run-name").value = uniqueRunName(suggestedRunName(datasetPack || state.selectedRun || "picochat"));
  $("launch-min-score").value = minScore;
}

function readInitialViewMode() {
  try {
    return localStorage.getItem("picochat:view-mode") === "inspect" ? "inspect" : "learn";
  } catch {
    return "learn";
  }
}

function readInitialAppView() {
  try {
    const value = localStorage.getItem("picochat:app-view");
    return APP_VIEWS.includes(value) ? value : "home";
  } catch {
    return "home";
  }
}

function readInitialTheme() {
  try {
    const savedTheme = localStorage.getItem("picochat:theme");
    if (savedTheme === "classic" || savedTheme === "paper") return savedTheme;
    return "paper";
  } catch {
    return "paper";
  }
}

function setAppView(view, options = {}) {
  const nextView = APP_VIEWS.includes(view) ? view : "home";
  state.activeView = nextView;
  document.body.classList.toggle("home-active", nextView === "home");
  document.body.classList.toggle("guide-active", nextView === "guide");
  document.body.classList.toggle("workbench-active", nextView === "workbench");
  document.body.classList.toggle("scale-active", nextView === "scale");
  document.querySelectorAll(".app-view").forEach((node) => {
    node.classList.toggle("active", node.id === `${nextView}-view`);
  });
  document.querySelectorAll("[data-app-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.appView === nextView);
  });
  if (options.persist !== false) {
    try {
      localStorage.setItem("picochat:app-view", nextView);
    } catch {
      // localStorage can be unavailable in restricted browser contexts.
    }
  }
  if (options.render !== false) {
    renderGuide();
    renderScalePlan();
    renderStatus();
  }
}

function setViewMode(mode, options = {}) {
  const nextMode = mode === "inspect" ? "inspect" : "learn";
  state.viewMode = nextMode;
  document.body.classList.toggle("learn-mode", nextMode === "learn");
  document.body.classList.toggle("inspect-mode", nextMode === "inspect");
  $("learn-mode-button").classList.toggle("active", nextMode === "learn");
  $("inspect-mode-button").classList.toggle("active", nextMode === "inspect");
  if (options.persist !== false) {
    try {
      localStorage.setItem("picochat:view-mode", nextMode);
    } catch {
      // localStorage can be unavailable in restricted browser contexts.
    }
  }
  if (options.render !== false) {
    renderStartHere();
    renderPanelGuide();
    renderStatus();
  }
}

function setTheme(theme, options = {}) {
  const nextTheme = theme === "paper" ? "paper" : "classic";
  state.theme = nextTheme;
  document.body.classList.toggle("paper-theme", nextTheme === "paper");
  document.body.classList.toggle("classic-theme", nextTheme === "classic");
  $("classic-theme-button").classList.toggle("active", nextTheme === "classic");
  $("paper-theme-button").classList.toggle("active", nextTheme === "paper");
  if (options.persist !== false) {
    try {
      localStorage.setItem("picochat:theme", nextTheme);
    } catch {
      // localStorage can be unavailable in restricted browser contexts.
    }
  }
}

function renderAll() {
  renderPanelGuide();
  renderPipeline();
  renderDataset();
  renderStartHere();
  renderGuide();
  renderTokenizer();
  renderTraining();
  renderGenerationDeck();
  renderEval();
  renderReportList();
  renderCompareControls();
  if (state.activePanel === "report") {
    loadReport().catch((error) => renderReportError(error));
  } else if (state.activePanel === "compare") {
    loadComparison().catch((error) => renderCompareError(error));
  }
  renderStatus();
}

function setStage(name, options = {}) {
  state.activeStage = name;
  const panel = {
    dataset: "dataset",
    tokenizer: "tokenizer",
    base: "training",
    sft: "training",
    eval: "eval",
    chat: "generation",
    report: "report",
  }[name];
  if (panel) setPanel(panel, { focus: options.focus, focusTarget: stageFocusTarget(name) });
  renderPipeline();
}

function setPanel(name, options = {}) {
  state.activePanel = name;
  const stage = {
    dataset: "dataset",
    tokenizer: "tokenizer",
    training: state.activeStage === "sft" ? "sft" : "base",
    generation: "chat",
    eval: "eval",
    report: "report",
  }[name];
  if (stage) state.activeStage = stage;
  document.querySelectorAll("[data-panel]").forEach((button) => {
    button.classList.toggle("active", button.dataset.panel === name);
  });
  document.querySelectorAll(".panel-screen").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `panel-${name}`);
  });
  renderPanelGuide();
  renderPipeline();
  if (name === "report") {
    loadReport().catch((error) => renderReportError(error));
  } else if (name === "compare") {
    loadComparison().catch((error) => renderCompareError(error));
  }
  renderStartHere();
  renderStatus();
  if (options.focus) {
    window.setTimeout(() => focusGuideTarget(options.focusTarget || `panel-${name}`), 80);
  }
}

function stageFocusTarget(stage) {
  if (state.viewMode === "learn") {
    return {
      dataset: "dataset-source-card",
      tokenizer: "tokenizer-input",
      base: "base-loss-chart",
      sft: "sft-loss-chart",
      eval: "score-table",
      chat: "prompt-input",
      report: "report-select",
    }[stage] || "panel-dataset";
  }
  return {
    dataset: "panel-dataset",
    tokenizer: "panel-tokenizer",
    base: "panel-training",
    sft: "panel-training",
    eval: "panel-eval",
    chat: "panel-generation",
    report: "panel-report",
  }[stage] || "panel-dataset";
}

function renderStartHere() {
  const container = $("start-here");
  if (!container) return;
  const steps = startHereSteps();
  const nextStep = steps.find((step) => step.status !== "done") || steps.at(-1);
  container.innerHTML = `
    <div class="start-here-head">
      <div>
        <p class="kicker">START HERE</p>
        <h2>GUIDED PATH: BEGINNER FIRST, RESEARCH ONLY TO LAUNCH</h2>
      </div>
      <span>${escapeHtml(nextStep ? `NEXT: ${nextStep.label}` : "READY")}</span>
    </div>
    <div class="walkthrough-strip">
      <div>
        <label>BEGINNER WALKTHROUGH</label>
        <strong>${escapeHtml(nextStep?.label || "Ready")}</strong>
        <p>${escapeHtml(nextStep?.action || "Pick a step below to inspect the exact place in the factory.")}</p>
        ${nextStep?.mode === "inspect" ? '<em>SHOW ME switches to Research mode because this step edits files or launches training.</em>' : ""}
      </div>
      <button type="button" data-guide-panel="${escapeHtml(nextStep?.panel || "dataset")}" data-guide-mode="${escapeHtml(nextStep?.mode || "learn")}" data-guide-target="${escapeHtml(nextStep?.target || "panel-dataset")}">SHOW ME</button>
    </div>
    <div class="start-here-grid">
      ${steps.map((step, index) => `
        <button class="start-step ${escapeHtml(step.status)} ${step.status === "next" ? "active" : ""}" type="button" data-guide-panel="${escapeHtml(step.panel)}" data-guide-mode="${escapeHtml(step.mode || "learn")}" data-guide-target="${escapeHtml(step.target || "")}">
          <span>${String(index + 1).padStart(2, "0")}</span>
          <strong>${escapeHtml(step.label)}</strong>
          <em>${escapeHtml(step.signal)}</em>
          <p>${escapeHtml(step.note)}</p>
        </button>
      `).join("")}
    </div>
  `;
}

function setGuideStep(index) {
  const bounded = Math.max(0, Math.min(GUIDE_STEPS.length - 1, Number(index) || 0));
  state.guideStep = bounded;
  renderGuide();
}

function renderGuide() {
  const list = $("guide-step-list");
  const content = $("guide-content");
  if (!list || !content) return;
  const statuses = guideStepStatuses();
  list.innerHTML = GUIDE_STEPS.map((step, index) => `
    <button class="guide-step ${index === state.guideStep ? "active" : ""} ${escapeHtml(statuses[index])}" type="button" data-guide-step="${index}">
      <span>${String(index + 1).padStart(2, "0")}</span>
      <strong>${escapeHtml(step.label)}</strong>
      <em>${escapeHtml(statuses[index])}</em>
      <p>${escapeHtml(step.note)}</p>
    </button>
  `).join("");
  content.innerHTML = guideStepContent(state.guideStep);
  $("guide-back-button").disabled = state.guideStep <= 0;
  $("guide-next-button").disabled = state.guideStep >= GUIDE_STEPS.length - 1;
}

function guideStepStatuses() {
  const hasSource = hasGuideDatasetSource();
  const checked = Boolean(state.datasetFlightPlan || state.corpusSourcePreview);
  const sftReady = Boolean(state.sftStarter || currentChatRowCount() >= 8);
  const evalReady = Boolean(state.evalStarter || currentEvalRowCount() >= 4);
  const tuningChecked = Boolean(state.tuningInspection);
  const launcherReady = launchReadiness().status !== "blocked";
  const guideRunMatches = Boolean(state.guideRunName && state.runJob?.run_name === state.guideRunName);
  const hasRun = guideRunMatches;
  const hasEval = Boolean(guideRunMatches && state.runJob?.summary?.eval?.num_examples);
  return [
    hasSource ? "done" : "next",
    checked ? "done" : hasSource ? "next" : "todo",
    sftReady ? "done" : checked ? "next" : "todo",
    evalReady ? "done" : sftReady ? "next" : "todo",
    tuningChecked ? "done" : evalReady ? "next" : "todo",
    hasRun ? "done" : launcherReady ? "next" : "todo",
    hasEval ? "done" : hasRun ? "next" : "todo",
  ];
}

function hasGuideDatasetSource() {
  return Boolean(
    $("flight-pack-path")?.value.trim()
    || $("flight-input-path")?.value.trim()
    || state.datasetFlightPlan?.dataset_pack
    || state.hfImport?.dataset_pack
  );
}

function currentChatRowCount() {
  return Math.max(
    Number(state.datasetFlightPlan?.chat_data?.num_examples) || 0,
    Number(state.tuningInspection?.chat_data?.num_examples) || 0,
    Number(state.packEditor?.chat_lines) || 0,
    Number(state.sftStarter?.num_rows) || 0,
  );
}

function currentEvalRowCount() {
  return Math.max(
    Number(state.datasetFlightPlan?.eval_data?.num_items) || 0,
    Number(state.tuningInspection?.eval_data?.num_items) || 0,
    Number(state.packEditor?.eval_lines) || 0,
    Number(state.evalStarter?.num_rows) || 0,
  );
}

function guideStepContent(index) {
  return [
    guideChooseDatasetContent,
    guideCheckCorpusContent,
    guideCreateSftContent,
    guideCreateEvalContent,
    guideEditValidateContent,
    guideSmokeTrainContent,
    guideReadResultContent,
  ][index]?.() || "";
}

function guideChooseDatasetContent() {
  return `
    <div class="guide-page">
      <div class="guide-page-head">
        <p class="kicker">STEP 01</p>
        <h2>Choose the text your tiny model will learn from.</h2>
        <p>A dataset is just source material. Picochat converts it into local documents and a dataset pack so the rest of training is reproducible.</p>
        <button type="button" data-guide-action="reset-guide">START A FRESH GUIDED BUILD</button>
      </div>
      <div class="guide-source-grid">
        <div class="guide-card primary">
          <strong>Import from Hugging Face</strong>
          <p>Best when you have a dataset URL or repo id. Picochat writes local documents and a dataset pack.</p>
          <label for="guide-hf-dataset">DATASET URL / ID</label>
          <input id="guide-hf-dataset" type="text" spellcheck="false" value="${escapeHtml($("hf-dataset-input")?.value || "")}" placeholder="HuggingFaceTB/smollm-corpus">
          <div class="guide-inline-grid">
            <div>
              <label for="guide-hf-config">CONFIG</label>
              <input id="guide-hf-config" type="text" spellcheck="false" value="${escapeHtml($("hf-config-name")?.value || "")}" placeholder="optional">
            </div>
            <div>
              <label for="guide-hf-split">SPLIT</label>
              <input id="guide-hf-split" type="text" spellcheck="false" value="${escapeHtml($("hf-split")?.value || "train")}">
            </div>
            <div>
              <label for="guide-hf-text-column">TEXT FIELD</label>
              <input id="guide-hf-text-column" type="text" spellcheck="false" value="${escapeHtml($("hf-text-column")?.value || "text")}">
            </div>
            <div>
              <label for="guide-hf-max-rows">MAX ROWS</label>
              <input id="guide-hf-max-rows" type="number" min="1" max="100000" value="${escapeHtml($("hf-max-rows")?.value || "1000")}">
            </div>
            <div>
              <label for="guide-hf-shards">CLIMBMIX SHARDS</label>
              <input id="guide-hf-shards" type="number" min="1" max="6543" value="${escapeHtml($("hf-shards")?.value || "1")}">
            </div>
          </div>
          <label for="guide-hf-out-dir">LOCAL OUT FOLDER</label>
          <input id="guide-hf-out-dir" type="text" spellcheck="false" value="${escapeHtml($("hf-out-dir")?.value || "")}" placeholder="runs/hf-my-dataset-1000">
          <button type="button" data-guide-action="import-hf">IMPORT AND CONTINUE</button>
        </div>
        <div class="guide-card">
          <strong>Use local docs</strong>
          <p>Use this for a folder of text/markdown/pdf/docx files or a single .txt file.</p>
          <label for="guide-local-path">LOCAL PATH</label>
          <input id="guide-local-path" type="text" spellcheck="false" value="${escapeHtml($("flight-input-path")?.value || "")}" placeholder="my_docs/">
          <button type="button" data-guide-action="use-local">USE LOCAL PATH</button>
        </div>
        <div class="guide-card">
          <strong>Use an existing pack</strong>
          <p>Use this when you already have dataset_pack.json connecting corpus, SFT, and eval.</p>
          <label for="guide-pack-path">DATASET PACK</label>
          <input id="guide-pack-path" type="text" spellcheck="false" value="${escapeHtml($("flight-pack-path")?.value || "")}" placeholder="my_pack/dataset_pack.json">
          <button type="button" data-guide-action="use-pack">USE PACK</button>
        </div>
        <div class="guide-card">
          <strong>I do not have a dataset yet</strong>
          <p>Load Picochat's sample pack to learn the flow first. This is for practice, not a serious model.</p>
          <button type="button" data-guide-action="use-sample">USE SAMPLE DATASET</button>
        </div>
      </div>
      ${guideNotice()}
    </div>
  `;
}

function guideCheckCorpusContent() {
  const report = state.datasetFlightPlan;
  const coach = flightCoach(report);
  return `
    <div class="guide-page">
      <div class="guide-page-head">
        <p class="kicker">STEP 02</p>
        <h2>Check the corpus before training.</h2>
        <p>This reads your selected source, counts documents and characters, catches missing files, and checks whether SFT/eval files are real or just scaffolds.</p>
      </div>
      <div class="guide-current-source">
        <label>CURRENT SOURCE</label>
        <strong>${escapeHtml(currentGuideSource())}</strong>
        <p>${escapeHtml(coach.evidence)}</p>
      </div>
      <div class="guide-action-card ${escapeHtml(coach.status)}">
        <strong>${escapeHtml(coach.happened)}</strong>
        <p>${escapeHtml(coach.detail)}</p>
        <button type="button" data-guide-action="check-corpus">CHECK CORPUS NOW</button>
      </div>
      ${report ? guideReportStats(report) : ""}
      ${guideNotice()}
    </div>
  `;
}

function guideCreateSftContent() {
  const chatRows = currentChatRowCount();
  const requestedRows = boundedNumberInput("flight-sft-max-items", STARTER_ROW_TARGETS.sft, 8, 2000);
  const curriculumSource = $("flight-benchmark-source")?.value || "offline";
  const curriculumProfile = $("flight-benchmark-profile")?.value || "behavior";
  const skillAnswerStyle = $("flight-skill-answer-style")?.value || "direct";
  return `
    <div class="guide-page">
      <div class="guide-page-head">
        <p class="kicker">STEP 03</p>
        <h2>Create the chat SFT starter.</h2>
        <p>SFT teaches response style and behavior. It does not replace base training. Starter rows are scaffolds; you edit them before trusting a run.</p>
      </div>
      <div class="guide-action-card ${chatRows >= 8 ? "ready" : "caution"}">
        <strong>${chatRows ? `${fmtInt(chatRows)} current SFT rows detected.` : "No usable SFT rows detected yet."}</strong>
        <p>${chatRows < STARTER_ROW_TARGETS.sft ? `For a meaningful medium run, generate about ${fmtInt(STARTER_ROW_TARGETS.sft)} rows. Smaller counts are only smoke-test scaffolds.` : "This is enough starter volume for the next medium experiment; still inspect and edit the rows before trusting a run."}</p>
        <div class="guide-inline-fields">
          <div>
            <label for="guide-sft-max-items">ROWS TO GENERATE</label>
            <input id="guide-sft-max-items" type="number" min="8" max="2000" value="${escapeHtml(String(requestedRows))}">
          </div>
          <div>
            <label for="guide-benchmark-source">CURATED SOURCE</label>
            <select id="guide-benchmark-source">
              <option value="offline" ${curriculumSource === "offline" ? "selected" : ""}>OFFLINE SAFE</option>
              <option value="auto" ${curriculumSource === "auto" ? "selected" : ""}>HF AUTO</option>
              <option value="hf" ${curriculumSource === "hf" ? "selected" : ""}>HF REQUIRED</option>
            </select>
          </div>
          <div>
            <label for="guide-benchmark-profile">CURATED PROFILE</label>
            <select id="guide-benchmark-profile">
              <option value="release_behavior" ${curriculumProfile === "release_behavior" ? "selected" : ""}>RELEASE BEHAVIOR</option>
              <option value="behavior" ${curriculumProfile === "behavior" ? "selected" : ""}>BEHAVIOR FIRST</option>
              <option value="weak_skills" ${curriculumProfile === "weak_skills" ? "selected" : ""}>WEAK SKILLS</option>
              <option value="full" ${curriculumProfile === "full" ? "selected" : ""}>FULL MIX</option>
            </select>
          </div>
          <div>
            <label for="guide-skill-answer-style">SKILL ANSWERS</label>
            <select id="guide-skill-answer-style">
              <option value="direct" ${skillAnswerStyle === "direct" ? "selected" : ""}>DIRECT</option>
              <option value="scratchpad" ${skillAnswerStyle === "scratchpad" ? "selected" : ""}>SCRATCHPAD</option>
            </select>
          </div>
        </div>
        <label class="checkbox-line guide-starter-overwrite" for="guide-starter-force">
          <input id="guide-starter-force" type="checkbox" ${$("flight-starter-force")?.checked ? "checked" : ""}>
          OVERWRITE EXISTING STARTER FILE
        </label>
        <button type="button" data-guide-action="create-sft">CREATE SFT STARTER</button>
        <button type="button" data-guide-action="create-benchmark-pack">CREATE CURATED SFT + EVAL</button>
      </div>
      <div id="guide-sft-mirror" class="guide-mirror">${$("flight-sft-result")?.innerHTML || ""}</div>
      ${guideNotice()}
    </div>
  `;
}

function guideCreateEvalContent() {
  const evalRows = currentEvalRowCount();
  const requestedRows = boundedNumberInput("flight-eval-max-items", STARTER_ROW_TARGETS.eval, 4, 500);
  const curriculumSource = $("flight-benchmark-source")?.value || "offline";
  const curriculumProfile = $("flight-benchmark-profile")?.value || "behavior";
  const skillAnswerStyle = $("flight-skill-answer-style")?.value || "direct";
  return `
    <div class="guide-page">
      <div class="guide-page-head">
        <p class="kicker">STEP 04</p>
        <h2>Create the eval starter before scaling.</h2>
        <p>Eval is the scoreboard. It should include answerable questions, refusals, and memorization probes that do not appear in SFT.</p>
      </div>
      <div class="guide-action-card ${evalRows >= 4 ? "ready" : "caution"}">
        <strong>${evalRows ? `${fmtInt(evalRows)} current eval rows detected.` : "No usable eval rows detected yet."}</strong>
        <p>${evalRows < STARTER_ROW_TARGETS.eval ? `Use about ${fmtInt(STARTER_ROW_TARGETS.eval)} eval rows before judging a medium run. The score should cover recall, transfer, refusals, and memorization probes.` : "This is enough eval volume to compare medium experiments; still inspect leakage and weak categories."}</p>
        <div class="guide-inline-fields">
          <div>
            <label for="guide-eval-max-items">ROWS TO GENERATE</label>
            <input id="guide-eval-max-items" type="number" min="4" max="500" value="${escapeHtml(String(requestedRows))}">
          </div>
          <div>
            <label for="guide-benchmark-source">CURATED SOURCE</label>
            <select id="guide-benchmark-source">
              <option value="offline" ${curriculumSource === "offline" ? "selected" : ""}>OFFLINE SAFE</option>
              <option value="auto" ${curriculumSource === "auto" ? "selected" : ""}>HF AUTO</option>
              <option value="hf" ${curriculumSource === "hf" ? "selected" : ""}>HF REQUIRED</option>
            </select>
          </div>
          <div>
            <label for="guide-benchmark-profile">CURATED PROFILE</label>
            <select id="guide-benchmark-profile">
              <option value="release_behavior" ${curriculumProfile === "release_behavior" ? "selected" : ""}>RELEASE BEHAVIOR</option>
              <option value="behavior" ${curriculumProfile === "behavior" ? "selected" : ""}>BEHAVIOR FIRST</option>
              <option value="weak_skills" ${curriculumProfile === "weak_skills" ? "selected" : ""}>WEAK SKILLS</option>
              <option value="full" ${curriculumProfile === "full" ? "selected" : ""}>FULL MIX</option>
            </select>
          </div>
          <div>
            <label for="guide-skill-answer-style">SKILL ANSWERS</label>
            <select id="guide-skill-answer-style">
              <option value="direct" ${skillAnswerStyle === "direct" ? "selected" : ""}>DIRECT</option>
              <option value="scratchpad" ${skillAnswerStyle === "scratchpad" ? "selected" : ""}>SCRATCHPAD</option>
            </select>
          </div>
        </div>
        <label class="checkbox-line guide-starter-overwrite" for="guide-starter-force">
          <input id="guide-starter-force" type="checkbox" ${$("flight-starter-force")?.checked ? "checked" : ""}>
          OVERWRITE EXISTING STARTER FILE
        </label>
        <button type="button" data-guide-action="create-eval">CREATE EVAL STARTER</button>
        <button type="button" data-guide-action="create-benchmark-pack">CREATE CURATED SFT + EVAL</button>
      </div>
      <div id="guide-eval-mirror" class="guide-mirror">${$("flight-eval-result")?.innerHTML || ""}</div>
      ${guideNotice()}
    </div>
  `;
}

function guideEditValidateContent() {
  const report = state.tuningInspection;
  const status = report?.status || "waiting";
  return `
    <div class="guide-page">
      <div class="guide-page-head">
        <p class="kicker">STEP 05</p>
        <h2>Edit the starter rows and validate them.</h2>
        <p>This is the human part. Replace generic rows with real questions, correct answers, refusals, and held-out eval checks.</p>
      </div>
      <div class="guide-split-actions">
        <div class="guide-action-card">
          <strong>Open the dashboard JSONL editor.</strong>
          <p>This stays in the browser. No CLI needed. Save after editing, then validate again.</p>
          <button type="button" data-guide-action="open-editor">OPEN EDITOR</button>
        </div>
        <div class="guide-action-card ${escapeHtml(status)}">
          <strong>Tuning status: ${escapeHtml(String(status).toUpperCase())}</strong>
          <p>${escapeHtml(report?.summary || "Run validation after editing starter rows.")}</p>
          <button type="button" data-guide-action="inspect-tuning">VALIDATE TUNING DATA</button>
        </div>
      </div>
      ${report ? renderTuningPaths(report) : ""}
      ${report ? renderTuningPreflight(report.chat_data, report.eval_data) : ""}
      ${guideNotice()}
    </div>
  `;
}

function guideSmokeTrainContent() {
  const readiness = launchReadiness();
  return `
    <div class="guide-page">
      <div class="guide-page-head">
        <p class="kicker">STEP 06</p>
        <h2>Launch a smoke run.</h2>
        <p>A smoke run proves the pipeline is wired. It is not the final model. Scale only after eval and trust checks make sense.</p>
      </div>
      <div class="guide-action-card ${escapeHtml(readiness.status)}">
        <strong>${escapeHtml(readiness.title)}</strong>
        <p>${escapeHtml(readiness.notes.slice(0, 3).join(" | "))}</p>
        <div class="button-row">
          <button type="button" data-guide-action="apply-plan">APPLY CHECKED PLAN</button>
          <button type="button" data-guide-action="launch-smoke">LAUNCH SMOKE RUN</button>
          <button type="button" data-guide-action="open-launcher">OPEN LAUNCH CONSOLE</button>
        </div>
      </div>
      ${state.runJob ? renderGuideRunJob() : ""}
      ${guideNotice()}
    </div>
  `;
}

function guideReadResultContent() {
  const summary = state.detail?.summary || state.runJob?.summary || {};
  const evalSummary = summary.eval || {};
  return `
    <div class="guide-page">
      <div class="guide-page-head">
        <p class="kicker">STEP 07</p>
        <h2>Read the result like an experiment.</h2>
        <p>Do not judge by one chat sample. Check eval pass rate, BPB/loss, memorization warnings, and compare against a previous run.</p>
      </div>
      <div class="guide-result-grid">
        <div>
          <label>EVAL</label>
          <strong>${fmtInt(evalSummary.num_passed)}/${fmtInt(evalSummary.num_examples)}</strong>
          <p>${fmtPercent(evalSummary.pass_rate)}</p>
        </div>
        <div>
          <label>BASE BPB</label>
          <strong>${fmtLoss(summary.base?.final_val_bpb)}</strong>
          <p>Lower is better on the same tokenizer/eval setup.</p>
        </div>
        <div>
          <label>SFT BPB</label>
          <strong>${fmtLoss(summary.sft?.final_val_bpb)}</strong>
          <p>Watch for overfitting and tiny SFT files.</p>
        </div>
      </div>
      <div class="button-row">
        <button type="button" data-guide-action="open-eval">OPEN EVAL SCOREBOARD</button>
        <button type="button" data-guide-action="open-compare">COMPARE RUNS</button>
        <button type="button" data-guide-action="open-chat">TRY CHAT</button>
      </div>
      ${guideNotice()}
    </div>
  `;
}

function guideReportStats(report) {
  const stats = report.stats || {};
  return `
    <div class="guide-result-grid">
      <div><label>DOCUMENTS</label><strong>${fmtInt(stats.num_documents)}</strong><p>Separate source units.</p></div>
      <div><label>CHARACTERS</label><strong>${fmtInt(stats.num_characters)}</strong><p>Raw text size.</p></div>
      <div><label>DUPLICATES</label><strong>${fmtPercent(stats.duplicate_document_rate || 0)}</strong><p>Repeated docs can fake progress.</p></div>
    </div>
  `;
}

function renderGuideRunJob() {
  return `
    <div class="guide-current-source">
      <label>RUN STATUS</label>
      <strong>${escapeHtml(state.runJob.run_name || "--")} | ${escapeHtml(state.runJob.state || "--")}</strong>
      <p>${escapeHtml(state.runJob.summary_path ? "Summary is available after the run finishes." : "Training logs appear in the workbench launch console.")}</p>
    </div>
  `;
}

function currentGuideSource() {
  return state.hfImport?.dataset_pack
    || $("flight-pack-path")?.value.trim()
    || $("flight-input-path")?.value.trim()
    || "No dataset selected yet.";
}

function guideNotice() {
  return '<div id="guide-status" class="guide-status">READY.</div>';
}

function renderGuideError(error) {
  const target = $("guide-status");
  if (target) target.textContent = `FAULT: ${error.message}`;
  flashStatus(`GUIDE FAULT. | ${error.message}`);
}

function syncGuideInputs(event) {
  const target = event.target;
  if (!target?.id) return;
  const map = {
    "guide-hf-dataset": "hf-dataset-input",
    "guide-hf-out-dir": "hf-out-dir",
    "guide-hf-config": "hf-config-name",
    "guide-hf-split": "hf-split",
    "guide-hf-text-column": "hf-text-column",
    "guide-hf-max-rows": "hf-max-rows",
    "guide-hf-shards": "hf-shards",
    "guide-local-path": "flight-input-path",
    "guide-pack-path": "flight-pack-path",
    "guide-sft-max-items": "flight-sft-max-items",
    "guide-eval-max-items": "flight-eval-max-items",
    "guide-benchmark-source": "flight-benchmark-source",
    "guide-benchmark-profile": "flight-benchmark-profile",
    "guide-skill-answer-style": "flight-skill-answer-style",
    "guide-starter-force": "flight-starter-force",
  };
  const destination = map[target.id];
  if (!destination || !$(destination)) return;
  if (target.type === "checkbox") {
    $(destination).checked = target.checked;
  } else {
    $(destination).value = target.value;
  }
  if (target.id === "guide-hf-dataset") seedHfOutDirFromDataset();
  if (target.id === "guide-local-path" && target.value.trim()) $("flight-pack-path").value = "";
  if (target.id === "guide-pack-path" && target.value.trim()) $("flight-input-path").value = "";
  syncFlightStarterDefaults();
}

async function handleGuideAction(action) {
  if (action === "reset-guide") {
    resetGuidedWorkflow();
    setGuideStep(0);
    flashStatus("GUIDE RESET. | CHOOSE A DATASET SOURCE.");
    return;
  }
  if (action === "import-hf") {
    syncGuideInputValues();
    await importHfDataset();
    setGuideStep(1);
    return;
  }
  if (action === "use-local") {
    syncGuideInputValues();
    if (!$("flight-input-path").value.trim()) throw new Error("enter a local corpus path first");
    $("flight-pack-path").value = "";
    syncFlightStarterDefaults();
    setGuideStep(1);
    flashStatus("LOCAL SOURCE SELECTED. | CHECK CORPUS NEXT.");
    return;
  }
  if (action === "use-pack") {
    syncGuideInputValues();
    if (!$("flight-pack-path").value.trim()) throw new Error("enter a dataset pack path first");
    $("flight-input-path").value = "";
    syncFlightStarterDefaults();
    setGuideStep(1);
    flashStatus("DATASET PACK SELECTED. | CHECK CORPUS NEXT.");
    return;
  }
  if (action === "use-sample") {
    await useSampleDataset();
    setGuideStep(1);
    return;
  }
  if (action === "check-corpus") {
    await checkDatasetFlightPlan();
    setGuideStep(2);
    return;
  }
  if (action === "create-sft") {
    await createSftStarter();
    setGuideStep(3);
    return;
  }
  if (action === "create-eval") {
    await createEvalStarter();
    setGuideStep(4);
    return;
  }
  if (action === "create-benchmark-pack") {
    await createBenchmarkTuningPack();
    setGuideStep(4);
    return;
  }
  if (action === "open-editor") {
    await loadPackEditor();
    setAppView("workbench");
    setViewMode("inspect");
    setPanel("dataset", { focus: true, focusTarget: "pack-editor-card" });
    return;
  }
  if (action === "inspect-tuning") {
    await inspectTuningData();
    renderGuide();
    return;
  }
  if (action === "apply-plan") {
    applyFlightPlanToLauncher();
    renderGuide();
    return;
  }
  if (action === "launch-smoke") {
    if (state.datasetFlightPlan) applyFlightPlanToLauncher(true);
    if (state.runPresets.smoke) {
      $("launch-preset").value = "smoke";
      applyLaunchPreset(true);
    }
    await launchRun();
    state.guideRunName = state.runJob?.run_name || $("launch-run-name").value.trim() || null;
    setGuideStep(6);
    return;
  }
  if (action === "open-launcher") {
    if (state.datasetFlightPlan) applyFlightPlanToLauncher(true);
    setAppView("workbench");
    setViewMode("inspect");
    setPanel("dataset", { focus: true, focusTarget: "run-launcher-card" });
    return;
  }
  if (action === "open-eval") {
    setAppView("workbench");
    setViewMode("learn");
    setPanel("eval", { focus: true, focusTarget: "panel-eval" });
    return;
  }
  if (action === "open-compare") {
    setAppView("workbench");
    setViewMode("learn");
    setPanel("compare", { focus: true, focusTarget: "panel-compare" });
    return;
  }
  if (action === "open-chat") {
    setAppView("workbench");
    setViewMode("learn");
    setPanel("generation", { focus: true, focusTarget: "panel-generation" });
  }
}

function resetGuidedWorkflow() {
  state.datasetFlightPlan = null;
  state.corpusSourcePreview = null;
  state.hfImport = null;
  state.sftStarter = null;
  state.evalStarter = null;
  state.tuningInspection = null;
  state.packEditor = null;
  state.runJob = null;
  state.guideRunName = null;
  [
    "flight-pack-path",
    "flight-input-path",
    "flight-chat-path",
    "flight-sft-out-path",
    "flight-eval-path",
    "flight-eval-out-path",
    "hf-dataset-input",
    "hf-out-dir",
    "hf-config-name",
    "preview-pack-path",
    "preview-input-path",
    "preview-chat-path",
    "preview-eval-path",
    "tuning-pack-path",
    "tuning-chat-path",
    "tuning-eval-path",
    "editor-pack-path",
    "editor-chat-path",
    "editor-eval-path",
    "launch-pack-path",
    "scale-dataset-pack",
    "scale-run-name",
    "scale-import-source",
    "scale-import-name",
  ].forEach((id) => {
    if ($(id)) $(id).value = "";
  });
  $("hf-split").value = "train";
  $("hf-text-column").value = "text";
  $("hf-max-rows").value = "1000";
  if ($("hf-shards")) $("hf-shards").value = "1";
  $("hf-min-chars").value = "20";
  $("flight-sft-max-items").value = String(STARTER_ROW_TARGETS.sft);
  $("flight-eval-max-items").value = String(STARTER_ROW_TARGETS.eval);
  if ($("flight-benchmark-source")) $("flight-benchmark-source").value = "offline";
  if ($("flight-benchmark-profile")) $("flight-benchmark-profile").value = "release_behavior";
  if ($("flight-skill-answer-style")) $("flight-skill-answer-style").value = "direct";
  $("flight-starter-force").checked = false;
  $("flight-min-score").value = "0";
  $("launch-run-name").value = "";
  if ($("launch-long-run-gate-profile")) $("launch-long-run-gate-profile").value = "research";
  renderDatasetFlightPlan(null);
  renderHfImport(null);
  renderSftStarter(null);
  renderEvalStarter(null);
  renderCorpusSourcePreview(null);
  renderTuningInspection(null);
  renderPackEditor(null);
  renderRunJob(null);
  renderLaunchReadiness();
  renderScalePlan();
}

function fillClimbMixImport() {
  $("hf-dataset-input").value = "karpathy/climbmix-400b-shuffle";
  $("hf-out-dir").value = "runs/climbmix-1shard-1k";
  $("hf-config-name").value = "";
  $("hf-split").value = "train";
  $("hf-text-column").value = "text";
  $("hf-max-rows").value = "1000";
  if ($("hf-shards")) $("hf-shards").value = "1";
  $("hf-min-chars").value = "20";
  if ($("scale-dataset-pack")) $("scale-dataset-pack").value = "runs/climbmix-1shard-1k/dataset_pack.json";
  if ($("scale-run-name")) $("scale-run-name").value = "climbmix-pilot-v1";
  if ($("scale-preset")) $("scale-preset").value = "climbmix-pilot";
  if ($("scale-device")) $("scale-device").value = "auto";
  renderScalePlan();
  renderGuide();
  flashStatus("CLIMBMIX READY TO IMPORT. | Click IMPORT HF DATASET when ready.");
}

function syncGuideInputValues() {
  [
    "guide-hf-dataset",
    "guide-hf-out-dir",
    "guide-hf-config",
    "guide-hf-split",
    "guide-hf-text-column",
    "guide-hf-max-rows",
    "guide-hf-shards",
    "guide-local-path",
    "guide-pack-path",
    "guide-sft-max-items",
    "guide-eval-max-items",
    "guide-starter-force",
  ].forEach((id) => {
    const node = $(id);
    if (node) syncGuideInputs({ target: node });
  });
}

function focusGuideTarget(targetId) {
  if (!targetId) return;
  const target = $(targetId);
  if (!target) return;
  const context = target.closest(".terminal-card, .panel-screen, .start-here, .assembly-line");
  target.scrollIntoView({ behavior: "smooth", block: "center" });
  target.classList.remove("guide-target-focus");
  if (context && context !== target) context.classList.remove("guide-target-context");
  window.requestAnimationFrame(() => {
    if (context && context !== target) context.classList.add("guide-target-context");
    target.classList.add("guide-target-focus");
    const focusable = target.classList.contains("panel-screen")
      ? null
      : target.matches("input:not([type='hidden']), textarea, select, button")
        ? target
        : target.querySelector("input:not([type='hidden']), textarea, select, button");
    if (focusable && !focusable.disabled) {
      focusable.focus({ preventScroll: true });
    }
    window.setTimeout(() => {
      target.classList.remove("guide-target-focus");
      if (context && context !== target) context.classList.remove("guide-target-context");
    }, 2400);
  });
}

function startHereSteps() {
  const flightPack = $("flight-pack-path")?.value.trim();
  const flightInput = $("flight-input-path")?.value.trim();
  const hasDatasetSource = Boolean(flightPack || flightInput || state.datasetFlightPlan?.dataset_pack || state.hfImport?.dataset_pack);
  const checkedDataset = Boolean(state.datasetFlightPlan || state.corpusSourcePreview);
  const sftReady = Boolean(state.sftStarter || currentChatRowCount() >= 8);
  const evalReady = Boolean(state.evalStarter || currentEvalRowCount() >= 4);
  const tuningReady = state.tuningInspection?.status === "ready";
  const hasRun = Boolean(state.detail?.summary);
  const evalSummary = state.detail?.eval_reports?.at(-1)?.report?.summary || state.detail?.summary?.eval;
  const hasEval = Boolean(evalSummary?.num_examples);
  const canCompare = state.runs.length >= 2;
  const comparedRuns = state.comparison?.rows?.length || 0;
  const hasComparison = comparedRuns >= 2;
  const steps = [
    {
      label: "Choose dataset",
      panel: "dataset",
      target: "dataset-source-card",
      status: hasDatasetSource ? "done" : "todo",
      signal: hasDatasetSource ? "Source selected" : "HF, local docs, or sample",
      note: "A dataset is the raw text Picochat learns from. Hugging Face import is one way to get it.",
      action: "Choose Hugging Face, local files, an existing pack, or the sample pack if you do not have data yet.",
    },
    {
      label: "Check readiness",
      panel: "dataset",
      target: "flight-check-button",
      status: checkedDataset ? "done" : "todo",
      signal: checkedDataset ? "Corpus checked" : "Run dataset check",
      note: "This catches missing files, tiny corpora, duplicate text, and tuning-data blockers early.",
      action: "Press CHECK DATASET. Picochat will inspect the corpus before any training starts.",
    },
    {
      label: "Create SFT",
      panel: "dataset",
      target: "flight-sft-button",
      status: sftReady ? "done" : "todo",
      signal: sftReady ? `${fmtInt(currentChatRowCount())} chat rows` : "Generate starter chat rows",
      note: "SFT teaches chat behavior. It is not a substitute for base training knowledge.",
      action: "Generate starter chat rows, then edit them into real domain conversations before trusting a run.",
    },
    {
      label: "Create eval",
      panel: "dataset",
      target: "flight-eval-button",
      status: evalReady ? "done" : "todo",
      signal: evalReady ? `${fmtInt(currentEvalRowCount())} eval rows` : "Generate starter eval rows",
      note: "Eval is the scoreboard. Keep answerable, refusal, and memorization probes.",
      action: "Generate starter eval rows. These become the evidence that the model improved without cheating.",
    },
    {
      label: "Inspect/edit",
      panel: "dataset",
      mode: "inspect",
      target: "pack-editor-card",
      status: tuningReady ? "done" : "todo",
      signal: tuningReady ? "Tuning ready" : "Open JSONL editor",
      note: "Rewrite starter rows into real domain questions before trusting a run.",
      action: "Switch to Research only to inspect or edit JSONL. Replace starter rows with real examples from the dataset domain.",
    },
    {
      label: "Train smoke",
      panel: "dataset",
      mode: "inspect",
      target: "run-launcher-card",
      status: hasRun ? "done" : "todo",
      signal: hasRun ? "Run loaded" : "Launch a small run",
      note: "Smoke runs prove the wiring before you spend time on larger experiments.",
      action: "Switch to Research only for the run launcher. Launch a smoke run first; scale only after the wiring is proven.",
    },
    {
      label: "Evaluate/chat",
      panel: hasEval ? "eval" : "generation",
      target: hasEval ? "panel-eval" : "panel-generation",
      status: hasEval ? "done" : "todo",
      signal: hasEval ? `${evalSummary.num_passed}/${evalSummary.num_examples} pass` : "Inspect behavior",
      note: "Use eval for evidence and chat for qualitative failure discovery.",
      action: "Read the eval scoreboard first, then use chat samples to understand the failure cases.",
    },
    {
      label: "Compare",
      panel: "compare",
      target: "panel-compare",
      status: hasComparison ? "done" : "todo",
      signal: hasComparison ? `Compared ${comparedRuns} runs` : canCompare ? `${state.runs.length} runs available` : "Need two runs",
      note: "A better SLM is a measured improvement, not a single good sample.",
      action: "Compare runs side by side. Better means lower loss, stronger evals, and no trust regressions.",
    },
  ];
  let markedNext = false;
  return steps.map((step) => {
    if (step.status === "done") return step;
    if (!markedNext) {
      markedNext = true;
      return { ...step, status: "next" };
    }
    return { ...step, status: "todo" };
  });
}

function renderPanelGuide() {
  const guide = PANEL_GUIDES[state.activePanel] || PANEL_GUIDES.dataset;
  $("panel-guide").innerHTML = `
    <div>
      <label>LEARNING SIGNAL</label>
      <strong>${escapeHtml(guide.station)}</strong>
    </div>
    <p><b>Question</b> ${escapeHtml(guide.question)}</p>
    <p><b>Healthy</b> ${escapeHtml(guide.signal)}</p>
    <p><b>Watch</b> ${escapeHtml(guide.caution)}</p>
  `;
  renderMetricGlossary();
}

function renderMetricGlossary() {
  const metrics = PANEL_METRICS[state.activePanel] || PANEL_METRICS.dataset;
  $("metric-glossary").innerHTML = `
    <label>METRIC DECODER</label>
    <div class="metric-pill-row">
      ${metrics.map(([label, note]) => `
        <div class="metric-pill">
          <strong>${escapeHtml(label)}</strong>
          <span>${escapeHtml(note)}</span>
        </div>
      `).join("")}
    </div>
  `;
}

function renderPipeline() {
  const stages = pipelineStages();
  $("pipeline-run").textContent = state.selectedRun ? `RUN ${state.selectedRun}` : "NO RUN";
  $("pipeline-strip").innerHTML = stages.map((stage) => renderPipelineStage(stage)).join("");
  const active = stages.find((stage) => stage.id === state.activeStage) || stages[0];
  $("pipeline-verdict").innerHTML = learningVerdict(stages);
  $("run-storyline").innerHTML = runStoryTimeline(stages);
  $("run-trust-panel").innerHTML = runTrustPanel();
  $("pipeline-detail").innerHTML = active ? stageDetail(active) : "LOAD A RUN TO INSPECT THE PIPELINE.";
  $("run-doctor").innerHTML = runDoctor(stages);
}

function renderPipelineStage(stage) {
  const health = stageHealth(stage);
  return `
    <button class="pipeline-stage ${stage.id === state.activeStage ? "active" : ""}" type="button" data-stage="${stage.id}">
      <strong>${escapeHtml(stage.label)}</strong>
      <small>${escapeHtml(stage.summary)}</small>
      <em class="stage-health ${health.className}">${escapeHtml(health.label)}</em>
    </button>
  `;
}

function pipelineStages() {
  const detail = state.detail;
  const summary = detail?.summary || {};
  const artifacts = summary.artifacts || {};
  const corpus = summary.corpus || {};
  const config = summary.config || {};
  const outDir = config.out_dir || "runs/manual";
  const corpusPath = artifacts.corpus || `${outDir}/corpus.txt`;
  const tokenizerPath = artifacts.tokenizer || `${outDir}/tokenizer.json`;
  const baseCheckpoint = artifacts.base_eval_checkpoint || summary.base?.eval_checkpoint || summary.base?.checkpoint || `${outDir}/base/checkpoint`;
  const sftCheckpoint = summary.sft?.checkpoint || `${outDir}/sft/checkpoint`;
  const tokenizer = detail?.tokenizer_detail || summary.tokenizer || {};
  const tokenizerType = tokenizer.type || summary.tokenizer?.tokenizer_type || config.tokenizer_type || "char";
  const baseLast = detail?.base_report?.losses?.at(-1);
  const sftLast = detail?.sft_report?.losses?.at(-1);
  const evalReport = detail?.eval_reports?.at(-1)?.report;
  const evalSummary = evalReport?.summary || summary.eval;
  const honesty = summary.honesty || {};
  const reportCount = Object.values(detail?.reports || {}).filter((report) => report.exists).length;
  return [
    {
      id: "dataset",
      label: "DATASET",
      summary: corpus.num_characters ? `${fmtInt(corpus.num_characters)} chars / ${fmtInt(corpus.num_lines)} lines` : "not loaded",
      stats: [
        ["Input", config.corpus_input || "unknown"],
        ["Documents", fmtInt(corpus.num_documents)],
        ["Characters", fmtInt(corpus.num_characters)],
        ["Duplicate docs", fmtPercent(corpus.duplicate_document_rate || 0)],
        ["Duplicate lines", fmtPercent(corpus.duplicate_line_rate || 0)],
        ["Honesty", honesty.status || "--"],
      ],
      note: honesty.summary || (detail?.corpus_preview ? `Preview: ${compactPreview(detail.corpus_preview, 220)}` : "No corpus preview artifact found."),
      command: datasetCommand(config, artifacts),
      ledger: [
        artifactItem("INPUT", "Source", config.corpus_recipe || config.corpus_input || "examples/tiny_corpus.txt"),
        artifactItem("OUTPUT", "Corpus", corpusPath),
        artifactItem("OUTPUT", "Manifest", artifacts.corpus_manifest || `${outDir}/corpus_manifest.json`),
        artifactItem("OUTPUT", "Report", artifacts.corpus_report || `${outDir}/corpus_report.md`),
        artifactItem("OUTPUT", "Honesty", artifacts.honesty_report || `${outDir}/honesty/report.md`),
      ],
    },
    {
      id: "tokenizer",
      label: "TOKENIZER",
      summary: tokenizer.vocab_size ? `${tokenizerType} / ${tokenizer.vocab_size} vocab / ${tokenizer.special_tokens?.length || 0} special` : "not trained",
      stats: [
        ["Type", tokenizerType],
        ["Vocab", tokenizer.vocab_size ?? "--"],
        ["Special", tokenizer.special_tokens?.length ?? "--"],
        ["Text tokens", summary.tokenizer?.num_text_tokens ?? "--"],
      ],
      note: "Turns text into token IDs before the model ever sees it.",
      command: shellCommand([
        "PYTHONPATH=src", "python", "-m", "picochat.cli", "tok", "train",
        "--input", corpusPath,
        "--out", tokenizerPath,
        "--type", tokenizerType,
      ]),
      ledger: [
        artifactItem("INPUT", "Corpus", corpusPath),
        artifactItem("OUTPUT", "Tokenizer JSON", tokenizerPath),
      ],
    },
    {
      id: "base",
      label: "BASE TRAIN",
      summary: baseLast ? `val ${fmtLoss(baseLast.val_loss)} / bpb ${fmtLoss(baseLast.val_bpb)} / ${fmtInt(summary.base?.num_parameters)} params` : "no trace",
      stats: [
        ["Steps", config.base_steps ?? detail?.base_report?.config?.max_steps ?? "--"],
        ["Train loss", baseLast ? fmtLoss(baseLast.train_loss) : "--"],
        ["Val loss", baseLast ? fmtLoss(baseLast.val_loss) : "--"],
        ["Val BPB", baseLast ? fmtLoss(baseLast.val_bpb) : "--"],
        ["Stop", summary.base?.stop_reason || detail?.base_report?.stop_reason || "--"],
        ["Params", fmtInt(summary.base?.num_parameters)],
      ],
      note: "Learns next-token prediction from the corpus. This is the actual tiny language model training stage.",
      command: shellCommand([
        "PYTHONPATH=src", "python", "-m", "picochat.cli", "train", "base",
        "--corpus", corpusPath,
        "--tokenizer", tokenizerPath,
        "--out-dir", `${outDir}/base`,
        "--context-size", config.context_size ?? 128,
        "--n-embd", config.n_embd ?? 64,
        "--n-head", config.n_head ?? 4,
        "--n-kv-head", config.n_kv_head ?? config.n_head ?? 4,
        "--n-layer", config.n_layer ?? 2,
        "--norm-type", config.norm_type || "layernorm",
        "--position-encoding", config.position_encoding || "learned",
        "--activation", config.activation || "gelu",
        "--max-steps", config.base_steps ?? 300,
        "--batch-size", config.base_batch_size ?? 8,
        "--grad-accum-steps", config.base_grad_accum_steps ?? 1,
        "--learning-rate", config.base_learning_rate ?? "3e-4",
        "--early-stop-patience", config.base_early_stop_patience ?? 3,
        "--canary-count", config.canary_count ?? 1,
        "--seed", config.seed ?? 42,
        "--device", config.device || "cpu",
      ]),
      ledger: [
        artifactItem("INPUT", "Corpus", corpusPath),
        artifactItem("INPUT", "Tokenizer", tokenizerPath),
        artifactItem("OUTPUT", "Checkpoint", baseCheckpoint),
        artifactItem("OUTPUT", "Best checkpoint", artifacts.base_best_checkpoint || summary.base?.best_checkpoint?.path),
        artifactItem("OUTPUT", "Trace JSON", `${outDir}/base/train_report.json`),
        artifactItem("OUTPUT", "Report", artifacts.base_report || `${outDir}/base/report.md`),
        artifactItem("OUTPUT", "Sample", `${outDir}/base/sample.txt`),
      ],
    },
    {
      id: "sft",
      label: "CHAT SFT",
      summary: sftLast ? `gap ${fmtLoss(sftLast.val_loss - sftLast.train_loss)} / ${fmtInt(detail?.sft_report?.dataset?.num_examples)} chats` : "no trace",
      stats: [
        ["Steps", config.sft_steps ?? detail?.sft_report?.config?.max_steps ?? "--"],
        ["Train loss", sftLast ? fmtLoss(sftLast.train_loss) : "--"],
        ["Val loss", sftLast ? fmtLoss(sftLast.val_loss) : "--"],
        ["Val BPB", sftLast ? fmtLoss(sftLast.val_bpb) : "--"],
        ["Stop", summary.sft?.stop_reason || detail?.sft_report?.stop_reason || "--"],
        ["Sampling", detail?.sft_report?.dataset?.sampling || config.sft_sampling || "--"],
        ["Packing", detail?.sft_report?.dataset?.packing || config.sft_packing || "--"],
        ["Truncated", summary.sft?.truncated_examples ?? "--"],
        ["Skipped", summary.sft?.skipped_long_examples ?? detail?.sft_report?.dataset?.skipped_long_examples ?? "--"],
      ],
      note: "Tunes the base model on User/Assistant examples. A large loss gap is a memorization warning.",
      command: shellCommand([
        "PYTHONPATH=src", "python", "-m", "picochat.cli", "train", "sft",
        "--input", config.chat_input || "examples/tiny_chat.jsonl",
        "--tokenizer", tokenizerPath,
        "--checkpoint", baseCheckpoint,
        "--out-dir", `${outDir}/sft`,
        "--max-steps", config.sft_steps ?? 600,
        "--batch-size", config.sft_batch_size ?? 7,
        "--grad-accum-steps", config.sft_grad_accum_steps ?? 1,
        "--learning-rate", config.sft_learning_rate ?? "1e-3",
        "--early-stop-patience", config.sft_early_stop_patience ?? 4,
        "--sampling", config.sft_sampling || detail?.sft_report?.dataset?.sampling || "uniform",
        "--packing", config.sft_packing || detail?.sft_report?.dataset?.packing || "separate",
        "--seed", config.seed ?? 42,
        "--device", config.device || "cpu",
      ]),
      ledger: [
        artifactItem("INPUT", "Chat JSONL", config.chat_input || "examples/tiny_chat.jsonl"),
        artifactItem("INPUT", "Base checkpoint", baseCheckpoint),
        artifactItem("INPUT", "Tokenizer", tokenizerPath),
        artifactItem("OUTPUT", "Checkpoint", sftCheckpoint),
        artifactItem("OUTPUT", "Trace JSON", `${outDir}/sft/sft_report.json`),
        artifactItem("OUTPUT", "Report", artifacts.sft_report || `${outDir}/sft/report.md`),
        artifactItem("OUTPUT", "Sample", `${outDir}/sft/sample.txt`),
      ],
    },
    {
      id: "eval",
      label: "EVAL",
      summary: evalSummary ? `${evalSummary.num_passed}/${evalSummary.num_examples} pass` : "no report",
      stats: [
        ["Examples", evalSummary?.num_examples ?? "--"],
        ["Passed", evalSummary?.num_passed ?? "--"],
        ["Failed", evalSummary?.num_failed ?? "--"],
        ["Pass rate", evalSummary ? fmtPercent(evalSummary.pass_rate) : "--"],
      ],
      note: "Checks model replies against answerable and unanswerable prompts.",
      command: shellCommand([
        "PYTHONPATH=src", "python", "-m", "picochat.cli", "eval", "chat",
        "--input", config.eval_input || "examples/tiny_eval.jsonl",
        "--checkpoint", sftCheckpoint,
        "--tokenizer", tokenizerPath,
        "--out-dir", `${outDir}/eval`,
        "--max-new-tokens", config.eval_max_new_tokens ?? 120,
        "--seed", config.seed ?? 42,
        "--device", config.device || "cpu",
      ]),
      ledger: [
        artifactItem("INPUT", "Eval JSONL", config.eval_input || "examples/tiny_eval.jsonl"),
        artifactItem("INPUT", "SFT checkpoint", sftCheckpoint),
        artifactItem("INPUT", "Tokenizer", tokenizerPath),
        artifactItem("OUTPUT", "Eval JSON", `${outDir}/eval/eval_report.json`),
        artifactItem("OUTPUT", "Report", artifacts.eval_report || `${outDir}/eval/report.md`),
      ],
    },
    {
      id: "chat",
      label: "CLI/WEB CHAT",
      summary: `${detail?.sft_sample ? "SFT sample" : "no SFT sample"} / seed ${padSeed(config.seed)}`,
      stats: [
        ["Base sample", detail?.base_sample ? "present" : "missing"],
        ["SFT sample", detail?.sft_sample ? "present" : "missing"],
        ["Context", config.context_size ?? "--"],
        ["Seed", padSeed(config.seed)],
      ],
      note: "Runs live generation from the selected checkpoint through /api/generate.",
      command: shellCommand([
        "PYTHONPATH=src", "python", "-m", "picochat.cli", "chat",
        "--checkpoint", sftCheckpoint,
        "--tokenizer", tokenizerPath,
        "--seed", config.seed ?? 42,
        "--device", config.device || "cpu",
      ]),
      ledger: [
        artifactItem("INPUT", "Checkpoint", sftCheckpoint),
        artifactItem("INPUT", "Tokenizer", tokenizerPath),
        artifactItem("OUTPUT", "Terminal reply", "stdout"),
        artifactItem("OUTPUT", "Web reply", "/api/generate"),
      ],
    },
    {
      id: "report",
      label: "REPORT",
      summary: `${reportCount}/5 markdown reports`,
      stats: [
        ["Summary", detail?.reports?.summary?.exists ? "ready" : "missing"],
        ["Honesty", detail?.reports?.honesty?.exists ? "ready" : "missing"],
        ["Base", detail?.reports?.base?.exists ? "ready" : "missing"],
        ["SFT", detail?.reports?.sft?.exists ? "ready" : "missing"],
        ["Eval", detail?.reports?.eval?.exists ? "ready" : "missing"],
      ],
      note: "Collects the run into human-readable artifacts so the experiment can be inspected later.",
      command: shellCommand([
        "PYTHONPATH=src", "python", "-m", "picochat.cli", "web",
        "--runs-dir", "runs",
        "--port", 8765,
      ]),
      ledger: [
        artifactItem("INPUT", "Summary JSON", detail?.artifact_inventory?.items?.find((item) => item.key === "summary_json")?.path || `${outDir}/summary.json`),
        artifactItem("INPUT", "Honesty report", detail?.reports?.honesty?.path || artifacts.honesty_report || `${outDir}/honesty/report.md`),
        artifactItem("INPUT", "Base report", detail?.reports?.base?.path || artifacts.base_report || `${outDir}/base/report.md`),
        artifactItem("INPUT", "SFT report", detail?.reports?.sft?.path || artifacts.sft_report || `${outDir}/sft/report.md`),
        artifactItem("INPUT", "Eval report", detail?.reports?.eval?.path || artifacts.eval_report || `${outDir}/eval/report.md`),
        artifactItem("OUTPUT", "Summary report", detail?.reports?.summary?.path || `${outDir}/summary.md`),
        artifactItem("OUTPUT", "Workbench", "http://127.0.0.1:8765/"),
      ],
    },
  ];
}

function artifactItem(role, label, path) {
  return { role, label, path };
}

function compactPreview(text, maxLength) {
  const normalized = String(text || "").replace(/\s+/g, " ").trim();
  if (normalized.length <= maxLength) return normalized;
  const slice = normalized.slice(0, maxLength + 1);
  const lastSpace = slice.lastIndexOf(" ");
  const end = lastSpace > maxLength * 0.6 ? lastSpace : maxLength;
  return `${normalized.slice(0, end).trim()}...`;
}

function artifactStatus(path) {
  const text = String(path || "");
  const status = state.detail?.artifact_inventory?.by_path?.[text];
  if (status) return status;
  if (text === "stdout" || text.startsWith("/api/") || text.startsWith("http://") || text.startsWith("https://")) {
    return { exists: true, kind: "virtual", size_bytes: null };
  }
  return { exists: false, kind: "unknown", size_bytes: null };
}

function artifactStatusText(status) {
  if (status.kind === "virtual") return "VIRTUAL";
  if (!status.exists) return "MISSING";
  return `${status.kind === "directory" ? "DIR" : "READY"} ${fmtBytes(status.size_bytes)}`;
}

function stageHealth(stage) {
  const ledger = stage.ledger || [];
  const materialOutputs = ledger
    .filter((item) => item.role === "OUTPUT")
    .filter((item) => artifactStatus(item.path).kind !== "virtual");
  const checkedItems = materialOutputs.length
    ? materialOutputs
    : ledger
      .filter((item) => item.role === "INPUT")
      .filter((item) => artifactStatus(item.path).kind !== "virtual");

  if (!checkedItems.length) {
    return { label: "LIVE", className: "live" };
  }

  const ready = checkedItems.filter((item) => artifactStatus(item.path).exists).length;
  if (ready === checkedItems.length) {
    return { label: "READY", className: "ready" };
  }
  if (ready === 0) {
    return { label: "MISSING", className: "missing" };
  }
  return { label: "PARTIAL", className: "partial" };
}

function materialLedgerItems(stage, role) {
  return (stage.ledger || [])
    .filter((item) => item.role === role)
    .filter((item) => artifactStatus(item.path).kind !== "virtual");
}

function doctorRows(stages) {
  return stages.map((stage) => {
    const health = stageHealth(stage);
    const inputs = materialLedgerItems(stage, "INPUT");
    const outputs = materialLedgerItems(stage, "OUTPUT");
    const missingInputs = inputs.filter((item) => !artifactStatus(item.path).exists);
    const missingOutputs = outputs.filter((item) => !artifactStatus(item.path).exists);
    let message = "Artifacts are present.";
    if (health.className === "live") {
      message = "Runtime stage. Use it when checkpoint inputs are ready.";
    } else if (missingInputs.length) {
      message = `Waiting on input: ${missingInputs.map((item) => item.label).join(", ")}.`;
    } else if (missingOutputs.length) {
      message = `Needs output: ${missingOutputs.map((item) => item.label).join(", ")}.`;
    }
    return { stage, health, missingInputs, missingOutputs, message };
  });
}

function runDoctor(stages) {
  if (!state.detail) return "RUN DOCTOR WAITING FOR A RUN.";
  const rows = doctorRows(stages);
  const ready = rows.filter((row) => row.health.className === "ready").length;
  const live = rows.filter((row) => row.health.className === "live").length;
  const attention = rows.length - ready - live;
  const next = rows.find((row) =>
    row.health.className !== "ready" &&
    row.health.className !== "live" &&
    !row.missingInputs.length
  ) || rows.find((row) =>
    row.health.className !== "ready" &&
    row.health.className !== "live"
  );
  const nextStage = next?.stage;
  return `
    <div class="doctor-head">
      <div>
        <label>RUN DOCTOR</label>
        <strong>${ready}/${rows.length} READY | ${attention} NEED ATTENTION | ${live} LIVE</strong>
      </div>
      <span>${escapeHtml(nextStage ? `NEXT ${nextStage.label}` : "RUN COMPLETE")}</span>
    </div>
    <div class="doctor-body">
      <div class="doctor-next">
        <div class="command-head">
          <label>${nextStage ? "NEXT COMMAND" : "NEXT STEP"}</label>
          ${nextStage ? copyCommandButton(nextStage.command) : ""}
        </div>
        <code>${escapeHtml(nextStage?.command || "All material stages look ready. Open Generation Deck or Report Vault.")}</code>
      </div>
      <div class="doctor-list">
        ${rows.map((row) => `
          <button class="doctor-row ${row.health.className}" type="button" data-stage="${row.stage.id}">
            <span>${escapeHtml(row.health.label)}</span>
            <strong>${escapeHtml(row.stage.label)}</strong>
            <small>${escapeHtml(row.message)}</small>
          </button>
        `).join("")}
      </div>
    </div>
  `;
}

function learningVerdict(stages) {
  if (!state.detail) return "LOAD A RUN TO SEE THE EXPERIMENT VERDICT.";
  const rows = doctorRows(stages);
  const summary = state.detail?.summary || {};
  const evalReport = state.detail?.eval_reports?.at(-1)?.report;
  const evalSummary = evalReport?.summary || summary.eval || {};
  const checks = trustChecks();
  const levelRows = evalReport ? evalLevelRows(evalReport) : [];
  const strongest = bestLevel(levelRows, "high");
  const weakest = bestLevel(levelRows, "low");
  const firstRecommendation = evalReport?.analysis?.recommendations?.[0];
  const baseStatus = summary.base?.loss_diagnostics?.status || state.detail?.base_report?.loss_diagnostics?.status || "--";
  const sftStatus = summary.sft?.loss_diagnostics?.status || state.detail?.sft_report?.loss_diagnostics?.status || "--";
  const next = rows.find((row) =>
    row.health.className !== "ready" &&
    row.health.className !== "live" &&
    !row.missingInputs.length
  ) || rows.find((row) =>
    row.health.className !== "ready" &&
    row.health.className !== "live"
  );
  const passRate = Number.isFinite(Number(evalSummary.pass_rate)) ? fmtPercent(evalSummary.pass_rate) : "--";
  const headline = passRate === "--"
    ? "This run has not reached eval yet."
    : `This run passes ${passRate} of the visible eval.`;
  const weakText = weakest
    ? `${weakest.level} is weakest at ${fmtPercent(weakest.passRate)}.`
    : "No eval ladder yet.";
  const strongText = strongest
    ? `${strongest.level} is strongest at ${fmtPercent(strongest.passRate)}.`
    : "Train and eval to reveal strengths.";
  const nextAction = firstRecommendation?.action || next?.message || "Open Inspect mode for artifacts and exact commands.";
  const nextLabel = firstRecommendation?.area
    ? String(firstRecommendation.area).toUpperCase()
    : next?.stage?.label || "INSPECT";
  return `
    <div class="verdict-card">
      <div>
        <label>EXPERIMENT VERDICT</label>
        <strong>${escapeHtml(headline)}</strong>
        <p>${escapeHtml(weakText)}</p>
      </div>
      <div>
        <label>LEARNED</label>
        <strong>${escapeHtml(strongText)}</strong>
        <p>Base ${escapeHtml(baseStatus)} | SFT ${escapeHtml(sftStatus)}</p>
      </div>
      <div>
        <label>NEXT MOVE</label>
        <strong>${escapeHtml(nextLabel)}</strong>
        <p>${escapeHtml(nextAction)}</p>
      </div>
    </div>
    <div class="decision-grid">
      ${runDecisionCards(summary, evalSummary, checks).map((card) => `
        <div class="decision-card ${escapeHtml(card.status)}">
          <label>${escapeHtml(card.label)}</label>
          <strong>${escapeHtml(card.title)}</strong>
          <p>${escapeHtml(card.message)}</p>
        </div>
      `).join("")}
    </div>
  `;
}

function runDecisionCards(summary, evalSummary, checks) {
  const hasEval = Number(evalSummary.num_examples || 0) > 0;
  const passRate = hasEval ? Number(evalSummary.pass_rate || 0) : null;
  const trustFails = checks.filter((check) => check.status === "fail");
  const trustWarns = checks.filter((check) => check.status === "warn");
  const baseStatus = summary.base?.loss_diagnostics?.status || state.detail?.base_report?.loss_diagnostics?.status || "";
  const sftStatus = summary.sft?.loss_diagnostics?.status || state.detail?.sft_report?.loss_diagnostics?.status || "";
  const baseBpb = Number(summary.base?.val_bpb);
  const sftBpb = Number(summary.sft?.val_bpb);
  const learningImproved = Number.isFinite(baseBpb) && Number.isFinite(sftBpb) && sftBpb <= baseBpb;
  const trustCard = trustFails.length
    ? {
        label: "TRUST GATE",
        status: "fail",
        title: "Do not trust yet",
        message: `Fix ${trustFails[0].label.toLowerCase()} before treating this score as evidence.`,
      }
    : trustWarns.length
      ? {
          label: "TRUST GATE",
          status: "warn",
          title: "Usable with caution",
          message: `Review ${trustWarns[0].label.toLowerCase()} before scaling the experiment.`,
        }
      : {
          label: "TRUST GATE",
          status: "pass",
          title: "No obvious cheating signal",
          message: "Leakage, memorization, support, and echo checks are clean enough for this run.",
        };
  const learningCard = !hasEval
    ? {
        label: "LEARNING GATE",
        status: "warn",
        title: "Eval missing",
        message: "Loss can look healthy while behavior is wrong. Run eval before judging the model.",
      }
    : passRate >= 0.7
      ? {
          label: "LEARNING GATE",
          status: "pass",
          title: `${fmtPercent(passRate)} visible eval pass`,
          message: learningImproved ? "Behavior and SFT loss both improved; inspect weakest categories next." : "Eval improved, but inspect loss diagnostics before scaling.",
        }
      : passRate >= 0.35
        ? {
            label: "LEARNING GATE",
            status: "warn",
            title: `${fmtPercent(passRate)} visible eval pass`,
            message: "The model is learning something. Add targeted SFT rows for the weakest eval groups.",
          }
        : {
            label: "LEARNING GATE",
            status: "fail",
            title: `${fmtPercent(passRate)} visible eval pass`,
            message: "Treat this as a diagnostic run. Improve SFT/eval data before longer training.",
          };
  const scaleCard = !hasEval
    ? {
        label: "SCALE GATE",
        status: "warn",
        title: "Not ready to scale",
        message: "Complete eval and compare against a baseline before spending more compute.",
      }
    : trustFails.length
      ? {
          label: "SCALE GATE",
          status: "fail",
          title: "Blocked by trust",
          message: "Longer training can make a bad signal look stronger. Fix trust failures first.",
        }
      : passRate >= 0.7 && !trustWarns.length
        ? {
            label: "SCALE GATE",
            status: "pass",
            title: "Scale candidate",
            message: "Run a harder eval or larger preset, then compare against this checkpoint.",
          }
        : {
            label: "SCALE GATE",
            status: "warn",
            title: "Improve before scaling",
            message: `${baseStatus || "base"} / ${sftStatus || "SFT"} diagnostics are not enough; use failures to edit data first.`,
          };
  return [trustCard, learningCard, scaleCard];
}

function runStoryTimeline(stages) {
  if (!state.detail) return "LOAD A RUN TO SEE THE TRAINING STORY.";
  return `
    <div class="story-head">
      <div>
        <label>RUN STORY TIMELINE</label>
        <strong>${escapeHtml(state.selectedRun || "current run")}</strong>
      </div>
      <span>${escapeHtml(currentRunOneLineVerdict())}</span>
    </div>
    <div class="story-grid">
      ${stages.map((stage, index) => {
        const health = stageHealth(stage);
        return `
          <button class="story-step ${health.className} ${stage.id === state.activeStage ? "active" : ""}" type="button" data-stage="${stage.id}">
            <span>${String(index + 1).padStart(2, "0")}</span>
            <strong>${escapeHtml(stage.label)}</strong>
            <em>${escapeHtml(stageLearningSignal(stage))}</em>
            <p>${escapeHtml(STAGE_LESSONS[stage.id] || stage.note || "")}</p>
          </button>
        `;
      }).join("")}
    </div>
  `;
}

function currentRunOneLineVerdict() {
  const summary = state.detail?.summary || {};
  const evalSummary = state.detail?.eval_reports?.at(-1)?.report?.summary || summary.eval;
  if (!evalSummary) return "Train, tune, then eval to get a verdict.";
  const passRate = Number(evalSummary.pass_rate || 0);
  if (passRate >= 0.7) return "Promising run; inspect weak levels before scaling.";
  if (passRate >= 0.35) return "Learning signal exists; target the weakest eval group.";
  return "Early run; use failures to build better SFT/eval data.";
}

function stageLearningSignal(stage) {
  const summary = state.detail?.summary || {};
  const corpus = summary.corpus || {};
  const config = summary.config || {};
  const evalSummary = state.detail?.eval_reports?.at(-1)?.report?.summary || summary.eval;
  if (stage.id === "dataset") {
    return `${fmtInt(corpus.num_documents)} docs / dup ${fmtPercent(corpus.duplicate_line_rate || 0)}`;
  }
  if (stage.id === "tokenizer") {
    return `${config.tokenizer_type || summary.tokenizer?.tokenizer_type || "--"} / vocab ${summary.tokenizer?.vocab_size ?? "--"}`;
  }
  if (stage.id === "base") {
    const loss = summary.base?.val_bpb ?? state.detail?.base_report?.losses?.at(-1)?.val_bpb;
    return `base BPB ${fmtLoss(loss)}`;
  }
  if (stage.id === "sft") {
    const loss = summary.sft?.val_bpb ?? state.detail?.sft_report?.losses?.at(-1)?.val_bpb;
    const truncated = summary.sft?.truncated_examples ?? state.detail?.sft_report?.dataset?.truncated_examples;
    const skipped = summary.sft?.skipped_long_examples ?? state.detail?.sft_report?.dataset?.skipped_long_examples;
    return `SFT BPB ${fmtLoss(loss)} / trunc ${truncated ?? "--"} / skip ${skipped ?? "--"}`;
  }
  if (stage.id === "eval") {
    return evalSummary ? `${evalSummary.num_passed}/${evalSummary.num_examples} pass` : "not evaluated";
  }
  if (stage.id === "chat") {
    return `seed ${padSeed(config.seed)} / ctx ${config.context_size ?? "--"}`;
  }
  return `${Object.values(state.detail?.reports || {}).filter((report) => report.exists).length}/5 reports`;
}

function runTrustPanel() {
  if (!state.detail) return "LOAD A RUN TO SEE TRUST CHECKS.";
  const checks = trustChecks();
  return `
    <div class="trust-head">
      <div>
        <label>TRUST PANEL</label>
        <strong>${escapeHtml(trustVerdict(checks))}</strong>
      </div>
      <span>${checks.filter((check) => check.status === "pass").length}/${checks.length} CLEAN</span>
    </div>
    <div class="trust-grid">
      ${checks.map((check) => `
        <div class="trust-check ${check.status}">
          <span>${escapeHtml(check.status.toUpperCase())}</span>
          <strong>${escapeHtml(check.label)}</strong>
          <p>${escapeHtml(check.value)}</p>
        </div>
      `).join("")}
    </div>
  `;
}

function trustChecks() {
  const summary = state.detail?.summary || {};
  const corpus = summary.corpus || {};
  const honesty = summary.honesty || {};
  const evalReport = state.detail?.eval_reports?.at(-1)?.report;
  const evalSummary = evalReport?.summary || summary.eval || {};
  const baseMem = summary.base?.memorization || state.detail?.base_report?.memorization || {};
  const sft = summary.sft || {};
  const duplicateDocs = Number(corpus.duplicate_document_rate || 0);
  const duplicateLines = Number(corpus.duplicate_line_rate || 0);
  const unsupportedRate = Number(evalSummary.unsupported_claim_rate || 0);
  const missingSupportRate = Number(evalSummary.missing_support_rate || 0);
  const promptEchoRate = Number(evalSummary.prompt_echo_rate || 0);
  const truncation = Number(sft.truncated_examples || 0);
  const skippedLong = Number(sft.skipped_long_examples || 0);
  const domainPassRate = optionalNumber(evalSummary.domain_pass_rate);
  const refusalPassRate = optionalNumber(evalSummary.refusal_pass_rate);
  const leakageSummary = honesty.summary || "No honesty report summary found.";
  const leakageClean = /no obvious eval leakage/i.test(leakageSummary) || honesty.status === "ready";
  return [
    {
      label: "Eval leakage",
      status: leakageClean ? "pass" : "warn",
      value: leakageSummary,
    },
    {
      label: "Duplicate data",
      status: duplicateDocs <= 0.01 && duplicateLines <= 0.02 ? "pass" : duplicateDocs <= 0.05 && duplicateLines <= 0.08 ? "warn" : "fail",
      value: `docs ${fmtPercent(duplicateDocs)} / lines ${fmtPercent(duplicateLines)}`,
    },
    {
      label: "Memorization",
      status: String(baseMem.status || "low") === "low" ? "pass" : String(baseMem.status || "").includes("high") ? "fail" : "warn",
      value: baseMem.summary || "No memorization probe summary found.",
    },
    {
      label: "Domain answers",
      status: domainPassRate == null ? "warn" : domainPassRate >= 0.5 ? "pass" : domainPassRate >= 0.25 ? "warn" : "fail",
      value: domainPassRate == null ? "No domain-answer gate found in this eval." : `${fmtPercent(domainPassRate)} domain-answer pass rate.`,
    },
    {
      label: "Refusal boundary",
      status: refusalPassRate == null ? "warn" : refusalPassRate >= 0.8 ? "pass" : refusalPassRate >= 0.5 ? "warn" : "fail",
      value: refusalPassRate == null ? "No refusal/boundary gate found in this eval." : `${fmtPercent(refusalPassRate)} refusal/boundary pass rate.`,
    },
    {
      label: "Unsupported claims",
      status: unsupportedRate <= 0.05 ? "pass" : unsupportedRate <= 0.15 ? "warn" : "fail",
      value: `${fmtPercent(unsupportedRate)} of eval replies triggered unsupported-claim checks.`,
    },
    {
      label: "Missing support",
      status: missingSupportRate <= 0.1 ? "pass" : missingSupportRate <= 0.25 ? "warn" : "fail",
      value: `${fmtPercent(missingSupportRate)} missing required support phrases/entities.`,
    },
    {
      label: "Prompt echo",
      status: promptEchoRate <= 0.03 ? "pass" : promptEchoRate <= 0.1 ? "warn" : "fail",
      value: `${fmtPercent(promptEchoRate)} prompt echo rate.`,
    },
    {
      label: "SFT truncation",
      status: truncation === 0 ? "pass" : truncation <= 3 ? "warn" : "fail",
      value: `${fmtInt(truncation)} examples truncated during SFT.`,
    },
    {
      label: "SFT too long",
      status: skippedLong === 0 ? "pass" : skippedLong <= 5 ? "warn" : "fail",
      value: `${fmtInt(skippedLong)} examples skipped because they exceeded context.`,
    },
  ];
}

function trustVerdict(checks) {
  const fail = checks.filter((check) => check.status === "fail").length;
  const warn = checks.filter((check) => check.status === "warn").length;
  if (fail) return "Do not trust this run yet; inspect failed trust checks.";
  if (warn) return "Mostly usable, but verify warnings before making claims.";
  return "No obvious cheating signals in available artifacts.";
}

function bestLevel(rows, direction) {
  const candidates = rows.filter((row) => row.numExamples > 0);
  if (!candidates.length) return null;
  const sorted = [...candidates].sort((left, right) => left.passRate - right.passRate);
  return direction === "high" ? sorted.at(-1) : sorted[0];
}

function datasetCommand(config, artifacts) {
  const output = artifacts.corpus || `${config.out_dir || "runs/manual"}/corpus.txt`;
  if (config.corpus_recipe) {
    return shellCommand([
      "PYTHONPATH=src", "python", "-m", "picochat.cli", "data", "build",
      "--recipe", config.corpus_recipe,
      "--out", output,
    ]);
  }
  return shellCommand([
    "PYTHONPATH=src", "python", "-m", "picochat.cli", "data", "build",
    "--input", config.corpus_input || "examples/tiny_corpus.txt",
    "--out", output,
  ]);
}

function stageDetail(stage) {
  return `
    <div><strong>${escapeHtml(stage.label)}</strong> / ${escapeHtml(stage.note)}</div>
    <div class="pipeline-detail-grid">
      ${stage.stats.map(([label, value]) => `
        <div class="pipeline-stat">
          <label>${escapeHtml(label)}</label>
          <span>${escapeHtml(value)}</span>
        </div>
      `).join("")}
    </div>
    <div class="artifact-ledger">
      <label>ARTIFACT LEDGER</label>
      ${(stage.ledger || []).map((item) => `
        <div class="artifact-row ${item.role.toLowerCase()} ${artifactStatus(item.path).exists ? "ready" : "missing"}">
          <span>${escapeHtml(item.role)}</span>
          <strong>${escapeHtml(item.label)}</strong>
          <code>${escapeHtml(item.path)}</code>
          <em>${escapeHtml(artifactStatusText(artifactStatus(item.path)))}</em>
        </div>
      `).join("")}
    </div>
    <div class="command-tape">
      <div class="command-head">
        <label>COMMAND TAPE</label>
        ${copyCommandButton(stage.command)}
      </div>
      <code>${escapeHtml(stage.command || "NO COMMAND AVAILABLE.")}</code>
    </div>
  `;
}

function copyCommandButton(command) {
  if (!command) return "";
  return `<button class="copy-command" type="button" data-copy-command="${escapeHtml(command)}">COPY</button>`;
}

async function copyCommand(command, button) {
  if (!command) {
    flashStatus("COPY FAULT. | no command found");
    return;
  }
  const previous = button.textContent;
  button.textContent = "COPYING";
  button.disabled = true;
  try {
    await writeClipboard(command);
    button.textContent = "COPIED";
    flashStatus("COPIED COMMAND. | Paste it in your terminal from the repo root.");
  } catch (error) {
    button.textContent = "FAILED";
    flashStatus(`COPY FAULT. | ${error.message}`);
  }
  window.setTimeout(() => {
    button.textContent = previous;
    button.disabled = false;
  }, 1200);
}

async function writeClipboard(text) {
  let clipboardError = null;
  if (navigator.clipboard?.writeText) {
    try {
      await withTimeout(navigator.clipboard.writeText(text), 700);
      return;
    } catch (error) {
      clipboardError = error;
    }
  }
  if (fallbackCopy(text)) return;
  throw clipboardError || new Error("clipboard unavailable");
}

function withTimeout(promise, timeoutMs) {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => reject(new Error("clipboard timeout")), timeoutMs);
    promise.then(
      (value) => {
        window.clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        window.clearTimeout(timer);
        reject(error);
      },
    );
  });
}

function fallbackCopy(text) {
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  document.body.removeChild(textarea);
  return copied;
}

function renderDataset() {
  const summary = state.detail?.summary || {};
  const corpus = summary.corpus || {};
  const config = summary.config || {};
  const baseDataset = state.detail?.base_report?.dataset || {};
  const manifest = state.detail?.corpus_manifest;
  seedPackBuilderInputs(config);
  seedDatasetFlightInputs(config);
  seedTuningInspectorInputs(config);
  seedPackEditorInputs(config);
  seedRunLauncherInputs(config);
  seedSourcePreviewInputs(config);
  $("dataset-summary").textContent = corpus.num_characters
    ? `${fmtInt(corpus.num_characters)} CHARS | ${fmtInt(corpus.num_lines)} LINES`
    : "DATA --";
  $("dataset-stats").innerHTML = statCards([
    ["Input", config.corpus_input || "unknown"],
    ["Built corpus", summary.artifacts?.corpus || "unknown"],
    ["Files", fmtInt(corpus.num_files)],
    ["Documents", fmtInt(corpus.num_documents)],
    ["Characters", fmtInt(corpus.num_characters)],
    ["Lines", fmtInt(corpus.num_lines)],
    ["Avg doc chars", fmtLoss(corpus.average_document_chars || 0)],
    ["Context", config.context_size ?? "--"],
  ]);
  $("dataset-warnings").innerHTML = qualityChecks(corpus);
  $("dataset-windows").innerHTML = statCards([
    ["Token count", fmtInt(baseDataset.num_tokens)],
    ["Context size", baseDataset.context_size ?? config.context_size ?? "--"],
    ["Windows", fmtInt(baseDataset.num_sequences)],
    ["Train windows", fmtInt(baseDataset.train_sequences)],
    ["Val windows", fmtInt(baseDataset.val_sequences)],
    ["Split mode", baseDataset.split_mode || "--"],
    ["Held-out docs", baseDataset.val_documents == null ? "--" : `${baseDataset.val_documents}/${baseDataset.num_documents}`],
    ["Val fraction", baseDataset.num_sequences ? fmtPercent((baseDataset.val_sequences || 0) / baseDataset.num_sequences) : "--"],
  ]);
  $("corpus-files").innerHTML = renderCorpusFiles(manifest?.files || []);
  renderDatasetFlightPlan(state.datasetFlightPlan);
  renderHfImport(state.hfImport);
  renderSftStarter(state.sftStarter);
  renderEvalStarter(state.evalStarter);
  renderCorpusSourcePreview(state.corpusSourcePreview);
  renderDatasetPackInit(state.datasetPackInit);
  renderTuningInspection(state.tuningInspection);
  renderPackEditor(state.packEditor);
  renderRunJob(state.runJob);
  renderRunJobList();
  $("corpus-preview").textContent = state.detail?.corpus_preview || "NO CORPUS PREVIEW ARTIFACT FOUND.";
}

function seedHfOutDirFromDataset() {
  const outInput = $("hf-out-dir");
  if (outInput.value.trim()) return;
  const dataset = normalizeHfInput($("hf-dataset-input").value);
  if (!dataset) return;
  outInput.value = `runs/hf-${slugify(dataset)}-${$("hf-max-rows").value || 1000}`;
}

async function useSampleDataset() {
  $("sample-dataset-button").disabled = true;
  $("flight-status").innerHTML = 'CHECKING SAMPLE DATASET<span class="cursor"></span>';
  state.hfImport = null;
  state.datasetPackInit = null;
  state.sftStarter = null;
  state.evalStarter = null;
  applyDatasetPackToWorkflow({
    datasetPack: SAMPLE_DATASET_PACK,
    chatInput: SAMPLE_CHAT_INPUT,
    evalInput: SAMPLE_EVAL_INPUT,
    runNameSeed: "sample-smoke",
  });
  renderHfImport(null);
  renderSftStarter(null);
  renderEvalStarter(null);
  try {
    await refreshDatasetFlightPlanAfterChange();
    applyFlightPlanToLauncher(true);
    flashStatus("SAMPLE DATASET READY. | Run smoke train when you are ready.");
  } finally {
    $("sample-dataset-button").disabled = false;
  }
}

function applyDatasetPackToWorkflow({ datasetPack, chatInput = "", evalInput = "", runNameSeed = "" }) {
  if (!datasetPack) return;
  $("flight-pack-path").value = datasetPack;
  $("flight-input-path").value = "";
  $("flight-chat-path").value = chatInput || "";
  $("flight-sft-out-path").value = suggestedSftStarterPath(chatInput || datasetPack);
  $("flight-eval-path").value = evalInput || "";
  $("flight-eval-out-path").value = suggestedEvalStarterPath(evalInput || datasetPack);

  $("preview-pack-path").value = datasetPack;
  $("preview-recipe-path").value = "";
  $("preview-input-path").value = "";
  $("preview-chat-path").value = chatInput || "";
  $("preview-eval-path").value = evalInput || "";

  $("tuning-pack-path").value = datasetPack;
  $("tuning-chat-path").value = "";
  $("tuning-eval-path").value = "";

  $("editor-pack-path").value = datasetPack;
  $("editor-chat-path").value = "";
  $("editor-eval-path").value = "";

  $("launch-pack-path").value = datasetPack;
  $("launch-run-name").value = uniqueRunName(runNameSeed || suggestedRunName(datasetPack));
  renderLaunchReadiness();
}

function syncFlightStarterDefaults() {
  const packPath = $("flight-pack-path").value.trim();
  const inputPath = $("flight-input-path").value.trim();
  const chatPath = $("flight-chat-path").value.trim();
  const evalPath = $("flight-eval-path").value.trim();
  const sourcePath = packPath || inputPath;
  if (sourcePath && !$("flight-sft-out-path").value.trim()) {
    $("flight-sft-out-path").value = suggestedSftStarterPath(chatPath || sourcePath);
  }
  if (sourcePath && !$("flight-eval-out-path").value.trim()) {
    $("flight-eval-out-path").value = suggestedEvalStarterPath(evalPath || sourcePath);
  }
  if (packPath) {
    if (!$("preview-pack-path").value.trim()) $("preview-pack-path").value = packPath;
    if (!$("tuning-pack-path").value.trim()) $("tuning-pack-path").value = packPath;
    if (!$("editor-pack-path").value.trim()) $("editor-pack-path").value = packPath;
    if (!$("launch-pack-path").value.trim()) $("launch-pack-path").value = packPath;
    if (!$("launch-run-name").value.trim()) $("launch-run-name").value = suggestedRunName(packPath);
    renderLaunchReadiness();
  } else if (inputPath && !$("preview-input-path").value.trim()) {
    $("preview-input-path").value = inputPath;
  }
}

function prepareDatasetSourceMode(mode) {
  const messages = {
    hf: "SOURCE MODE: HUGGING FACE. | Paste a dataset repo or URL, then import.",
    local: "SOURCE MODE: LOCAL DOCS. | Enter a folder or file path, then check readiness.",
    pack: "SOURCE MODE: DATASET PACK. | Enter dataset_pack.json, then check readiness.",
  };
  if (mode === "local") {
    $("flight-pack-path").value = "";
  } else if (mode === "pack") {
    $("flight-input-path").value = "";
  }
  if (messages[mode]) {
    $("flight-status").textContent = messages[mode];
    flashStatus(messages[mode]);
  }
  syncFlightStarterDefaults();
  renderStartHere();
}

async function importHfDataset() {
  const dataset = $("hf-dataset-input").value.trim();
  const outDir = $("hf-out-dir").value.trim();
  const configName = $("hf-config-name").value.trim();
  const split = $("hf-split").value.trim() || "train";
  const textColumn = $("hf-text-column").value.trim() || "text";
  const maxRows = Number($("hf-max-rows").value || 1000);
  const shards = Number($("hf-shards")?.value || 1);
  const minChars = Number($("hf-min-chars").value || 20);
  const force = $("hf-force").checked;
  if (!dataset) throw new Error("enter a Hugging Face dataset URL or id");

  $("hf-import-button").disabled = true;
  $("hf-import-status").innerHTML = 'IMPORTING HF DATASET<span class="cursor"></span>';
  $("hf-import-result").innerHTML = "";
  if ($("flight-coach")) {
    $("flight-coach").className = "flight-coach caution";
    $("flight-coach").innerHTML = `
      <div>
        <label>WHAT JUST HAPPENED</label>
        <strong>Hugging Face import is running.</strong>
        <p>Picochat is copying dataset rows into local documents and making a dataset pack.</p>
      </div>
      <div>
        <label>NEXT CLICK</label>
        <strong>Wait for import, then read the check.</strong>
        <p>The imported pack is automatically sent through the same readiness report.</p>
      </div>
      <div>
        <label>WHY IT MATTERS</label>
        <strong>HF data becomes local corpus files.</strong>
        <p>Training reads local text artifacts, not the website directly.</p>
      </div>
    `;
  }
  try {
    const report = await postJson("/api/hf/import", {
      dataset_url: dataset,
      out_dir: outDir || null,
      config_name: configName || null,
      split,
      text_column: textColumn,
      max_rows: maxRows,
      shards,
      min_chars: minChars,
      streaming: true,
      force,
    });
    state.hfImport = report;
    state.datasetFlightPlan = report.preview;
    state.corpusSourcePreview = report.preview;
    state.tuningInspection = tuningInspectionFromPreview(report.preview);
    applyDatasetPackToWorkflow({
      datasetPack: report.dataset_pack || "",
      chatInput: report.chat_input || "",
      evalInput: report.eval_input || "",
      runNameSeed: suggestedRunName(report.dataset_pack || report.dataset),
    });
    renderHfImport(report);
    renderDatasetFlightPlan(report.preview);
    renderCorpusSourcePreview(report.preview);
    renderTuningInspection(state.tuningInspection);
    renderScalePlan();
    renderStartHere();
  } finally {
    $("hf-import-button").disabled = false;
  }
}

function renderHfImport(report) {
  if (!report) {
    $("hf-import-status").textContent = "NO HUGGING FACE DATASET IMPORTED.";
    $("hf-import-result").innerHTML = "";
    return;
  }
  $("hf-import-status").textContent =
    `HF IMPORT READY | ${escapeHtml(report.dataset)} | ${fmtInt(report.rows_written)} ROWS | ${fmtInt(report.characters_written)} CHARS`;
  $("hf-import-result").innerHTML = `
    <div class="hf-import-grid">
      <div><strong>${escapeHtml(report.dataset)}</strong><span>dataset</span></div>
      <div><strong>${fmtInt(report.rows_written)}</strong><span>rows written</span></div>
      <div><strong>${escapeHtml(shortPath(report.documents_dir))}</strong><span>documents</span></div>
      <div><strong>${escapeHtml(shortPath(report.dataset_pack))}</strong><span>dataset pack</span></div>
    </div>
    <div class="command-tape source-command">
      <div class="command-head">
        <label>HF IMPORT COMMAND</label>
        ${copyCommandButton(report.command)}
      </div>
      <code>${escapeHtml(report.command || "")}</code>
      ${(report.next_actions || []).map((action) => `<p>${escapeHtml(action)}</p>`).join("")}
    </div>
  `;
}

function renderHfImportError(error) {
  $("hf-import-button").disabled = false;
  const availableSplits = error.payload?.available_splits || splitListFromMessage(error.message);
  const isSplitIssue = availableSplits.length > 0;
  $("hf-import-status").textContent = isSplitIssue ? "HF IMPORT NEEDS SPLIT" : "HF IMPORT FAULT";
  const requestedSplit = error.payload?.requested_split || $("hf-split").value.trim() || "train";
  const splitHint = availableSplits.length
    ? hfSplitHint(availableSplits, requestedSplit)
    : "";
  const dependencyHint = hfDependencyHint(error);
  $("hf-import-result").innerHTML = `
    <div class="notice">${isSplitIssue ? "SPLIT ISSUE" : "FAULT"}: ${escapeHtml(error.message)}</div>
    ${splitHint}
    ${dependencyHint}
  `;
}

function hfDependencyHint(error) {
  const message = String(error.message || "");
  if (error.payload?.error_type !== "HFDatasetsMissingError" && !message.includes(".[hf]")) {
    return "";
  }
  return '<p class="helper-copy">Install Picochat inside the venv with <code>pip install -e ".[hf]"</code>.</p>';
}

function splitListFromMessage(message) {
  const match = String(message || "").match(/Available splits:\s*(\[[^\]]*\])/);
  if (!match) return [];
  return Array.from(match[1].matchAll(/'([^']+)'|"([^"]+)"/g), (item) => item[1] || item[2]);
}

function hfSplitHint(availableSplits, requestedSplit) {
  const splitText = availableSplits.join(", ");
  if (availableSplits.length === 1) {
    $("hf-split").value = availableSplits[0];
    return `
      <div class="starter-handoff caution">
        <strong>SPLIT UPDATED TO ${escapeHtml(availableSplits[0].toUpperCase())}</strong>
        <span>This dataset does not have ${escapeHtml(requestedSplit)}. Picochat filled the only available split; click import again if you mean to use it.</span>
        <em>If the split is named test, treat it carefully. It may be benchmark/eval data, not normal pretraining data.</em>
      </div>
    `;
  }
  return `
    <div class="starter-handoff caution">
      <strong>CHOOSE A VALID SPLIT</strong>
      <span>This dataset does not have ${escapeHtml(requestedSplit)}. Available splits: ${escapeHtml(splitText)}.</span>
      <em>Set SPLIT to one of those values, then import again.</em>
    </div>
  `;
}

function normalizeHfInput(value) {
  const text = String(value || "").trim();
  if (!text) return "";
  try {
    if (text.includes("://")) {
      const url = new URL(text);
      const parts = url.pathname.split("/").filter(Boolean);
      const start = parts[0] === "datasets" ? 1 : 0;
      return parts.slice(start, start + 2).join("/");
    }
  } catch {
    return text;
  }
  return text.replace(/^datasets\//, "").replace(/^\/+|\/+$/g, "");
}

function seedPackBuilderInputs(config) {
  const nameInput = $("pack-name");
  const corpusInput = $("pack-corpus-path");
  const outInput = $("pack-out-dir");
  const descriptionInput = $("pack-description");
  if (nameInput.value || corpusInput.value || outInput.value || descriptionInput.value) return;
  nameInput.value = "my-domain-pack";
  corpusInput.value = config.corpus_input || config.corpus_recipe || "my_docs/";
  outInput.value = "my_pack/";
  descriptionInput.value = "Starter Picochat dataset pack.";
}

function seedDatasetFlightInputs(config) {
  const packInput = $("flight-pack-path");
  const sourceInput = $("flight-input-path");
  const chatInput = $("flight-chat-path");
  const sftOutInput = $("flight-sft-out-path");
  const evalInput = $("flight-eval-path");
  const evalOutInput = $("flight-eval-out-path");
  const minScoreInput = $("flight-min-score");
  if (packInput.value || sourceInput.value || chatInput.value || sftOutInput.value || evalInput.value || evalOutInput.value) return;
  packInput.value = config.dataset_pack || "";
  sourceInput.value = config.dataset_pack ? "" : config.corpus_input || "";
  chatInput.value = config.chat_input || "";
  sftOutInput.value = suggestedSftStarterPath(config.chat_input || config.dataset_pack || "my_pack/chat.jsonl");
  evalInput.value = config.eval_input || "";
  evalOutInput.value = suggestedEvalStarterPath(config.eval_input || config.dataset_pack || "my_pack/eval.jsonl");
  minScoreInput.value = config.min_quality_score || 0;
}

function seedTuningInspectorInputs(config) {
  const packInput = $("tuning-pack-path");
  const chatInput = $("tuning-chat-path");
  const evalInput = $("tuning-eval-path");
  if (packInput.value || chatInput.value || evalInput.value) return;
  packInput.value = config.dataset_pack || "";
  chatInput.value = config.chat_input || "examples/tiny_chat.jsonl";
  evalInput.value = config.eval_input || "examples/tiny_eval.jsonl";
}

function seedPackEditorInputs(config) {
  const packInput = $("editor-pack-path");
  const chatInput = $("editor-chat-path");
  const evalInput = $("editor-eval-path");
  if (packInput.value || chatInput.value || evalInput.value) return;
  packInput.value = config.dataset_pack || "";
  chatInput.value = config.chat_input || "examples/tiny_chat.jsonl";
  evalInput.value = config.eval_input || "examples/tiny_eval.jsonl";
}

async function loadRunPresets() {
  const payload = await fetchJson("/api/run/presets");
  state.runPresets = payload.presets || {};
  renderRunPresetOptions();
  applyLaunchPreset(true);
}

function renderRunPresetOptions() {
  const select = $("launch-preset");
  const selected = select.value || "smoke";
  const presets = Object.entries(state.runPresets);
  if (!presets.length) return;
  select.innerHTML = presets.map(([key, preset]) => `
    <option value="${escapeHtml(key)}">${escapeHtml(String(preset.label || key).toUpperCase())}</option>
  `).join("");
  select.value = state.runPresets[selected] ? selected : "smoke";
}

function seedRunLauncherInputs(config) {
  const packInput = $("launch-pack-path");
  const runNameInput = $("launch-run-name");
  const minScoreInput = $("launch-min-score");
  if (!packInput.value) packInput.value = config.dataset_pack || "";
  if (!runNameInput.value) runNameInput.value = suggestedRunName(packInput.value || "picochat");
  else if (runNameExists(runNameInput.value)) runNameInput.value = uniqueRunName(runNameInput.value);
  if (config.min_quality_score && (!minScoreInput.value || minScoreInput.value === "0")) {
    minScoreInput.value = config.min_quality_score;
  }
  renderLaunchReadiness();
}

function seedSourcePreviewInputs(config) {
  const packInput = $("preview-pack-path");
  const recipeInput = $("preview-recipe-path");
  const sourceInput = $("preview-input-path");
  const chatInput = $("preview-chat-path");
  const evalInput = $("preview-eval-path");
  if (packInput.value || recipeInput.value || sourceInput.value || chatInput.value || evalInput.value) return;
  packInput.value = config.dataset_pack || "";
  recipeInput.value = config.corpus_recipe || "examples/corpus_recipe.json";
  sourceInput.value = config.corpus_input || "";
  chatInput.value = config.chat_input || "examples/tiny_chat.jsonl";
  evalInput.value = config.eval_input || "examples/tiny_eval.jsonl";
}

async function initDatasetPack() {
  const name = $("pack-name").value.trim() || "picochat-pack";
  const corpusPath = $("pack-corpus-path").value.trim();
  const outDir = $("pack-out-dir").value.trim();
  const description = $("pack-description").value.trim() || "Starter Picochat dataset pack.";
  const force = $("pack-force").checked;
  if (!corpusPath || !outDir) {
    throw new Error("enter corpus path and output folder");
  }

  $("init-pack-button").disabled = true;
  $("pack-builder-status").innerHTML = 'BUILDING PACK FILES<span class="cursor"></span>';
  $("pack-builder-result").innerHTML = "";
  try {
    const report = await postJson("/api/dataset-pack/init", {
      name,
      description,
      corpus_path: corpusPath,
      out_dir: outDir,
      force,
    });
    state.datasetPackInit = report;
    applyDatasetPackToWorkflow({
      datasetPack: report.dataset_pack || "",
      chatInput: report.chat_input || "",
      evalInput: report.eval_input || "",
      runNameSeed: suggestedRunName(report.dataset_pack || "picochat"),
    });
    renderDatasetPackInit(report);
    inspectTuningData().catch((error) => renderTuningInspectionError(error));
    loadPackEditor().catch((error) => renderPackEditorError(error));
  } finally {
    $("init-pack-button").disabled = false;
  }
}

function renderDatasetPackInit(report) {
  if (!report) {
    $("pack-builder-status").textContent = "NO PACK INIT REQUESTED.";
    $("pack-builder-result").innerHTML = "";
    return;
  }
  const created = report.created || [];
  const overwritten = report.overwritten || [];
  const files = [
    ["Dataset pack", report.dataset_pack],
    ["Corpus recipe", report.corpus_recipe],
    ["Chat SFT JSONL", report.chat_input],
    ["Eval JSONL", report.eval_input],
  ];
  $("pack-builder-status").textContent =
    `PACK READY | ${fmtInt(created.length)} CREATED | ${fmtInt(overwritten.length)} OVERWRITTEN`;
  $("pack-builder-result").innerHTML = `
    <div class="command-head">
      <label>PACK FILES</label>
      ${report.preview_command ? copyCommandButton(report.preview_command) : ""}
    </div>
    <div class="pack-file-list">
      ${files.map(([label, path]) => `
        <div>
          <strong>${escapeHtml(label)}</strong>
          <code>${escapeHtml(path || "--")}</code>
        </div>
      `).join("")}
    </div>
    <label>NEXT COMMAND</label>
    <code>${escapeHtml(report.preview_command || "NO PREVIEW COMMAND AVAILABLE.")}</code>
    <p>Edit the starter chat and eval rows, then preview the pack before training.</p>
  `;
}

function renderDatasetPackInitError(error) {
  $("init-pack-button").disabled = false;
  $("pack-builder-status").textContent = "PACK BUILDER FAULT";
  $("pack-builder-result").innerHTML = `
    <label>ERROR</label>
    <code>FAULT: ${escapeHtml(error.message)}</code>
    <p>Use FORCE only when you mean to overwrite the four starter pack files.</p>
  `;
}

async function inspectTuningData() {
  const packPath = $("tuning-pack-path").value.trim();
  const chatInput = $("tuning-chat-path").value.trim();
  const evalInput = $("tuning-eval-path").value.trim();
  $("inspect-tuning-button").disabled = true;
  $("tuning-inspector-status").innerHTML = 'INSPECTING TUNING DATA<span class="cursor"></span>';
  $("tuning-inspector-actions").innerHTML = "";
  $("tuning-inspector-result").innerHTML = "";
  $("tuning-inspector-command").innerHTML = "";
  try {
    const report = await postJson("/api/tuning/inspect", {
      dataset_pack: packPath || null,
      chat_input: packPath ? null : chatInput || null,
      eval_input: packPath ? null : evalInput || null,
    });
    state.tuningInspection = report;
    renderTuningInspection(report);
    renderStartHere();
  } finally {
    $("inspect-tuning-button").disabled = false;
  }
}

function renderTuningInspection(report) {
  if (!report) {
    $("tuning-inspector-status").textContent = "NO TUNING INSPECTION REQUESTED.";
    $("tuning-inspector-actions").innerHTML = "";
    $("tuning-inspector-result").innerHTML = "";
    $("tuning-inspector-command").innerHTML = "";
    return;
  }
  const status = String(report.status || "unknown").toUpperCase();
  $("tuning-inspector-status").textContent =
    `TUNING ${status} | CHAT ${escapeHtml(report.chat_data?.status || "--")} | EVAL ${escapeHtml(report.eval_data?.status || "--")}`;
  $("tuning-inspector-actions").innerHTML = renderTuningActions(report);
  $("tuning-inspector-result").innerHTML = `${renderTuningPaths(report)}${renderTuningPreflight(report.chat_data, report.eval_data)}`;
  $("tuning-inspector-command").innerHTML = report.preview_command ? `
    <div class="command-head">
      <label>NEXT PREVIEW COMMAND</label>
      ${copyCommandButton(report.preview_command)}
    </div>
    <div class="command-meta">
      <span>PACK ${escapeHtml(report.dataset_pack || "--")}</span>
      <span>CHAT ${escapeHtml(report.chat_input || "--")}</span>
      <span>EVAL ${escapeHtml(report.eval_input || "--")}</span>
    </div>
    <code>${escapeHtml(report.preview_command)}</code>
  ` : "";
}

function renderTuningActions(report) {
  const status = escapeHtml(report.status || "unknown");
  const rowStatus = report.status === "ready" ? "pass" : report.status === "caution" ? "warn" : "fail";
  const actions = report.next_actions || [];
  return `
    <div class="readiness-summary ${status}">
      <strong>TUNING ${escapeHtml(String(report.status || "--").toUpperCase())}</strong>
      <span>${escapeHtml(report.summary || "")}</span>
    </div>
    ${actions.map((action, index) => `
      <div class="readiness-row ${rowStatus}">
        <strong>ACTION ${index + 1}</strong>
        <span>${escapeHtml(report.training_ready ? "ready" : report.can_train ? "can-run" : "blocked")}</span>
        <p>${escapeHtml(action)}</p>
      </div>
    `).join("")}
  `;
}

function renderTuningPaths(report) {
  const chatPath = report?.chat_input || "--";
  const evalPath = report?.eval_input || "--";
  const packPath = report?.dataset_pack || $("flight-pack-path")?.value.trim() || "--";
  const hasRepeatedStarter = /_starter_starter/i.test(`${chatPath} ${evalPath}`);
  const status = hasRepeatedStarter ? "caution" : "ready";
  const message = hasRepeatedStarter ? "Path needs cleanup before launch." : "Paths are clean for launch.";
  return `
    <div class="tuning-path-card ${status}">
      <div>
        <label>TUNING FILE PATHS</label>
        <strong>${escapeHtml(message)}</strong>
      </div>
      <div class="command-meta">
        <span>PACK ${escapeHtml(packPath)}</span>
        <span>SFT ${escapeHtml(chatPath)}</span>
        <span>EVAL ${escapeHtml(evalPath)}</span>
      </div>
    </div>
  `;
}

function renderTuningInspectionError(error) {
  $("inspect-tuning-button").disabled = false;
  $("tuning-inspector-status").textContent = "TUNING INSPECTION FAULT";
  $("tuning-inspector-actions").innerHTML = "";
  $("tuning-inspector-result").innerHTML = "";
  $("tuning-inspector-command").innerHTML = `
    <label>ERROR</label>
    <code>FAULT: ${escapeHtml(error.message)}</code>
  `;
}

async function loadPackEditor() {
  const packPath = $("editor-pack-path").value.trim();
  const chatInput = $("editor-chat-path").value.trim();
  const evalInput = $("editor-eval-path").value.trim();
  $("load-editor-button").disabled = true;
  $("pack-editor-status").innerHTML = 'LOADING JSONL<span class="cursor"></span>';
  try {
    const report = await postJson("/api/pack/editor/load", {
      dataset_pack: packPath || null,
      chat_input: packPath ? null : chatInput || null,
      eval_input: packPath ? null : evalInput || null,
    });
    state.packEditor = report;
    $("editor-chat-jsonl").value = report.chat_text || "";
    $("editor-eval-jsonl").value = report.eval_text || "";
    renderPackEditor(report);
    renderStartHere();
  } finally {
    $("load-editor-button").disabled = false;
  }
}

async function savePackEditor() {
  const packPath = $("editor-pack-path").value.trim();
  const chatInput = $("editor-chat-path").value.trim();
  const evalInput = $("editor-eval-path").value.trim();
  $("save-editor-button").disabled = true;
  $("pack-editor-status").innerHTML = 'SAVING JSONL<span class="cursor"></span>';
  try {
    const report = await postJson("/api/pack/editor/save", {
      dataset_pack: packPath || null,
      chat_input: packPath ? null : chatInput || null,
      eval_input: packPath ? null : evalInput || null,
      chat_text: $("editor-chat-jsonl").value,
      eval_text: $("editor-eval-jsonl").value,
    });
    state.packEditor = report;
    state.tuningInspection = editorToTuningInspection(report);
    $("tuning-pack-path").value = report.dataset_pack || "";
    $("tuning-chat-path").value = report.dataset_pack ? "" : report.chat_input || "";
    $("tuning-eval-path").value = report.dataset_pack ? "" : report.eval_input || "";
    $("preview-pack-path").value = report.dataset_pack || $("preview-pack-path").value;
    $("launch-pack-path").value = report.dataset_pack || $("launch-pack-path").value;
    if (report.dataset_pack && !$("launch-run-name").value) {
      $("launch-run-name").value = suggestedRunName(report.dataset_pack);
    }
    renderPackEditor(report);
    renderTuningInspection(state.tuningInspection);
    renderStartHere();
  } finally {
    $("save-editor-button").disabled = false;
  }
}

function renderPackEditor(report) {
  if (!report) {
    $("pack-editor-status").textContent = "NO JSONL LOADED.";
    return;
  }
  const saved = report.saved ? "SAVED" : "LOADED";
  $("pack-editor-status").textContent =
    `${saved} | CHAT ${fmtInt(report.chat_lines)} LINES | EVAL ${fmtInt(report.eval_lines)} LINES | TUNING ${String(report.status || "--").toUpperCase()}`;
}

function renderPackEditorError(error) {
  $("load-editor-button").disabled = false;
  $("save-editor-button").disabled = false;
  $("pack-editor-status").textContent = `JSONL EDITOR FAULT | ${error.message}`;
}

function addChatEditorRow() {
  appendJsonlLine("editor-chat-jsonl", {
    user: "Replace with a real user question.",
    assistant: "Replace with the answer you want the model to learn.",
  });
}

function addEvalEditorRow() {
  appendJsonlLine("editor-eval-jsonl", {
    user: "Replace with a question your model should answer.",
    category: "starter",
    answerable: true,
    must_include: ["Replace with a required phrase"],
  });
}

function appendJsonlLine(id, row) {
  const field = $(id);
  const prefix = field.value.trim() ? `${field.value.trimEnd()}\n` : "";
  field.value = `${prefix}${JSON.stringify(row)}`;
}

function editorToTuningInspection(report) {
  return {
    status: report.status,
    summary: report.summary,
    training_ready: report.status === "ready",
    can_train: report.status !== "blocked",
    dataset_pack: report.dataset_pack || null,
    chat_input: report.chat_input || null,
    eval_input: report.eval_input || null,
    chat_data: report.chat_data,
    eval_data: report.eval_data,
    next_actions: report.next_actions || [],
    preview_command: report.dataset_pack
      ? shellCommand(["PYTHONPATH=src", "python", "-m", "picochat.cli", "data", "preview", "--dataset-pack", report.dataset_pack])
      : null,
  };
}

function applyLaunchPreset(quiet = false) {
  const preset = $("launch-preset").value;
  const values = state.runPresets[preset];
  if (!values) return;
  $("launch-context-size").value = values.context_size;
  $("launch-n-embd").value = values.n_embd;
  $("launch-n-head").value = values.n_head;
  $("launch-n-kv-head").value = values.n_kv_head || values.n_head;
  $("launch-n-layer").value = values.n_layer;
  $("launch-norm-type").value = values.norm_type || "layernorm";
  $("launch-position-encoding").value = values.position_encoding || "learned";
  $("launch-activation").value = values.activation || "gelu";
  $("launch-tie-embeddings").checked = Boolean(values.tie_embeddings);
  $("launch-qk-norm").checked = Boolean(values.qk_norm);
  $("launch-parallel-residual").checked = Boolean(values.parallel_residual);
  $("launch-base-steps").value = values.base_steps;
  $("launch-sft-steps").value = values.sft_steps;
  $("launch-base-batch-size").value = values.base_batch_size;
  $("launch-sft-batch-size").value = values.sft_batch_size;
  $("launch-base-learning-rate").value = values.base_learning_rate;
  $("launch-sft-learning-rate").value = values.sft_learning_rate;
  $("launch-base-lr-decay").value = values.base_lr_decay || "none";
  $("launch-sft-lr-decay").value = values.sft_lr_decay || "none";
  $("launch-base-lr-warmup-steps").value = values.base_lr_warmup_steps;
  $("launch-sft-lr-warmup-steps").value = values.sft_lr_warmup_steps;
  $("launch-base-grad-clip").value = values.base_grad_clip;
  $("launch-sft-grad-clip").value = values.sft_grad_clip;
  $("launch-base-grad-accum-steps").value = values.base_grad_accum_steps || 1;
  $("launch-base-dataset-mode").value = values.base_dataset_mode || "memory";
  $("launch-base-shard-token-size").value = values.base_shard_token_size || 1000000;
  $("launch-base-shard-cache-size").value = values.base_shard_cache_size || 2;
  $("launch-sft-grad-accum-steps").value = values.sft_grad_accum_steps || 1;
  $("launch-base-optimizer").value = values.base_optimizer || "adamw";
  $("launch-sft-optimizer").value = values.sft_optimizer || "adamw";
  $("launch-base-muon-learning-rate").value = values.base_muon_learning_rate || 0.02;
  $("launch-sft-muon-learning-rate").value = values.sft_muon_learning_rate || 0.02;
  $("launch-base-ema-decay").value = values.base_ema_decay || 0;
  $("launch-sft-ema-decay").value = values.sft_ema_decay || 0;
  $("launch-device").value = values.device || "cpu";
  $("launch-precision").value = values.precision || "float32";
  $("launch-matmul-precision").value = values.matmul_precision || "default";
  $("launch-attn-backend").value = values.attn_backend || "auto";
  $("launch-torch-compile").checked = Boolean(values.torch_compile);
  $("launch-torch-compile-mode").value = values.torch_compile_mode || "default";
  $("launch-gradient-checkpointing").checked = Boolean(values.gradient_checkpointing);
  $("launch-auto-lr-scaling").checked = Boolean(values.auto_lr_scaling);
  $("launch-loss-spike-rollback").checked = Boolean(values.loss_spike_rollback);
  $("launch-base-early-stop-patience").value = values.base_early_stop_patience;
  $("launch-sft-early-stop-patience").value = values.sft_early_stop_patience;
  $("launch-sft-sampling").value = values.sft_sampling || "uniform";
  $("launch-sft-packing").value = values.sft_packing || "separate";
  $("launch-eval-max-new-tokens").value = values.eval_max_new_tokens;
  $("launch-target-param-data-ratio").value = values.target_param_data_ratio || 20;
  $("launch-long-run-gate-profile").value = values.long_run_gate_profile || (preset === "h100-pilot" ? "first_release" : "research");
  if (values.tokenizer_type) $("launch-tokenizer-type").value = values.tokenizer_type;
  $("launch-bpe-pretokenizer").value = values.bpe_pretokenizer || "regex";
  $("launch-tokenizer-vocab-size").value = values.tokenizer_vocab_size || "";
  if (!quiet) {
    flashStatus(`APPLIED ${String(values.label || preset).toUpperCase()} PRESET. | ${values.description || ""}`);
  }
  renderLaunchReadiness();
}

function applyPreviewBudgetToLauncher() {
  const budget = state.corpusSourcePreview?.budget;
  if (!budget) {
    flashStatus("BUDGET APPLY FAULT. | Run Source Preview first.");
    return;
  }
  $("launch-context-size").value = budget.suggested_context_size || $("launch-context-size").value;
  $("launch-base-steps").value = budget.suggested_base_steps || $("launch-base-steps").value;
  $("launch-sft-steps").value = Math.max(60, Number(budget.suggested_base_steps || 30) * 2);
  $("launch-min-score").value = state.corpusSourcePreview?.min_quality_score ?? $("launch-min-score").value;
  renderLaunchReadiness();
  flashStatus(`APPLIED PREVIEW BUDGET. | CTX ${$("launch-context-size").value} | BASE ${$("launch-base-steps").value} | SFT ${$("launch-sft-steps").value}`);
}

function launchNumber(id) {
  const value = Number($(id).value);
  return Number.isFinite(value) ? value : 0;
}

function boundedNumberInput(id, fallback, min, max) {
  const node = $(id);
  const raw = Number(node?.value);
  const value = Number.isFinite(raw) ? raw : fallback;
  const bounded = Math.min(max, Math.max(min, Math.round(value)));
  if (node && String(node.value) !== String(bounded)) node.value = String(bounded);
  return bounded;
}

function launchConfig() {
  const tokenizerVocab = $("launch-tokenizer-vocab-size").value.trim();
  return {
    dataset_pack: $("launch-pack-path").value.trim(),
    run_name: $("launch-run-name").value.trim(),
    preset: $("launch-preset").value,
    context_size: launchNumber("launch-context-size"),
    n_embd: launchNumber("launch-n-embd"),
    n_head: launchNumber("launch-n-head"),
    n_kv_head: launchNumber("launch-n-kv-head"),
    n_layer: launchNumber("launch-n-layer"),
    norm_type: $("launch-norm-type").value,
    position_encoding: $("launch-position-encoding").value,
    activation: $("launch-activation").value,
    tie_embeddings: $("launch-tie-embeddings").checked,
    qk_norm: $("launch-qk-norm").checked,
    parallel_residual: $("launch-parallel-residual").checked,
    base_steps: launchNumber("launch-base-steps"),
    sft_steps: launchNumber("launch-sft-steps"),
    base_batch_size: launchNumber("launch-base-batch-size"),
    sft_batch_size: launchNumber("launch-sft-batch-size"),
    base_learning_rate: launchNumber("launch-base-learning-rate"),
    sft_learning_rate: launchNumber("launch-sft-learning-rate"),
    base_lr_decay: $("launch-base-lr-decay").value,
    sft_lr_decay: $("launch-sft-lr-decay").value,
    base_lr_warmup_steps: launchNumber("launch-base-lr-warmup-steps"),
    sft_lr_warmup_steps: launchNumber("launch-sft-lr-warmup-steps"),
    base_grad_clip: launchNumber("launch-base-grad-clip"),
    sft_grad_clip: launchNumber("launch-sft-grad-clip"),
    base_grad_accum_steps: boundedNumberInput("launch-base-grad-accum-steps", 1, 1, 128),
    base_dataset_mode: $("launch-base-dataset-mode").value,
    base_shard_token_size: boundedNumberInput("launch-base-shard-token-size", 1000000, 1000, 50000000),
    base_shard_cache_size: boundedNumberInput("launch-base-shard-cache-size", 2, 1, 16),
    sft_grad_accum_steps: boundedNumberInput("launch-sft-grad-accum-steps", 1, 1, 128),
    base_optimizer: $("launch-base-optimizer").value,
    sft_optimizer: $("launch-sft-optimizer").value,
    base_muon_learning_rate: launchNumber("launch-base-muon-learning-rate"),
    sft_muon_learning_rate: launchNumber("launch-sft-muon-learning-rate"),
    base_ema_decay: launchNumber("launch-base-ema-decay"),
    sft_ema_decay: launchNumber("launch-sft-ema-decay"),
    device: $("launch-device").value,
    precision: $("launch-precision").value,
    matmul_precision: $("launch-matmul-precision").value,
    attn_backend: $("launch-attn-backend").value,
    torch_compile: $("launch-torch-compile").checked,
    torch_compile_mode: $("launch-torch-compile-mode").value,
    gradient_checkpointing: $("launch-gradient-checkpointing").checked,
    auto_lr_scaling: $("launch-auto-lr-scaling").checked,
    loss_spike_rollback: $("launch-loss-spike-rollback").checked,
    base_early_stop_patience: launchNumber("launch-base-early-stop-patience"),
    sft_early_stop_patience: launchNumber("launch-sft-early-stop-patience"),
    sft_sampling: $("launch-sft-sampling").value,
    sft_packing: $("launch-sft-packing").value,
    eval_max_new_tokens: launchNumber("launch-eval-max-new-tokens"),
    target_param_data_ratio: launchNumber("launch-target-param-data-ratio"),
    long_run_gate_profile: $("launch-long-run-gate-profile").value,
    seed: launchNumber("launch-seed"),
    tokenizer_type: $("launch-tokenizer-type").value,
    bpe_pretokenizer: $("launch-bpe-pretokenizer").value,
    tokenizer_vocab_size: tokenizerVocab ? Number(tokenizerVocab) : null,
    min_quality_score: launchNumber("launch-min-score"),
  };
}

function launchReadiness(config = launchConfig()) {
  const blockers = [];
  const cautions = [];
  const notes = [];
  if (!config.dataset_pack) blockers.push("Choose or import a dataset pack first.");
  if (!config.run_name) blockers.push("Name the run so artifacts land in a unique folder.");
  if (config.context_size < 8) blockers.push("Context size must be at least 8.");
  if (config.n_embd < 1 || config.n_head < 1 || config.n_layer < 1) blockers.push("Model shape needs positive embed, heads, and layers.");
  if (config.n_head > 0 && config.n_embd % config.n_head !== 0) {
    blockers.push(`N EMBED ${config.n_embd} must divide evenly by HEADS ${config.n_head}.`);
  }
  if (config.n_kv_head > 0 && config.n_head % config.n_kv_head !== 0) {
    blockers.push(`HEADS ${config.n_head} must divide evenly by KV HEADS ${config.n_kv_head}.`);
  }
  if (config.position_encoding === "rope" && config.n_head > 0 && Math.floor(config.n_embd / config.n_head) % 2 !== 0) {
    blockers.push("RoPE requires an even attention head dimension.");
  }
  if (config.base_steps < 1 || config.sft_steps < 1) blockers.push("Base and SFT steps must be at least 1.");
  if (config.base_batch_size < 1 || config.sft_batch_size < 1) blockers.push("Batch sizes must be at least 1.");
  if (config.base_grad_accum_steps < 1 || config.sft_grad_accum_steps < 1) blockers.push("Gradient accumulation must be at least 1.");
  if (!["memory", "sharded"].includes(config.base_dataset_mode)) blockers.push("Base data mode is invalid.");
  if (config.base_dataset_mode === "sharded" && config.base_shard_token_size < config.context_size * 4) {
    blockers.push("Sharded token size should be at least 4x context size.");
  }
  if (config.base_learning_rate <= 0 || config.sft_learning_rate <= 0) blockers.push("Learning rates must be above zero.");
  if (!["adamw", "muon"].includes(config.base_optimizer) || !["adamw", "muon"].includes(config.sft_optimizer)) {
    blockers.push("Optimizer must be ADAMW or MUON.");
  }
  if (!["float32", "bf16", "fp16", "auto"].includes(config.precision)) blockers.push("Precision mode is invalid.");
  if (!["default", "highest", "high", "medium"].includes(config.matmul_precision)) blockers.push("Matmul precision mode is invalid.");
  if (!["auto", "flash", "efficient", "math", "cudnn"].includes(config.attn_backend)) blockers.push("Attention backend is invalid.");
  if (!["default", "reduce-overhead", "max-autotune"].includes(config.torch_compile_mode)) blockers.push("Torch compile mode is invalid.");
  if (!["research", "first_release"].includes(config.long_run_gate_profile)) blockers.push("Gate profile is invalid.");
  if (config.attn_backend === "flash" && !["auto", "cuda"].includes(config.device)) {
    blockers.push("Flash attention requires CUDA or AUTO device selection.");
  }
  if (!["separate", "bos_bestfit"].includes(config.sft_packing)) blockers.push("SFT packing mode is invalid.");
  if (config.base_muon_learning_rate <= 0 || config.sft_muon_learning_rate <= 0) blockers.push("Muon learning rates must be above zero.");
  if (config.base_ema_decay < 0 || config.base_ema_decay >= 1 || config.sft_ema_decay < 0 || config.sft_ema_decay >= 1) {
    blockers.push("EMA decay must be at least 0 and below 1.");
  }
  if (config.eval_max_new_tokens < 1) blockers.push("Eval tokens must be at least 1.");
  if (config.target_param_data_ratio < 1) blockers.push("Tokens / param target must be at least 1.");
  const usesBpe = ["bpe", "hf_bpe"].includes(config.tokenizer_type);
  if (usesBpe && !config.tokenizer_vocab_size) {
    cautions.push("BPE vocab is empty, so the backend default will decide tokenizer size.");
  }
  if (!usesBpe && config.tokenizer_vocab_size) {
    cautions.push("Vocab size only changes BPE; char and byte tokenizers ignore it.");
  }
  const starterWarning = starterSizeWarning(currentChatRowCount(), currentEvalRowCount());
  if (starterWarning) {
    cautions.push(`${starterWarning}. Use Guide Me to regenerate roughly ${fmtInt(STARTER_ROW_TARGETS.sft)} SFT rows and ${fmtInt(STARTER_ROW_TARGETS.eval)} eval rows before judging a medium run.`);
  }
  if (config.base_lr_warmup_steps > config.base_steps) cautions.push("Base warmup is longer than base training.");
  if (config.sft_lr_warmup_steps > config.sft_steps) cautions.push("SFT warmup is longer than SFT training.");
  if (config.base_optimizer === "muon" || config.sft_optimizer === "muon") {
    cautions.push("Muon is experimental here; compare against an AdamW baseline before trusting a win.");
  }
  if (config.torch_compile && config.device === "cpu") {
    cautions.push("torch.compile can be slow to warm up on CPU; reserve it for CUDA/MPS timing checks.");
  }
  if (config.gradient_checkpointing && config.context_size <= 512) {
    cautions.push("Gradient checkpointing saves memory but can slow small local runs.");
  }
  if (config.base_dataset_mode === "sharded") {
    cautions.push("Sharded base data avoids giant token tensors but validates by token shard, not complete source document.");
  }
  const usingBenchmarkPack = Boolean(state.benchmarkPack && state.benchmarkPack.dataset_pack === config.dataset_pack);
  const benchmarkProfile = state.benchmarkPack?.profile || "";
  if (config.long_run_gate_profile === "first_release" && usingBenchmarkPack && benchmarkProfile && benchmarkProfile !== "release_behavior") {
    cautions.push("FIRST RELEASE gate should normally use a RELEASE BEHAVIOR benchmark pack.");
  }
  if (usingBenchmarkPack && config.sft_steps > 500) {
    cautions.push("Curated benchmark SFT usually overfits past a few hundred steps; start near 300, then compare.");
  }
  if (config.sft_steps > config.base_steps * 2) cautions.push("SFT is much longer than base; watch eval leakage and overfitting.");
  notes.push(`${config.n_layer}L x ${config.n_embd} embd / ${config.n_head} heads / ${config.n_kv_head} kv`);
  notes.push(`${config.norm_type} / ${config.position_encoding} / ${config.activation}`);
  if (config.tie_embeddings || config.qk_norm || config.parallel_residual) {
    notes.push(`modern flags ${[
      config.tie_embeddings ? "tied" : null,
      config.qk_norm ? "qk-norm" : null,
      config.parallel_residual ? "parallel" : null,
    ].filter(Boolean).join("/")}`);
  }
  notes.push(`${String(config.tokenizer_type).toUpperCase()} tokenizer${config.tokenizer_vocab_size ? ` vocab ${config.tokenizer_vocab_size}` : ""}`);
  if (usesBpe) notes.push(`BPE split ${config.bpe_pretokenizer}`);
  notes.push(`base ${config.base_steps} / sft ${config.sft_steps}`);
  if (usingBenchmarkPack) notes.push("clean benchmark pack active");
  notes.push(`optimizer ${config.base_optimizer}/${config.sft_optimizer}`);
  if (config.base_ema_decay > 0 || config.sft_ema_decay > 0) notes.push(`EMA ${config.base_ema_decay}/${config.sft_ema_decay}`);
  notes.push(`effective batch ${config.base_batch_size * config.base_grad_accum_steps} / ${config.sft_batch_size * config.sft_grad_accum_steps}`);
  notes.push(`base data ${config.base_dataset_mode}`);
  notes.push(`target ${config.target_param_data_ratio} tok/param`);
  notes.push(`gate ${config.long_run_gate_profile.replace("_", " ")}`);
  notes.push(`device ${String(config.device || "cpu").toUpperCase()}`);
  notes.push(`${config.precision} / matmul ${config.matmul_precision} / attn ${config.attn_backend}`);
  if (config.torch_compile) notes.push(`compile ${config.torch_compile_mode}`);
  if (config.gradient_checkpointing) notes.push("gradient checkpointing");
  if (config.auto_lr_scaling) notes.push("auto LR scaling");
  if (config.loss_spike_rollback) notes.push("loss rollback");
  notes.push(`LR ${config.base_learning_rate} -> ${config.sft_learning_rate}`);
  notes.push(`SFT ${config.sft_sampling.replace("_", " ")}`);
  notes.push(`packing ${config.sft_packing.replace("_", " ")}`);
  const status = blockers.length ? "blocked" : cautions.length ? "caution" : "ready";
  const title = status === "ready" ? "LAUNCH CHECK READY" : status === "caution" ? "LAUNCH CHECK CAUTION" : "LAUNCH CHECK BLOCKED";
  return { status, title, notes: [...blockers, ...cautions, ...notes] };
}

function renderLaunchReadiness() {
  const target = $("launch-readiness");
  if (!target) return;
  const readiness = launchReadiness();
  target.className = `readiness-summary ${readiness.status}`;
  target.innerHTML = `
    <strong>${escapeHtml(readiness.title)}</strong>
    <span>${readiness.notes.map((note) => escapeHtml(note)).join(" | ")}</span>
    <em>Research mode launches runs. Beginner mode explains the result and the next safe step.</em>
  `;
  renderLaunchCommandPreview();
}

function launchPreviewCommand(config = launchConfig()) {
  const outDir = config.run_name ? `runs/${config.run_name}` : "runs/my-run";
  const usesDdp = DDP_SCALE_PRESETS.has(config.preset);
  const envParts = [
    ...(usesDdp ? ["OMP_NUM_THREADS=1"] : []),
    ...(usesDdp ? ["PICOCHAT_DDP_TIMEOUT_MINUTES=120"] : []),
    ...((usesDdp || config.device === "cuda") ? ["PYTORCH_ALLOC_CONF=expandable_segments:True"] : []),
    "PYTHONUNBUFFERED=1",
    "PYTHONPATH=src",
  ];
  const parts = usesDdp ? [
    ...envParts,
    "torchrun",
    "--standalone",
    "--nproc_per_node=8",
    "-m",
    "picochat.cli",
  ] : [
    ...envParts,
    "python",
    "-m",
    "picochat.cli",
  ];
  parts.push(
    "run",
    "tiny",
    "--out-dir",
    outDir,
    "--dataset-pack",
    config.dataset_pack || "<dataset-pack>",
    "--context-size",
    config.context_size,
    "--n-embd",
    config.n_embd,
    "--n-head",
    config.n_head,
    "--n-kv-head",
    config.n_kv_head,
    "--n-layer",
    config.n_layer,
    "--norm-type",
    config.norm_type,
    "--position-encoding",
    config.position_encoding,
    "--activation",
    config.activation,
    "--attn-backend",
    config.attn_backend,
    "--base-steps",
    config.base_steps,
    "--sft-steps",
    config.sft_steps,
    "--base-batch-size",
    config.base_batch_size,
    "--sft-batch-size",
    config.sft_batch_size,
    "--base-learning-rate",
    config.base_learning_rate,
    "--sft-learning-rate",
    config.sft_learning_rate,
    "--seed",
    config.seed,
    "--eval-max-new-tokens",
    config.eval_max_new_tokens,
    "--precision",
    config.precision,
    "--matmul-precision",
    config.matmul_precision,
    "--tokenizer-type",
    config.tokenizer_type,
    "--bpe-pretokenizer",
    config.bpe_pretokenizer,
    "--base-lr-warmup-steps",
    config.base_lr_warmup_steps,
    "--sft-lr-warmup-steps",
    config.sft_lr_warmup_steps,
    "--base-lr-decay",
    config.base_lr_decay,
    "--sft-lr-decay",
    config.sft_lr_decay,
    "--base-grad-clip",
    config.base_grad_clip,
    "--sft-grad-clip",
    config.sft_grad_clip,
    "--base-grad-accum-steps",
    config.base_grad_accum_steps,
    "--base-dataset-mode",
    config.base_dataset_mode,
    "--sft-grad-accum-steps",
    config.sft_grad_accum_steps,
    "--base-optimizer",
    config.base_optimizer,
    "--sft-optimizer",
    config.sft_optimizer,
    "--base-muon-learning-rate",
    config.base_muon_learning_rate,
    "--sft-muon-learning-rate",
    config.sft_muon_learning_rate,
    "--base-ema-decay",
    config.base_ema_decay,
    "--sft-ema-decay",
    config.sft_ema_decay,
    "--base-early-stop-patience",
    config.base_early_stop_patience,
    "--sft-early-stop-patience",
    config.sft_early_stop_patience,
    "--sft-sampling",
    config.sft_sampling,
    "--sft-packing",
    config.sft_packing,
    "--target-param-data-ratio",
    config.target_param_data_ratio,
    "--long-run-gate-profile",
    config.long_run_gate_profile,
    "--split-mode",
    "document",
    "--min-score",
    config.min_quality_score,
    "--device",
    config.device || "cpu",
  );
  if (state.runPresets[config.preset]) parts.push("--scale", config.preset);
  if (config.tie_embeddings) parts.push("--tie-embeddings");
  if (config.qk_norm) parts.push("--qk-norm");
  if (config.parallel_residual) parts.push("--parallel-residual");
  if (config.torch_compile) parts.push("--torch-compile", "--torch-compile-mode", config.torch_compile_mode);
  if (config.gradient_checkpointing) parts.push("--gradient-checkpointing");
  if (config.auto_lr_scaling) parts.push("--auto-lr-scaling");
  if (config.loss_spike_rollback) parts.push("--loss-spike-rollback");
  if (usesDdp) parts.push("--ddp");
  if (config.tokenizer_vocab_size) parts.push("--tokenizer-vocab-size", config.tokenizer_vocab_size);
  if (config.base_dataset_mode === "sharded") {
    parts.push("--base-shard-token-size", config.base_shard_token_size);
    parts.push("--base-shard-cache-size", config.base_shard_cache_size);
  }
  return shellCommand(parts);
}

function renderLaunchCommandPreview() {
  const target = $("launch-command-preview");
  if (!target) return;
  const config = launchConfig();
  const readiness = launchReadiness(config);
  const command = launchPreviewCommand(config);
  target.innerHTML = `
    <div class="command-head">
      <label>LAUNCH COMMAND PREVIEW</label>
      ${readiness.status === "blocked" ? "" : copyCommandButton(command)}
    </div>
    <div class="command-meta">
      <span>${readiness.status === "blocked" ? "FIX BLOCKERS FIRST" : "READY TO COPY OR LAUNCH"}</span>
      <span>OUT ${escapeHtml(config.run_name ? `runs/${config.run_name}` : "runs/my-run")}</span>
      <span>SOURCE WEB LAUNCHER</span>
    </div>
    <code>${escapeHtml(command)}</code>
    <p>${readiness.status === "blocked"
      ? "This is the CLI shape for the current fields, but blocked checks must be fixed before launch."
      : "This mirrors the current web launcher settings so you can learn or reproduce the run from terminal."}</p>
  `;
}

async function launchRun() {
  const config = launchConfig();
  const readiness = launchReadiness(config);
  renderLaunchReadiness();
  if (readiness.status === "blocked") throw new Error(readiness.notes[0] || "fix launch settings");
  $("launch-run-button").disabled = true;
  state.runJob = null;
  state.runJobLoaded = false;
  renderRunJob(null);
  renderRunJobList();
  $("run-launch-status").innerHTML = 'LAUNCHING RUN<span class="cursor"></span>';
  try {
    const payload = await startRunWithRetry(config);
    state.runJob = payload.job;
    state.runJobs = payload.jobs || [payload.job];
    state.runJobLoaded = false;
    renderRunJob(state.runJob);
    renderRunJobList();
    renderStartHere();
    startRunPolling();
  } finally {
    $("launch-run-button").disabled = false;
  }
}

async function startRunWithRetry(config) {
  try {
    return await postJson("/api/run/start", runStartPayload(config));
  } catch (error) {
    if (!/run output already exists/i.test(String(error?.message || ""))) throw error;
    $("launch-run-name").value = uniqueRunName(config.run_name || suggestedRunName(config.dataset_pack || "picochat"));
    const retryConfig = launchConfig();
    renderLaunchReadiness();
    flashStatus(`RUN NAME EXISTS. | RETRYING AS ${retryConfig.run_name}`);
    return postJson("/api/run/start", runStartPayload(retryConfig));
  }
}

function runStartPayload(config) {
  return {
    dataset_pack: config.dataset_pack,
    run_name: config.run_name,
    preset: config.preset,
    context_size: config.context_size,
    n_embd: config.n_embd,
    n_head: config.n_head,
    n_kv_head: config.n_kv_head,
    n_layer: config.n_layer,
    norm_type: config.norm_type,
    position_encoding: config.position_encoding,
    activation: config.activation,
    tie_embeddings: config.tie_embeddings,
    qk_norm: config.qk_norm,
    parallel_residual: config.parallel_residual,
    base_steps: config.base_steps,
    sft_steps: config.sft_steps,
    base_batch_size: config.base_batch_size,
    sft_batch_size: config.sft_batch_size,
    base_learning_rate: config.base_learning_rate,
    sft_learning_rate: config.sft_learning_rate,
    base_lr_decay: config.base_lr_decay,
    sft_lr_decay: config.sft_lr_decay,
    base_lr_warmup_steps: config.base_lr_warmup_steps,
    sft_lr_warmup_steps: config.sft_lr_warmup_steps,
    base_grad_clip: config.base_grad_clip,
    sft_grad_clip: config.sft_grad_clip,
    base_grad_accum_steps: config.base_grad_accum_steps,
    base_dataset_mode: config.base_dataset_mode,
    base_shard_token_size: config.base_shard_token_size,
    base_shard_cache_size: config.base_shard_cache_size,
    sft_grad_accum_steps: config.sft_grad_accum_steps,
    base_optimizer: config.base_optimizer,
    sft_optimizer: config.sft_optimizer,
    base_muon_learning_rate: config.base_muon_learning_rate,
    sft_muon_learning_rate: config.sft_muon_learning_rate,
    base_ema_decay: config.base_ema_decay,
    sft_ema_decay: config.sft_ema_decay,
    precision: config.precision,
    matmul_precision: config.matmul_precision,
    attn_backend: config.attn_backend,
    torch_compile: config.torch_compile,
    torch_compile_mode: config.torch_compile_mode,
    gradient_checkpointing: config.gradient_checkpointing,
    auto_lr_scaling: config.auto_lr_scaling,
    loss_spike_rollback: config.loss_spike_rollback,
    base_early_stop_patience: config.base_early_stop_patience,
    sft_early_stop_patience: config.sft_early_stop_patience,
    sft_sampling: config.sft_sampling,
    sft_packing: config.sft_packing,
    target_param_data_ratio: config.target_param_data_ratio,
    long_run_gate_profile: config.long_run_gate_profile,
    eval_max_new_tokens: config.eval_max_new_tokens,
    seed: config.seed,
    tokenizer_type: config.tokenizer_type,
    bpe_pretokenizer: config.bpe_pretokenizer,
    tokenizer_vocab_size: config.tokenizer_vocab_size,
    min_quality_score: config.min_quality_score,
    device: config.device,
  };
}

function seedScaleFromLauncher() {
  const config = launchConfig();
  if ($("scale-dataset-pack")) $("scale-dataset-pack").value = config.dataset_pack || "";
  if ($("scale-run-name")) $("scale-run-name").value = config.run_name || suggestedRunName(config.dataset_pack || "climbmix");
  if ($("scale-preset")) $("scale-preset").value = SCALE_PRESETS.includes(config.preset) ? config.preset : "h100-pilot";
  if ($("scale-device")) $("scale-device").value = config.device === "cpu" ? "auto" : config.device || "auto";
  applyScalePresetDefaults();
  renderScalePlan();
  flashStatus("SCALE PLAN SEEDED. | Commands now mirror the launcher where possible.");
}

function applyScalePresetDefaults() {
  const preset = $("scale-preset")?.value || "h100-100m";
  const defaults = SCALE_IMPORT_DEFAULTS[preset];
  if (!defaults) return;
  if ($("scale-climbmix-shards")) $("scale-climbmix-shards").value = String(defaults.shards);
  if ($("scale-max-rows")) $("scale-max-rows").value = String(defaults.maxRows);
}

function scaleConfig() {
  const datasetPack = $("scale-dataset-pack")?.value.trim() || state.hfImport?.dataset_pack || $("launch-pack-path")?.value.trim() || "";
  const runName = $("scale-run-name")?.value.trim() || $("launch-run-name")?.value.trim() || suggestedRunName(datasetPack || "climbmix");
  const preset = $("scale-preset")?.value || "h100-pilot";
  return {
    dataset_pack: datasetPack,
    run_name: runName,
    preset,
    device: $("scale-device")?.value || "auto",
    long_run_gate_profile: H100_SCALE_PRESETS.has(preset)
      ? "first_release"
      : $("launch-long-run-gate-profile")?.value || "research",
    shards: boundedScaleNumber("scale-climbmix-shards", 1, 1, 6543),
    max_rows: boundedScaleNumber("scale-max-rows", 1000, 1, 1000000),
  };
}

function boundedScaleNumber(id, fallback, min, max) {
  const node = $(id);
  if (!node) return fallback;
  const raw = Number(node.value);
  const value = Number.isFinite(raw) ? Math.round(raw) : fallback;
  const bounded = Math.min(max, Math.max(min, value));
  if (String(node.value) !== String(bounded)) node.value = String(bounded);
  return bounded;
}

function renderScalePlan() {
  if (!$("scale-readiness")) return;
  const config = scaleConfig();
  if ($("scale-dataset-pack") && !$("scale-dataset-pack").value && config.dataset_pack) {
    $("scale-dataset-pack").value = config.dataset_pack;
  }
  if ($("scale-run-name") && !$("scale-run-name").value && config.run_name) {
    $("scale-run-name").value = config.run_name;
  }
  if ($("scale-import-name") && !$("scale-import-name").value && config.run_name) {
    $("scale-import-name").value = config.run_name;
  }
  const blockers = [];
  if (!config.dataset_pack) blockers.push("Choose or import a dataset pack first.");
  if (!config.run_name) blockers.push("Name the GPU run.");
  const status = blockers.length ? "blocked" : "ready";
  const localProofPreset = H100_SCALE_PRESETS.has(config.preset) ? "mps-local" : config.preset;
  const localProofRunName = H100_SCALE_PRESETS.has(config.preset)
    ? `${config.run_name}-local-proof`
    : config.run_name;
  const localProofNote = H100_SCALE_PRESETS.has(config.preset)
    ? " | local proof uses MPS LOCAL to avoid H100-only FlashAttention settings"
    : "";
  $("scale-readiness").className = `readiness-summary ${status}`;
  $("scale-readiness").innerHTML = `
    <strong>${status === "ready" ? "GPU PLAN READY" : "GPU PLAN BLOCKED"}</strong>
    <span>${escapeHtml(blockers.join(" | ") || `${config.preset} | ${config.device.toUpperCase()} | ${config.dataset_pack}${localProofNote}`)}</span>
  `;

  const mpsParts = [
    "PYTHONPATH=src",
    "python",
    "-m",
    "picochat.cli",
    "run",
    "tiny",
    "--out-dir",
    `runs/${localProofRunName}`,
    "--dataset-pack",
    config.dataset_pack || "<dataset-pack>",
    "--scale",
    localProofPreset,
    "--device",
    config.device === "cuda" ? "auto" : config.device,
  ];
  if (config.long_run_gate_profile === "first_release") {
    mpsParts.push("--long-run-gate-profile", "first_release");
  }
  const mpsCommand = shellCommand(mpsParts);
  const remoteSetup = [
    `git clone ${PICOCHAT_REPO_URL}`,
    "cd picochat",
    "git checkout develop",
    "sudo apt-get update",
    "sudo apt-get install -y python3.10-venv",
    "python3 -m venv .venv",
    "source .venv/bin/activate",
    "python -m pip install --upgrade pip",
    "python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1",
    `python -m pip install -e ".[hf,dev]"`,
  ].join("\n");
  const remoteSanity = [
    "mkdir -p logs",
    `${shellCommand([
      "PYTHONUNBUFFERED=1",
      "PYTHONPATH=src",
      "python",
      "-m",
      "picochat.cli",
      "sanity",
      "preh100",
      "--out-dir",
      "runs/h100-sanity-v1",
      "--device",
      "cuda",
      "--precision",
      "bf16",
      "--matmul-precision",
      "high",
      "--attn-backend",
      "flash",
      "--include-compile",
    ])} 2>&1 | tee logs/preh100-sanity.log`,
  ].join("\n");
  const remoteImport = [
    "mkdir -p logs",
    `${shellCommand([
      "PYTHONUNBUFFERED=1",
      "PYTHONPATH=src",
      "python",
      "-m",
      "picochat.cli",
      "data",
      "climbmix-import",
      "--out-dir",
      "runs/climbmix-cuda",
      "--shards",
      config.shards,
      "--max-rows",
      config.max_rows,
      "--min-chars",
      100,
      "--document-shard-rows",
      1000,
      "--force",
    ])} 2>&1 | tee logs/import-climbmix.log`,
  ].join("\n");
  const remoteBenchmark = `${shellCommand([
    "PYTHONUNBUFFERED=1",
    "PYTHONPATH=src",
    "python",
    "-m",
    "picochat.cli",
    "data",
    "benchmark-pack",
    "--dataset-pack",
    "runs/climbmix-cuda/dataset_pack.json",
    "--sft-rows",
    1600,
    "--eval-rows",
    320,
    "--profile",
    "release_behavior",
    "--skill-answer-style",
    "direct",
    "--source",
    "offline",
    "--force",
  ])} 2>&1 | tee logs/benchmark-pack-cuda.log`;
  const usesDdp = DDP_SCALE_PRESETS.has(config.preset);
  const remoteRunArgs = [
    "run",
    "tiny",
    "--out-dir",
    `runs/${config.run_name}`,
    "--dataset-pack",
    "runs/climbmix-cuda/dataset_pack.json",
    "--scale",
    config.preset,
    "--device",
    "cuda",
  ];
  if (config.long_run_gate_profile === "first_release") {
    remoteRunArgs.push("--long-run-gate-profile", "first_release");
  }
  const remotePreflightParts = [
    ...(usesDdp ? ["OMP_NUM_THREADS=1"] : []),
    ...(usesDdp ? ["PICOCHAT_DDP_TIMEOUT_MINUTES=120"] : []),
    "PYTORCH_ALLOC_CONF=expandable_segments:True",
    "PYTHONUNBUFFERED=1",
    "PYTHONPATH=src",
    "python",
    "-m",
    "picochat.cli",
    ...remoteRunArgs,
  ];
  if (usesDdp) remotePreflightParts.push("--ddp", "--ddp-world-size", "8");
  const remoteRunParts = usesDdp
    ? [
      "OMP_NUM_THREADS=1",
      "PICOCHAT_DDP_TIMEOUT_MINUTES=120",
      "PYTORCH_ALLOC_CONF=expandable_segments:True",
      "PYTHONUNBUFFERED=1",
      "PYTHONPATH=src",
      "torchrun",
      "--standalone",
      "--nproc_per_node=8",
      "-m",
      "picochat.cli",
      ...remoteRunArgs,
      "--ddp",
    ]
    : [
      "PYTORCH_ALLOC_CONF=expandable_segments:True",
      "PYTHONUNBUFFERED=1",
      "PYTHONPATH=src",
      "python",
      "-m",
      "picochat.cli",
      ...remoteRunArgs,
    ];
  const remotePreflight = `${shellCommand([...remotePreflightParts, "--preflight-only"])} 2>&1 | tee logs/preflight-${config.run_name}.log`;
  const remoteRun = `${shellCommand(remoteRunParts)} 2>&1 | tee logs/train-${config.run_name}.log`;
  const bundleParts = [
    "PYTHONPATH=src",
    "python",
    "-m",
    "picochat.cli",
    "run",
    "bundle",
    "--run-dir",
    `runs/${config.run_name}`,
    "--out",
    `${config.run_name}.tgz`,
    "--logs-dir",
    "logs",
    "--strict",
  ];
  const remoteReturn = [
    shellCommand(bundleParts),
    `# Copy ${config.run_name}.tgz back to this Mac, extract it, then paste the run folder below.`,
  ].join("\n");
  renderScaleCommand(
    "scale-mps-command",
    H100_SCALE_PRESETS.has(config.preset) ? "LOCAL PROOF COMMAND" : "MPS / LOCAL COMMAND",
    mpsCommand,
  );
  renderScaleCommand("scale-remote-setup-command", "REMOTE SETUP", remoteSetup);
  renderScaleCommand("scale-remote-sanity-command", "PRE-H100 SANITY", remoteSanity);
  renderScaleCommand("scale-remote-import-command", "REMOTE CLIMBMIX IMPORT", remoteImport);
  renderScaleCommand("scale-remote-benchmark-command", "REMOTE RELEASE BEHAVIOR PACK", remoteBenchmark);
  renderScaleCommand("scale-remote-preflight-command", "REMOTE PREFLIGHT", remotePreflight);
  renderScaleCommand("scale-remote-run-command", "REMOTE TRAIN", remoteRun);
  renderScaleCommand("scale-remote-return-command", "REMOTE RETURN TAR", remoteReturn);
}

function renderScaleCommand(id, label, command) {
  const target = $(id);
  if (!target) return;
  target.innerHTML = `
    <div class="command-head">
      <label>${escapeHtml(label)}</label>
      ${copyCommandButton(command)}
    </div>
    <code>${escapeHtml(command || "NO COMMAND AVAILABLE.")}</code>
  `;
}

async function importCompletedRun() {
  const sourcePath = $("scale-import-source")?.value.trim();
  const runName = $("scale-import-name")?.value.trim() || $("scale-run-name")?.value.trim();
  if (!sourcePath) throw new Error("paste the completed run folder path first");
  $("scale-import-button").disabled = true;
  $("scale-import-status").innerHTML = 'IMPORTING RUN<span class="cursor"></span>';
  try {
    const report = await postJson("/api/run/import", {
      source_path: sourcePath,
      run_name: runName || null,
    });
    $("scale-import-status").textContent = `${report.imported ? "IMPORTED" : "ALREADY PRESENT"} | ${report.run_name} | ${report.message}`;
    state.selectedRun = report.run_name || state.selectedRun;
    await loadRuns();
    renderScalePlan();
  } finally {
    $("scale-import-button").disabled = false;
  }
}

function renderScaleImportError(error) {
  $("scale-import-button").disabled = false;
  $("scale-import-status").textContent = `IMPORT FAULT: ${error.message}`;
}

async function refreshRunJob() {
  if (!state.runJob?.id) throw new Error("no run job to refresh");
  const payload = await fetchJson(`/api/run/status?job=${encodeURIComponent(state.runJob.id)}`);
  state.runJob = payload.job;
  state.runJobs = mergeRunJobs(state.runJobs, payload.job ? [payload.job] : []);
  renderRunJob(state.runJob);
  renderRunJobList();
  if (state.runJob?.state === "running") startRunPolling();
}

async function loadRunJobs() {
  const scope = runJobStatusScope();
  if (!scope) {
    state.runJobs = [];
    state.runJob = null;
    renderRunJob(null);
    renderRunJobList();
    return;
  }
  let payload;
  try {
    payload = await fetchJson(`/api/run/status?job=${encodeURIComponent(scope)}`);
  } catch (error) {
    if (!/unknown run job/i.test(String(error?.message || ""))) throw error;
    state.runJobs = [];
    state.runJob = null;
    renderRunJob(null);
    renderRunJobList();
    return;
  }
  state.runJobs = payload.jobs || [];
  if (state.runJob) {
    const refreshed = state.runJobs.find((job) => job.id === state.runJob.id || job.run_name === state.runJob.run_name);
    state.runJob = refreshed || payload.job || null;
  } else {
    state.runJob = payload.job || null;
  }
  keepLauncherRunNameFresh();
  renderRunJob(state.runJob);
  renderRunJobList();
  if (state.runJob?.state === "running") startRunPolling();
}

function runJobStatusScope() {
  if (state.runJob?.state === "running" && state.runJob.id) return state.runJob.id;
  return state.selectedRun || "";
}

async function cancelRunJob() {
  if (!state.runJob?.id) throw new Error("no active run job selected");
  if (!state.runJob.can_cancel) throw new Error("selected run cannot be cancelled");
  $("cancel-run-job-button").disabled = true;
  try {
    const payload = await postJson("/api/run/cancel", { job_id: state.runJob.id });
    state.runJob = payload.job;
    state.runJobs = mergeRunJobs(state.runJobs, payload.job ? [payload.job] : []);
    renderRunJob(state.runJob);
    renderRunJobList();
  } finally {
    $("cancel-run-job-button").disabled = false;
  }
}

function startRunPolling() {
  window.clearInterval(state.runPollTimer);
  state.runPollTimer = window.setInterval(() => {
    refreshRunJob().catch((error) => {
      window.clearInterval(state.runPollTimer);
      state.runPollTimer = null;
      renderRunJobError(error);
    });
  }, 1500);
}

function renderRunJob(job) {
  if (!job) {
    $("run-launch-status").textContent = "NO RUN LAUNCHED.";
    $("run-launch-progress").innerHTML = "";
    $("run-launch-command").innerHTML = "";
    $("run-launch-log").textContent = "READY.";
    $("cancel-run-job-button").disabled = true;
    return;
  }
  $("run-launch-status").textContent =
    `RUN ${String(job.state || "--").toUpperCase()} | ${escapeHtml(job.run_name)} | ${job.elapsed_seconds == null ? "--" : fmtLoss(job.elapsed_seconds)}S | PID ${escapeHtml(job.pid || "--")}`;
  $("cancel-run-job-button").disabled = !job.can_cancel;
  $("run-launch-progress").innerHTML = renderRunProgress(job.progress, job.state);
  $("run-launch-command").innerHTML = `
    <div class="command-head">
      <label>RUN COMMAND</label>
      ${copyCommandButton(job.command)}
    </div>
    <div class="command-meta">
      <span>OUT ${escapeHtml(job.out_dir)}</span>
      <span>LOG ${escapeHtml(job.log_path)}</span>
      <span>SUMMARY ${job.summary_exists ? "READY" : "PENDING"}</span>
      <span>SOURCE ${escapeHtml(job.source || "--")}</span>
      <span>PRESET ${escapeHtml(job.preset || "--")}</span>
      <span>MIN SCORE ${escapeHtml(job.min_quality_score ?? "--")}</span>
    </div>
    ${renderLaunchPreflight(job.launch_preflight)}
    <code>${escapeHtml(job.command || "")}</code>
  `;
  $("run-launch-log").textContent = job.log_tail || "WAITING FOR LOG OUTPUT.";
  if (job.state !== "running") {
    window.clearInterval(state.runPollTimer);
    state.runPollTimer = null;
    if (!state.runJobLoaded && job.summary_exists) {
      state.runJobLoaded = true;
      state.selectedRun = job.run_name;
      loadRuns().catch(() => {});
    }
  }
}

function renderLaunchPreflight(preflight) {
  if (!preflight) return "";
  const budget = preflight.budget || {};
  const blockers = preflight.blocking_checks || [];
  const warnings = preflight.warning_checks || [];
  const visibleChecks = blockers.length ? blockers : warnings.slice(0, 4);
  return `
    <div class="readiness-summary ${escapeHtml(preflight.status || "unknown")}">
      <strong>LONG-RUN PREFLIGHT ${escapeHtml(String(preflight.status || "--").toUpperCase())}</strong>
      <span>${escapeHtml(preflight.summary || "")}</span>
    </div>
    <div class="command-meta">
      <span>PARAMS ${fmtInt(budget.estimated_parameters)}</span>
      <span>TARGET ${fmtInt(budget.target_training_tokens)} TOK</span>
      <span>PLAN/TARGET ${fmtLoss(budget.planned_to_target_ratio)}</span>
      <span>REC ${fmtInt(budget.recommended_base_steps)} STEPS</span>
      <span>BASE EPOCHS ${fmtLoss(budget.estimated_base_epochs)}</span>
      <span>SFT EPOCHS ${fmtLoss(budget.estimated_sft_example_epochs)}</span>
      <span>${budget.long_run ? "LONG RUN" : "LOCAL RUN"}</span>
    </div>
    ${visibleChecks.map((check) => `
      <div class="readiness-row ${escapeHtml(check.status)}">
        <strong>${escapeHtml(check.name)}</strong>
        <span>${escapeHtml(check.metric)} / ${escapeHtml(check.threshold)}</span>
        <p>${escapeHtml(check.message)}</p>
      </div>
    `).join("")}
  `;
}

function renderRunProgress(progress, state) {
  if (!progress) {
    return `
      <div class="run-progress-head">
        <strong>WAITING FOR PROGRESS</strong>
        <span>Launch a run to see stage, loss, and eval movement here.</span>
      </div>
    `;
  }
  const stage = progress.stage || {};
  return `
    <div class="run-progress-head ${escapeHtml(stage.id || "waiting")}">
      <div>
        <label>RUN PROGRESS</label>
        <strong>${escapeHtml(stage.label || String(state || "waiting").toUpperCase())}</strong>
      </div>
      <span>${escapeHtml(stage.message || "Reading the run log.")}</span>
      <em>${escapeHtml(stage.index && stage.total ? `STAGE ${stage.index}/${stage.total}` : String(state || "--").toUpperCase())}</em>
    </div>
    <div class="run-progress-grid">
      ${renderRunProgressCard("base", "BASE TRAIN", progress.base, stage.id)}
      ${renderRunProgressCard("sft", "CHAT SFT", progress.sft, stage.id)}
      ${renderEvalProgressCard(progress.eval, stage.id)}
    </div>
  `;
}

function renderRunProgressCard(id, label, phase, activeStage) {
  const active = activeStage === id;
  const percent = progressWidth(phase?.percent);
  const stepText = phase ? `${fmtInt(phase.current)} / ${fmtInt(phase.total)} steps` : "waiting";
  const lossText = phase
    ? `train ${fmtLoss(phase.train_loss)} | val ${fmtLoss(phase.val_loss)} | bpb ${fmtLoss(phase.val_bpb)}`
    : "loss will appear after the first checkpoint row";
  return `
    <div class="run-progress-card ${active ? "active" : ""}">
      <label>${escapeHtml(label)}</label>
      <strong>${escapeHtml(stepText)}</strong>
      <div class="run-progress-meter"><span style="width:${percent}%"></span></div>
      <p>${escapeHtml(lossText)}</p>
    </div>
  `;
}

function renderEvalProgressCard(evalResult, activeStage) {
  const active = activeStage === "eval" || activeStage === "complete";
  const percent = progressWidth(evalResult?.pass_rate);
  const scoreText = evalResult ? `${fmtInt(evalResult.passed)} / ${fmtInt(evalResult.total)} passed` : "waiting";
  const helper = evalResult ? `${fmtLoss(evalResult.pass_rate)}% pass rate` : "eval score appears after generation checks finish";
  return `
    <div class="run-progress-card ${active ? "active" : ""}">
      <label>EVAL</label>
      <strong>${escapeHtml(scoreText)}</strong>
      <div class="run-progress-meter"><span style="width:${percent}%"></span></div>
      <p>${escapeHtml(helper)}</p>
    </div>
  `;
}

function progressWidth(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return 0;
  return Math.max(0, Math.min(100, number));
}

function renderRunJobList() {
  const jobs = state.runJobs || [];
  if (!jobs.length) {
    $("run-job-list").innerHTML = '<div class="empty">NO WEB-LAUNCHED RUNS FOUND.</div>';
    return;
  }
  $("run-job-list").innerHTML = jobs.slice(-8).reverse().map((job) => `
    <div class="run-job-row">
      <button class="run-job-button ${state.runJob?.id === job.id ? "active" : ""}" type="button" data-run-job="${escapeHtml(job.id)}">
        <strong>${escapeHtml(job.run_name)}</strong>
        <span>${escapeHtml(String(job.state || "--").toUpperCase())} | ${job.summary_exists ? "SUMMARY" : "NO SUMMARY"} | ${escapeHtml(job.source || "--")}</span>
      </button>
      ${job.can_cancel ? "" : `<button class="run-job-archive-button" type="button" data-archive-job-run="${escapeHtml(job.run_name)}">ARCHIVE</button>`}
    </div>
  `).join("");
  document.querySelectorAll("[data-run-job]").forEach((button) => {
    button.addEventListener("click", () => {
      const job = state.runJobs.find((item) => item.id === button.dataset.runJob);
      if (!job) return;
      state.runJob = job;
      renderRunJob(job);
      renderRunJobList();
      if (job.state === "running") startRunPolling();
      if (job.summary_exists) {
        state.selectedRun = job.run_name;
        loadRuns().catch((error) => renderRunJobError(error));
      }
    });
  });
  document.querySelectorAll("[data-archive-job-run]").forEach((button) => {
    button.addEventListener("click", async () => {
      const runName = button.dataset.archiveJobRun;
      if (!runName) return;
      button.disabled = true;
      button.textContent = "ARCHIVING";
      try {
        const payload = await postJson("/api/run/archive", { run_names: [runName] });
        const archivedNames = new Set((payload.archived_runs || []).map((run) => run.run_name));
        state.runJobs = (state.runJobs || []).filter((job) => !archivedNames.has(job.run_name));
        if (state.runJob && archivedNames.has(state.runJob.run_name)) state.runJob = null;
        if (archivedNames.has(state.selectedRun)) state.selectedRun = null;
        renderRunJob(state.runJob);
        renderRunJobList();
        flashStatus(`ARCHIVED ${runName}. | ${payload.archive_root || "ARCHIVE READY"}`);
        await loadRuns();
      } catch (error) {
        button.disabled = false;
        button.textContent = "ARCHIVE";
        flashStatus(`ARCHIVE FAULT. | ${error.message}`);
      }
    });
  });
}

function mergeRunJobs(existing, incoming) {
  const byId = new Map((existing || []).map((job) => [job.id, job]));
  (incoming || []).forEach((job) => {
    if (job?.id) byId.set(job.id, job);
  });
  return [...byId.values()];
}

function renderRunJobError(error) {
  $("launch-run-button").disabled = false;
  $("run-launch-status").textContent = `RUN LAUNCH FAULT | ${error.message}`;
  $("run-launch-progress").innerHTML = "";
  $("run-launch-command").innerHTML = "";
  $("run-launch-log").textContent = "Fix the launch fault, then start a new run.";
  $("cancel-run-job-button").disabled = true;
}

function suggestedRunName(packPath) {
  const parts = String(packPath || "picochat").split("/").filter(Boolean);
  const last = parts.at(-1) || "picochat";
  const parent = parts.length > 1 ? parts.at(-2) : last.replace(/\.[^.]+$/, "");
  return uniqueRunName(`${slugify(parent || "picochat")}-v1`);
}

function slugify(value) {
  return String(value || "picochat")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "") || "picochat";
}

function runNameExists(name) {
  const normalized = String(name || "").trim();
  if (!normalized) return false;
  return state.runs.some((run) => run.name === normalized) ||
    state.runJobs.some((job) => job.run_name === normalized);
}

function uniqueRunName(base) {
  const match = String(base || "picochat").trim().match(/^(.*?)-v(\d+)$/);
  const stem = match ? match[1] : base;
  const normalized = slugify(stem) || "picochat";
  const usedNames = new Set([
    ...state.runs.map((run) => run.name),
    ...state.runJobs.map((job) => job.run_name),
  ]);
  let index = match ? Number(match[2]) : 1;
  let candidate = `${normalized}-v${index}`;
  while (usedNames.has(candidate)) {
    index += 1;
    candidate = `${normalized}-v${index}`;
  }
  return candidate;
}

function keepLauncherRunNameFresh() {
  const input = $("launch-run-name");
  if (!input.value || !runNameExists(input.value)) return;
  input.value = uniqueRunName(input.value);
}

function statCards(rows) {
  return rows.map(([label, value]) => `
    <div class="pipeline-stat">
      <label>${escapeHtml(label)}</label>
      <span>${escapeHtml(value)}</span>
    </div>
  `).join("");
}

function qualityChecks(corpus) {
  const manifestWarnings = state.detail?.corpus_manifest?.warnings || [];
  const checks = [
    qualityCheck("Duplicate docs", corpus.duplicate_document_rate || 0, 0.05, "Repeated full documents can inflate learning and memorization signals."),
    qualityCheck("Duplicate lines", corpus.duplicate_line_rate || 0, 0.15, "Repeated lines can make a tiny model memorize phrasing."),
    qualityCheck("Empty lines", corpus.empty_line_rate || 0, 0.35, "Too many empty lines waste context windows."),
    qualityCheck("Non-ASCII chars", corpus.non_ascii_rate || 0, 0.05, "High non-ASCII rate is fine only if intentional."),
  ];
  const qualityHtml = checks.map((check) => `
    <div class="quality-row ${check.warn ? "warn" : "pass"}">
      <div>
        <strong>${escapeHtml(check.label)}</strong>
        <span>${fmtPercent(check.value)}</span>
      </div>
      <div class="quality-meter"><i style="width:${Math.min(100, Math.round(check.value * 100))}%"></i></div>
      <p>${escapeHtml(check.warn ? check.message : "Looks acceptable for this tiny run.")}</p>
    </div>
  `).join("");
  if (!manifestWarnings.length) return qualityHtml;
  return `${qualityHtml}
    <div class="quality-row warn">
      <div><strong>Manifest warnings</strong><span>${manifestWarnings.length}</span></div>
      ${manifestWarnings.map((warning) => `<p>${escapeHtml(warning)}</p>`).join("")}
    </div>
  `;
}

function qualityCheck(label, value, threshold, message) {
  return {
    label,
    value,
    warn: value > threshold,
    message,
  };
}

function renderCorpusFiles(files) {
  if (!files.length) {
    return '<div class="empty">NO CORPUS MANIFEST FOUND FOR THIS RUN.</div>';
  }
  return files.map((file) => `
    <div class="source-row ${file.included ? "included" : "skipped"}">
      <div>
        <strong>${escapeHtml(shortPath(file.path))}</strong>
        <span>${escapeHtml(file.included ? "INCLUDED" : "SKIPPED")} | Q${escapeHtml(file.quality_score ?? "--")}</span>
      </div>
      <small>${escapeHtml(file.extension)}${file.label ? ` | label:${escapeHtml(file.label)}` : ""} | ${fmtInt(file.num_characters)} chars | ${fmtInt(file.num_lines)} lines | ${escapeHtml(file.reason)}${file.quality_flags?.length ? ` | flags:${escapeHtml(file.quality_flags.join(","))}` : ""}</small>
    </div>
  `).join("");
}

async function checkDatasetFlightPlan() {
  const payload = datasetFlightPayload();
  if (!payload.dataset_pack && !payload.recipe_path && !payload.input_path) {
    throw new Error("enter a dataset pack or corpus path");
  }
  $("flight-check-button").disabled = true;
  $("flight-status").innerHTML = 'CHECKING DATASET<span class="cursor"></span>';
  $("flight-coach").className = "flight-coach caution";
  $("flight-coach").innerHTML = `
    <div>
      <label>WHAT JUST HAPPENED</label>
      <strong>Dataset check is running.</strong>
      <p>Picochat is reading files, measuring size, and checking SFT/eval readiness.</p>
    </div>
    <div>
      <label>NEXT CLICK</label>
      <strong>Wait for the report.</strong>
      <p>The next safe action appears here when the check finishes.</p>
    </div>
    <div>
      <label>WHY IT MATTERS</label>
      <strong>Cheap checks come before training.</strong>
      <p>This prevents wasting a run on broken paths or unusable tuning files.</p>
    </div>
  `;
  $("flight-plan").innerHTML = "";
  $("flight-command").innerHTML = "";
  try {
    return await refreshDatasetFlightPlanAfterChange(payload);
  } finally {
    $("flight-check-button").disabled = false;
  }
}

async function refreshDatasetFlightPlanAfterChange(payload = datasetFlightPayload()) {
  const report = await postJson("/api/corpus/preview", {
    ...payload,
    preview_chars: 900,
  });
  state.datasetFlightPlan = report;
  state.corpusSourcePreview = report;
  state.tuningInspection = tuningInspectionFromPreview(report);
  if (report.training_command?.chat_input && !$("flight-sft-out-path").value.trim()) {
    $("flight-sft-out-path").value = suggestedSftStarterPath(report.training_command.chat_input);
  }
  if (report.training_command?.eval_input && !$("flight-eval-out-path").value.trim()) {
    $("flight-eval-out-path").value = suggestedEvalStarterPath(report.training_command.eval_input);
  }
  if (report.dataset_pack) {
    $("launch-pack-path").value = report.dataset_pack;
    $("preview-pack-path").value = report.dataset_pack;
  }
  renderLaunchReadiness();
  renderDatasetFlightPlan(report);
  renderCorpusSourcePreview(report);
  renderTuningInspection(state.tuningInspection);
  renderStartHere();
  return report;
}

function datasetFlightPayload() {
  const packPath = $("flight-pack-path").value.trim();
  const inputPath = $("flight-input-path").value.trim();
  const chatInput = $("flight-chat-path").value.trim();
  const evalInput = $("flight-eval-path").value.trim();
  const minQualityScore = Number($("flight-min-score").value || 0);
  return {
    dataset_pack: packPath || null,
    recipe_path: null,
    input_path: packPath ? null : inputPath || null,
    chat_input: packPath ? null : chatInput || null,
    eval_input: packPath ? null : evalInput || null,
    min_quality_score: minQualityScore,
  };
}

function renderDatasetFlightPlan(report) {
  if (!report) {
    $("flight-status").textContent = "NO DATASET CHECKED.";
    renderFlightCoach(null);
    renderSmokeReadiness(null);
    $("flight-plan").innerHTML = "";
    $("flight-command").innerHTML = "";
    return;
  }
  const plan = trainingPlan(report);
  const stats = report.stats || {};
  $("flight-status").textContent =
    `${escapeHtml(plan.status.toUpperCase())} | ${readinessBadge(report.readiness)} | ${fmtInt(stats.num_documents)} DOCS | ${fmtInt(stats.num_characters)} CHARS`;
  $("flight-plan").innerHTML = `
    <div class="flight-grid">
      <div class="${escapeHtml(plan.status)}">
        <label>GO / NO-GO</label>
        <strong>${escapeHtml(plan.verdict)}</strong>
        <p>${escapeHtml(plan.reason)}</p>
      </div>
      <div>
        <label>FIRST RUN</label>
        <strong>${escapeHtml(plan.firstRun)}</strong>
        <p>${escapeHtml(plan.runtime)}</p>
      </div>
      <div>
        <label>TUNING DATA</label>
        <strong>${escapeHtml(plan.tuningStatus)}</strong>
        <p>${escapeHtml(plan.tuningAction)}</p>
      </div>
      <div>
        <label>EVAL STARTER</label>
        <strong>${escapeHtml(plan.evalStatus)}</strong>
        <p>${escapeHtml(plan.evalAction)}</p>
      </div>
    </div>
    <div class="flight-steps">
      ${plan.steps.map((step, index) => `
        <div class="${escapeHtml(step.status)}">
          <span>${String(index + 1).padStart(2, "0")}</span>
          <strong>${escapeHtml(step.label)}</strong>
          <p>${escapeHtml(step.message)}</p>
        </div>
      `).join("")}
    </div>
  `;
  $("flight-command").innerHTML = renderTrainingCommand(report.training_command);
  renderFlightCoach(report);
  renderSmokeReadiness(report);
}

function renderFlightCoach(report) {
  const target = $("flight-coach");
  if (!target) return;
  const coach = flightCoach(report);
  target.className = `flight-coach ${coach.status}`;
  target.innerHTML = `
    <div>
      <label>WHAT JUST HAPPENED</label>
      <strong>${escapeHtml(coach.happened)}</strong>
      <p>${escapeHtml(coach.evidence)}</p>
    </div>
    <div>
      <label>NEXT CLICK</label>
      <strong>${escapeHtml(coach.next)}</strong>
      <p>${escapeHtml(coach.detail)}</p>
    </div>
    <div>
      <label>WHY IT MATTERS</label>
      <strong>${escapeHtml(coach.why)}</strong>
      <p>${escapeHtml(coach.warning)}</p>
    </div>
  `;
}

function flightCoach(report) {
  if (!report) {
    return {
      status: "blocked",
      happened: "No dataset has been checked yet.",
      evidence: "Choose Hugging Face, local docs, an existing pack, or the sample dataset.",
      next: "Choose a source, then CHECK DATASET.",
      detail: "If you do not have data yet, use the sample dataset to learn the flow first.",
      why: "Training starts with corpus quality.",
      warning: "Skipping this check can waste time on missing files, repeated text, or tiny data.",
    };
  }
  const plan = trainingPlan(report);
  const stats = report.stats || {};
  const readiness = readinessBadge(report.readiness);
  const datasetPack = report.dataset_pack || $("flight-pack-path")?.value.trim();
  const launchPack = $("launch-pack-path")?.value.trim();
  const launcherReady = Boolean(datasetPack && launchPack === datasetPack);
  const chatRows = dataRowCount(report.chat_data);
  const evalRows = dataRowCount(report.eval_data);
  const starterWarning = starterSizeWarning(chatRows, evalRows);
  const source = state.hfImport?.preview === report
    ? `HF import ${state.hfImport.dataset}`
    : datasetPack
      ? shortPath(datasetPack)
      : shortPath($("flight-input-path")?.value.trim() || "current source");
  if (plan.status === "blocked") {
    return {
      status: "blocked",
      happened: `${source} was checked and is blocked.`,
      evidence: `${readiness} | ${fmtInt(stats.num_documents)} docs | ${fmtInt(stats.num_characters)} chars`,
      next: "Fix the blocked check shown below.",
      detail: plan.reason,
      why: "A blocked dataset cannot produce a trustworthy run.",
      warning: "Do not launch until corpus, SFT, and eval checks are usable.",
    };
  }
  if (!launcherReady) {
    return {
      status: "caution",
      happened: `${source} was checked and can be used for a first run.`,
      evidence: `${readiness} | ${fmtInt(stats.num_documents)} docs | ${fmtInt(stats.num_characters)} chars${starterWarning ? ` | ${starterWarning}` : ""}`,
      next: "Click APPLY PLAN.",
      detail: "That copies the checked dataset pack and budget into the Research-mode run launcher.",
      why: "The launcher should use exactly the dataset you just inspected.",
      warning: starterWarning || "A smoke run proves wiring. It is not the final model.",
    };
  }
  return {
    status: plan.status === "ready" && !starterWarning ? "ready" : "caution",
    happened: `${source} is connected to the run launcher.`,
    evidence: `${readiness} | ${fmtInt(stats.num_documents)} docs | ${fmtInt(stats.num_characters)} chars${starterWarning ? ` | ${starterWarning}` : ""}`,
    next: "Switch to Research and launch a smoke run.",
    detail: "Use the run launcher only to start or tune runs. Come back to Beginner to read the pipeline results.",
    why: "Small smoke runs catch bad SFT/eval wiring before larger training.",
    warning: starterWarning || "After it finishes, evaluate, compare, and only then scale.",
  };
}

function dataRowCount(data) {
  if (!data) return 0;
  const candidates = [data.num_examples, data.num_rows, data.rows, data.count];
  for (const value of candidates) {
    const number = Number(value);
    if (Number.isFinite(number) && number > 0) return number;
  }
  return 0;
}

function starterSizeWarning(chatRows, evalRows) {
  const warnings = [];
  if (chatRows > 0 && chatRows < STARTER_ROW_TARGETS.sft) {
    warnings.push(`${fmtInt(chatRows)} SFT rows < ${fmtInt(STARTER_ROW_TARGETS.sft)}`);
  }
  if (evalRows > 0 && evalRows < STARTER_ROW_TARGETS.eval) {
    warnings.push(`${fmtInt(evalRows)} eval rows < ${fmtInt(STARTER_ROW_TARGETS.eval)}`);
  }
  if (!warnings.length) return "";
  return `starter-sized ${warnings.join(" / ")}`;
}

function renderSmokeReadiness(report) {
  const target = $("flight-smoke-status");
  if (!target) return;
  if (!report) {
    target.className = "readiness-summary blocked";
    target.innerHTML = `
      <strong>SMOKE TRAIN WAITING</strong>
      <span>Choose a dataset source, then run the readiness check.</span>
    `;
    return;
  }
  const plan = trainingPlan(report);
  const launchPack = $("launch-pack-path").value.trim();
  const datasetPack = report.dataset_pack || $("flight-pack-path").value.trim();
  const canTrain = plan.status !== "blocked";
  const launcherReady = Boolean(datasetPack && launchPack === datasetPack);
  const starterWarning = starterSizeWarning(dataRowCount(report.chat_data), dataRowCount(report.eval_data));
  const className = !canTrain ? "blocked" : launcherReady ? "ready" : "caution";
  const title = !canTrain
    ? "SMOKE TRAIN BLOCKED"
    : launcherReady
      ? starterWarning ? "SMOKE ONLY: STARTERS NEED WORK" : "READY TO SMOKE TRAIN"
      : "APPLY PLAN NEXT";
  const message = !canTrain
    ? "Fix the failed corpus, SFT, or eval checks before launching."
    : launcherReady
      ? starterWarning
        ? `${starterWarning}. Launch only to test wiring; edit and validate SFT/eval before trusting scores.`
        : "Launcher has this dataset pack. Use the smoke preset first, then compare results."
      : "Click APPLY PLAN to move these checked paths into the run launcher.";
  target.className = `readiness-summary ${className}`;
  target.innerHTML = `
    <strong>${escapeHtml(title)}</strong>
    <span>${escapeHtml(message)}</span>
  `;
}

function trainingPlan(report) {
  const readinessStatus = report.readiness?.status || "blocked";
  const chatStatus = report.chat_data?.status || "blocked";
  const evalStatus = report.eval_data?.status || "blocked";
  const budget = report.budget || {};
  const blocked = readinessStatus === "blocked" || chatStatus === "blocked" || evalStatus === "blocked";
  const caution = readinessStatus === "caution" || chatStatus === "caution" || evalStatus === "caution";
  const status = blocked ? "blocked" : caution ? "caution" : "ready";
  const firstRun = budget.preset === "small-preview"
    ? "small-local"
    : budget.preset === "overfit-check"
      ? "tiny"
      : budget.preset || "smoke";
  return {
    status,
    verdict: blocked ? "Blocked before training" : caution ? "Preview run first" : "Ready for a serious tiny run",
    reason: blocked
      ? "Fix failed corpus or tuning checks before launching."
      : caution
        ? "You can run, but do not trust the score until cautions are handled."
        : "Corpus, SFT, and eval checks are clean enough for the next run.",
    firstRun,
    runtime: runtimeHint(budget),
    tuningStatus: `${String(chatStatus).toUpperCase()} / ${String(evalStatus).toUpperCase()}`,
    tuningAction: tuningActionText(report.chat_data, report.eval_data),
    evalStatus: String(evalStatus).toUpperCase(),
    evalAction: evalStatus === "ready"
      ? "Keep it, then add harder transfer/adversarial items after the first run."
      : "Create a starter eval, edit it, then inspect tuning again.",
    steps: [
      {
        label: "Inspect data",
        status: readinessStatus === "ready" ? "pass" : readinessStatus === "caution" ? "warn" : "fail",
        message: report.readiness?.summary || "No readiness report.",
      },
      {
        label: "Build eval",
        status: evalStatus === "ready" ? "pass" : evalStatus === "caution" ? "warn" : "fail",
        message: report.eval_data?.summary || "Eval JSONL missing or not inspected.",
      },
      {
        label: "First train",
        status: blocked ? "fail" : "pass",
        message: blocked ? "Blocked until checks pass." : `Start with ${firstRun}; compare before scaling.`,
      },
      {
        label: "Compare and chat",
        status: "warn",
        message: "Only trust improvements that show better eval behavior without worse trust checks.",
      },
    ],
  };
}

function runtimeHint(budget) {
  const preset = budget?.preset || "unknown";
  if (preset === "blocked") return "No runtime estimate until usable text exists.";
  if (preset === "smoke") return "Usually seconds to a minute on a local Mac.";
  if (preset === "overfit-check" || preset === "tiny") return "Usually minutes; good for first proof.";
  if (preset === "small-preview") return "Usually longer; start here before hour-scale runs.";
  return "Runtime depends on context, steps, and model size.";
}

function tuningActionText(chatData, evalData) {
  if (chatData?.status === "blocked") return `Fix chat SFT: ${chatData.summary}`;
  if (evalData?.status === "blocked") return `Fix eval: ${evalData.summary}`;
  if (chatData?.status === "caution") return `Improve chat SFT: ${chatData.summary}`;
  if (evalData?.status === "caution") return `Improve eval: ${evalData.summary}`;
  return "Tuning data is ready for a first run.";
}

async function createSftStarter() {
  syncFlightStarterDefaults();
  const packPath = $("flight-pack-path").value.trim();
  const inputPath = $("flight-input-path").value.trim();
  const outPath = $("flight-sft-out-path").value.trim();
  const maxItems = boundedNumberInput("flight-sft-max-items", STARTER_ROW_TARGETS.sft, 8, 300);
  if (!packPath && !inputPath) {
    throw new Error("enter a dataset pack or corpus path first");
  }
  if (!outPath) {
    throw new Error("enter an SFT starter output path");
  }
  $("flight-sft-button").disabled = true;
  $("flight-sft-result").innerHTML = 'CREATING SFT STARTER<span class="cursor"></span>';
  try {
    const report = await postJson("/api/sft/starter", {
      dataset_pack: packPath || null,
      input_path: packPath ? null : inputPath,
      out_path: outPath,
      max_items: maxItems,
      seed: state.detail?.summary?.config?.seed ?? 42,
      force: Boolean($("flight-starter-force")?.checked),
      promote_to_pack: Boolean(packPath),
    });
    state.sftStarter = report;
    const chatPath = report.pack_chat_input || report.output_path || outPath;
    const evalPath = report.pack_eval_input || $("flight-eval-path").value.trim();
    $("flight-chat-path").value = chatPath;
    $("preview-chat-path").value = chatPath;
    if (report.promoted_to_pack && packPath) {
      $("tuning-pack-path").value = packPath;
      $("tuning-chat-path").value = "";
      $("tuning-eval-path").value = "";
      $("editor-pack-path").value = packPath;
      $("editor-chat-path").value = "";
      $("editor-eval-path").value = "";
      $("launch-pack-path").value = packPath;
    } else {
      $("tuning-pack-path").value = "";
      $("tuning-chat-path").value = chatPath;
      if (!$("tuning-eval-path").value.trim()) $("tuning-eval-path").value = evalPath;
      $("editor-pack-path").value = "";
      $("editor-chat-path").value = chatPath;
      if (!$("editor-eval-path").value.trim()) $("editor-eval-path").value = evalPath;
    }
    renderSftStarter(report);
    await refreshDatasetFlightPlanAfterChange();
    if (report.promoted_to_pack && packPath) {
      loadPackEditor().catch((error) => renderPackEditorError(error));
    }
    renderLaunchReadiness();
    renderStartHere();
  } finally {
    $("flight-sft-button").disabled = false;
  }
}

function renderSftStarter(report) {
  if (!report) {
    $("flight-sft-result").innerHTML = "";
    return;
  }
  $("flight-sft-result").innerHTML = `
    <label>SFT STARTER RESULT</label>
    <div class="flight-eval-grid">
      <div><strong>${fmtInt(report.num_rows)}</strong><span>chat rows</span></div>
      <div><strong>${fmtInt(report.num_sentences)}</strong><span>candidate sentences</span></div>
      <div><strong>${escapeHtml(shortPath(report.output_path))}</strong><span>jsonl</span></div>
      <div><strong>${escapeHtml(shortPath(report.report_path))}</strong><span>report</span></div>
    </div>
    <div class="mini-stat-row">
      ${report.promoted_to_pack ? "<span>connected to dataset pack</span>" : ""}
      ${Object.entries(report.categories || {}).map(([name, count]) => `<span>${escapeHtml(name)} ${fmtInt(count)}</span>`).join("")}
    </div>
    ${starterHandoff(report, "sft")}
    <div class="command-tape source-command">
      <div class="command-head">
        <label>SFT STARTER COMMAND</label>
        ${copyCommandButton(report.command)}
      </div>
      <code>${escapeHtml(report.command || "")}</code>
      ${(report.next_actions || []).map((action) => `<p>${escapeHtml(action)}</p>`).join("")}
    </div>
  `;
}

async function createEvalStarter() {
  syncFlightStarterDefaults();
  const packPath = $("flight-pack-path").value.trim();
  const inputPath = $("flight-input-path").value.trim();
  const outPath = $("flight-eval-out-path").value.trim();
  const maxItems = boundedNumberInput("flight-eval-max-items", STARTER_ROW_TARGETS.eval, 4, 200);
  if (!packPath && !inputPath) {
    throw new Error("enter a dataset pack or corpus path first");
  }
  if (!outPath) {
    throw new Error("enter an eval starter output path");
  }
  $("flight-eval-button").disabled = true;
  $("flight-eval-result").innerHTML = 'CREATING EVAL STARTER<span class="cursor"></span>';
  try {
    const report = await postJson("/api/eval/starter", {
      dataset_pack: packPath || null,
      input_path: packPath ? null : inputPath,
      out_path: outPath,
      max_items: maxItems,
      seed: state.detail?.summary?.config?.seed ?? 42,
      force: Boolean($("flight-starter-force")?.checked),
      promote_to_pack: Boolean(packPath),
    });
    state.evalStarter = report;
    const chatPath = report.pack_chat_input || $("flight-chat-path").value.trim();
    const evalPath = report.pack_eval_input || report.output_path || outPath;
    $("flight-eval-path").value = evalPath;
    $("preview-eval-path").value = evalPath;
    if (report.promoted_to_pack && packPath) {
      $("tuning-pack-path").value = packPath;
      $("tuning-chat-path").value = "";
      $("tuning-eval-path").value = "";
      $("editor-pack-path").value = packPath;
      $("editor-chat-path").value = "";
      $("editor-eval-path").value = "";
      $("launch-pack-path").value = packPath;
    } else {
      $("tuning-pack-path").value = "";
      if (!$("tuning-chat-path").value.trim()) $("tuning-chat-path").value = chatPath;
      $("tuning-eval-path").value = evalPath;
      $("editor-pack-path").value = "";
      if (!$("editor-chat-path").value.trim()) $("editor-chat-path").value = chatPath;
      $("editor-eval-path").value = evalPath;
    }
    renderEvalStarter(report);
    await refreshDatasetFlightPlanAfterChange();
    if (report.promoted_to_pack && packPath) {
      loadPackEditor().catch((error) => renderPackEditorError(error));
    }
    renderLaunchReadiness();
    renderStartHere();
  } finally {
    $("flight-eval-button").disabled = false;
  }
}

function renderEvalStarter(report) {
  if (!report) {
    $("flight-eval-result").innerHTML = "";
    return;
  }
  $("flight-eval-result").innerHTML = `
    <label>EVAL STARTER RESULT</label>
    <div class="flight-eval-grid">
      <div><strong>${fmtInt(report.num_rows)}</strong><span>eval rows</span></div>
      <div><strong>${fmtInt(report.num_sentences)}</strong><span>candidate sentences</span></div>
      <div><strong>${escapeHtml(shortPath(report.output_path))}</strong><span>jsonl</span></div>
      <div><strong>${escapeHtml(shortPath(report.report_path))}</strong><span>report</span></div>
    </div>
    <div class="mini-stat-row">
      ${report.promoted_to_pack ? "<span>connected to dataset pack</span>" : ""}
      ${Object.entries(report.categories || {}).map(([name, count]) => `<span>${escapeHtml(name)} ${fmtInt(count)}</span>`).join("")}
    </div>
    ${starterHandoff(report, "eval")}
    <div class="command-tape source-command">
      <div class="command-head">
        <label>EVAL STARTER COMMAND</label>
        ${copyCommandButton(report.command)}
      </div>
      <code>${escapeHtml(report.command || "")}</code>
      ${(report.next_actions || []).map((action) => `<p>${escapeHtml(action)}</p>`).join("")}
    </div>
  `;
}

async function createBenchmarkTuningPack() {
  syncFlightStarterDefaults();
  const packPath = $("flight-pack-path").value.trim();
  if (!packPath) {
    throw new Error("enter a dataset pack first; curated SFT/eval needs a pack to update");
  }
  const maxSftRows = boundedNumberInput("flight-sft-max-items", STARTER_ROW_TARGETS.sft, 32, 2000);
  const maxEvalRows = boundedNumberInput("flight-eval-max-items", STARTER_ROW_TARGETS.eval, 16, 500);
  const source = $("flight-benchmark-source")?.value || "offline";
  const profile = $("flight-benchmark-profile")?.value || "behavior";
  const skillAnswerStyle = $("flight-skill-answer-style")?.value || "direct";
  const chatOut = benchmarkOutputPath(packPath, "chat_benchmark.jsonl");
  const evalOut = benchmarkOutputPath(packPath, "eval_benchmark.jsonl");
  $("flight-benchmark-button").disabled = true;
  $("flight-sft-result").innerHTML = 'CREATING CURATED SFT + EVAL<span class="cursor"></span>';
  $("flight-eval-result").innerHTML = "";
  try {
    const report = await postJson("/api/tuning/benchmark-pack", {
      dataset_pack: packPath,
      chat_out: chatOut,
      eval_out: evalOut,
      sft_rows: maxSftRows,
      eval_rows: maxEvalRows,
      source,
      profile,
      skill_answer_style: skillAnswerStyle,
      seed: state.detail?.summary?.config?.seed ?? 42,
      force: Boolean($("flight-starter-force")?.checked),
      promote_to_pack: true,
    });
    state.benchmarkPack = report;
    const chatPath = report.pack_chat_input || report.chat_output_path || chatOut;
    const evalPath = report.pack_eval_input || report.eval_output_path || evalOut;
    $("flight-chat-path").value = chatPath;
    $("flight-sft-out-path").value = chatPath;
    $("flight-eval-path").value = evalPath;
    $("flight-eval-out-path").value = evalPath;
    $("preview-chat-path").value = chatPath;
    $("preview-eval-path").value = evalPath;
    $("tuning-pack-path").value = packPath;
    $("tuning-chat-path").value = "";
    $("tuning-eval-path").value = "";
    $("editor-pack-path").value = packPath;
    $("editor-chat-path").value = "";
    $("editor-eval-path").value = "";
    $("launch-pack-path").value = packPath;
    if ($("launch-long-run-gate-profile") && report.profile === "release_behavior") {
      $("launch-long-run-gate-profile").value = "first_release";
    } else if ($("launch-long-run-gate-profile")?.value === "first_release") {
      $("launch-long-run-gate-profile").value = "research";
    }
    renderBenchmarkTuningPack(report);
    await refreshDatasetFlightPlanAfterChange();
    loadPackEditor().catch((error) => renderPackEditorError(error));
    renderLaunchReadiness();
    renderStartHere();
  } finally {
    $("flight-benchmark-button").disabled = false;
  }
}

function renderBenchmarkTuningPack(report) {
  if (!report) {
    $("flight-sft-result").innerHTML = "";
    $("flight-eval-result").innerHTML = "";
    return;
  }
  $("flight-sft-result").innerHTML = `
    <label>CURATED SFT + EVAL RESULT</label>
    <div class="flight-eval-grid">
      <div><strong>${fmtInt(report.sft_rows)}</strong><span>chat rows</span></div>
      <div><strong>${fmtInt(report.eval_rows)}</strong><span>eval rows</span></div>
      <div><strong>${escapeHtml(report.profile || "full")}</strong><span>profile</span></div>
      <div><strong>${escapeHtml(report.skill_answer_style || "direct")}</strong><span>skill answers</span></div>
      <div><strong>${escapeHtml(report.source_status || report.source_mode || "offline")}</strong><span>source</span></div>
      <div><strong>${escapeHtml(report.contamination?.status || "unknown")}</strong><span>contamination</span></div>
      <div><strong>${escapeHtml(shortPath(report.chat_output_path))}</strong><span>SFT jsonl</span></div>
      <div><strong>${escapeHtml(shortPath(report.eval_output_path))}</strong><span>eval jsonl</span></div>
    </div>
    <div class="starter-handoff ready">
      <strong>NANOCHAT-STYLE TRACK CONNECTED</strong>
      <span>Base training still uses your selected corpus. Chat SFT now uses a curated instruction mix, and eval uses held-out benchmark/refusal rows.</span>
      <em>Do not copy held-out eval prompts into SFT; use failed categories to create separate practice rows.</em>
    </div>
    <div class="mini-stat-row">
      ${Object.entries(report.chat_categories || {}).map(([name, count]) => `<span>SFT ${escapeHtml(name)} ${fmtInt(count)}</span>`).join("")}
    </div>
    <div class="mini-stat-row">
      ${Object.entries(report.source_datasets || {}).map(([name, count]) => `<span>${escapeHtml(name)} ${fmtInt(count)}</span>`).join("")}
    </div>
    ${report.fallback_reason ? `<div class="notice caution">SOURCE FALLBACK: ${escapeHtml(report.fallback_reason)}</div>` : ""}
    <div class="command-tape source-command">
      <div class="command-head">
        <label>BENCHMARK PACK COMMAND</label>
        ${copyCommandButton(report.command)}
      </div>
      <code>${escapeHtml(report.command || "")}</code>
      ${(report.next_actions || []).map((action) => `<p>${escapeHtml(action)}</p>`).join("")}
    </div>
  `;
  $("flight-eval-result").innerHTML = `
    <label>CURATED EVAL MIX</label>
    <div class="mini-stat-row">
      ${Object.entries(report.eval_categories || {}).map(([name, count]) => `<span>${escapeHtml(name)} ${fmtInt(count)}</span>`).join("")}
    </div>
    ${renderTuningPreflight(report.chat_data, report.eval_data)}
  `;
}

function benchmarkOutputPath(packPath, filename) {
  const index = packPath.lastIndexOf("/");
  const dir = index >= 0 ? packPath.slice(0, index + 1) : "";
  return `${dir}${filename}`;
}

function starterHandoff(report, kind) {
  const isEval = kind === "eval";
  const connected = Boolean(report.promoted_to_pack);
  const path = isEval
    ? report.pack_eval_input || report.output_path
    : report.pack_chat_input || report.output_path;
  const title = connected ? "CONNECTED TO TRAINING PACK" : "SAVED BUT NOT CONNECTED";
  const body = connected
    ? `${isEval ? "Eval" : "SFT"} now points at ${shortPath(path)}. Open the JSONL editor next, edit the starter rows, then run tuning inspection before training.`
    : `${isEval ? "Eval" : "SFT"} was written to ${shortPath(path)}. Because no dataset pack was selected, use this path explicitly or create a pack before training.`;
  const caution = isEval
    ? "Keep eval held out; do not copy these prompts into SFT."
    : "Use non-eval examples here; do not train on held-out eval prompts.";
  return `
    <div class="starter-handoff ${connected ? "ready" : "caution"}">
      <strong>${escapeHtml(title)}</strong>
      <span>${escapeHtml(body)}</span>
      <em>${escapeHtml(caution)}</em>
    </div>
  `;
}

function applyFlightPlanToLauncher(quiet = false) {
  const report = state.datasetFlightPlan;
  if (!report) {
    flashStatus("APPLY FAULT. | Check a dataset first.");
    return;
  }
  if (report.dataset_pack) {
    $("launch-pack-path").value = report.dataset_pack;
    const currentRunName = $("launch-run-name").value.trim();
    if (!currentRunName || runNameExists(currentRunName)) {
      $("launch-run-name").value = uniqueRunName(suggestedRunName(report.dataset_pack));
    }
  }
  const budget = report.budget || {};
  const preset = budget.preset === "small-preview" ? "small-local" : budget.preset === "overfit-check" ? "tiny" : budget.preset;
  if (state.runPresets[preset]) {
    $("launch-preset").value = preset;
    applyLaunchPreset(true);
  }
  $("launch-context-size").value = budget.suggested_context_size || $("launch-context-size").value;
  $("launch-base-steps").value = budget.suggested_base_steps || $("launch-base-steps").value;
  $("launch-sft-steps").value = Math.max(60, Number(budget.suggested_base_steps || 30));
  applyBenchmarkSftDefaults(report);
  $("launch-min-score").value = report.min_quality_score ?? ($("flight-min-score").value || 0);
  if (!quiet) flashStatus("APPLIED PLAN. | Review launcher values before starting the run.");
  renderFlightCoach(report);
  renderSmokeReadiness(report);
  renderLaunchReadiness();
  renderStartHere();
}

function applyBenchmarkSftDefaults(report = state.datasetFlightPlan) {
  const packPath = report?.dataset_pack || $("launch-pack-path")?.value.trim() || "";
  const benchmark = state.benchmarkPack;
  if (!benchmark || !packPath || benchmark.dataset_pack !== packPath) return false;
  $("launch-sft-steps").value = String(BENCHMARK_SFT_DEFAULTS.sftSteps);
  $("launch-sft-learning-rate").value = String(BENCHMARK_SFT_DEFAULTS.sftLearningRate);
  $("launch-sft-early-stop-patience").value = String(BENCHMARK_SFT_DEFAULTS.sftPatience);
  $("launch-sft-sampling").value = "category_sqrt";
  return true;
}

function applyMuonEmaTrial() {
  $("launch-base-optimizer").value = MUON_EMA_TRIAL_DEFAULTS.baseOptimizer;
  $("launch-sft-optimizer").value = MUON_EMA_TRIAL_DEFAULTS.sftOptimizer;
  $("launch-base-muon-learning-rate").value = String(MUON_EMA_TRIAL_DEFAULTS.baseMuonLearningRate);
  $("launch-sft-muon-learning-rate").value = String(MUON_EMA_TRIAL_DEFAULTS.sftMuonLearningRate);
  $("launch-base-ema-decay").value = String(MUON_EMA_TRIAL_DEFAULTS.baseEmaDecay);
  $("launch-sft-ema-decay").value = String(MUON_EMA_TRIAL_DEFAULTS.sftEmaDecay);
  renderLaunchReadiness();
  renderLaunchCommandPreview();
  flashStatus("APPLIED MUON/EMA TRIAL. | Compare it against the AdamW baseline.");
}

function renderDatasetFlightPlanError(error) {
  $("flight-check-button").disabled = false;
  $("flight-status").textContent = "DATASET CHECK FAULT";
  renderFlightCoach(null);
  renderSmokeReadiness(null);
  $("flight-plan").innerHTML = "";
  $("flight-command").innerHTML = "";
  $("flight-sft-result").innerHTML = "";
  $("flight-eval-result").innerHTML = `<div class="notice">FAULT: ${escapeHtml(error.message)}</div>`;
}

function renderSftStarterError(error) {
  $("flight-sft-button").disabled = false;
  $("flight-sft-result").innerHTML = `<div class="notice">SFT STARTER FAULT: ${escapeHtml(error.message)}</div>`;
}

function renderEvalStarterError(error) {
  $("flight-eval-button").disabled = false;
  $("flight-eval-result").innerHTML = `<div class="notice">EVAL STARTER FAULT: ${escapeHtml(error.message)}</div>`;
}

function renderBenchmarkPackError(error) {
  $("flight-benchmark-button").disabled = false;
  $("flight-sft-result").innerHTML = `<div class="notice">CURATED PACK FAULT: ${escapeHtml(error.message)}</div>`;
}

function suggestedSftStarterPath(path) {
  return suggestedStarterPath(path, "chat");
}

function suggestedEvalStarterPath(path) {
  return suggestedStarterPath(path, "eval");
}

function suggestedStarterPath(path, kind) {
  const text = String(path || "").trim();
  if (!text) return `my_pack/${kind}_starter.jsonl`;
  const slash = text.lastIndexOf("/");
  const dir = slash >= 0 ? text.slice(0, slash + 1) : "";
  const file = slash >= 0 ? text.slice(slash + 1) : text;
  let stem = file.replace(/\.jsonl$/i, "").replace(/\.json$/i, "") || kind;
  if (stem === "dataset_pack" || stem === "corpus_recipe") stem = kind;
  stem = stem.replace(/(_starter)+$/i, "_starter");
  if (/_starter$/i.test(stem)) return `${dir}${stem}.jsonl`;
  return `${dir}${stem}_starter.jsonl`;
}

async function previewCorpusSources() {
  const packPath = $("preview-pack-path").value.trim();
  const recipePath = $("preview-recipe-path").value.trim();
  const inputPath = $("preview-input-path").value.trim();
  const chatInput = $("preview-chat-path").value.trim();
  const evalInput = $("preview-eval-path").value.trim();
  const minQualityScore = Number($("preview-min-score").value || 0);
  if (!packPath && !recipePath && !inputPath) {
    throw new Error("enter a dataset pack, recipe path, or input path");
  }

  $("preview-corpus-button").disabled = true;
  $("source-preview-status").innerHTML = 'PREVIEWING SOURCES<span class="cursor"></span>';
  $("source-preview-readiness").innerHTML = "";
  $("source-preview-budget").innerHTML = "";
  $("source-preview-command").innerHTML = "";
  $("source-preview-tuning").innerHTML = "";
  $("source-preview-stats").innerHTML = "";
  $("source-preview-files").innerHTML = "";
  $("source-preview-text").textContent = "";
  try {
    const report = await postJson("/api/corpus/preview", {
      dataset_pack: packPath || null,
      recipe_path: packPath ? null : recipePath || null,
      input_path: packPath ? null : inputPath || null,
      chat_input: packPath ? null : chatInput || null,
      eval_input: packPath ? null : evalInput || null,
      preview_chars: 1400,
      min_quality_score: minQualityScore,
    });
    state.corpusSourcePreview = report;
    state.datasetFlightPlan = report;
    state.tuningInspection = tuningInspectionFromPreview(report);
    if (report.dataset_pack) {
      $("launch-pack-path").value = report.dataset_pack;
      $("launch-min-score").value = report.min_quality_score ?? minQualityScore;
      if (!$("launch-run-name").value) $("launch-run-name").value = suggestedRunName(report.dataset_pack);
    }
    renderLaunchReadiness();
    renderDatasetFlightPlan(report);
    renderCorpusSourcePreview(report);
    renderTuningInspection(state.tuningInspection);
    renderStartHere();
  } finally {
    $("preview-corpus-button").disabled = false;
  }
}

function renderCorpusSourcePreview(report) {
  if (!report) {
    $("source-preview-status").textContent = "NO SOURCE PREVIEW REQUESTED.";
    $("source-preview-readiness").innerHTML = "";
    $("source-preview-budget").innerHTML = "";
    $("source-preview-command").innerHTML = "";
    $("source-preview-tuning").innerHTML = "";
    $("source-preview-stats").innerHTML = "";
    $("source-preview-files").innerHTML = '<div class="empty">NO PREVIEW PLAN LOADED.</div>';
    $("source-preview-text").textContent = "READY.";
    return;
  }
  const files = report.files || [];
  const included = files.filter((file) => file.included);
  const skipped = files.length - included.length;
  const stats = report.stats || {};
  const qualityScore = averageQualityScore(included);
  $("source-preview-status").textContent =
    `${readinessBadge(report.readiness)} | ${fmtInt(included.length)} INCLUDED | ${fmtInt(skipped)} SKIPPED | Q${qualityScore ?? "--"} | ${fmtInt(stats.num_characters)} CHARS`;
  $("source-preview-readiness").innerHTML = renderReadiness(report.readiness);
  $("source-preview-budget").innerHTML = renderBudget(report.budget);
  $("source-preview-command").innerHTML = renderTrainingCommand(report.training_command);
  $("source-preview-tuning").innerHTML = renderTuningPreflight(report.chat_data, report.eval_data);
  $("source-preview-stats").innerHTML = statCards([
    ["Pack", report.dataset_pack || "none"],
    ["Input", report.input_path || "unknown"],
    ["Recipe", report.recipe_path || "none"],
    ["Chat SFT", report.training_command?.chat_input || "examples/tiny_chat.jsonl"],
    ["Eval", report.training_command?.eval_input || "examples/tiny_eval.jsonl"],
    ["Min score", report.min_quality_score ?? 0],
    ["Avg score", qualityScore ?? "--"],
    ["Files", fmtInt(stats.num_files)],
    ["Documents", fmtInt(stats.num_documents)],
    ["Characters", fmtInt(stats.num_characters)],
    ["Lines", fmtInt(stats.num_lines)],
  ]);
  $("source-preview-files").innerHTML = renderCorpusFiles(files);
  $("source-preview-text").textContent = report.preview || "(EMPTY)";
}

function averageQualityScore(files) {
  const scores = files
    .map((file) => Number(file.quality_score))
    .filter((score) => Number.isFinite(score));
  if (!scores.length) return null;
  return Math.round(scores.reduce((total, score) => total + score, 0) / scores.length);
}

function tuningInspectionFromPreview(report) {
  const chatStatus = report.chat_data?.status || "unknown";
  const evalStatus = report.eval_data?.status || "unknown";
  const status = chatStatus === "blocked" || evalStatus === "blocked"
    ? "blocked"
    : chatStatus === "caution" || evalStatus === "caution"
      ? "caution"
      : "ready";
  return {
    status,
    summary: status === "ready"
      ? "Chat SFT and eval files look ready for a tiny run."
      : status === "caution"
        ? "Files are readable, but improve them before trusting a run."
        : "Fix blocked chat/eval data before training.",
    training_ready: status === "ready",
    can_train: status !== "blocked",
    dataset_pack: report.dataset_pack || null,
    chat_input: report.training_command?.chat_input || null,
    eval_input: report.training_command?.eval_input || null,
    chat_data: report.chat_data,
    eval_data: report.eval_data,
    next_actions: compactTuningActions(report.chat_data, report.eval_data, Boolean(report.dataset_pack)),
    preview_command: report.dataset_pack
      ? shellCommand(["PYTHONPATH=src", "python", "-m", "picochat.cli", "data", "preview", "--dataset-pack", report.dataset_pack])
      : null,
  };
}

function compactTuningActions(chatData, evalData, fromPack) {
  const actions = [];
  [
    ["Chat SFT", chatData],
    ["Eval", evalData],
  ].forEach(([label, report]) => {
    if (report?.status === "blocked") {
      actions.push(`Fix ${label}: ${report.summary}`);
    } else if (report?.status === "caution") {
      actions.push(`Improve ${label}: ${report.summary}`);
    }
  });
  if (!actions.length) actions.push("Tuning data is ready for the next tiny run.");
  if (fromPack) actions.push("Run Source Preview next to inspect corpus readiness and get the training command.");
  return actions;
}

function readinessBadge(readiness) {
  if (!readiness) return "READINESS --";
  return `READINESS ${String(readiness.status || "--").toUpperCase()}`;
}

function renderReadiness(readiness) {
  if (!readiness) return "";
  const checks = readiness.checks || [];
  return `
    <div class="readiness-summary ${escapeHtml(readiness.status || "unknown")}">
      <strong>${escapeHtml(readinessBadge(readiness))}</strong>
      <span>${escapeHtml(readiness.summary || "")}</span>
    </div>
    ${checks.map((check) => `
      <div class="readiness-row ${escapeHtml(check.status)}">
        <strong>${escapeHtml(check.name)}</strong>
        <span>${escapeHtml(check.metric)} / ${escapeHtml(check.threshold)}</span>
        <p>${escapeHtml(check.message)}</p>
      </div>
    `).join("")}
  `;
}

function renderBudget(budget) {
  if (!budget) return "";
  return `
    <label>TRAINING BUDGET ESTIMATE</label>
    <div class="budget-grid">
      <div><strong>${escapeHtml(budget.preset)}</strong><span>preset</span></div>
      <div><strong>${fmtInt(budget.estimated_tokens)}</strong><span>char-token est</span></div>
      <div><strong>${escapeHtml(budget.suggested_context_size)}</strong><span>ctx</span></div>
      <div><strong>${fmtInt(budget.estimated_windows)}</strong><span>windows</span></div>
      <div><strong>${escapeHtml(budget.suggested_batch_size)}</strong><span>batch</span></div>
      <div><strong>${escapeHtml(budget.suggested_base_steps)}</strong><span>base steps</span></div>
      <div><strong>${fmtLoss(budget.estimated_passes)}</strong><span>rough passes</span></div>
    </div>
    <p>${escapeHtml(budget.note || "")}</p>
  `;
}

function renderTrainingCommand(trainingCommand) {
  if (!trainingCommand) return "";
  const command = trainingCommand.command || "# Fix corpus readiness issues before running training.";
  return `
    <div class="command-head">
      <label>SUGGESTED RUN COMMAND</label>
      ${trainingCommand.command ? copyCommandButton(trainingCommand.command) : ""}
    </div>
    <div class="command-meta">
      ${trainingCommand.dataset_pack ? `<span>PACK ${escapeHtml(trainingCommand.dataset_pack)}</span>` : ""}
      <span>CHAT ${escapeHtml(trainingCommand.chat_input || "--")}</span>
      <span>EVAL ${escapeHtml(trainingCommand.eval_input || "--")}</span>
    </div>
    <code>${escapeHtml(command)}</code>
    <p>${escapeHtml(trainingCommand.note || "")}</p>
  `;
}

function renderTuningPreflight(chatData, evalData) {
  if (!chatData && !evalData) return "";
  return `
    <div class="tuning-card ${escapeHtml(chatData?.status || "unknown")}">
      <div>
        <label>CHAT SFT PREFLIGHT</label>
        <strong>${escapeHtml(String(chatData?.status || "--").toUpperCase())}</strong>
      </div>
      <p>${escapeHtml(chatData?.summary || "")}</p>
      <div class="mini-stat-row">
        <span>${fmtInt(chatData?.num_examples)} usable</span>
        <span>${fmtInt(chatData?.invalid_rows)} invalid</span>
        <span>${fmtPercent(chatData?.duplicate_user_rate || 0)} dup prompts</span>
        <span>${fmtInt(chatData?.near_duplicate_user_pairs || 0)} near dup</span>
        <span>${escapeHtml(chatData?.curriculum_label || "--")}</span>
      </div>
      <div class="mini-stat-row">
        <span>entropy ${fmtLoss(chatData?.category_entropy_normalized || 0)}</span>
        <span>avg answer words ${fmtLoss(chatData?.assistant_length_distribution?.avg_words || 0)}</span>
      </div>
      ${renderCategoryCounts(chatData?.categories)}
      ${renderTemplateCounts(chatData?.template_families)}
      ${renderQualityWarnings(chatData?.quality_warnings || [])}
      ${renderIssues(chatData?.issues || [])}
      ${renderChatPreview(chatData?.preview || [])}
    </div>
    <div class="tuning-card ${escapeHtml(evalData?.status || "unknown")}">
      <div>
        <label>EVAL PREFLIGHT</label>
        <strong>${escapeHtml(String(evalData?.status || "--").toUpperCase())}</strong>
      </div>
      <p>${escapeHtml(evalData?.summary || "")}</p>
      <div class="mini-stat-row">
        <span>${fmtInt(evalData?.num_items)} items</span>
        <span>${fmtInt(evalData?.answerable_items)} answerable</span>
        <span>${fmtInt(evalData?.unanswerable_items)} unanswerable</span>
        <span>${fmtInt((evalData?.must_include_rules || 0) + (evalData?.must_include_any_groups || 0) + (evalData?.must_not_include_rules || 0))} rules</span>
        <span>${fmtInt(evalData?.near_duplicate_user_pairs || 0)} near dup</span>
        <span>${escapeHtml(evalData?.curriculum_label || "--")}</span>
      </div>
      <div class="mini-stat-row">
        <span>entropy ${fmtLoss(evalData?.category_entropy_normalized || 0)}</span>
        <span>avg answer words ${fmtLoss(evalData?.answer_length_distribution?.avg_words || 0)}</span>
      </div>
      ${renderCategoryCounts(evalData?.categories)}
      ${renderCategoryCounts(evalData?.heldout_categories)}
      ${renderTemplateCounts(evalData?.template_families)}
      ${renderSplitCounts(evalData?.splits)}
      ${renderQualityWarnings(evalData?.quality_warnings || [])}
      ${renderIssues(evalData?.issues || [])}
      ${renderEvalPreview(evalData?.preview || [])}
    </div>
  `;
}

function renderCategoryCounts(categories) {
  const entries = Object.entries(categories || {});
  if (!entries.length) return "";
  return `
    <div class="mini-stat-row category-counts">
      ${entries.map(([name, count]) => `<span>${escapeHtml(name)} ${fmtInt(count)}</span>`).join("")}
    </div>
  `;
}

function renderSplitCounts(splits) {
  const entries = Object.entries(splits || {});
  if (!entries.length) return "";
  return `
    <div class="mini-stat-row category-counts">
      ${entries.map(([name, count]) => `<span>split:${escapeHtml(name)} ${fmtInt(count)}</span>`).join("")}
    </div>
  `;
}

function renderTemplateCounts(families) {
  const entries = Object.entries(families || {});
  if (!entries.length) return "";
  return `
    <div class="mini-stat-row category-counts">
      ${entries.slice(0, 8).map(([name, count]) => `<span>template ${escapeHtml(name)} ${fmtInt(count)}</span>`).join("")}
    </div>
  `;
}

function renderQualityWarnings(warnings) {
  if (!warnings?.length) return "";
  return `
    <div class="notice-list">
      ${warnings.slice(0, 4).map((warning) => `<p>${escapeHtml(warning)}</p>`).join("")}
    </div>
  `;
}

function renderIssues(issues) {
  if (!issues.length) return '<div class="mini-ok">NO SCHEMA ISSUES FOUND.</div>';
  return `
    <div class="issue-list">
      ${issues.slice(0, 4).map((issue) => `
        <div><strong>LINE ${escapeHtml(issue.line)}</strong> ${escapeHtml(issue.message)}</div>
      `).join("")}
    </div>
  `;
}

function renderChatPreview(rows) {
  if (!rows.length) return "";
  return `
    <div class="preview-list">
      ${rows.map((row) => `
        <div><strong>U</strong> ${escapeHtml(compactPreview(row.user || "", 90))}</div>
        <div><strong>A</strong> ${escapeHtml(compactPreview(row.assistant || "", 90))}</div>
      `).join("")}
    </div>
  `;
}

function renderEvalPreview(rows) {
  if (!rows.length) return "";
  return `
    <div class="preview-list">
      ${rows.map((row) => `
        <div><strong>${escapeHtml(row.category || "eval")}</strong> ${escapeHtml(compactPreview(row.user || "", 120))}</div>
      `).join("")}
    </div>
  `;
}

function renderCorpusSourcePreviewError(error) {
  $("preview-corpus-button").disabled = false;
  $("source-preview-status").textContent = "SOURCE PREVIEW FAULT";
  $("source-preview-readiness").innerHTML = "";
  $("source-preview-budget").innerHTML = "";
  $("source-preview-command").innerHTML = "";
  $("source-preview-tuning").innerHTML = "";
  $("source-preview-stats").innerHTML = "";
  $("source-preview-files").innerHTML = "";
  $("source-preview-text").textContent = `FAULT: ${error.message}`;
}

function shortPath(path) {
  const pieces = String(path || "").split("/");
  return pieces.slice(-3).join("/");
}

function renderTokenizer() {
  const tokenizer = state.detail?.tokenizer_detail;
  const summary = state.detail?.summary;
  $("tokenizer-summary").textContent = tokenizer
    ? `${(tokenizer.type || summary?.tokenizer?.tokenizer_type || "TOK").toUpperCase()} ${tokenizer.vocab_size} | SPECIAL ${tokenizer.special_tokens.length}`
    : "TOK --";
  $("special-tokens").innerHTML = (tokenizer?.special_tokens || [])
    .map((token) => `<span class="token-pill special">${escapeHtml(token)}:${tokenizer.token_to_id[token]}</span>`)
    .join("");
  if (!$("tokenizer-input").value && summary?.config?.out_dir) {
    $("tokenizer-input").value = "Picochat";
  }
  animateTokenizer();
}

function animateTokenizer() {
  const tokenizer = state.detail?.tokenizer_detail;
  if (!tokenizer?.token_to_id) {
    $("token-stream").innerHTML = 'TOKENIZER MAP NOT FOUND<span class="cursor"></span>';
    return;
  }
  clearInterval(state.tokenTimer);
  const text = $("tokenizer-input").value;
  const units = tokenizer.type === "byte"
    ? byteTokenUnits(text)
    : tokenizer.type === "bpe"
      ? bpeTokenUnits(text, tokenizer)
      : charTokenUnits(text);
  const stream = $("token-stream");
  stream.innerHTML = "";
  let index = 0;
  state.tokenTimer = setInterval(() => {
    if (index >= units.length) {
      clearInterval(state.tokenTimer);
      stream.insertAdjacentHTML("beforeend", '<span class="cursor"></span>');
      return;
    }
    const unit = units[index];
    const id = tokenizer.token_to_id[unit.token] ?? tokenizer.token_to_id["<unk>"];
    stream.insertAdjacentHTML(
      "beforeend",
      `<span class="token-step"><b>${escapeHtml(unit.label)}</b><em>${id}</em></span>`
    );
    index += 1;
  }, 80);
}

function charTokenUnits(text) {
  return [...text].map((char) => ({
    token: char,
    label: tokenLabel(char),
  }));
}

function byteTokenUnits(text) {
  return Array.from(new TextEncoder().encode(text)).map((byte) => {
    const hex = byte.toString(16).padStart(2, "0");
    const printable = byte >= 32 && byte <= 126 ? String.fromCharCode(byte) : `0x${hex}`;
    return { token: `<byte:${hex}>`, label: printable === " " ? "space" : printable };
  });
}

function bpeTokenUnits(text, tokenizer) {
  const output = [];
  for (const piece of bpePretokenPieces(text, tokenizer)) {
    let units = [...piece];
    for (const pair of tokenizer.merges || []) {
      if (!Array.isArray(pair) || pair.length !== 2) continue;
      const [left, right] = pair;
      const next = [];
      for (let index = 0; index < units.length; index += 1) {
        if (index < units.length - 1 && units[index] === left && units[index + 1] === right) {
          next.push(left + right);
          index += 1;
        } else {
          next.push(units[index]);
        }
      }
      units = next;
    }
    output.push(...units);
  }
  return output.map((token) => ({ token, label: tokenLabel(token) }));
}

function bpePretokenPieces(text, tokenizer) {
  if (tokenizer.pretokenizer !== "regex") return text ? [text] : [];
  const pattern = /'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\d+(?:[.,]\d+)*%?| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/giu;
  const pieces = [];
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    const piece = match[0];
    const start = match.index ?? cursor;
    if (start > cursor) pieces.push(...[...text.slice(cursor, start)]);
    pieces.push(piece);
    cursor = start + piece.length;
  }
  if (cursor < text.length) pieces.push(...[...text.slice(cursor)]);
  return pieces.filter(Boolean);
}

function tokenLabel(token) {
  return token === " " ? "space" : token === "\n" ? "\\n" : token;
}

function renderTraining() {
  const detail = state.detail;
  const baseLosses = detail?.base_report?.losses || [];
  const sftLosses = detail?.sft_report?.losses || [];
  const baseMemorization = detail?.base_report?.memorization;
  const sftLast = sftLosses.at(-1);
  const overfit = sftLast && sftLast.val_loss > sftLast.train_loss + 1;
  const memorized = baseMemorization?.status === "high" || overfit;
  $("training-badge").textContent = memorized ? "MEMORIZATION WARNING" : "LOSS TRACE READY";
  $("training-badge").classList.toggle("warning", Boolean(memorized));
  $("base-loss-chart").textContent = asciiLossChart(baseLosses);
  $("sft-loss-chart").textContent = asciiLossChart(sftLosses);
  $("training-table").innerHTML = trainingDiagnostics(detail) + trainingRows(baseLosses, sftLosses);
}

function asciiLossChart(losses) {
  if (!losses.length) return "NO LOSS ARTIFACT FOUND.";
  const vals = losses.flatMap((row) => [row.train_loss, row.val_loss]);
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const width = 28;
  return losses.map((row) => {
    const train = bar(row.train_loss, min, max, width, "█");
    const val = bar(row.val_loss, min, max, width, "▓");
    return `STEP ${String(row.step).padStart(4, "0")}  TRN ${train} ${fmtLoss(row.train_loss)}\n           VAL ${val} ${fmtLoss(row.val_loss)}`;
  }).join("\n");
}

function bar(value, min, max, width, glyph) {
  const spread = Math.max(0.0001, max - min);
  const count = Math.max(1, Math.round(((value - min) / spread) * width));
  return glyph.repeat(count).padEnd(width, ".");
}

function trainingRows(baseLosses, sftLosses) {
  const rows = [
    ["BASE", baseLosses.at(-1)],
    ["SFT", sftLosses.at(-1)],
  ];
  return `
    <label>LAST CHECKPOINT ROWS</label>
    <table>
      <thead><tr><th>Stage</th><th>Step</th><th>Train</th><th>Val</th><th>BPB</th><th>Gap</th></tr></thead>
      <tbody>
        ${rows.map(([name, row]) => row ? `
          <tr>
            <td>${name}</td>
            <td>${row.step}</td>
            <td>${fmtLoss(row.train_loss)}</td>
            <td>${fmtLoss(row.val_loss)}</td>
            <td>${fmtLoss(row.val_bpb)}</td>
            <td>${fmtLoss(row.val_loss - row.train_loss)}</td>
          </tr>
        ` : "").join("")}
      </tbody>
    </table>
  `;
}

function trainingDiagnostics(detail) {
  const dataset = detail?.base_report?.dataset || {};
  const coverage = detail?.base_report?.coverage || {};
  const memory = detail?.base_report?.memorization;
  const loss = detail?.base_report?.loss_diagnostics;
  if (!memory && !loss && !dataset.split_mode && !coverage.actual_steps) return "";
  return `
    <label>LEARNING CHECKS</label>
    <div class="stat-grid">
      ${statCards([
        ["Split", dataset.split_mode || "--"],
        ["Held-out docs", dataset.val_documents == null ? "--" : `${dataset.val_documents}/${dataset.num_documents}`],
        ["Loss status", loss?.status || "--"],
        ["Stop", detail?.base_report?.stop_reason || "--"],
        ["Train epochs", fmtLoss(coverage.estimated_train_epochs)],
        ["Canaries", dataset.canary_values?.length ?? 0],
        ["Copy risk", memory?.status || "--"],
        ["Train copy", memory ? fmtPercent(memory.train_overlap_rate) : "--"],
        ["Held-out overlap", memory ? fmtPercent(memory.validation_overlap_rate) : "--"],
      ])}
    </div>
    <p class="notice">${escapeHtml(memory?.summary || loss?.summary || "Diagnostics will appear after the base training report is written.")}</p>
  `;
}

function renderGenerationDeck() {
  $("seed-display").textContent = `SEED ${padSeed(state.detail?.summary?.config?.seed)}`;
  $("generation-output").textContent = "READY";
  $("logprob-bars").innerHTML = '<div class="notice">LIVE /api/generate READY. LOGPROBS WILL APPEAR AFTER GENERATION.</div>';
}

function renderReportList() {
  const reports = state.detail?.reports || {};
  const rows = [
    ["summary", "SUMMARY"],
    ["honesty", "DATA HONESTY"],
    ["base", "BASE TRAINING"],
    ["sft", "CHAT SFT"],
    ["eval", "EVAL"],
  ];
  $("report-select").value = state.activeReport;
  $("report-list").innerHTML = rows.map(([key, label]) => {
    const status = reports[key]?.exists ? "ready" : "missing";
    return `
      <div class="report-item ${status} ${key === state.activeReport ? "selected" : ""}">
        <strong>${escapeHtml(label)}</strong>
        <p>${escapeHtml(status.toUpperCase())}</p>
        <small>${escapeHtml(reports[key]?.path || "no path")}</small>
      </div>
    `;
  }).join("");
  const ready = Object.values(reports).filter((report) => report.exists).length;
  $("report-status").textContent = `${ready}/5 REPORTS`;
}

function renderCompareControls() {
  if (!$("compare-run-list")) return;
  $("compare-run-list").innerHTML = state.runs.map((run) => `
    <label class="compare-option">
      <input type="checkbox" data-compare-run="${escapeHtml(run.name)}" ${state.compareRuns.includes(run.name) ? "checked" : ""}>
      <span>${escapeHtml(run.name)}</span>
      <small>${escapeHtml(run.eval_score)} | ${fmtPercent(run.pass_rate)} | CTX ${escapeHtml(run.context_size)}</small>
    </label>
  `).join("");
  $("compare-status").textContent = `${state.compareRuns.length} SELECTED`;
}

function resetCompareLearningPanels() {
  state.comparison = null;
  if ($("experiment-notebook")) {
    $("experiment-notebook").textContent = "RUN SET CHANGED. PRESS COMPARE SELECTED TO REBUILD THE EXPERIMENT NOTEBOOK.";
  }
  if ($("config-diff")) {
    $("config-diff").textContent = "RUN SET CHANGED. PRESS COMPARE SELECTED TO REBUILD CONFIG DIFFS.";
  }
}

async function loadComparison() {
  if (!state.compareRuns.length) {
    throw new Error("select at least one run");
  }
  $("compare-status").textContent = "COMPARING";
  const query = state.compareRuns.map((run) => `run=${encodeURIComponent(run)}`).join("&");
  const [comparison, detailEntries] = await Promise.all([
    fetchJson(`/api/compare?${query}`),
    Promise.all(state.compareRuns.map(async (run) => {
      try {
        return [run, await fetchJson(`/api/run?name=${encodeURIComponent(run)}`)];
      } catch {
        return [run, null];
      }
    })),
  ]);
  state.compareDetails = Object.fromEntries(detailEntries);
  state.comparison = comparison;
  renderComparison(comparison);
  renderStartHere();
}

function renderComparison(comparison) {
  $("compare-status").textContent = `BEST ${comparison.best_run}`;
  $("compare-summary").innerHTML = compareSummary(comparison);
  $("experiment-notebook").innerHTML = experimentNotebook(comparison);
  $("config-diff").innerHTML = configDiff(comparison);
  $("compare-table").innerHTML = `
    <label>COMPARISON TABLE</label>
    <table>
      <thead>
        <tr>
          <th>Run</th>
          <th>Tok</th>
          <th>Eval</th>
          <th>Pass</th>
          <th>Domain</th>
          <th>Refusal</th>
          <th>Base BPB</th>
          <th>SFT BPB</th>
          <th>Base Val</th>
          <th>SFT Val</th>
          <th>Best</th>
          <th>Stop</th>
          <th>Mem</th>
          <th>Params</th>
          <th>Ctx</th>
          <th>Trunc</th>
          <th>Skip</th>
        </tr>
      </thead>
      <tbody>
        ${comparison.rows.map((row) => `
          <tr class="${row.run === comparison.best_run ? "best-row" : ""}">
            <td>${escapeHtml(row.run)}</td>
            <td>${escapeHtml(row.tokenizer_type || "--")}</td>
            <td>${escapeHtml(row.eval_score)}</td>
            <td>${fmtPercent(row.pass_rate)}</td>
            <td>${fmtPercent(row.domain_pass_rate)}</td>
            <td>${fmtPercent(row.refusal_pass_rate)}</td>
            <td>${fmtLoss(row.base_val_bpb)}</td>
            <td>${fmtLoss(row.sft_val_bpb)}</td>
            <td>${fmtLoss(row.base_val_loss)}</td>
            <td>${fmtLoss(row.sft_val_loss)}</td>
            <td>${escapeHtml(compareBestSteps(row))}</td>
            <td>${escapeHtml(compareStopReasons(row))}</td>
            <td>${escapeHtml(row.memorization_status || "--")}</td>
            <td>${fmtInt(row.num_parameters)}</td>
            <td>${escapeHtml(row.context_size)}</td>
            <td>${escapeHtml(row.truncated_examples)}</td>
            <td>${escapeHtml(row.skipped_long_examples ?? 0)}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function compareSummary(comparison) {
  const rows = [...comparison.rows].sort((a, b) => b.pass_rate - a.pass_rate);
  const best = rows.find((row) => row.run === comparison.best_run) || rows[0];
  const bestBaseBpb = rows.find((row) => row.run === comparison.best_base_bpb_run);
  const bestSftBpb = rows.find((row) => row.run === comparison.best_sft_bpb_run);
  const baseline = rows[0]?.run === best?.run ? rows[1] : rows[0];
  if (!best) return "NO COMPARISON ROWS.";
  const passDelta = baseline ? best.pass_rate - baseline.pass_rate : 0;
  const sftBpbDelta = baseline && best.sft_val_bpb != null && baseline.sft_val_bpb != null
    ? best.sft_val_bpb - baseline.sft_val_bpb
    : null;
  return `
    <div class="compare-cards">
      <div class="pipeline-stat">
        <label>Best eval</label>
        <span>${escapeHtml(best.run)}</span>
      </div>
      <div class="pipeline-stat">
        <label>Pass rate</label>
        <span>${fmtPercent(best.pass_rate)}</span>
      </div>
      <div class="pipeline-stat">
        <label>Pass delta</label>
        <span>${baseline ? signedPercent(passDelta) : "--"}</span>
      </div>
      <div class="pipeline-stat">
        <label>Best base BPB</label>
        <span>${bestBaseBpb ? `${escapeHtml(bestBaseBpb.run)} / ${fmtLoss(bestBaseBpb.base_val_bpb)}` : "--"}</span>
      </div>
      <div class="pipeline-stat">
        <label>SFT BPB delta</label>
        <span>${sftBpbDelta == null ? "--" : signedLoss(sftBpbDelta)}</span>
      </div>
    </div>
    ${compareDecisionBoard(comparison.decision, best, baseline, bestBaseBpb, bestSftBpb)}
    <p class="notice">${baseline ? `Compared against ${escapeHtml(baseline.run)}. Higher pass rate is good; use BPB, not raw loss, when comparing tokenizers. Best SFT BPB: ${bestSftBpb ? escapeHtml(bestSftBpb.run) : "--"}.` : "Only one run selected."}</p>
  `;
}

function compareDecisionBoard(decision, best, baseline, bestBaseBpb, bestSftBpb) {
  const view = decision || buildCompareDecision(best, baseline, bestBaseBpb, bestSftBpb);
  const championStatus = view.champion_status || "warn";
  const regressionStatus = view.regression_status || "warn";
  const nextStatus = view.next_status || "warn";
  return `
    <div class="compare-decision-grid">
      <div class="compare-decision-card ${escapeHtml(championStatus)}">
        <label>CHAMPION GATE</label>
        <strong>${escapeHtml(view.champion_title || "Need a baseline")}</strong>
        <p>${escapeHtml(view.champion_message || "Compare at least two runs before promoting a checkpoint.")}</p>
      </div>
      <div class="compare-decision-card ${escapeHtml(regressionStatus)}">
        <label>REGRESSION WATCH</label>
        <strong>${escapeHtml(view.regression_title || "No regression check")}</strong>
        <p>${escapeHtml(view.regression_message || "Regression checks need both a candidate and a baseline.")}</p>
      </div>
      <div class="compare-decision-card ${escapeHtml(nextStatus)}">
        <label>NEXT EXPERIMENT</label>
        <strong>${escapeHtml(view.next_title || "Add a comparison run")}</strong>
        <p>${escapeHtml(view.next_message || "Compare before changing model size or training time.")}</p>
      </div>
    </div>
  `;
}

function buildCompareDecision(best, baseline, bestBaseBpb, bestSftBpb) {
  const issues = compareRegressionIssues(best, baseline);
  const passDelta = baseline ? Number(best.pass_rate || 0) - Number(baseline.pass_rate || 0) : null;
  const championStatus = !baseline
    ? "warn"
    : issues.some((issue) => issue.severity === "fail")
      ? "fail"
      : issues.length
        ? "warn"
        : "pass";
  const championTitle = !baseline
    ? "Need a baseline"
    : championStatus === "pass"
      ? "Promote as reference"
      : championStatus === "warn"
        ? "Promising, inspect regressions"
        : "Do not promote yet";
  const championMessage = !baseline
    ? "Select at least two runs so the winner has something to beat."
    : `${best.run} is ${signedPercent(passDelta)} eval pass versus ${baseline.run}.`;
  const regressionTitle = !baseline
    ? "No regression check"
    : issues.length
      ? `${issues.length} watch item${issues.length === 1 ? "" : "s"}`
      : "No obvious regression";
  const regressionMessage = !baseline
    ? "Regression checks need a candidate and baseline."
    : issues.length
      ? issues.map((issue) => issue.message).join(" ")
      : "Eval pass, support, echo, SFT BPB, truncation, and memorization look acceptable.";
  const next = compareNextExperiment(best, baseline, issues, bestBaseBpb, bestSftBpb);
  return {
    champion_status: championStatus,
    champion_title: championTitle,
    champion_message: championMessage,
    regression_status: !baseline ? "warn" : issues.some((issue) => issue.severity === "fail") ? "fail" : issues.length ? "warn" : "pass",
    regression_title: regressionTitle,
    regression_message: regressionMessage,
    next_status: next.status,
    next_title: next.title,
    next_message: next.message,
  };
}

function compareRegressionIssues(best, baseline) {
  if (!baseline) return [];
  const issues = [];
  const passDelta = Number(best.pass_rate || 0) - Number(baseline.pass_rate || 0);
  const domainDelta = optionalNumberDelta(best.domain_pass_rate, baseline.domain_pass_rate);
  const refusalDelta = optionalNumberDelta(best.refusal_pass_rate, baseline.refusal_pass_rate);
  const supportDelta = optionalNumberDelta(best.support_match_rate, baseline.support_match_rate);
  const echoDelta = optionalNumberDelta(best.prompt_echo_rate, baseline.prompt_echo_rate);
  const sftBpbDelta = optionalNumberDelta(best.sft_val_bpb, baseline.sft_val_bpb);
  if (passDelta < 0.02) {
    issues.push({ severity: "warn", message: "Eval gain is under +2 points." });
  }
  if (domainDelta != null && domainDelta < 0.02) {
    issues.push({ severity: "warn", message: "Domain-answer gain is under +2 points." });
  }
  if (refusalDelta != null && refusalDelta < -0.05) {
    issues.push({ severity: "fail", message: `Refusal/boundary pass dropped ${signedPercent(refusalDelta)}.` });
  }
  if (supportDelta != null && supportDelta < -0.05) {
    issues.push({ severity: "fail", message: `Support match dropped ${signedPercent(supportDelta)}.` });
  }
  if (echoDelta != null && echoDelta > 0.02) {
    issues.push({ severity: "fail", message: `Prompt echo worsened ${signedPercent(echoDelta)}.` });
  }
  if (sftBpbDelta != null && sftBpbDelta > 0.10) {
    issues.push({ severity: "warn", message: `SFT BPB rose ${signedLoss(sftBpbDelta)}.` });
  }
  if (Number(best.truncated_examples || 0) > Number(baseline.truncated_examples || 0)) {
    issues.push({ severity: "warn", message: "More SFT rows were truncated." });
  }
  if (Number(best.skipped_long_examples || 0) > Number(baseline.skipped_long_examples || 0)) {
    issues.push({ severity: "warn", message: "More SFT rows were skipped for exceeding context." });
  }
  if (String(best.memorization_status || "").toLowerCase() !== "low") {
    issues.push({ severity: "fail", message: `Memorization status is ${best.memorization_status || "unknown"}.` });
  }
  return issues;
}

function compareNextExperiment(best, baseline, issues, bestBaseBpb, bestSftBpb) {
  if (!baseline) {
    return {
      status: "warn",
      title: "Add a comparison run",
      message: "Compare against a previous run before changing model size or training time.",
    };
  }
  if (issues.some((issue) => issue.severity === "fail")) {
    return {
      status: "fail",
      title: "Repair before scaling",
      message: "Use Eval Repair Board on the candidate run; fix trust regressions before longer runs.",
    };
  }
  if (best.run !== bestBaseBpb?.run && bestBaseBpb) {
    return {
      status: "warn",
      title: "Separate compression from behavior",
      message: `${bestBaseBpb.run} has better base BPB; compare tokenizer/model settings before scaling ${best.run}.`,
    };
  }
  if (best.run !== bestSftBpb?.run && bestSftBpb) {
    return {
      status: "warn",
      title: "SFT quality mismatch",
      message: `${bestSftBpb.run} has better SFT BPB; inspect SFT curriculum before choosing a champion.`,
    };
  }
  if (Number(best.pass_rate || 0) >= 0.7 && (best.domain_pass_rate == null || Number(best.domain_pass_rate) >= 0.5)) {
    return {
      status: "pass",
      title: "Attack harder eval",
      message: "Keep this as reference, add harder eval rows, then run a stronger preset.",
    };
  }
  if (best.domain_pass_rate != null && Number(best.domain_pass_rate) < 0.25) {
    return {
      status: "warn",
      title: "Improve answer data",
      message: "The model is not passing enough domain-answer rows yet; improve SFT/eval data before scaling.",
    };
  }
  return {
    status: "warn",
    title: "Improve data next",
    message: "Use failed eval categories to add targeted SFT rows before changing architecture.",
  };
}

function experimentNotebook(comparison) {
  const rows = comparison.rows || [];
  if (!rows.length) return "SELECT RUNS TO BUILD THE EXPERIMENT NOTEBOOK.";
  const sorted = rows
    .map((row) => ({ row, detail: state.compareDetails[row.run] }))
    .sort((left, right) => state.compareRuns.indexOf(left.row.run) - state.compareRuns.indexOf(right.row.run));
  return `
    <label>EXPERIMENT NOTEBOOK</label>
    <div class="notebook-grid">
      ${sorted.map((item, index) => {
        const previous = sorted[index - 1];
        return `
          <div class="notebook-entry ${item.row.run === comparison.best_run ? "best" : ""}">
            <div>
              <span>RUN ${String(index + 1).padStart(2, "0")}</span>
              <strong>${escapeHtml(item.row.run)}</strong>
            </div>
            <p><b>Hypothesis</b> ${escapeHtml(runHypothesis(item, previous))}</p>
            <p><b>Config</b> ${escapeHtml(runConfigLine(item))}</p>
            <p><b>Result</b> ${escapeHtml(runResultLine(item.row))}</p>
            <p><b>Verdict</b> ${escapeHtml(runNotebookVerdict(item.row, comparison.best_run))}</p>
          </div>
        `;
      }).join("")}
    </div>
  `;
}

function runHypothesis(item, previous) {
  if (!previous) return "Baseline for this comparison set.";
  const current = item.detail?.summary?.config || {};
  const prior = previous.detail?.summary?.config || {};
  const changes = importantConfigChanges(prior, current);
  if (!changes.length) return "Same visible config; result difference may be data, seed, runtime, or artifact changes.";
  if (changes.some((change) => change.key.includes("tokenizer"))) return "Test whether tokenizer changes improve compression and downstream eval.";
  if (changes.some((change) => ["n_embd", "n_layer", "n_head", "n_kv_head"].includes(change.key))) return "Test whether more model capacity improves learning without extra memorization.";
  if (changes.some((change) => change.key.includes("steps"))) return "Test whether more optimization time improves validation and eval.";
  if (changes.some((change) => change.key.includes("learning_rate") || change.key.includes("lr_"))) return "Test whether the optimizer schedule trains more cleanly.";
  if (changes.some((change) => change.key.includes("dataset") || change.key.includes("input"))) return "Test whether different data improves the behavior target.";
  return `Test config changes: ${changes.slice(0, 3).map((change) => change.label).join(", ")}.`;
}

function runConfigLine(item) {
  const config = item.detail?.summary?.config || {};
  const row = item.row || {};
  const shape = [config.n_embd, config.n_layer, config.n_head].every((value) => value !== undefined)
    ? `${config.n_embd}x${config.n_layer} h${config.n_head}/kv${config.n_kv_head || config.n_head}`
    : `${fmtInt(row.num_parameters)} params`;
  return `${config.tokenizer_type || row.tokenizer_type || "--"} vocab ${config.tokenizer_vocab_size || "--"} | ${shape} | ctx ${config.context_size || row.context_size || "--"} | base ${config.base_steps || "--"} / sft ${config.sft_steps || "--"}`;
}

function runResultLine(row) {
  return `${row.eval_score || "--"} (${fmtPercent(row.pass_rate)}) | base BPB ${fmtLoss(row.base_val_bpb)} | SFT BPB ${fmtLoss(row.sft_val_bpb)} | mem ${row.memorization_status || "--"}`;
}

function runNotebookVerdict(row, bestRun) {
  if (row.run === bestRun && Number(row.pass_rate || 0) >= 0.7) return "Keep as current reference, then attack weakest eval category.";
  if (row.run === bestRun) return "Best selected run, but still needs targeted eval/SFT work.";
  if (Number(row.pass_rate || 0) < 0.1) return "Reject for now; use failures to improve data before scaling.";
  return "Archive as comparison evidence unless it beats the reference on a specific metric.";
}

function configDiff(comparison) {
  const selected = state.compareRuns
    .map((run) => ({ run, detail: state.compareDetails[run], row: (comparison.rows || []).find((item) => item.run === run) }))
    .filter((item) => item.detail || item.row);
  if (selected.length < 2) {
    return "SELECT AT LEAST TWO RUNS TO INSPECT CONFIG DIFFS.";
  }
  const baseline = selected[0];
  const candidate = selected.at(-1);
  const changes = importantConfigChanges(
    baseline.detail?.summary?.config || {},
    candidate.detail?.summary?.config || {},
  );
  const metricRows = [
    ["eval pass", fmtPercent(baseline.row?.pass_rate), fmtPercent(candidate.row?.pass_rate), signedPercent(Number(candidate.row?.pass_rate || 0) - Number(baseline.row?.pass_rate || 0))],
    ["base BPB", fmtLoss(baseline.row?.base_val_bpb), fmtLoss(candidate.row?.base_val_bpb), signedLoss(Number(candidate.row?.base_val_bpb || 0) - Number(baseline.row?.base_val_bpb || 0))],
    ["SFT BPB", fmtLoss(baseline.row?.sft_val_bpb), fmtLoss(candidate.row?.sft_val_bpb), signedLoss(Number(candidate.row?.sft_val_bpb || 0) - Number(baseline.row?.sft_val_bpb || 0))],
  ];
  return `
    <label>CONFIG DIFF INSPECTOR</label>
    <div class="diff-head">
      <strong>${escapeHtml(baseline.run)}</strong>
      <span>VERSUS</span>
      <strong>${escapeHtml(candidate.run)}</strong>
    </div>
    <table class="config-diff-table">
      <thead><tr><th>Field</th><th>Baseline</th><th>Candidate</th><th>Meaning</th></tr></thead>
      <tbody>
        ${changes.length ? changes.map((change) => `
          <tr>
            <td>${escapeHtml(change.label)}</td>
            <td>${escapeHtml(change.before)}</td>
            <td>${escapeHtml(change.after)}</td>
            <td>${escapeHtml(changeTeachingNote(change.key))}</td>
          </tr>
        `).join("") : `<tr><td colspan="4">No visible config differences in the selected run summaries.</td></tr>`}
      </tbody>
    </table>
    <label>METRIC MOVEMENT</label>
    <table class="config-diff-table">
      <thead><tr><th>Metric</th><th>Baseline</th><th>Candidate</th><th>Delta</th></tr></thead>
      <tbody>
        ${metricRows.map(([metric, before, after, delta]) => `
          <tr><td>${escapeHtml(metric)}</td><td>${escapeHtml(before)}</td><td>${escapeHtml(after)}</td><td>${escapeHtml(delta)}</td></tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function importantConfigChanges(before, after) {
  const fields = [
    ["dataset_pack", "Dataset pack"],
    ["corpus_input", "Corpus input"],
    ["chat_input", "SFT data"],
    ["eval_input", "Eval data"],
    ["tokenizer_type", "Tokenizer"],
    ["tokenizer_vocab_size", "Vocab size"],
    ["context_size", "Context"],
    ["n_embd", "Embedding"],
    ["n_layer", "Layers"],
    ["n_head", "Heads"],
    ["n_kv_head", "KV Heads"],
    ["norm_type", "Norm"],
    ["position_encoding", "Position"],
    ["activation", "Activation"],
    ["tie_embeddings", "Tied embeddings"],
    ["qk_norm", "QK norm"],
    ["parallel_residual", "Parallel residual"],
    ["precision", "Precision"],
    ["matmul_precision", "Matmul precision"],
    ["attn_backend", "Attention backend"],
    ["torch_compile", "Torch compile"],
    ["torch_compile_mode", "Compile mode"],
    ["gradient_checkpointing", "Gradient checkpointing"],
    ["auto_lr_scaling", "Auto LR scaling"],
    ["loss_spike_rollback", "Loss rollback"],
    ["base_steps", "Base steps"],
    ["sft_steps", "SFT steps"],
    ["base_learning_rate", "Base LR"],
    ["sft_learning_rate", "SFT LR"],
    ["base_lr_decay", "Base LR decay"],
    ["sft_lr_decay", "SFT LR decay"],
    ["base_grad_clip", "Base grad clip"],
    ["sft_grad_clip", "SFT grad clip"],
    ["seed", "Seed"],
  ];
  return fields
    .map(([key, label]) => ({
      key,
      label,
      before: before[key] == null || before[key] === "" ? "--" : String(before[key]),
      after: after[key] == null || after[key] === "" ? "--" : String(after[key]),
    }))
    .filter((change) => change.before !== change.after);
}

function changeTeachingNote(key) {
  if (key.includes("tokenizer")) return "Changes compression and model vocabulary.";
  if (key === "context_size") return "Changes how much text the model sees at once.";
  if (["n_embd", "n_layer", "n_head", "n_kv_head"].includes(key)) return "Changes model capacity and compute cost.";
  if (["norm_type", "position_encoding", "activation", "tie_embeddings", "qk_norm", "parallel_residual"].includes(key)) return "Changes the Transformer architecture.";
  if (["precision", "matmul_precision", "attn_backend", "torch_compile", "torch_compile_mode", "gradient_checkpointing"].includes(key)) return "Changes GPU/runtime behavior.";
  if (key === "auto_lr_scaling") return "Changes automatic optimizer scaling from effective batch.";
  if (key === "loss_spike_rollback") return "Changes training stability recovery.";
  if (key.includes("steps")) return "Changes optimization time.";
  if (key.includes("learning_rate") || key.includes("lr_")) return "Changes optimizer behavior.";
  if (key.includes("data") || key.includes("input") || key.includes("corpus")) return "Changes what behavior/data the run can learn.";
  if (key.includes("grad_clip")) return "Controls unstable gradient spikes.";
  if (key === "seed") return "Changes randomness; useful for reproducibility checks.";
  return "Inspect whether this explains the metric movement.";
}

function compareBestSteps(row) {
  return `${row.base_best_step ?? "--"}/${row.sft_best_step ?? "--"}`;
}

function compareStopReasons(row) {
  return `${shortStopReason(row.base_stop_reason)}/${shortStopReason(row.sft_stop_reason)}`;
}

function shortStopReason(reason) {
  if (reason === "max_steps") return "max";
  if (reason === "max_minutes") return "time";
  if (reason === "early_stop") return "early";
  if (!reason || reason === "unknown") return "--";
  return reason;
}

function signedPercent(value) {
  const sign = value > 0 ? "+" : "";
  return `${sign}${fmtPercent(value)}`;
}

function signedLoss(value) {
  const sign = value > 0 ? "+" : "";
  return `${sign}${fmtLoss(value)}`;
}

function optionalNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function optionalNumberDelta(after, before) {
  const afterNumber = Number(after);
  const beforeNumber = Number(before);
  if (!Number.isFinite(afterNumber) || !Number.isFinite(beforeNumber)) return null;
  return afterNumber - beforeNumber;
}

function renderCompareError(error) {
  $("compare-status").textContent = "COMPARE FAULT";
  $("compare-summary").textContent = `FAULT: ${error.message}`;
  $("compare-table").innerHTML = "";
  $("experiment-notebook").textContent = "COMPARISON FAILED. FIX THE RUN SET, THEN TRY AGAIN.";
  $("config-diff").textContent = "COMPARISON FAILED. CONFIG DIFF UNAVAILABLE.";
}

async function loadReport() {
  if (!state.selectedRun) return;
  $("report-status").textContent = "LOADING";
  $("report-viewer").innerHTML = 'LOADING REPORT<span class="cursor"></span>';
  const report = await fetchJson(`/api/report?name=${encodeURIComponent(state.selectedRun)}&report=${encodeURIComponent(state.activeReport)}`);
  $("report-status").textContent = `${report.report.toUpperCase()} READY`;
  $("report-path").textContent = report.path;
  $("report-viewer").innerHTML = renderMarkdown(report.markdown);
}

function renderReportError(error) {
  $("report-status").textContent = "REPORT FAULT";
  $("report-viewer").textContent = `FAULT: ${error.message}`;
}

function renderMarkdown(markdown) {
  const lines = String(markdown || "").split("\n");
  const html = [];
  let inList = false;
  let inCode = false;
  let codeLines = [];
  let tableRows = [];

  for (const line of lines) {
    if (line.startsWith("```")) {
      if (inCode) {
        html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        codeLines = [];
        inCode = false;
      } else {
        closeList();
        closeTable();
        inCode = true;
      }
      continue;
    }
    if (inCode) {
      codeLines.push(line);
      continue;
    }
    if (line.startsWith("# ")) {
      closeList();
      closeTable();
      html.push(`<h1>${inlineMarkdown(line.slice(2))}</h1>`);
    } else if (line.startsWith("## ")) {
      closeList();
      closeTable();
      html.push(`<h2>${inlineMarkdown(line.slice(3))}</h2>`);
    } else if (line.startsWith("### ")) {
      closeList();
      closeTable();
      html.push(`<h3>${inlineMarkdown(line.slice(4))}</h3>`);
    } else if (line.startsWith("- ")) {
      closeTable();
      if (!inList) {
        html.push("<ul>");
        inList = true;
      }
      html.push(`<li>${inlineMarkdown(line.slice(2))}</li>`);
    } else if (line.startsWith("|") && line.endsWith("|")) {
      closeList();
      if (!/^\|\s*:?-+/.test(line)) {
        tableRows.push(line.split("|").slice(1, -1).map((cell) => cell.trim()));
      }
    } else if (line.trim()) {
      closeList();
      closeTable();
      html.push(`<p>${inlineMarkdown(line)}</p>`);
    } else {
      closeList();
      closeTable();
    }
  }
  if (inCode) html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
  closeList();
  closeTable();
  return html.join("");

  function closeList() {
    if (inList) {
      html.push("</ul>");
      inList = false;
    }
  }

  function closeTable() {
    if (!tableRows.length) return;
    const [header, ...body] = tableRows;
    html.push("<table>");
    html.push(`<thead><tr>${header.map((cell) => `<th>${inlineMarkdown(cell)}</th>`).join("")}</tr></thead>`);
    html.push(`<tbody>${body.map((row) => `<tr>${row.map((cell) => `<td>${inlineMarkdown(cell)}</td>`).join("")}</tr>`).join("")}</tbody>`);
    html.push("</table>");
    tableRows = [];
  }
}

function inlineMarkdown(text) {
  return escapeHtml(text).replace(/`([^`]+)`/g, "<code>$1</code>");
}

async function animateGeneration() {
  clearInterval(state.generationTimer);
  const output = $("generation-output");
  const button = $("generate-button");
  button.disabled = true;
  output.innerHTML = 'GENERATING<span class="cursor"></span>';
  $("logprob-bars").innerHTML = '<div class="notice">POST /api/generate</div>';

  const result = await postJson("/api/generate", {
    run: state.selectedRun,
    checkpoint: $("checkpoint-select").value,
    prompt: $("prompt-input").value,
    temperature: Number($("temperature-slider").value),
    top_k: Number($("topk-slider").value),
    top_p: Number($("topp-slider").value),
    repetition_penalty: Number($("repeat-slider").value),
    max_new_tokens: Number($("max-tokens-slider").value),
    seed: state.detail?.summary?.config?.seed ?? 42,
  });

  const text = result.completion === undefined
    ? result.text || "NO TOKENS GENERATED."
    : result.completion || "NO COMPLETION TOKENS GENERATED.";
  output.textContent = "";
  $("logprob-bars").innerHTML = "";
  let index = 0;
  state.generationTimer = setInterval(() => {
    if (index >= text.length) {
      clearInterval(state.generationTimer);
      button.disabled = false;
      renderLogprobs(result.generated_tokens || []);
      return;
    }
    output.textContent += text[index];
    index += 1;
  }, 18);
}

function renderLogprobs(tokens) {
  if (!tokens.length) {
    $("logprob-bars").innerHTML = '<div class="notice">NO GENERATED TOKEN METADATA RETURNED.</div>';
    return;
  }
  $("logprob-bars").innerHTML = `
    <label>GENERATED TOKEN LOGPROBS</label>
    ${tokens.slice(0, 64).map((token) => {
      const probability = Math.max(0, Math.min(1, Number(token.probability || 0)));
      const width = Math.max(2, Math.round(probability * 100));
      return `
        <div class="logprob-row">
          <span>${escapeHtml(renderTokenLabel(token.token))}</span>
          <div class="logprob-track"><i style="width:${width}%"></i></div>
          <em>${fmtLoss(token.logprob)}</em>
        </div>
      `;
    }).join("")}
  `;
}

function renderTokenLabel(token) {
  if (token === " ") return "space";
  if (token === "\n") return "\\n";
  return token;
}

function renderEval() {
  const reports = state.detail?.eval_reports || [];
  if (!reports.length) {
    $("eval-status").textContent = "NO EVAL";
    $("score-table").innerHTML = "NO EVAL REPORT FOUND.";
    $("eval-results").innerHTML = "";
    return;
  }
  const latest = reports.at(-1);
  const report = latest.report;
  const honesty = evalHonestySummary(report);
  const categoryRows = evalCategoryRows(report);
  const splitRows = evalSplitRows(report);
  const levelRows = evalLevelRows(report);
  $("eval-status").textContent = `${latest.name.toUpperCase()} ${report.summary.num_passed}/${report.summary.num_examples}`;
  $("score-table").innerHTML = `
    ${renderEvalReadout(report, levelRows)}
    ${renderEvalLearningFocus(report, levelRows)}
    ${renderEvalRepairBoard(report, categoryRows, levelRows)}
    <label>HONESTY SUMMARY</label>
    <div class="eval-summary-cards">
      <div class="pipeline-stat">
        <label>Pass rate</label>
        <span>${fmtPercent(report.summary.pass_rate)}</span>
      </div>
      <div class="pipeline-stat">
        <label>Unsupported claims</label>
        <span>${fmtPercent(honesty.unsupportedClaimRate)}</span>
      </div>
      <div class="pipeline-stat">
        <label>Missing support</label>
        <span>${fmtPercent(honesty.missingSupportRate)}</span>
      </div>
      <div class="pipeline-stat">
        <label>Prompt echo</label>
        <span>${fmtPercent(honesty.promptEchoRate)}</span>
      </div>
      <div class="pipeline-stat">
        <label>Corpus support</label>
        <span>${report.summary.average_corpus_support_rate === null || report.summary.average_corpus_support_rate === undefined ? "--" : fmtPercent(report.summary.average_corpus_support_rate)}</span>
      </div>
      <div class="pipeline-stat">
        <label>Unanswerable</label>
        <span>${honesty.numUnanswerable}/${honesty.numExamples}</span>
      </div>
    </div>
    <div class="inspect-only">
      ${renderEvalCategoryTable(categoryRows)}
      ${renderEvalSplitTable(splitRows)}
      ${renderEvalLevelTable(levelRows)}
      <label>ARCADE SCORE TABLE</label>
      <table>
        <thead><tr><th>Rank</th><th>Prompt</th><th>Kind</th><th>Level</th><th>Status</th><th>Support</th><th>Echo</th><th>Forbidden</th></tr></thead>
        <tbody>
          ${report.examples.map((item, index) => `
            <tr>
              <td>${String(index + 1).padStart(2, "0")}</td>
              <td>${escapeHtml(item.user)}</td>
              <td>${evalKindTag(item)}</td>
              <td>${escapeHtml(item.level || "heldout")}</td>
              <td class="${item.passed ? "pass-text" : "fail-text"}">${item.passed ? "PASS" : "FAIL"}</td>
              <td class="${hasMissingSupport(item) ? "fail-text" : "pass-text"}">${hasMissingSupport(item) ? "MISSING" : "COVERED"}</td>
              <td class="${hasPromptEcho(item) ? "fail-text" : "pass-text"}">${hasPromptEcho(item) ? "ECHO" : "CLEAR"}</td>
              <td class="${hasForbiddenClaim(item) ? "fail-text" : "pass-text"}">${hasForbiddenClaim(item) ? "FOUND" : "CLEAR"}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
  $("eval-results").innerHTML = report.examples.map((item, index) => evalCard(item, index)).join("");
}

function renderEvalReadout(report, levelRows) {
  const weakest = levelRows
    .filter((row) => row.numExamples > 0)
    .sort((left, right) => left.passRate - right.passRate)[0];
  const support = report.summary?.average_corpus_support_rate;
  const supportText = support === null || support === undefined ? "no support corpus" : `${fmtPercent(support)} corpus overlap`;
  return `
    <div class="teaching-note">
      <strong>READOUT</strong>
      <span>Pass rate is the broad score; ladder shows the failure type.</span>
      <span>Weakest level: ${escapeHtml(weakest ? `${weakest.level} ${fmtPercent(weakest.passRate)}` : "--")}</span>
      <span>${escapeHtml(supportText)}; this checks overlap, not semantic truth.</span>
    </div>
  `;
}

function renderEvalLearningFocus(report, levelRows) {
  const failed = evalFailureCoachRows(report).slice(0, 4);
  const weakest = levelRows
    .filter((row) => row.numExamples > 0)
    .sort((left, right) => left.passRate - right.passRate)[0];
  const recommendation = report.analysis?.recommendations?.[0];
  return `
    <div class="learning-focus learn-only">
      <div>
        <label>WEAKEST LEVEL</label>
        <strong>${escapeHtml(weakest ? weakest.level : "--")}</strong>
        <p>${weakest ? `${weakest.numPassed}/${weakest.numExamples} passed. Improve this before scaling.` : "No ladder data found."}</p>
      </div>
      <div>
        <label>NEXT DATA FIX</label>
        <strong>${escapeHtml(recommendation?.area || "curriculum")}</strong>
        <p>${escapeHtml(recommendation?.action || "Inspect failed examples and add targeted SFT rows.")}</p>
      </div>
      <div>
        <label>FAILURE COACH</label>
        ${failed.length ? failed.map((item) => `
          <p><b>#${escapeHtml(item.index)}</b> ${escapeHtml(item.reason)} -> ${escapeHtml(item.fix)}</p>
        `).join("") : "<p>No failed examples in this report.</p>"}
      </div>
    </div>
  `;
}

function renderEvalRepairBoard(report, categoryRows, levelRows) {
  const category = weakestBreakdownRow(categoryRows, "category");
  const level = weakestBreakdownRow(levelRows, "level");
  const failureMix = topFailureMix(report).slice(0, 4);
  const failures = evalFailureCoachRows(report).slice(0, 3);
  const recommendations = evalRepairRecommendations(report, category, failureMix).slice(0, 3);
  return `
    <div class="repair-board learn-only">
      <div class="repair-card">
        <label>REPAIR TARGET</label>
        <strong>${escapeHtml(category ? category.category : "--")}</strong>
        <p>${category ? `${category.numPassed}/${category.numExamples} passed. Fix this category before training longer.` : "No category breakdown found."}</p>
        <p>${level ? `Weakest ladder: ${level.level} at ${fmtPercent(level.passRate)}.` : "No eval ladder found."}</p>
      </div>
      <div class="repair-card">
        <label>FAILURE MIX</label>
        ${failureMix.length ? failureMix.map((item) => `
          <p><b>${escapeHtml(item.count)}</b> ${escapeHtml(item.label)}</p>
        `).join("") : "<p>No failed checks in this eval report.</p>"}
      </div>
      <div class="repair-card repair-queue">
        <label>REPAIR QUEUE</label>
        ${recommendations.length ? recommendations.map((item) => `
          <p><b>${escapeHtml(String(item.priority || "fix").toUpperCase())}</b> ${escapeHtml(item.action || item.message || "Inspect failed examples.")}</p>
        `).join("") : "<p>Add harder eval prompts before scaling this recipe.</p>"}
      </div>
    </div>
    <div class="repair-examples learn-only">
      <label>TURN FAILURE PATTERNS INTO NEW TRAINING ROWS</label>
      <p class="helper-copy">Do not copy these held-out eval prompts into SFT. Use the category, missing behavior, and wording pattern to write separate non-eval examples.</p>
      ${failures.length ? failures.map((item) => `
        <div class="repair-example">
          <strong>#${escapeHtml(item.index)} ${escapeHtml(item.category)} / ${escapeHtml(item.reason)}</strong>
          <p>${escapeHtml(item.user)}</p>
          <p><b>Curriculum fix</b> ${escapeHtml(item.fix)}</p>
          <p><b>New rows should practice</b> ${escapeHtml(item.missing.length ? item.missing.join(" | ") : "the expected behavior without copying the eval prompt or answer")}</p>
        </div>
      `).join("") : "<p class=\"helper-copy\">No failed examples. Add harder held-out eval rows before scaling.</p>"}
    </div>
  `;
}

function weakestBreakdownRow(rows, key) {
  return [...(rows || [])]
    .filter((row) => row.numExamples > 0)
    .sort((left, right) => (
      (left.passRate - right.passRate) ||
      ((right.missingSupport + right.unsupportedClaims + right.promptEchoes) - (left.missingSupport + left.unsupportedClaims + left.promptEchoes)) ||
      String(left[key] || "").localeCompare(String(right[key] || ""))
    ))[0];
}

function topFailureMix(report) {
  const counts = report.analysis?.failure_counts || {};
  const clusters = report.analysis?.cluster_counts || {};
  const rows = Object.entries(counts).map(([name, count]) => ({
    label: humanizeEvalKey(name),
    count,
  }));
  if (!rows.length) {
    const clusterRows = Object.entries(clusters).map(([name, count]) => ({
      label: humanizeEvalKey(name),
      count,
    }));
    if (clusterRows.length) return clusterRows.sort((left, right) => Number(right.count) - Number(left.count));
    const fallback = new Map();
    for (const item of evalFailureCoachRows(report)) {
      fallback.set(item.reason, (fallback.get(item.reason) || 0) + 1);
    }
    return Array.from(fallback.entries())
      .map(([label, count]) => ({ label: humanizeEvalKey(label), count }))
      .sort((left, right) => Number(right.count) - Number(left.count));
  }
  return rows.sort((left, right) => Number(right.count) - Number(left.count));
}

function evalRepairRecommendations(report, weakestCategory, failureMix) {
  const existing = report.analysis?.recommendations || [];
  if (existing.length) return existing;
  if (!(report.examples || []).some((item) => !item.passed)) {
    return [{
      priority: "medium",
      action: "Add harder held-out eval prompts before scaling this recipe.",
    }];
  }
  const primaryFailure = failureMix[0]?.label || "failed behavior";
  const category = weakestCategory?.category || "weakest category";
  return [
    {
      priority: "high",
      action: `Add 5-10 SFT rows for ${category}, targeting ${primaryFailure.toLowerCase()}.`,
    },
    {
      priority: "medium",
      action: "Keep eval prompts held out; do not copy failed eval answers directly into SFT.",
    },
    {
      priority: "medium",
      action: "Rerun SFT and eval before increasing base steps or model size.",
    },
  ];
}

function humanizeEvalKey(value) {
  return String(value || "failure")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function evalFailureCoachRows(report) {
  const analysisFailures = report.analysis?.failed_examples || [];
  const directFailures = (report.examples || [])
    .map((item, index) => ({ ...item, index: index + 1 }))
    .filter((item) => !item.passed);
  const failures = analysisFailures.length ? analysisFailures : directFailures;
  const examplesByIndex = new Map((report.examples || []).map((item, index) => [index + 1, item]));
  return failures.map((failure) => {
    const index = Number(failure.index);
    const item = examplesByIndex.get(index) || failure;
    return {
      index: Number.isFinite(index) ? index : "?",
      category: item.category || failure.category || "eval",
      level: item.level || failure.level || "heldout",
      user: item.user || failure.user || "Prompt unavailable.",
      missing: evalMissingTerms(item),
      reason: evalFailureReason(item, failure),
      fix: evalFailureFix(item, failure),
    };
  });
}

function evalMissingTerms(item) {
  return [
    ...(item.missing || []),
    ...((item.missing_any || []).flat()),
    ...(item.missing_entities || []),
    ...(item.found_forbidden || []).map((value) => `avoid: ${value}`),
    ...(item.prompt_echo_reasons || []).map((value) => `echo: ${value}`),
  ].filter(Boolean);
}

function evalFailureReason(item, failure = {}) {
  const clusters = [...(failure.clusters || []), ...(failure.reasons || [])].join(" ").toLowerCase();
  if (hasForbiddenClaim(item) || clusters.includes("unsupported") || clusters.includes("forbidden")) return "unsupported claim";
  if (hasPromptEcho(item) || clusters.includes("echo")) return "prompt echo";
  if (hasMissingSupport(item) || clusters.includes("missing")) return "missed required evidence";
  if (String(item.category || "").includes("memorization")) return "memorization refusal too weak";
  if (!isAnswerable(item)) return "refusal behavior too weak";
  return item.level ? `${item.level} generalization gap` : "behavior mismatch";
}

function evalFailureFix(item, failure = {}) {
  const category = String(item.category || failure.category || "").toLowerCase();
  const level = String(item.level || failure.level || "").toLowerCase();
  if (category.includes("memorization")) return "add refusal rows for verbatim-copy requests";
  if (category.includes("refusal") || !isAnswerable(item)) return "add refusal rows with short safe answers";
  if (level.includes("transfer")) return "add paraphrased versions, not duplicate target answers";
  if (level.includes("adversarial")) return "add harder negatives and format traps";
  if (hasMissingSupport(item)) return "add non-eval SFT rows that practice the same missing phrases/entities";
  if (hasPromptEcho(item)) return "add examples that answer without repeating the prompt";
  return "add 5-10 targeted non-eval SFT rows, then rerun eval";
}

function evalHonestySummary(report) {
  const examples = report.examples || [];
  const summary = report.summary || {};
  const numExamples = summary.num_examples ?? examples.length;
  const unsupportedClaims = summary.unsupported_claims ?? examples.filter(hasForbiddenClaim).length;
  const missingSupport = summary.missing_support ?? examples.filter(hasMissingSupport).length;
  const promptEchoes = summary.prompt_echoes ?? examples.filter(hasPromptEcho).length;
  const numUnanswerable = summary.num_unanswerable ?? examples.filter((item) => !isAnswerable(item)).length;
  return {
    numExamples,
    numUnanswerable,
    unsupportedClaimRate: summary.unsupported_claim_rate ?? unsupportedClaims / Math.max(1, numExamples),
    missingSupportRate: summary.missing_support_rate ?? missingSupport / Math.max(1, numExamples),
    promptEchoRate: summary.prompt_echo_rate ?? promptEchoes / Math.max(1, numExamples),
  };
}

function evalCategoryRows(report) {
  const breakdown = report.summary?.category_breakdown || {};
  const rows = Object.entries(breakdown).map(([category, row]) => ({
    category,
    numExamples: row.num_examples ?? 0,
    numPassed: row.num_passed ?? 0,
    passRate: row.pass_rate ?? 0,
    missingSupport: row.missing_support ?? 0,
    promptEchoes: row.prompt_echoes ?? 0,
    unsupportedClaims: row.unsupported_claims ?? 0,
  }));
  if (rows.length) {
    return rows.sort((left, right) => left.category.localeCompare(right.category));
  }

  const buckets = new Map();
  for (const item of report.examples || []) {
    const category = item.category || (isAnswerable(item) ? "answerable" : "unanswerable");
    if (!buckets.has(category)) {
      buckets.set(category, {
        category,
        numExamples: 0,
        numPassed: 0,
        passRate: 0,
        missingSupport: 0,
        promptEchoes: 0,
        unsupportedClaims: 0,
      });
    }
    const bucket = buckets.get(category);
    bucket.numExamples += 1;
    bucket.numPassed += item.passed ? 1 : 0;
    bucket.missingSupport += hasMissingSupport(item) ? 1 : 0;
    bucket.promptEchoes += hasPromptEcho(item) ? 1 : 0;
    bucket.unsupportedClaims += hasForbiddenClaim(item) ? 1 : 0;
  }
  return Array.from(buckets.values())
    .map((row) => ({
      ...row,
      passRate: row.numPassed / Math.max(1, row.numExamples),
    }))
    .sort((left, right) => left.category.localeCompare(right.category));
}

function renderEvalCategoryTable(rows) {
  if (!rows.length) return "";
  return renderEvalBreakdownTable("CATEGORY BREAKDOWN", "Category", rows, "category");
}

function evalSplitRows(report) {
  const breakdown = report.summary?.split_breakdown || {};
  const rows = Object.entries(breakdown).map(([split, row]) => ({
    split,
    numExamples: row.num_examples ?? 0,
    numPassed: row.num_passed ?? 0,
    passRate: row.pass_rate ?? 0,
    missingSupport: row.missing_support ?? 0,
    promptEchoes: row.prompt_echoes ?? 0,
    unsupportedClaims: row.unsupported_claims ?? 0,
  }));
  if (rows.length) {
    return rows.sort((left, right) => left.split.localeCompare(right.split));
  }

  const buckets = new Map();
  for (const item of report.examples || []) {
    const split = item.split || "default";
    if (!buckets.has(split)) {
      buckets.set(split, {
        split,
        numExamples: 0,
        numPassed: 0,
        passRate: 0,
        missingSupport: 0,
        promptEchoes: 0,
        unsupportedClaims: 0,
      });
    }
    const bucket = buckets.get(split);
    bucket.numExamples += 1;
    bucket.numPassed += item.passed ? 1 : 0;
    bucket.missingSupport += hasMissingSupport(item) ? 1 : 0;
    bucket.promptEchoes += hasPromptEcho(item) ? 1 : 0;
    bucket.unsupportedClaims += hasForbiddenClaim(item) ? 1 : 0;
  }
  return Array.from(buckets.values())
    .map((row) => ({
      ...row,
      passRate: row.numPassed / Math.max(1, row.numExamples),
    }))
    .sort((left, right) => left.split.localeCompare(right.split));
}

function renderEvalSplitTable(rows) {
  if (!rows.length) return "";
  return renderEvalBreakdownTable("SPLIT BREAKDOWN", "Split", rows, "split");
}

function evalLevelRows(report) {
  const breakdown = report.summary?.level_breakdown || {};
  const rows = Object.entries(breakdown).map(([level, row]) => ({
    level,
    numExamples: row.num_examples ?? 0,
    numPassed: row.num_passed ?? 0,
    passRate: row.pass_rate ?? 0,
    missingSupport: row.missing_support ?? 0,
    promptEchoes: row.prompt_echoes ?? 0,
    unsupportedClaims: row.unsupported_claims ?? 0,
  }));
  if (rows.length) {
    return rows.sort((left, right) => left.level.localeCompare(right.level));
  }
  return [];
}

function renderEvalLevelTable(rows) {
  if (!rows.length) return "";
  return renderEvalBreakdownTable("EVAL LADDER", "Level", rows, "level");
}

function renderEvalBreakdownTable(label, firstColumn, rows, key) {
  return `
    <label>${label}</label>
    <table class="eval-category-table">
      <thead><tr><th>${firstColumn}</th><th>Passed</th><th>Pass</th><th>Missing</th><th>Echo</th><th>Forbidden</th></tr></thead>
      <tbody>
        ${rows.map((row) => `
          <tr>
            <td>${escapeHtml(row[key])}</td>
            <td>${row.numPassed}/${row.numExamples}</td>
            <td>${fmtPercent(row.passRate)}</td>
            <td>${row.missingSupport}/${row.numExamples}</td>
            <td>${row.promptEchoes}/${row.numExamples}</td>
            <td>${row.unsupportedClaims}/${row.numExamples}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function evalCard(item, index) {
  const matched = item.passed ? "MATCHED" : "INSPECT";
  return `
    <details class="eval-card ${item.passed ? "pass" : "fail"}">
      <summary>
        <span>${String(index + 1).padStart(2, "0")} ${escapeHtml(item.user)}</span>
        <b>${matched}</b>
      </summary>
      <div class="phrase-grid">
        <p>CATEGORY: ${escapeHtml(item.category || "answerable")}</p>
        <p>LEVEL: ${escapeHtml(item.level || "heldout")}</p>
        <p>SPLIT: ${escapeHtml(item.split || "default")}</p>
        <p>ANSWERABLE: ${escapeHtml(isAnswerable(item) ? "yes" : "no")}</p>
        <p>REQUIRED: ${escapeHtml((item.must_include || []).join(" | ") || "none")}</p>
        <p>ENTITIES: ${escapeHtml((item.required_entities || []).join(" | ") || "none")}</p>
        <p>ANY: ${escapeHtml((item.must_include_any || []).map((group) => `[${group.join(" / ")}]`).join(" ") || "none")}</p>
        <p>FORBIDDEN: ${escapeHtml((item.must_not_include || []).join(" | ") || "none")}</p>
        <p>MISSING: ${escapeHtml([...(item.missing || []), ...((item.missing_any || []).flat()), ...(item.missing_entities || [])].join(" | ") || "none")}</p>
        <p>PROMPT ECHO: ${escapeHtml((item.prompt_echo_reasons || []).join(" | ") || "none")}</p>
        <p>FOUND FORBIDDEN: ${escapeHtml((item.found_forbidden || []).join(" | ") || "none")}</p>
        <p>REF F1: ${item.reference_token_f1 === null || item.reference_token_f1 === undefined ? "--" : fmtPercent(item.reference_token_f1)}</p>
        <p>CORPUS SUPPORT: ${item.corpus_support_rate === null || item.corpus_support_rate === undefined ? "--" : fmtPercent(item.corpus_support_rate)}</p>
        <p>LENGTH: ${escapeHtml(`${item.word_count ?? "--"} words / ${item.char_count ?? "--"} chars`)}</p>
      </div>
      <pre>${escapeHtml(item.reply)}</pre>
    </details>
  `;
}

function evalKindTag(item) {
  const answerable = isAnswerable(item);
  const category = item.category || (answerable ? "answerable" : "unanswerable");
  return `<span class="eval-tag ${answerable ? "answerable" : "unanswerable"}">${escapeHtml(category)}</span>`;
}

function isAnswerable(item) {
  return item.answerable !== false;
}

function hasMissingSupport(item) {
  return Boolean((item.missing || []).length || (item.missing_any || []).length);
}

function hasForbiddenClaim(item) {
  return Boolean((item.found_forbidden || []).length);
}

function hasPromptEcho(item) {
  return Boolean(item.prompt_echo);
}

function flashStatus(message) {
  window.clearTimeout(state.statusTimer);
  $("status-line").textContent = message;
  state.statusTimer = window.setTimeout(() => {
    state.statusTimer = null;
    renderStatus();
  }, 1600);
}

function renderStatus() {
  const summary = state.detail?.summary;
  const tok = state.detail?.tokenizer_detail?.vocab_size ?? "--";
  const ctx = summary?.config?.context_size ?? "--";
  const seed = padSeed(summary?.config?.seed);
  const run = state.selectedRun || "--";
  $("status-line").textContent = `READY. | VIEW ${state.activeView.toUpperCase()} | MODE ${state.viewMode.toUpperCase()} | PANEL ${state.activePanel.toUpperCase()} | RUN ${run} | CTX ${ctx} | TOK ${tok} | SEED ${seed}`;
}

boot().catch((error) => {
  $("run-count").textContent = "FAULT";
  $("status-line").textContent = `FAULT. | ${error.message}`;
});

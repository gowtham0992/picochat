const state = {
  runs: [],
  selectedRun: null,
  detail: null,
  activePanel: "dataset",
  activeStage: "dataset",
  viewMode: readInitialViewMode(),
  activeReport: "summary",
  compareRuns: [],
  compareDetails: {},
  corpusSourcePreview: null,
  datasetFlightPlan: null,
  evalStarter: null,
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

async function fetchJson(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}

async function boot() {
  bindControls();
  setViewMode(state.viewMode, { persist: false, render: false });
  await loadRunPresets();
  await loadRuns();
  await loadRunJobs();
}

function bindControls() {
  $("learn-mode-button").addEventListener("click", () => setViewMode("learn"));
  $("inspect-mode-button").addEventListener("click", () => setViewMode("inspect"));
  $("refresh-button").addEventListener("click", loadRuns);
  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-copy-command]");
    if (!button) return;
    copyCommand(button.dataset.copyCommand || "", button);
  });
  document.querySelectorAll("[data-panel]").forEach((button) => {
    button.addEventListener("click", () => setPanel(button.dataset.panel));
  });
  $("pipeline-strip").addEventListener("click", (event) => {
    const button = event.target.closest("[data-stage]");
    if (!button) return;
    setStage(button.dataset.stage);
  });
  $("run-storyline").addEventListener("click", (event) => {
    const button = event.target.closest("[data-stage]");
    if (!button) return;
    setStage(button.dataset.stage);
  });
  $("run-doctor").addEventListener("click", (event) => {
    const button = event.target.closest("[data-stage]");
    if (!button) return;
    setStage(button.dataset.stage);
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
    }
  });
  $("compare-button").addEventListener("click", () => {
    loadComparison().catch((error) => renderCompareError(error));
  });
  $("preview-corpus-button").addEventListener("click", () => {
    previewCorpusSources().catch((error) => renderCorpusSourcePreviewError(error));
  });
  $("flight-check-button").addEventListener("click", () => {
    checkDatasetFlightPlan().catch((error) => renderDatasetFlightPlanError(error));
  });
  $("flight-eval-button").addEventListener("click", () => {
    createEvalStarter().catch((error) => renderEvalStarterError(error));
  });
  $("flight-apply-button").addEventListener("click", () => {
    applyFlightPlanToLauncher();
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
  if (!state.selectedRun && state.runs.length) {
    state.selectedRun = state.runs[state.runs.length - 1].name;
  }
  const runNames = state.runs.map((run) => run.name);
  state.compareRuns = state.compareRuns.filter((name) => runNames.includes(name));
  if (!state.compareRuns.length) {
    state.compareRuns = state.runs.slice(-2).map((run) => run.name);
  }
  renderRuns();
  renderCompareControls();
  if (state.selectedRun) {
    await loadRun(state.selectedRun);
  }
}

function renderRuns() {
  const list = $("run-list");
  if (!state.runs.length) {
    list.innerHTML = '<div class="empty">NO RUN ARTIFACTS FOUND.</div>';
    return;
  }
  list.innerHTML = state.runs.map((run) => `
    <button class="run-button ${run.name === state.selectedRun ? "active" : ""}" type="button" data-run="${escapeHtml(run.name)}">
      <span>${escapeHtml(run.name)}</span>
      <small>${escapeHtml(run.eval_score)} | ${fmtPercent(run.pass_rate)} | CTX ${escapeHtml(run.context_size)}</small>
    </button>
  `).join("");
  list.querySelectorAll("[data-run]").forEach((button) => {
    button.addEventListener("click", async () => {
      state.selectedRun = button.dataset.run;
      renderRuns();
      await loadRun(state.selectedRun);
    });
  });
}

async function loadRun(name) {
  state.detail = await fetchJson(`/api/run?name=${encodeURIComponent(name)}`);
  renderAll();
}

function readInitialViewMode() {
  try {
    return localStorage.getItem("picochat:view-mode") === "inspect" ? "inspect" : "learn";
  } catch {
    return "learn";
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
    renderPanelGuide();
    renderStatus();
  }
}

function renderAll() {
  renderPanelGuide();
  renderPipeline();
  renderDataset();
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

function setStage(name) {
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
  if (panel) setPanel(panel);
  renderPipeline();
}

function setPanel(name) {
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
  renderStatus();
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
        "--n-layer", config.n_layer ?? 2,
        "--max-steps", config.base_steps ?? 300,
        "--batch-size", config.base_batch_size ?? 8,
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
        ["Truncated", summary.sft?.truncated_examples ?? "--"],
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
        "--learning-rate", config.sft_learning_rate ?? "1e-3",
        "--early-stop-patience", config.sft_early_stop_patience ?? 4,
        "--sampling", config.sft_sampling || detail?.sft_report?.dataset?.sampling || "uniform",
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
        artifactItem("INPUT", "Summary JSON", `${outDir}/summary.json`),
        artifactItem("INPUT", "Honesty report", artifacts.honesty_report || `${outDir}/honesty/report.md`),
        artifactItem("INPUT", "Base report", artifacts.base_report || `${outDir}/base/report.md`),
        artifactItem("INPUT", "SFT report", artifacts.sft_report || `${outDir}/sft/report.md`),
        artifactItem("INPUT", "Eval report", artifacts.eval_report || `${outDir}/eval/report.md`),
        artifactItem("OUTPUT", "Summary report", `${outDir}/summary.md`),
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
  `;
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
    return `SFT BPB ${fmtLoss(loss)} / trunc ${truncated ?? "--"}`;
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
  renderEvalStarter(state.evalStarter);
  renderCorpusSourcePreview(state.corpusSourcePreview);
  renderDatasetPackInit(state.datasetPackInit);
  renderTuningInspection(state.tuningInspection);
  renderPackEditor(state.packEditor);
  renderRunJob(state.runJob);
  renderRunJobList();
  $("corpus-preview").textContent = state.detail?.corpus_preview || "NO CORPUS PREVIEW ARTIFACT FOUND.";
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
  const evalInput = $("flight-eval-path");
  const evalOutInput = $("flight-eval-out-path");
  const minScoreInput = $("flight-min-score");
  if (packInput.value || sourceInput.value || chatInput.value || evalInput.value || evalOutInput.value) return;
  packInput.value = config.dataset_pack || "";
  sourceInput.value = config.dataset_pack ? "" : config.corpus_input || "";
  chatInput.value = config.chat_input || "";
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
    $("preview-pack-path").value = report.dataset_pack || "";
    $("preview-recipe-path").value = "";
    $("preview-input-path").value = "";
    $("preview-chat-path").value = "";
    $("preview-eval-path").value = "";
    $("flight-pack-path").value = report.dataset_pack || "";
    $("flight-input-path").value = "";
    $("flight-chat-path").value = "";
    $("flight-eval-path").value = "";
    $("flight-eval-out-path").value = report.eval_input ? suggestedEvalStarterPath(report.eval_input) : "";
    $("tuning-pack-path").value = report.dataset_pack || "";
    $("tuning-chat-path").value = "";
    $("tuning-eval-path").value = "";
    $("editor-pack-path").value = report.dataset_pack || "";
    $("editor-chat-path").value = "";
    $("editor-eval-path").value = "";
    $("launch-pack-path").value = report.dataset_pack || "";
    $("launch-run-name").value = suggestedRunName(report.dataset_pack || "picochat");
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
  $("tuning-inspector-result").innerHTML = renderTuningPreflight(report.chat_data, report.eval_data);
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
  $("launch-base-steps").value = values.base_steps;
  $("launch-sft-steps").value = values.sft_steps;
  if (values.tokenizer_type) $("launch-tokenizer-type").value = values.tokenizer_type;
  $("launch-tokenizer-vocab-size").value = values.tokenizer_vocab_size || "";
  if (!quiet) {
    flashStatus(`APPLIED ${String(values.label || preset).toUpperCase()} PRESET. | ${values.description || ""}`);
  }
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
  flashStatus(`APPLIED PREVIEW BUDGET. | CTX ${$("launch-context-size").value} | BASE ${$("launch-base-steps").value} | SFT ${$("launch-sft-steps").value}`);
}

async function launchRun() {
  const datasetPack = $("launch-pack-path").value.trim();
  const runName = $("launch-run-name").value.trim();
  if (!datasetPack) throw new Error("enter a dataset pack");
  if (!runName) throw new Error("enter a run name");
  $("launch-run-button").disabled = true;
  $("run-launch-status").innerHTML = 'LAUNCHING RUN<span class="cursor"></span>';
  try {
    const payload = await postJson("/api/run/start", {
      dataset_pack: datasetPack,
      run_name: runName,
      preset: $("launch-preset").value,
      context_size: Number($("launch-context-size").value),
      base_steps: Number($("launch-base-steps").value),
      sft_steps: Number($("launch-sft-steps").value),
      seed: Number($("launch-seed").value),
      tokenizer_type: $("launch-tokenizer-type").value,
      tokenizer_vocab_size: $("launch-tokenizer-vocab-size").value
        ? Number($("launch-tokenizer-vocab-size").value)
        : null,
      min_quality_score: Number($("launch-min-score").value || 0),
    });
    state.runJob = payload.job;
    state.runJobs = payload.jobs || [payload.job];
    state.runJobLoaded = false;
    renderRunJob(state.runJob);
    renderRunJobList();
    startRunPolling();
  } finally {
    $("launch-run-button").disabled = false;
  }
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
  const payload = await fetchJson("/api/run/status");
  state.runJobs = payload.jobs || [];
  state.runJob = state.runJob || payload.job;
  if (state.runJob) {
    const refreshed = state.runJobs.find((job) => job.id === state.runJob.id || job.run_name === state.runJob.run_name);
    if (refreshed) state.runJob = refreshed;
  }
  keepLauncherRunNameFresh();
  renderRunJob(state.runJob);
  renderRunJobList();
  if (state.runJob?.state === "running") startRunPolling();
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
    $("run-launch-command").innerHTML = "";
    $("run-launch-log").textContent = "READY.";
    $("cancel-run-job-button").disabled = true;
    return;
  }
  $("run-launch-status").textContent =
    `RUN ${String(job.state || "--").toUpperCase()} | ${escapeHtml(job.run_name)} | ${job.elapsed_seconds == null ? "--" : fmtLoss(job.elapsed_seconds)}S | PID ${escapeHtml(job.pid || "--")}`;
  $("cancel-run-job-button").disabled = !job.can_cancel;
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

function renderRunJobList() {
  const jobs = state.runJobs || [];
  if (!jobs.length) {
    $("run-job-list").innerHTML = '<div class="empty">NO WEB-LAUNCHED RUNS FOUND.</div>';
    return;
  }
  $("run-job-list").innerHTML = jobs.slice(-8).reverse().map((job) => `
    <button class="run-job-button ${state.runJob?.id === job.id ? "active" : ""}" type="button" data-run-job="${escapeHtml(job.id)}">
      <strong>${escapeHtml(job.run_name)}</strong>
      <span>${escapeHtml(String(job.state || "--").toUpperCase())} | ${job.summary_exists ? "SUMMARY" : "NO SUMMARY"} | ${escapeHtml(job.source || "--")}</span>
    </button>
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
  $("flight-plan").innerHTML = "";
  $("flight-command").innerHTML = "";
  const report = await postJson("/api/corpus/preview", {
    ...payload,
    preview_chars: 900,
  });
  state.datasetFlightPlan = report;
  state.corpusSourcePreview = report;
  state.tuningInspection = tuningInspectionFromPreview(report);
  if (report.training_command?.eval_input && !$("flight-eval-out-path").value.trim()) {
    $("flight-eval-out-path").value = suggestedEvalStarterPath(report.training_command.eval_input);
  }
  if (report.dataset_pack) {
    $("launch-pack-path").value = report.dataset_pack;
    $("preview-pack-path").value = report.dataset_pack;
  }
  renderDatasetFlightPlan(report);
  renderCorpusSourcePreview(report);
  renderTuningInspection(state.tuningInspection);
  $("flight-check-button").disabled = false;
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

async function createEvalStarter() {
  const packPath = $("flight-pack-path").value.trim();
  const inputPath = $("flight-input-path").value.trim();
  const outPath = $("flight-eval-out-path").value.trim();
  if (!packPath && !inputPath) {
    throw new Error("enter a dataset pack or corpus path first");
  }
  if (!outPath) {
    throw new Error("enter an eval starter output path");
  }
  $("flight-eval-button").disabled = true;
  $("flight-eval-result").innerHTML = 'CREATING EVAL STARTER<span class="cursor"></span>';
  const report = await postJson("/api/eval/starter", {
    dataset_pack: packPath || null,
    input_path: packPath ? null : inputPath,
    out_path: outPath,
    max_items: 24,
    seed: state.detail?.summary?.config?.seed ?? 42,
    force: false,
  });
  state.evalStarter = report;
  $("flight-eval-path").value = report.output_path || outPath;
  $("preview-eval-path").value = report.output_path || outPath;
  $("tuning-eval-path").value = packPath ? "" : report.output_path || outPath;
  $("editor-eval-path").value = packPath ? "" : report.output_path || outPath;
  renderEvalStarter(report);
  $("flight-eval-button").disabled = false;
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
      ${Object.entries(report.categories || {}).map(([name, count]) => `<span>${escapeHtml(name)} ${fmtInt(count)}</span>`).join("")}
    </div>
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

function applyFlightPlanToLauncher() {
  const report = state.datasetFlightPlan;
  if (!report) {
    flashStatus("APPLY FAULT. | Check a dataset first.");
    return;
  }
  if (report.dataset_pack) {
    $("launch-pack-path").value = report.dataset_pack;
    $("launch-run-name").value = uniqueRunName(suggestedRunName(report.dataset_pack));
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
  $("launch-min-score").value = report.min_quality_score ?? ($("flight-min-score").value || 0);
  flashStatus("APPLIED PLAN. | Review launcher values before starting the run.");
}

function renderDatasetFlightPlanError(error) {
  $("flight-check-button").disabled = false;
  $("flight-status").textContent = "DATASET CHECK FAULT";
  $("flight-plan").innerHTML = "";
  $("flight-command").innerHTML = "";
  $("flight-eval-result").innerHTML = `<div class="notice">FAULT: ${escapeHtml(error.message)}</div>`;
}

function renderEvalStarterError(error) {
  $("flight-eval-button").disabled = false;
  $("flight-eval-result").innerHTML = `<div class="notice">EVAL STARTER FAULT: ${escapeHtml(error.message)}</div>`;
}

function suggestedEvalStarterPath(path) {
  const text = String(path || "").trim();
  if (!text) return "my_pack/eval_starter.jsonl";
  const slash = text.lastIndexOf("/");
  const dir = slash >= 0 ? text.slice(0, slash + 1) : "";
  const file = slash >= 0 ? text.slice(slash + 1) : text;
  const stem = file.replace(/\.jsonl$/i, "").replace(/\.json$/i, "") || "eval";
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
  renderDatasetFlightPlan(report);
  renderCorpusSourcePreview(report);
  renderTuningInspection(state.tuningInspection);
  $("preview-corpus-button").disabled = false;
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
      </div>
      ${renderCategoryCounts(chatData?.categories)}
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
      </div>
      ${renderCategoryCounts(evalData?.categories)}
      ${renderSplitCounts(evalData?.splits)}
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
  let units = [...text];
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
  return units.map((token) => ({ token, label: tokenLabel(token) }));
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
  renderComparison(comparison);
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
        </tr>
      </thead>
      <tbody>
        ${comparison.rows.map((row) => `
          <tr class="${row.run === comparison.best_run ? "best-row" : ""}">
            <td>${escapeHtml(row.run)}</td>
            <td>${escapeHtml(row.tokenizer_type || "--")}</td>
            <td>${escapeHtml(row.eval_score)}</td>
            <td>${fmtPercent(row.pass_rate)}</td>
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
    <p class="notice">${baseline ? `Compared against ${escapeHtml(baseline.run)}. Higher pass rate is good; use BPB, not raw loss, when comparing tokenizers. Best SFT BPB: ${bestSftBpb ? escapeHtml(bestSftBpb.run) : "--"}.` : "Only one run selected."}</p>
  `;
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
  if (changes.some((change) => ["n_embd", "n_layer", "n_head"].includes(change.key))) return "Test whether more model capacity improves learning without extra memorization.";
  if (changes.some((change) => change.key.includes("steps"))) return "Test whether more optimization time improves validation and eval.";
  if (changes.some((change) => change.key.includes("learning_rate") || change.key.includes("lr_"))) return "Test whether the optimizer schedule trains more cleanly.";
  if (changes.some((change) => change.key.includes("dataset") || change.key.includes("input"))) return "Test whether different data improves the behavior target.";
  return `Test config changes: ${changes.slice(0, 3).map((change) => change.label).join(", ")}.`;
}

function runConfigLine(item) {
  const config = item.detail?.summary?.config || {};
  const row = item.row || {};
  const shape = [config.n_embd, config.n_layer, config.n_head].every((value) => value !== undefined)
    ? `${config.n_embd}x${config.n_layer} h${config.n_head}`
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
  if (["n_embd", "n_layer", "n_head"].includes(key)) return "Changes model capacity and compute cost.";
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
      reason: evalFailureReason(item, failure),
      fix: evalFailureFix(item, failure),
    };
  });
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
  if (hasMissingSupport(item)) return "add SFT rows that include the missing phrases/entities";
  if (hasPromptEcho(item)) return "add examples that answer without repeating the prompt";
  return "add 5-10 targeted SFT rows, then rerun eval";
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
  $("status-line").textContent = `READY. | MODE ${state.viewMode.toUpperCase()} | PANEL ${state.activePanel.toUpperCase()} | RUN ${run} | CTX ${ctx} | TOK ${tok} | SEED ${seed}`;
}

boot().catch((error) => {
  $("run-count").textContent = "FAULT";
  $("status-line").textContent = `FAULT. | ${error.message}`;
});

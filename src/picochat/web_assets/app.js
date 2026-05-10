const state = {
  runs: [],
  selectedRun: null,
  detail: null,
  activePanel: "dataset",
  activeStage: "dataset",
  activeReport: "summary",
  compareRuns: [],
  corpusSourcePreview: null,
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
  await loadRunPresets();
  await loadRuns();
  await loadRunJobs();
}

function bindControls() {
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
    }
  });
  $("compare-button").addEventListener("click", () => {
    loadComparison().catch((error) => renderCompareError(error));
  });
  $("preview-corpus-button").addEventListener("click", () => {
    previewCorpusSources().catch((error) => renderCorpusSourcePreviewError(error));
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

function renderAll() {
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
  renderPipeline();
  if (name === "report") {
    loadReport().catch((error) => renderReportError(error));
  } else if (name === "compare") {
    loadComparison().catch((error) => renderCompareError(error));
  }
  renderStatus();
}

function renderPipeline() {
  const stages = pipelineStages();
  $("pipeline-run").textContent = state.selectedRun ? `RUN ${state.selectedRun}` : "NO RUN";
  $("pipeline-strip").innerHTML = stages.map((stage) => renderPipelineStage(stage)).join("");
  const active = stages.find((stage) => stage.id === state.activeStage) || stages[0];
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
        ["Duplicate lines", fmtPercent(corpus.duplicate_line_rate || 0)],
      ],
      note: detail?.corpus_preview ? `Preview: ${compactPreview(detail.corpus_preview, 220)}` : "No corpus preview artifact found.",
      command: datasetCommand(config, artifacts),
      ledger: [
        artifactItem("INPUT", "Source", config.corpus_recipe || config.corpus_input || "examples/tiny_corpus.txt"),
        artifactItem("OUTPUT", "Corpus", corpusPath),
        artifactItem("OUTPUT", "Manifest", artifacts.corpus_manifest || `${outDir}/corpus_manifest.json`),
        artifactItem("OUTPUT", "Report", artifacts.corpus_report || `${outDir}/corpus_report.md`),
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
        "--early-stop-patience", config.base_early_stop_patience ?? 6,
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
        "--early-stop-patience", config.sft_early_stop_patience ?? 6,
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
      summary: `${reportCount}/4 markdown reports`,
      stats: [
        ["Summary", detail?.reports?.summary?.exists ? "ready" : "missing"],
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
  state.tuningInspection = tuningInspectionFromPreview(report);
  if (report.dataset_pack) {
    $("launch-pack-path").value = report.dataset_pack;
    $("launch-min-score").value = report.min_quality_score ?? minQualityScore;
    if (!$("launch-run-name").value) $("launch-run-name").value = suggestedRunName(report.dataset_pack);
  }
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
      ${renderIssues(evalData?.issues || [])}
      ${renderEvalPreview(evalData?.preview || [])}
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
  const units = tokenizer.type === "byte" ? byteTokenUnits(text) : charTokenUnits(text);
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
    label: char === " " ? "space" : char === "\n" ? "\\n" : char,
  }));
}

function byteTokenUnits(text) {
  return Array.from(new TextEncoder().encode(text)).map((byte) => {
    const hex = byte.toString(16).padStart(2, "0");
    const printable = byte >= 32 && byte <= 126 ? String.fromCharCode(byte) : `0x${hex}`;
    return { token: `<byte:${hex}>`, label: printable === " " ? "space" : printable };
  });
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
  $("report-status").textContent = `${ready}/4 REPORTS`;
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

async function loadComparison() {
  if (!state.compareRuns.length) {
    throw new Error("select at least one run");
  }
  $("compare-status").textContent = "COMPARING";
  const query = state.compareRuns.map((run) => `run=${encodeURIComponent(run)}`).join("&");
  const comparison = await fetchJson(`/api/compare?${query}`);
  renderComparison(comparison);
}

function renderComparison(comparison) {
  $("compare-status").textContent = `BEST ${comparison.best_run}`;
  $("compare-summary").innerHTML = compareSummary(comparison);
  $("compare-table").innerHTML = `
    <label>COMPARISON TABLE</label>
    <table>
      <thead>
        <tr>
          <th>Run</th>
          <th>Eval</th>
          <th>Pass</th>
          <th>Base Val</th>
          <th>SFT Val</th>
          <th>Params</th>
          <th>Ctx</th>
          <th>Trunc</th>
        </tr>
      </thead>
      <tbody>
        ${comparison.rows.map((row) => `
          <tr class="${row.run === comparison.best_run ? "best-row" : ""}">
            <td>${escapeHtml(row.run)}</td>
            <td>${escapeHtml(row.eval_score)}</td>
            <td>${fmtPercent(row.pass_rate)}</td>
            <td>${fmtLoss(row.base_val_loss)}</td>
            <td>${fmtLoss(row.sft_val_loss)}</td>
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
  const baseline = rows[0]?.run === best?.run ? rows[1] : rows[0];
  if (!best) return "NO COMPARISON ROWS.";
  const passDelta = baseline ? best.pass_rate - baseline.pass_rate : 0;
  const sftDelta = baseline ? best.sft_val_loss - baseline.sft_val_loss : 0;
  return `
    <div class="compare-cards">
      <div class="pipeline-stat">
        <label>Best run</label>
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
        <label>SFT val delta</label>
        <span>${baseline ? signedLoss(sftDelta) : "--"}</span>
      </div>
    </div>
    <p class="notice">${baseline ? `Compared against ${escapeHtml(baseline.run)}. Higher pass rate is good; lower SFT validation loss is usually healthier.` : "Only one run selected."}</p>
  `;
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
  $("eval-status").textContent = `${latest.name.toUpperCase()} ${report.summary.num_passed}/${report.summary.num_examples}`;
  $("score-table").innerHTML = `
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
        <label>Unanswerable</label>
        <span>${honesty.numUnanswerable}/${honesty.numExamples}</span>
      </div>
    </div>
    <label>ARCADE SCORE TABLE</label>
    <table>
      <thead><tr><th>Rank</th><th>Prompt</th><th>Kind</th><th>Status</th><th>Support</th><th>Forbidden</th></tr></thead>
      <tbody>
        ${report.examples.map((item, index) => `
          <tr>
            <td>${String(index + 1).padStart(2, "0")}</td>
            <td>${escapeHtml(item.user)}</td>
            <td>${evalKindTag(item)}</td>
            <td class="${item.passed ? "pass-text" : "fail-text"}">${item.passed ? "PASS" : "FAIL"}</td>
            <td class="${hasMissingSupport(item) ? "fail-text" : "pass-text"}">${hasMissingSupport(item) ? "MISSING" : "COVERED"}</td>
            <td class="${hasForbiddenClaim(item) ? "fail-text" : "pass-text"}">${hasForbiddenClaim(item) ? "FOUND" : "CLEAR"}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
  $("eval-results").innerHTML = report.examples.map((item, index) => evalCard(item, index)).join("");
}

function evalHonestySummary(report) {
  const examples = report.examples || [];
  const summary = report.summary || {};
  const numExamples = summary.num_examples ?? examples.length;
  const unsupportedClaims = summary.unsupported_claims ?? examples.filter(hasForbiddenClaim).length;
  const missingSupport = summary.missing_support ?? examples.filter(hasMissingSupport).length;
  const numUnanswerable = summary.num_unanswerable ?? examples.filter((item) => !isAnswerable(item)).length;
  return {
    numExamples,
    numUnanswerable,
    unsupportedClaimRate: summary.unsupported_claim_rate ?? unsupportedClaims / Math.max(1, numExamples),
    missingSupportRate: summary.missing_support_rate ?? missingSupport / Math.max(1, numExamples),
  };
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
        <p>ANSWERABLE: ${escapeHtml(isAnswerable(item) ? "yes" : "no")}</p>
        <p>REQUIRED: ${escapeHtml((item.must_include || []).join(" | ") || "none")}</p>
        <p>ANY: ${escapeHtml((item.must_include_any || []).map((group) => `[${group.join(" / ")}]`).join(" ") || "none")}</p>
        <p>FORBIDDEN: ${escapeHtml((item.must_not_include || []).join(" | ") || "none")}</p>
        <p>MISSING: ${escapeHtml([...(item.missing || []), ...((item.missing_any || []).flat())].join(" | ") || "none")}</p>
        <p>FOUND FORBIDDEN: ${escapeHtml((item.found_forbidden || []).join(" | ") || "none")}</p>
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
  $("status-line").textContent = `READY. | PANEL ${state.activePanel.toUpperCase()} | RUN ${run} | CTX ${ctx} | TOK ${tok} | SEED ${seed}`;
}

boot().catch((error) => {
  $("run-count").textContent = "FAULT";
  $("status-line").textContent = `FAULT. | ${error.message}`;
});

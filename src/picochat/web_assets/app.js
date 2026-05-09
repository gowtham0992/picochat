const state = {
  runs: [],
  selectedRun: null,
  detail: null,
  activePanel: "dataset",
  activeStage: "dataset",
  activeReport: "summary",
  compareRuns: [],
  corpusSourcePreview: null,
  tokenTimer: null,
  generationTimer: null,
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
  await loadRuns();
}

function bindControls() {
  $("refresh-button").addEventListener("click", loadRuns);
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
  const baseCheckpoint = summary.base?.checkpoint || `${outDir}/base/checkpoint`;
  const sftCheckpoint = summary.sft?.checkpoint || `${outDir}/sft/checkpoint`;
  const tokenizer = detail?.tokenizer_detail || summary.tokenizer || {};
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
      note: detail?.corpus_preview ? `Preview: ${detail.corpus_preview.slice(0, 180).replace(/\s+/g, " ")}` : "No corpus preview artifact found.",
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
      summary: tokenizer.vocab_size ? `${tokenizer.vocab_size} vocab / ${tokenizer.special_tokens?.length || 0} special` : "not trained",
      stats: [
        ["Type", tokenizer.type || "unknown"],
        ["Vocab", tokenizer.vocab_size ?? "--"],
        ["Special", tokenizer.special_tokens?.length ?? "--"],
        ["Text tokens", summary.tokenizer?.num_text_tokens ?? "--"],
      ],
      note: "Turns text into token IDs before the model ever sees it.",
      command: shellCommand([
        "PYTHONPATH=src", "python", "-m", "picochat.cli", "tok", "train",
        "--input", corpusPath,
        "--out", tokenizerPath,
      ]),
      ledger: [
        artifactItem("INPUT", "Corpus", corpusPath),
        artifactItem("OUTPUT", "Tokenizer JSON", tokenizerPath),
      ],
    },
    {
      id: "base",
      label: "BASE TRAIN",
      summary: baseLast ? `val ${fmtLoss(baseLast.val_loss)} / ${fmtInt(summary.base?.num_parameters)} params` : "no trace",
      stats: [
        ["Steps", config.base_steps ?? detail?.base_report?.config?.max_steps ?? "--"],
        ["Train loss", baseLast ? fmtLoss(baseLast.train_loss) : "--"],
        ["Val loss", baseLast ? fmtLoss(baseLast.val_loss) : "--"],
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
        "--seed", config.seed ?? 42,
        "--device", config.device || "cpu",
      ]),
      ledger: [
        artifactItem("INPUT", "Corpus", corpusPath),
        artifactItem("INPUT", "Tokenizer", tokenizerPath),
        artifactItem("OUTPUT", "Checkpoint", baseCheckpoint),
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
        <label>${nextStage ? "NEXT COMMAND" : "NEXT STEP"}</label>
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
      <label>COMMAND TAPE</label>
      <code>${escapeHtml(stage.command || "NO COMMAND AVAILABLE.")}</code>
    </div>
  `;
}

function renderDataset() {
  const summary = state.detail?.summary || {};
  const corpus = summary.corpus || {};
  const config = summary.config || {};
  const baseDataset = state.detail?.base_report?.dataset || {};
  const manifest = state.detail?.corpus_manifest;
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
    ["Val fraction", baseDataset.num_sequences ? fmtPercent((baseDataset.val_sequences || 0) / baseDataset.num_sequences) : "--"],
  ]);
  $("corpus-files").innerHTML = renderCorpusFiles(manifest?.files || []);
  renderCorpusSourcePreview(state.corpusSourcePreview);
  $("corpus-preview").textContent = state.detail?.corpus_preview || "NO CORPUS PREVIEW ARTIFACT FOUND.";
}

function seedSourcePreviewInputs(config) {
  const recipeInput = $("preview-recipe-path");
  const sourceInput = $("preview-input-path");
  if (recipeInput.value || sourceInput.value) return;
  recipeInput.value = config.corpus_recipe || "examples/corpus_recipe.json";
  sourceInput.value = config.corpus_input || "";
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
        <span>${escapeHtml(file.included ? "INCLUDED" : "SKIPPED")}</span>
      </div>
      <small>${escapeHtml(file.extension)}${file.label ? ` | label:${escapeHtml(file.label)}` : ""} | ${fmtInt(file.num_characters)} chars | ${fmtInt(file.num_lines)} lines | ${escapeHtml(file.reason)}</small>
    </div>
  `).join("");
}

async function previewCorpusSources() {
  const recipePath = $("preview-recipe-path").value.trim();
  const inputPath = $("preview-input-path").value.trim();
  if (!recipePath && !inputPath) {
    throw new Error("enter a recipe path or input path");
  }

  $("preview-corpus-button").disabled = true;
  $("source-preview-status").innerHTML = 'PREVIEWING SOURCES<span class="cursor"></span>';
  $("source-preview-stats").innerHTML = "";
  $("source-preview-files").innerHTML = "";
  $("source-preview-text").textContent = "";
  const report = await postJson("/api/corpus/preview", {
    recipe_path: recipePath || null,
    input_path: inputPath || null,
    preview_chars: 1400,
  });
  state.corpusSourcePreview = report;
  renderCorpusSourcePreview(report);
  $("preview-corpus-button").disabled = false;
}

function renderCorpusSourcePreview(report) {
  if (!report) {
    $("source-preview-status").textContent = "NO SOURCE PREVIEW REQUESTED.";
    $("source-preview-stats").innerHTML = "";
    $("source-preview-files").innerHTML = '<div class="empty">NO PREVIEW PLAN LOADED.</div>';
    $("source-preview-text").textContent = "READY.";
    return;
  }
  const files = report.files || [];
  const included = files.filter((file) => file.included);
  const skipped = files.length - included.length;
  const stats = report.stats || {};
  $("source-preview-status").textContent =
    `${fmtInt(included.length)} INCLUDED | ${fmtInt(skipped)} SKIPPED | ${fmtInt(stats.num_characters)} CHARS`;
  $("source-preview-stats").innerHTML = statCards([
    ["Input", report.input_path || "unknown"],
    ["Recipe", report.recipe_path || "none"],
    ["Files", fmtInt(stats.num_files)],
    ["Documents", fmtInt(stats.num_documents)],
    ["Characters", fmtInt(stats.num_characters)],
    ["Lines", fmtInt(stats.num_lines)],
  ]);
  $("source-preview-files").innerHTML = renderCorpusFiles(files);
  $("source-preview-text").textContent = report.preview || "(EMPTY)";
}

function renderCorpusSourcePreviewError(error) {
  $("preview-corpus-button").disabled = false;
  $("source-preview-status").textContent = "SOURCE PREVIEW FAULT";
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
    ? `TOK ${tokenizer.vocab_size} | SPECIAL ${tokenizer.special_tokens.length}`
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
  const chars = [...text];
  const stream = $("token-stream");
  stream.innerHTML = "";
  let index = 0;
  state.tokenTimer = setInterval(() => {
    if (index >= chars.length) {
      clearInterval(state.tokenTimer);
      stream.insertAdjacentHTML("beforeend", '<span class="cursor"></span>');
      return;
    }
    const char = chars[index];
    const id = tokenizer.token_to_id[char] ?? tokenizer.token_to_id["<unk>"];
    const label = char === " " ? "space" : char === "\n" ? "\\n" : char;
    stream.insertAdjacentHTML(
      "beforeend",
      `<span class="token-step"><b>${escapeHtml(label)}</b><em>${id}</em></span>`
    );
    index += 1;
  }, 80);
}

function renderTraining() {
  const detail = state.detail;
  const baseLosses = detail?.base_report?.losses || [];
  const sftLosses = detail?.sft_report?.losses || [];
  const sftLast = sftLosses.at(-1);
  const overfit = sftLast && sftLast.val_loss > sftLast.train_loss + 1;
  $("training-badge").textContent = overfit ? "MEMORIZED WARNING" : "LOSS TRACE READY";
  $("training-badge").classList.toggle("warning", Boolean(overfit));
  $("base-loss-chart").textContent = asciiLossChart(baseLosses);
  $("sft-loss-chart").textContent = asciiLossChart(sftLosses);
  $("training-table").innerHTML = trainingRows(baseLosses, sftLosses);
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
      <thead><tr><th>Stage</th><th>Step</th><th>Train</th><th>Val</th><th>Gap</th></tr></thead>
      <tbody>
        ${rows.map(([name, row]) => row ? `
          <tr>
            <td>${name}</td>
            <td>${row.step}</td>
            <td>${fmtLoss(row.train_loss)}</td>
            <td>${fmtLoss(row.val_loss)}</td>
            <td>${fmtLoss(row.val_loss - row.train_loss)}</td>
          </tr>
        ` : "").join("")}
      </tbody>
    </table>
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

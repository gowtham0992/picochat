import type { RunDetail, RunSummary, Tone } from "./types";

export function percent(value?: number | null, digits = 1): string {
  if (value == null || Number.isNaN(value)) return "--";
  return `${(value * 100).toFixed(digits)}%`;
}

export function compactNumber(value?: number | null): string {
  if (value == null || Number.isNaN(value)) return "--";
  const abs = Math.abs(value);
  if (abs >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(1)}B`;
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (abs >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return String(value);
}

export function fixed(value?: number | null, digits = 4): string {
  if (value == null || Number.isNaN(value)) return "--";
  return value.toFixed(digits);
}

export function runTone(run?: RunSummary | null): Tone {
  if (!run) return "neutral";
  if ((run.pass_rate || 0) >= 0.5) return "pass";
  if ((run.pass_rate || 0) > 0) return "warn";
  return "block";
}

export function releaseTone(detail?: RunDetail | null): Tone {
  const summary = detail?.summary || {};
  const release = summary.release_gate || summary.long_run_gate || {};
  const status = String(release.status || release.verdict || "").toLowerCase();
  if (status.includes("approved") || status.includes("pass")) return "pass";
  if (status.includes("block") || status.includes("fail")) return "block";
  if (status.includes("warn") || status.includes("review")) return "warn";
  const evalSummary = summary.eval || {};
  if ((evalSummary.pass_rate || 0) >= 0.5) return "pass";
  if ((evalSummary.pass_rate || 0) > 0) return "warn";
  return "block";
}

export function statusLabel(tone: Tone): string {
  if (tone === "pass") return "Pass";
  if (tone === "warn") return "Review";
  if (tone === "block") return "Blocked";
  if (tone === "running") return "Running";
  return "Ready";
}

export function latestRun(runs: RunSummary[]): RunSummary | null {
  return runs.length ? runs[runs.length - 1] : null;
}

export function parseEvalScore(score?: string): { passed: number; total: number } {
  const [passed, total] = String(score || "0/0").split("/").map((part) => Number(part.trim()));
  return { passed: Number.isFinite(passed) ? passed : 0, total: Number.isFinite(total) ? total : 0 };
}

export function lossPoints(detail?: RunDetail | null): Array<{ step: number; train?: number; val?: number }> {
  const reports = [detail?.base_report, detail?.sft_report].filter(Boolean) as Array<Record<string, any>>;
  const rows: Array<{ step: number; train?: number; val?: number }> = [];
  for (const report of reports) {
    const history = report.history || report.steps || report.losses || [];
    if (Array.isArray(history)) {
      for (const row of history) {
        const step = Number(row.step ?? row.global_step ?? rows.length + 1);
        const train = Number(row.train_loss ?? row.train ?? row.loss);
        const val = Number(row.val_loss ?? row.val);
        rows.push({
          step,
          train: Number.isFinite(train) ? train : undefined,
          val: Number.isFinite(val) ? val : undefined
        });
      }
    }
  }
  if (!rows.length) {
    const base = Number(detail?.summary?.base?.final_val_loss);
    const sft = Number(detail?.summary?.sft?.final_val_loss);
    if (Number.isFinite(base)) rows.push({ step: 1, train: base, val: base });
    if (Number.isFinite(sft)) rows.push({ step: 2, train: sft, val: sft });
  }
  return rows;
}

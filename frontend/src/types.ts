export type Tone = "pass" | "warn" | "block" | "info" | "neutral" | "running";

export interface RunSummary {
  name: string;
  path: string;
  eval_score: string;
  pass_rate: number;
  domain_pass_rate?: number | null;
  refusal_pass_rate?: number | null;
  base_val_loss?: number | null;
  sft_val_loss?: number | null;
  num_parameters?: number | null;
  context_size?: number | null;
  truncated_examples?: number | null;
  skipped_long_examples?: number | null;
}

export interface ReportStatus {
  name?: string;
  path?: string;
  exists?: boolean;
}

export interface RunDetail {
  summary: Record<string, any>;
  base_report?: Record<string, any> | null;
  sft_report?: Record<string, any> | null;
  eval?: Record<string, any> | null;
  eval_reports?: Record<string, any> | null;
  preflight?: Record<string, any> | null;
  reports?: Record<string, ReportStatus>;
  artifact_inventory?: Record<string, any> | null;
  run_timeline?: Array<Record<string, any>>;
  handoff_packet?: Record<string, any> | null;
  release_repair_plan?: Record<string, any> | null;
  run_passport?: Record<string, any> | null;
}

export interface JobStatus {
  id: string;
  run_name: string;
  state: "running" | "succeeded" | "failed" | "cancelled" | string;
  out_dir: string;
  log_path?: string;
  updated_at?: number;
  progress?: Record<string, any> | null;
}

export interface RunLog {
  job_id?: string;
  run_name?: string;
  state?: string;
  running?: boolean;
  log_path?: string;
  log_tail?: string;
  progress?: Record<string, any> | null;
  updated_at?: number;
}

export interface GeneratedToken {
  token?: string;
  text?: string;
  prob?: number;
  logprob?: number;
}

export interface GenerateResult {
  run: string;
  checkpoint: "base" | "sft" | string;
  prompt: string;
  text: string;
  completion: string;
  generated_tokens?: GeneratedToken[];
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  finish_reason: "stop" | "length" | string;
  temperature?: number;
  top_k?: number;
  top_p?: number;
  max_new_tokens?: number;
  seed?: number;
}

export interface Preset {
  label: string;
  description: string;
  context_size?: number;
  base_steps?: number;
  sft_steps?: number;
  n_embd?: number;
  n_head?: number;
  n_layer?: number;
  device?: string;
  [key: string]: any;
}

export type PresetMap = Record<string, Preset>;

export interface ModelConfig {
  n_layer: number;
  n_head: number;
  n_embd: number;
  context_size: number;
  params?: number;
}

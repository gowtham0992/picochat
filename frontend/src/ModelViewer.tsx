import type { ModelConfig } from "./types";
import { compactNumber } from "./utils";

/**
 * ModelViewer — a crisp 2D schematic of the transformer the run produced.
 *
 * Reads straight from the run config: a token-embedding base, a stack of N
 * decoder blocks (each showing its attention heads), and the LM head on top,
 * threaded by the residual stream. `heat` (eval pass-rate) charges the stack
 * from the bottom up. No WebGL, no drag gimmick — it reads like a spec sheet,
 * scales to any size, and respects reduced motion.
 */
export default function ModelViewer({
  config,
  heat,
  accent
}: {
  config: ModelConfig;
  heat: number;
  accent: [number, number, number];
}) {
  const ac = rgb(accent);
  const layers = Math.max(1, config.n_layer || 1);
  const heads = Math.max(1, config.n_head || 1);
  const visible = Math.min(layers, 8);
  const collapsed = layers > visible;
  const litFrac = clamp01(heat);

  const W = 460;
  const H = 420;
  const cx = 170;
  const blockW = 188;
  const blockH = 26;
  const gap = 9;
  const headPips = Math.min(heads, 12);

  const stackH = visible * blockH + (visible - 1) * gap;
  const baseY = H - 96;
  const stackBottom = baseY - 34;
  const stackTop = stackBottom - stackH;

  // Blocks are drawn top→bottom but charged bottom→top.
  const blocks = Array.from({ length: visible }, (_, row) => {
    const y = stackTop + row * (blockH + gap);
    const fromBottom = visible - 1 - row;
    const lit = (fromBottom + 0.5) / visible <= litFrac + 1e-6;
    return { y, lit, fromBottom };
  });

  const headLabel = visible === layers ? `${layers}` : `${visible} of ${layers}`;

  return (
    <svg
      className="pc-model-svg"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={`Schematic of a ${layers}-layer, ${heads}-head transformer`}
      style={{ ["--ac" as string]: ac }}
    >
      <defs>
        <linearGradient id="pc-spine" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={ac} stopOpacity="0.85" />
          <stop offset="100%" stopColor={ac} stopOpacity="0.12" />
        </linearGradient>
      </defs>

      {/* residual stream */}
      <line x1={cx} y1={stackTop - 30} x2={cx} y2={baseY} stroke="url(#pc-spine)" strokeWidth="2.5" />

      {/* LM head */}
      <g>
        <path
          d={`M ${cx - 26} ${stackTop - 18} L ${cx + 26} ${stackTop - 18} L ${cx + 14} ${stackTop - 40} L ${cx - 14} ${stackTop - 40} Z`}
          fill="color-mix(in srgb, var(--ac) 18%, transparent)"
          stroke={ac}
          strokeWidth="1.5"
          strokeLinejoin="round"
        />
        <text className="pc-model-tag" x={cx} y={stackTop - 50} textAnchor="middle">LM head</text>
      </g>

      {/* decoder blocks */}
      {blocks.map((b, i) => (
        <g key={i} className={b.lit ? "pc-block lit" : "pc-block"} style={{ ["--d" as string]: `${b.fromBottom * 90}ms` }}>
          <rect
            x={cx - blockW / 2}
            y={b.y}
            width={blockW}
            height={blockH}
            rx="6"
            fill={b.lit ? "color-mix(in srgb, var(--ac) 16%, var(--panel-2, #15161d))" : "var(--panel-2, #15161d)"}
            stroke={b.lit ? ac : "var(--line-2, #2a2c38)"}
            strokeWidth="1.25"
          />
          {Array.from({ length: headPips }, (_, h) => {
            const span = blockW - 36;
            const hx = cx - span / 2 + (headPips === 1 ? span / 2 : (h / (headPips - 1)) * span);
            return (
              <circle
                key={h}
                cx={hx}
                cy={b.y + blockH / 2}
                r="2.4"
                fill={b.lit ? ac : "var(--faint, #5b5f72)"}
                opacity={b.lit ? 0.95 : 0.5}
              />
            );
          })}
        </g>
      ))}

      {collapsed ? (
        <text className="pc-model-tag" x={cx + blockW / 2 + 12} y={stackTop + 14} textAnchor="start">
          +{layers - visible}
        </text>
      ) : null}

      {/* embedding base */}
      <g>
        <rect
          x={cx - blockW / 2 - 8}
          y={baseY}
          width={blockW + 16}
          height={30}
          rx="8"
          fill="color-mix(in srgb, var(--ac) 10%, var(--panel-2, #15161d))"
          stroke="var(--line-2, #2a2c38)"
          strokeWidth="1.25"
        />
        <text className="pc-model-tag strong" x={cx} y={baseY + 19} textAnchor="middle">
          token + position embedding
        </text>
      </g>

      {/* annotations */}
      <g className="pc-model-ann">
        <Annot x={cx + blockW / 2 + 26} y={stackTop + stackH * 0.32} k="blocks" v={headLabel} ac={ac} />
        <Annot x={cx + blockW / 2 + 26} y={stackTop + stackH * 0.62} k="heads / block" v={`${heads}`} ac={ac} />
        <Annot x={cx + blockW / 2 + 26} y={stackTop + stackH * 0.92} k="d_model" v={`${config.n_embd || "--"}`} ac={ac} />
        <Annot x={cx - blockW / 2 - 86} y={stackTop + stackH * 0.32} k="context" v={`${config.context_size || "--"}`} ac={ac} right />
        <Annot x={cx - blockW / 2 - 86} y={stackTop + stackH * 0.62} k="params" v={compactNumber(config.params)} ac={ac} right />
        <Annot
          x={cx - blockW / 2 - 86}
          y={stackTop + stackH * 0.92}
          k="eval charge"
          v={`${Math.round(litFrac * 100)}%`}
          ac={ac}
          right
        />
      </g>
    </svg>
  );
}

function Annot({ x, y, k, v, ac, right }: { x: number; y: number; k: string; v: string; ac: string; right?: boolean }) {
  const anchor = right ? "start" : "start";
  return (
    <g transform={`translate(${x}, ${y})`}>
      <text className="pc-model-k" x="0" y="-7" textAnchor={anchor}>{k}</text>
      <text className="pc-model-v" x="0" y="9" textAnchor={anchor} style={{ fill: ac }}>{v}</text>
    </g>
  );
}

function rgb(c: [number, number, number]) {
  return `rgb(${Math.round(clamp01(c[0]) * 255)}, ${Math.round(clamp01(c[1]) * 255)}, ${Math.round(clamp01(c[2]) * 255)})`;
}

function clamp01(v: number) {
  return Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0));
}

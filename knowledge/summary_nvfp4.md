# Picochat Research Note: Pretraining Large Language Models With NVFP4

## Paper

- Title: Pretraining Large Language Models with NVFP4
- arXiv: 2509.25149
- Link: https://arxiv.org/abs/2509.25149
- Authors: NVIDIA et al.
- Source read: local TeX source cached at `~/.cache/codex/arxiv/2509.25149/`
- Related implementation: https://github.com/NVIDIA/TransformerEngine/pull/2177

## One-line Verdict

This is useful for Picochat's long-term precision roadmap, but it should not change the next H100 plan: run BF16 or FP8 first, keep the run honest, and treat NVFP4 as a future Blackwell/TransformerEngine experiment.

## Core Idea

The paper shows that 4-bit floating point pretraining can be made stable when FP4 is applied selectively and carefully:

- Linear GEMMs can use NVFP4.
- Sensitive layers stay high precision, especially final blocks.
- Random Hadamard Transforms are applied only to weight-gradient inputs.
- Weights use 2D 16x16 block scaling.
- Activations and gradients use 1D 1x16 scaling.
- Gradients use stochastic rounding.
- Forward-pass tensors use nearest rounding, not stochastic rounding.

The key lesson is not "turn everything into 4-bit." The lesson is "low precision only works when the training stack controls where quantization happens, how scaling is done, and which layers are protected."

## Evidence

NVIDIA reports a 12B model trained for 10T tokens with NVFP4, compared against an FP8 baseline. Their headline result is that validation loss and downstream metrics remain close to FP8.

Important reported results:

- MMLU-Pro: FP8 62.62%, NVFP4 62.58%.
- General benchmark average: FP8 68.99, NVFP4 69.82.
- Code was weaker under NVFP4: code average FP8 59.52, NVFP4 56.67.
- NVFP4 validation loss tracked FP8 closely, with relative loss error usually under about 1% during the stable phase and somewhat higher during decay.
- Switching from FP4 back to higher precision late in training can heal much of the remaining loss gap.

The paper also shows failure modes:

- Naive NVFP4 diverges.
- Quantizing every linear layer is unstable.
- Final layers are especially sensitive.
- RHT on forward or dgrad paths can hurt.
- Stochastic rounding is helpful for gradients but harmful for forward tensors.

## Picochat Fit

Picochat already has the right first-stage precision foundation:

- `src/picochat/precision.py` supports `float32`, `bf16`, `fp16`, and `auto` through `torch.autocast`.
- `src/picochat/model.py` already uses SDPA and supports RoPE, RMSNorm, ReLU2, KV cache, and gradient checkpointing.
- Current model linears are standard `nn.Linear` with bias and fused QKV.
- Picochat has no FP8, FP4, TransformerEngine, per-layer quantization policy, RHT, 2D scaling, or stochastic gradient rounding.

The direct NVFP4 implementation is not a Mac feature and not a normal H100 feature. It belongs behind a future optional backend, likely tied to NVIDIA TransformerEngine and Blackwell-class hardware. For the upcoming H100 path, the useful practical analog is FP8, not NVFP4.

What Picochat should adopt now from this paper:

- Add a precision roadmap that separates `bf16_baseline`, `fp8_experimental`, and `nvfp4_future`.
- Add a precision sensitivity diagnostic before any FP8/FP4 work: measure per-linear activation max, gradient max, and simulated quantization error.
- Keep embeddings, norms, attention softmax/QK/AV matmuls, output head, optimizer states, and the last N blocks in high precision for any narrow-precision experiment.
- Prefer opt-in precision experiments. Never make FP8/FP4 part of the first credibility run.
- Consider WSD learning-rate schedules and high-quality final-phase training as later recipe polish.

## Nanochat Comparison

Nanochat currently uses FP8 on H100 through `nanochat/fp8.py`, not NVFP4. Its speedrun enables `--fp8` for the large run, and its docs are honest that unsupported GPUs fall back to BF16.

This paper confirms nanochat's general approach: low precision is a speed tool, not a quality tool, and it must remain optional. Picochat should learn from that without becoming a clone:

- Nanochat optimizes for speedrun throughput.
- Picochat should optimize for a defensible domain SLM factory: data audit, preflight gates, contamination checks, resumability, export, model card, and reproducible run evidence.
- FP8/NVFP4 should be late-stage acceleration, not the identity of the project.

## Implementation Candidate

Do not implement NVFP4 in Picochat immediately.

Better sequence:

1. Add a short precision roadmap note to the repo and H100 handoff.
2. Add a precision sensitivity report for current BF16 runs.
3. Add a simple FP8 readiness gate that says whether hardware/software supports FP8.
4. After BF16 H100 baseline works, add optional FP8 Linear replacement or TransformerEngine integration.
5. Only after FP8 is proven, consider NVFP4 as a separate backend for Blackwell systems.

Concrete future API shape:

```text
--precision bf16
--precision fp8-experimental
--precision nvfp4-experimental
--low-precision-keep-final-blocks 4
--low-precision-sensitivity-report
```

The first real implementation target should be the sensitivity report, because it helps BF16, FP8, and future FP4 without risking training stability.

## Validation Plan

For Picochat:

1. Run current Mac/Colab/H100 BF16 baseline.
2. Record loss curve, validation BPB, throughput, VRAM, SFT fit, eval pass rate, contamination report, and export manifest.
3. Add low-precision sensitivity report and check whether final blocks show larger activation/gradient/quantization error.
4. Only then test FP8 on H100 against the exact BF16 baseline.
5. Accept FP8 only if it improves capability-matched wall-clock or memory without hurting eval confidence intervals.
6. Treat NVFP4 as publishable only if it has a clean BF16/FP8 comparison and hardware-specific reproducibility notes.

## Risks

- NVFP4 could distract us from the immediate goal: a credible first base SLM release.
- Implementing FP4 in pure PyTorch would be research-code theater, not a production training backend.
- H100 does not make this paper directly actionable in the way Blackwell/TransformerEngine does.
- Low precision can make losses look normal while hurting specific skills such as code.
- If Picochat adds precision tricks before locking data and eval, it will be harder to know whether an improvement came from the model, data, or noise.

## Decision

Keep the next serious Picochat run simple and defensible:

- BF16 on H100 first.
- Optional FP8 only after BF16 is a known baseline.
- NVFP4 tracked as future work.
- Immediate actionable work: precision sensitivity diagnostics and a written precision policy.

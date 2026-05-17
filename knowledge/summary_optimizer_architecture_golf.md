# Optimizer and Architecture Lessons from Nanochat and Parameter Golf

## One-line Verdict

Use Parameter Golf and nanochat ideas as opt-in ablation knobs first; do not silently replace the 1B release preset with unvalidated optimizer or evaluation-time tricks.

## Sources Checked

- OpenAI Parameter Golf retrospective: https://openai.com/index/what-parameter-golf-taught-us/
- nanochat GPT architecture and optimizer discussion: https://github.com/karpathy/nanochat/discussions/481
- Local nanochat code: `/Users/g/Documents/andrej/nanochat/nanochat/gpt.py`
- Local Parameter Golf record package: `/Users/g/Documents/andrej/parameter-golf-universal/records/track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/README.md`
- Muon distributed optimizer paper: https://openreview.net/pdf/1bff28e2c4dc62d54931563338d4dd25112a0203.pdf
- Dion parameter grouping guidance: https://github.com/microsoft/dion

## What Looks Worth Borrowing

1. LeakyReLU2 activation as an architecture ablation.
   - The strongest local Parameter Golf stack uses LeakyReLU(0.5)^2.
   - Picochat already supports `relu2`; `leaky_relu2` is a small opt-in extension with the same MLP parameter shape.
   - This is safe to test at 100M without changing data, eval, optimizer state, or release defaults.

2. Distributed Muon as a future serious ablation, not a default switch.
   - nanochat's optimizer uses AdamW for embeddings/scalars and a specialized distributed Muon path for matrix parameters.
   - Picochat's current Muon is intentionally transparent and rank-local under DDP.
   - The paper and nanochat discussion both point at real distributed-Muon engineering, not just parameter grouping.

3. Depth recurrence, SmearGate, sparse attention gates, and XSA as controlled model ablations.
   - Parameter Golf showed these can matter under tight artifact/time constraints.
   - They change training dynamics and may interact with document boundaries, BOS handling, and cache equivalence.
   - They should stay behind knobs and pass pre-H100 sanity before any paid long run.

## What Not to Copy Into Release Pretraining

1. Test-time training as a release-quality base model claim.
   - Parameter Golf TTT was valid under its rules, but Picochat's release goal is an honest closed-book base/SFT model.
   - TTT can be added later as an inference experiment, not as base-model intelligence.

2. GPTQ/LQER/compression stack as a training quality improvement.
   - These are mainly artifact and inference/eval packaging techniques.
   - Useful later for deployment, not for making the pretrained checkpoint smarter.

3. CaseOps/tokenizer sidecar without rebuilding the whole data/eval story.
   - It is interesting, but it changes byte accounting and tokenizer comparability.
   - Picochat should keep regex HF BPE as the release path until a tokenizer ablation proves otherwise.

## Picochat Decision

- Keep the H200 1B release preset conservative: AdamW, FA3, DDP8, regex HF BPE, modern transformer options, skill-release data/gate.
- Add `leaky_relu2` as an opt-in knob now.
- Keep DDP+Muon warning in preflight until Picochat has a real distributed Muon implementation or a validated 100M DDP ablation.

## Validation Plan

Run this before promoting any activation change to 1B:

1. Same 100M DDP8 data pack and run budget.
2. Compare `activation=swiglu`, `activation=relu2`, and `activation=leaky_relu2`.
3. Require:
   - lower or equal base validation BPB,
   - no SFT fit regression,
   - no held-out eval regression,
   - no throughput regression large enough to erase the BPB gain.

Only promote the winner if the result survives at least two seeds or a clear BPB margin larger than normal run noise.

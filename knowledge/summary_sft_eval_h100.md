# Picochat SFT/Eval Research Note For ClimbMix H100 Path

## Scope

Goal: train the strongest honest closed-book Picochat SLM we can on NVIDIA/Nemotron-ClimbMix after local and Colab tests, then spend one 8xH100 run only after the recipe passes readiness gates.

Current verdict: do not spend the H100 run yet. Picochat is directionally aligned with the research, but the next improvement should be SFT/eval protocol and data packing, not another blind architecture change.

## Paper: LIMA, Less Is More For Alignment

- arXiv: 2305.11206
- Link: https://arxiv.org/abs/2305.11206

### One-line Verdict

Implement the lesson now: SFT should be small, high-quality, diverse, and behavior-focused; it should not be expected to create knowledge the base model never learned.

### Core Idea

Most knowledge and capability comes from pretraining. SFT mostly teaches the model which response format and behavioral subdistribution to use.

### Evidence

The paper fine-tuned a 65B LLaMA with only about 1,000 carefully curated prompt/response pairs and no RLHF. The source argues that capability is learned almost entirely during pretraining, while alignment teaches format/style. Its ablations show diversity and response quality matter more than scaling SFT quantity alone. It also warns that held-out SFT perplexity can move in the wrong direction while generation quality improves.

### Picochat Fit

Maps to `src/picochat/benchmark_pack.py`, `src/picochat/sft.py`, and `src/picochat/run_preflight.py`.

Picochat should separate two SFT modes:

- behavior SFT: identity, formatting, refusal, multiple-choice protocol, concise math/spelling answer style.
- skill/data SFT: only when the base model has enough pretraining signal for that skill.

### Nanochat Comparison

Nanochat does this in practice: base training is the knowledge phase; SFT teaches chat format, identity, multiple-choice, tool/math style, and spelling behavior using broad curated mixtures.

### Implementation Candidate

Add an H100 gate that blocks long runs unless behavior SFT exact-fit is at least 70%, preferably 85%, on a held-out-clean behavior curriculum. Keep weak-skill SFT separate from behavior SFT so we can see whether a failure is format or knowledge.

### Validation Plan

Run a behavior-only SFT sweep against the same base checkpoint. Require improved SFT fit, stable or improved held-out eval, zero prompt echo, and no contamination warnings.

### Risks

High SFT fit can still be memorization if rows are duplicated or overlap eval prompts. SFT fit is diagnostic, not final intelligence.

## Paper: Training Compute-Optimal Large Language Models

- arXiv: 2203.15556
- Link: https://arxiv.org/abs/2203.15556

### One-line Verdict

Already partly implemented; tighten it into a hard H100 readiness gate.

### Core Idea

For a fixed compute budget, model size and training tokens should scale together. Undertraining a larger model is usually worse than training a smaller model longer.

### Evidence

The paper fits many models across model sizes and token budgets, then shows Chinchilla outperforming larger undertrained models with more training tokens. It explicitly emphasizes choosing model size and tokens before a single expensive run.

### Picochat Fit

Maps to `src/picochat/run_preflight.py`.

Picochat now estimates parameter count, corpus tokens, planned tokens, target token/parameter ratio, recommended steps, and epoch exposure. This is the right direction.

### Nanochat Comparison

Nanochat exposes `--target-param-data-ratio` and its speedrun uses that ratio to configure base training.

### Implementation Candidate

For H100 mode, block if planned/target tokens are outside the acceptable band, if corpus epochs are too high, or if token/parameter ratio is not explicitly recorded.

### Validation Plan

Before H100, run a 3-run local/Colab ablation with the same corpus/model but different planned/target ratios. Pick the best validation BPB per compute minute.

### Risks

The Chinchilla ratio is a guide, not magic. Tiny SLMs and synthetic/curated corpora may prefer different ratios, so Picochat must report the assumption.

## Paper: Textbooks Are All You Need / Phi-1

- arXiv: 2306.11644
- Link: https://arxiv.org/abs/2306.11644

### One-line Verdict

Implement as a data-quality strategy, not as a claim that synthetic rows alone solve everything.

### Core Idea

Small models can punch above their size when the training data is clear, self-contained, instructive, balanced, and diverse.

### Evidence

Phi-1 used a 1.3B model trained on about 7B high-quality textbook-style tokens plus a smaller exercise finetuning set, achieving strong code benchmarks. The paper shows filtered data beating much larger unfiltered data and stresses diversity in synthetic data.

### Picochat Fit

Maps to `src/picochat/data.py`, `src/picochat/benchmark_pack.py`, and future corpus-quality reports.

ClimbMix is a good base source, but Picochat should add curriculum-quality reports: length distribution, duplicate/similarity, topic diversity, answer style diversity, and source mixture.

### Nanochat Comparison

Nanochat uses broad curated post-training tasks rather than relying only on raw pretraining data. Picochat's benchmark pack is moving toward this but is still much smaller.

### Implementation Candidate

Add a "curriculum quality" report for benchmark/SFT packs: unique groups, duplicate prompts, answer length distribution, category entropy, near-duplicate answers, and held-out template families.

### Validation Plan

Compare SFT fit and held-out eval for same row count before/after quality filtering. Do not increase row count until quality metrics improve.

### Risks

Synthetic rows can become repetitive and can leak template structure into eval. Diversity must be measured, not assumed.

## Paper: TinyStories

- arXiv: 2305.07759
- Link: https://arxiv.org/abs/2305.07759

### One-line Verdict

Use its evaluation lesson now: small models need task-appropriate eval plus memorization/novelty checks.

### Core Idea

A small, refined dataset can make tiny models useful inside a narrow distribution, but evaluation must check diversity and memorization, not just loss.

### Evidence

TinyStories trains very small models to produce coherent stories and evaluates them with held-out prompts plus overlap/memorization analysis such as k-gram and nearest-story overlap.

### Picochat Fit

Maps to `src/picochat/eval.py`, honesty reports, and memorization reports.

Picochat already has canaries, prompt echo checks, support diagnostics, and leakage checks. The missing piece is a stronger nearest-neighbor novelty report for generated answers versus base corpus and SFT rows.

### Nanochat Comparison

Nanochat uses fixed benchmark tasks and centered metrics; TinyStories is more about small-model interpretability and novelty checks.

### Implementation Candidate

Add an eval novelty panel: max n-gram overlap and nearest SFT/corpus neighbor for each generated answer.

### Validation Plan

Any score improvement should be accepted only if novelty does not regress and prompt echo remains zero.

### Risks

Novel text can still be wrong; low overlap is not proof of understanding.

## Paper: Tulu 3 Open Post-Training

- arXiv: 2411.15124
- Link: https://arxiv.org/abs/2411.15124

### One-line Verdict

Test later, but adopt the process discipline now: clear post-training stages, eval suites, and data provenance.

### Core Idea

Open post-training should be reproducible: define mixtures, keep held-out evaluations, track changes, and report category-level failures.

### Evidence

Tulu 3 is an open post-training recipe with staged SFT/preference-style alignment and broad evaluation. The useful lesson for Picochat is not to copy scale, but to make every post-training dataset and evaluation auditable.

### Picochat Fit

Maps to benchmark pack reports, leaderboard, eval reports, and UI run readiness.

### Nanochat Comparison

Nanochat is also explicit about the SFT mixture and ChatCORE benchmark tasks. Picochat should expose the same clarity in its workbench.

### Implementation Candidate

Add a "post-training manifest" to every run: SFT source, row counts, category counts, split method, contamination status, and whether eval prompts/answers appear in SFT.

### Validation Plan

The UI and CLI should show one page answering: "what did this SFT teach, what was held out, and why should we trust the score?"

### Risks

Process discipline does not improve capability by itself; it prevents false confidence.

## Paper: Evaluation Contamination / Data Leakage Work

- Representative arXiv direction: benchmark contamination and eval trust in LLMs.

### One-line Verdict

Implement stronger contamination gates before H100.

### Core Idea

If benchmark prompts, answers, or near-duplicates appear in training data, scores can look like intelligence while measuring memorization.

### Evidence

Recent contamination papers repeatedly show that overlap can appear in subtle ways: exact prompt copies, paraphrases, answer leakage, and template reuse.

### Picochat Fit

Maps to `src/picochat/benchmark_pack.py`, `src/picochat/eval.py`, and honesty reports.

Picochat checks exact/near prompt overlap and answer overlap between SFT and eval. It should also compare eval prompts and expected answers against the base corpus and generated benchmark corpus.

### Nanochat Comparison

Nanochat uses known benchmark splits for SFT/eval tasks. Picochat needs to make this visible and enforceable because its user can create arbitrary domain packs.

### Implementation Candidate

Add a preflight contamination matrix:

- base corpus vs eval prompts
- base corpus vs eval answers
- SFT prompts vs eval prompts
- SFT answers vs eval answers
- generated answers vs nearest SFT/corpus row after eval

### Validation Plan

Long-run readiness fails if exact overlap exists and warns on high near-overlap. Reports should show samples.

### Risks

Aggressive decontamination can remove legitimate domain facts. The report should distinguish "leakage" from "domain overlap."

## Overall H100 Decision

Do not run 8xH100 yet.

The research says our bottleneck is not "find one clever optimizer." The bottleneck is making the training/eval protocol strong enough that a big run is meaningful:

1. Base pretraining must be budgeted by tokens/params and validation BPB.
2. SFT must teach behavior with high-quality diverse rows.
3. Eval must be held out, centered against random baseline, and contamination-audited.
4. Data packing/token utilization must be fixed before scale.

## Next Implementation Order

1. Add nanochat-style BOS-bestfit SFT packing.
2. Add eval random baseline and centered score in leaderboard/UI.
3. Add contamination matrix against base corpus, SFT, eval, and generated answers.
4. Add benchmark-pack diversity/quality report and fix the behavior profile uniqueness ceiling.
5. Add H100 readiness gate that requires: passed preflight, no leakage, SFT fit gate, choice score above random, stable BPB, and explicit token/parameter budget.
6. Only then run local/Colab ablations and pick one H100 recipe.

## One-Sentence Verdict

More arXiv research is useful only in this targeted way: it confirms we should harden SFT/eval/packing/readiness now, then stop reading and run controlled ablations.

# CoT Persona Probes

**Do a reasoning model's chain-of-thought activations predict the persona its final answer will express?**

While a reasoning model thinks, its final answer does not exist yet — but its hidden states might already encode *which persona is about to speak*. This project generates rollouts for conversations that naturally pull Qwen3-32B away from its default Assistant persona, scores each final answer in the Assistant Axis persona space of Lu et al. [1], collects chain-of-thought (CoT) activations at sentence boundaries, and fits probes that predict the answer's persona coordinates from the CoT — across layers and reasoning time.

This repository is a fork of the [future-probes codebase](https://github.com/kortukov/future_probes) (Kortukov et al. [2]); see the upstream repository for the original behavior-distribution and future-probe steering experiments. Everything below documents this fork's persona pipeline.

## Motivation

Kortukov et al. [2] show that intermediate CoT activations encode the *distribution over future behaviors* of the answer a reasoning model will eventually give. Related work shows that hidden states during reasoning encode the *correctness* of the eventual answer well before it is produced [11, 12, 13]. This project asks the analogous question for **persona**: while the model thinks, do its CoT activations already encode which persona the final answer will express — and how early, and at which layers?

Persona is a natural target because it is (a) linearly represented — character traits and roles correspond to directions in activation space [1, 6, 14] — and (b) *unstable over conversations*: models measurably drift away from their default "Assistant" persona over multi-turn dialogues, most strongly in emotionally or philosophically charged ones [1, 15]. Since Shanahan et al. [16] frame LLM chat behavior as role play over a superposition of characters, the question becomes: does the CoT reveal which character is about to speak?

We use **Qwen3-32B** [17] because the Assistant Axis project [1] publishes precomputed persona-space artifacts for exactly this model (assistant axis + 275 role vectors at all 64 layers), and because Qwen3's hybrid thinking mode gives clean `<think>...</think>` traces.

## Installation

We use `uv` to manage the environment ([install instructions](https://docs.astral.sh/uv/getting-started/installation/)). In the project root:

```bash
uv sync
```

## Pipeline overview

Per rollout, the pipeline:

1. **generates** a CoT and final answer for a persona-drift conversation prefix (`unsteered_generation.py`),
2. **scores the final answer** in the Assistant Axis persona space — an axis scalar plus multi-dimensional persona coordinates (`src/interp/persona_axis.py`),
3. **collects CoT activations** at sentence/step boundaries across layers (`gather_persona_drift_activations.py`),
4. **tests whether those CoT activations predict** the persona coordinates of the eventual answer, across layers and reasoning time, against baselines and shuffled controls (`analyze_persona_drift_probes.py`).

Multiple prompts and several stochastic rollouts per prompt give both **cross-prompt** variation (different conversations induce different personas) and **within-prompt** variation (stochastic sampling makes the same prefix land on different personas).

```
persona-drift transcripts ──▶ unsteered_generation.py ──▶ *_results.json / *_outputs.json
                                                                    │
     lu-christina/assistant-axis-vectors (HF) ──┐                   ▼
                                                ├─▶ gather_persona_drift_activations.py
                                                │            │
                                                │            ▼
                                                │   *_persona_activations.pt   (v1 format)
                                                │            │
                                                └────────────▼
                                       analyze_persona_drift_probes.py ──▶ *_probe_analysis.json
```

## Stage 1 — Persona-drift rollouts

### Why persona-drift conversations instead of role prompts

Assistant Axis persona-drift transcripts [1] are multi-turn conversations in which a frontier "auditor" model simulates a human user (with a hidden persona and topic) across four domains — coding, writing, therapy-like emotional support, and philosophical discussion — while the target model replies **without any system prompt**. Lu et al. show these conversations *naturally* pull the model away from its default Assistant persona, most strongly in the therapy and philosophy domains [1]. Li et al. [15] document the same instability for explicit instructions: persona/instruction adherence decays significantly within eight dialogue rounds. Using these conversations rather than explicit "You are a pirate" role prompts means the persona of the answer is *not stated anywhere in the context* — any predictive signal in the CoT must come from the model's own latent persona state, not from copying a role instruction. This mirrors how the persona-vectors line of work distinguishes *monitoring emergent persona shifts* from *inducing* them by prompt [14].

### Dataset construction (`src/custom_datasets/persona_drift.py`)

We load the upstream transcripts (`data/persona_drift/*.json`, from [safety-research/assistant-axis](https://github.com/safety-research/assistant-axis) `transcripts/persona_drift/`) and turn **every user turn into one prompt**: the example's messages are the conversation prefix through that user turn, so the model answers turn *t* with turns *1..t−1* as drift-inducing history. This follows Lu et al.'s turn-resolved drift measurement, which tracks the axis projection as a function of turn index [1]. Auditor-side metadata (persona, topic, domain) is kept for analysis but **never** shown to the model. Dataset names: `persona_drift` (all domains) or `persona_drift_<domain>`.

**Estimand note.** The four published transcripts were recorded with *other* target models (coding/therapy: Llama-3.3-70B; philosophy/writing: Gemma-2-27B). Reusing them as frozen prefixes therefore measures *Qwen3-32B conditioned on foreign-model conversation histories*, not Qwen drifting across its own conversation. With only four conversations (one per domain), cross-conversation results on this data are **pilot-scale only**; the within-prompt analysis (history held fixed) is unaffected by the history's provenance. Regenerating conversations with Qwen as the live target (protocol in `docs/persona_drift_generation_protocol.md`) removes both caveats.

The four checked-in transcripts (one per domain, 58 user-turn prefixes) are the only persona-drift transcripts released by the upstream project; the paper's full set (4 domains × 5 personas × 20 topics = 400 conversations, per auditor) is not published, and the upstream `pipeline/` is its role-vector extraction pipeline, not conversation generation. The protocol is fully reproducible, however: Appendix E of Lu et al. [1] publishes the verbatim topic-generation prompt and auditor system prompt (an LLM simulates the user given a domain + persona + topic; the target model gets no system prompt; up to 15 turns). To expand the prompt set, regenerate conversations with that protocol and drop the JSON files into `data/persona_drift/` (or pass `--dataset_path`) — the checked-in transcripts show the expected schema.

### Generation (`unsteered_generation.py`)

Rollouts are sampled with vLLM [18] at temperature 1.0 with `enable_thinking=True`, several samples per prompt (`--num_samples`), reusing the upstream rollout machinery. Two persona-drift-specific behaviors:

- **No behavior labels.** Persona-drift prompts have no binary behavior; the persona target is computed downstream in activation space. Behavior detection and stability plotting are skipped automatically (`supports_behavior_scoring = False`).
- **Exact tokenization capture.** The output JSON stores `prompt_token_ids` (as consumed by vLLM) and per-response `token_ids` so that activation gathering can *replay the exact rollout tokens*. Re-tokenizing concatenated text can silently move BPE boundaries; replaying stored ids guarantees that the token positions we probe are the token positions the model actually generated.
- **Sampling.** We default to temperature 1.0 with no nucleus/top-k truncation to maximize within-prompt rollout diversity — the within-prompt analysis needs stochastic variation. Qwen3's recommended thinking-mode decoding is temperature 0.6, top-p 0.95, top-k 20 [17]; pass `--temperature 0.6 --top_p 0.95 --top_k 20` to reproduce the recommended operating point instead (all three are logged in the results file).

```bash
uv run unsteered_generation.py \
    --model_name Qwen/Qwen3-32B \
    --dataset persona_drift \
    --subset 50 --seed 42 \
    --num_samples 8 \
    --max_new_tokens 4096 \
    --temperature 1.0 \
    --max_model_len 16384
```
Results land in `results/behavioral_stability/Qwen3-32B/base_model/persona_drift/…_results.json` (plus `…_outputs.json`). Long conversation prefixes are allowed: for `requires_long_context` datasets the vLLM context defaults to the model maximum rather than the short-prompt cap. On a single 80 GB GPU, Qwen3-32B needs an explicit `--max_model_len` cap (the 40k default does not leave enough KV-cache memory next to the bf16 weights); 16384 comfortably covers the longest transcript prefix (~8k tokens) plus generation.

## Stage 2 — Scoring answers in the Assistant Axis persona space

### The persona space (`src/interp/persona_axis.py`)

We adopt the Assistant Axis construction of Lu et al. [1] wholesale, using their published Qwen3-32B artifacts ([lu-christina/assistant-axis-vectors](https://huggingface.co/datasets/lu-christina/assistant-axis-vectors), tensors of shape `[64 layers, 5120]`):

- **Role vectors**: for each of 275 character roles, the mean post-MLP residual-stream activation over all response tokens of fully-role-playing responses (1,200 rollouts per role; responses filtered by an LLM role-adherence judge) [1].
- **Assistant axis**: a *difference of means* — the mean default-Assistant activation minus the mean of all role vectors, at every layer. It points from role-play toward the default Assistant, so a **higher projection = more Assistant-like**; drift shows up as a declining projection [1]. Difference-in-means is the standard way to extract such concept directions [7, 8].
- **Target layer**: the middle layer (**layer 32** of 64, zero-indexed decoder-layer outputs), following the upstream convention of using the middle residual-stream layer for analyses [1].

### Scoring a final answer

For each rollout we take the **mean residual-stream activation over the final-answer tokens** (the public answer after `</think>`; the CoT is excluded) at layer 32, and compute:

1. **`assistant_axis_score`** — the dot product with the unit-normalized assistant axis. This is exactly the upstream projection used to monitor drift, which also averages activations over response tokens before projecting [1].
2. **`persona_coordinates`** — coordinates in a persona subspace, in one of two modes:
   - `pca` (default): project the (role-mean-centered) answer activation onto principal components fitted on the mean-centered role vectors. Lu et al. run an analogous PCA over mean-centered role vectors — on a larger basis (n = 463 vectors for Qwen3-32B) than the 275 published per-role centroids used here — and find the assistant axis closely aligned with role-PC1 [1]; PCA over activation differences is likewise how Representation Engineering constructs its reading vectors [6]. On the published Qwen3-32B role centroids, PC1 explains 37.9% of variance and the top-8 PCs explain 72.6%. PC1 tracks the validated axis; treat PC2+ as exploratory coordinates.
   - `role_cosine`: cosine similarity to each of the 275 centered role centroids — a maximally interpretable (but high-dimensional, non-orthogonal) coordinate system.

Scoring answers (rather than thinking) also keeps the probe targets close to the distribution the artifacts were extracted from: the upstream vectors come from non-thinking response tokens [1].

The artifacts are auto-downloaded from HuggingFace on first use, or pass `--assistant_axis_path` to a local copy (`assistant_axis.pt` + `role_vectors/`).

**Sanity check.** A small Qwen3-32B pilot (6 prompts × 3 rollouts) reproduces the upstream drift phenomenon end-to-end through this scoring path: final-answer axis projections decline over the turns of a philosophy conversation (turn 2 ≈ −42 → turn 9 ≈ −53 → turn 13 ≈ −58) while coding/writing prefixes stay far more Assistant-like (≈ −18 to −22), matching the domain ordering reported by Lu et al. [1]. Reference artifacts (per-rollout scores, generation config, probe-analysis report) are checked in under `docs/pilot_qwen3_32b/`; probe metrics at this scale are pilot-only (see Limitations).

## Stage 3 — CoT activations at step boundaries

### Boundary definition (`gather_persona_drift_activations.py`)

Following Kortukov et al. [2] and Thought Anchors [10], the unit of CoT analysis is the **sentence**: the thinking block (`<think>…</think>`) is split with the NLTK Punkt sentence tokenizer [19], and each sentence's **end-of-sentence token** is a boundary. Kortukov et al. probe exactly these end-of-sentence token positions [2]; the last token of a segment is where transformer representations summarize the segment.

Character-level sentence spans are mapped back to token positions with the fast tokenizer's offset mapping (validated by re-encoding the decoded text and requiring the exact generated ids), with a binary-search-over-prefix-decodes fallback — so boundaries always index into the *replayed* rollout tokens.

### Extraction

For each rollout, one nnsight [20] forward pass over the exact `prompt_token_ids + token_ids` sequence collects, in a single trace (`src/interp/activations_nnsight.py::extract_boundary_and_span_activations`):

- the residual-stream state at every CoT boundary token, for a **set of layers spanning the model's depth** (`--layers auto`: 8 evenly spaced layers + the axis layer; probing across depth follows the layer-sweep methodology of linear-probe studies [3, 5]),
- the **prompt-end state** (last prompt token, before any generated token) at the same layers — the input for the context-only baseline probe,
- the **mean answer-token activation** at the axis layer (the Stage 2 target).

Each boundary also gets a `cot_progress` value — the fraction of thinking tokens completed — so trajectories of different lengths can be compared on a normalized reasoning-time axis (Stage 4).

Rollouts whose thinking never closes (no `</think>` within the token budget), or with empty thinking/answer, are skipped and logged with reasons.

```bash
uv run gather_persona_drift_activations.py \
    --results_file results/behavioral_stability/Qwen3-32B/base_model/persona_drift/n50_nsamp8_l4096_gumbel_s42_ss42_t1.0_results.json \
    --layers auto \
    --axis_layer 32 \
    --coordinate_mode pca
```

This writes a single `…_persona_activations.pt` (version-1 format):

```
{
  "format_version": 1,
  "layers": LongTensor[num_layers],
  "persona_coordinate_names": [str, ...],
  "prompts": [
    { "conversation_id": str, "turn_index": int, "metadata": {...},
      "rollouts": [
        { "rollout_index": int,
          "cot_activations": Tensor[boundaries, layers, hidden]  (fp16),
          "cot_progress": Tensor[boundaries],
          "assistant_axis_score": float,
          "persona_coordinates": Tensor[coordinates], ... }, ... ] }, ... ],
  "config": {...}
}
```

## Stage 4 — Probing: layers × reasoning time (`analyze_persona_drift_probes.py`)

### Probe

For every (layer, reasoning-time checkpoint) cell we fit a **multi-output ridge regression** from the CoT boundary state to the target vector `[assistant_axis_score, persona_coordinates…]`, with a standardization step on the inputs. (Targets — and the shuffled control targets — are columns of one joint solve; multi-output least squares/ridge with a shared design decomposes into independent per-target fits [27], the same vectorization voxelwise encoding models use to fit thousands of targets at once [28].) Ridge probes [21] are the standard choice for reading *continuous* quantities out of residual-stream activations — Gurnee & Tegmark use exactly L2-regularized linear probes to read continuous space/time coordinates from layer activations [5]; linear probes in general follow Alain & Bengio [3]. Reasoning time is normalized: checkpoint *t* ∈ (0,1] uses the boundary state at the last sentence completed within the first *t* fraction of thinking tokens (`--trajectory-representation latest`, or `cumulative_mean` for a running average); rollouts whose first sentence ends after *t* are excluded from that cell rather than represented by a future state, so early checkpoints never see later reasoning (note the evaluated cohort then changes across bins; `--cohort fixed` restricts every bin to rollouts valid at all checkpoints for a constant cohort). The ridge penalty can be tuned on training data only via `--ridge-alphas`, selected per fit by grouped cross-validation. Analyzing predictability as a function of position within the CoT follows [2, 11, 12, 13].

### Evaluations: cross-prompt and within-prompt

Two complementary splits, mirroring the two sources of variation in the data:

- **`cross_prompt`** holds out entire conversations by default (`--cross-group conversation`): because persona-drift prompts are nested prefixes of one transcript, holding out single prompts (`--cross-group prompt`) would leak conversation identity between train and evaluation. This measures whether a probe transfers to unseen conversations — grouped held-out evaluation guards against the probe memorizing prompt or conversation identity, one of the standard confounds in probing methodology [4, 9].
- **`within_prompt`** holds out stochastic rollouts *inside* each prompt, and centers both activations and targets by their *training-rollout* prompt means. Metrics are computed in this prompt-residual space, so the probe only gets credit for predicting *which way this particular rollout deviates* from the prompt's average persona — the per-question analogue of how self-verification probes predict the outcome of an individual reasoning trajectory [11, 12]. The centering itself is the classical within-transformation of panel-data econometrics [25]; per-prompt-group normalization of activations is likewise how CCS [23] and Cluster-Norm [24] keep probes from latching onto prompt/context identity, the dominant-feature failure mode documented by Farquhar et al. [29].

### Baselines and controls

Every probe result is reported against:

- **`context` probe cells** — for every layer, the same probe fitted on the **prompt-end activation** (before any reasoning token). This is the context-only baseline: the auditor-generated history itself carries persona/topic cues, so a CoT cell only demonstrates *incremental* predictive power by beating the context cell at the same layer. (Persona-vector work shows pooled correlations can be dominated by between-condition prompt differences [14] — this baseline separates them.)
- **`prompt_mean`** — predict each prompt's training-rollout mean target. Meaningful for `within_prompt`; under cross-prompt holdout every evaluation prompt is unseen, so `prompt_mean` degenerates to `global_mean` there — use the context probe as the cross-prompt context baseline.
- **`global_mean`** — predict the global training mean,
- **`target_shuffled_probe`** — the identical probe refit on permuted targets, following the control-task methodology of Hewitt & Liang [4]. For cross-prompt evaluation the permutation is a **block permutation over conversations** (whole groups swap target blocks), preserving within-conversation target correlation so the control is not artificially easy to beat — the exchangeability-block / multi-level block permutation practice of permutation inference for clustered data [26]; for within-prompt it permutes rollouts inside each prompt.

Metrics: **R²**, **MAE**, and **Pearson r** per target, plus persona-space macro averages — R² for continuous probe targets follows [5], MAE follows [2]. Repeated splits (`--split-repeats`) report mean ± std.

```bash
uv run analyze_persona_drift_probes.py \
    --activations results/behavioral_stability/Qwen3-32B/base_model/persona_drift/n50_nsamp8_l4096_gumbel_s42_ss42_t1.0_persona_activations.pt \
    --layers all --time-bins 4 \
    --splits cross_prompt,within_prompt \
    --split-repeats 5 --shuffle-repeats 5
```

The JSON report contains one entry per (split, layer, time-bin) with probe/baseline/control metrics and split diagnostics.

### Reading the results

- If a CoT cell beats the **context cell** and the shuffled control **cross-prompt**, CoT states carry transferable persona information beyond what the conversation context already predicts.
- If the probe beats the baselines **within-prompt** (in prompt-residual space), the CoT state predicts *rollout-specific* persona deviations — information associated with the sampled reasoning trajectory rather than the shared context. This is a predictive claim only: it establishes neither a causal role for the CoT nor CoT faithfulness [22].
- The layer × time grid shows *where* (depth) and *when* (reasoning time) the final-answer persona becomes readable — analogous to how outcome-correctness becomes predictable early in the CoT [12] and how look-ahead information is concentrated at pivot positions [13].

## Design decisions ← sources

| Decision | Follows |
|---|---|
| Persona-drift conversations, no role prompt, no system prompt | Lu et al. [1] persona-drift protocol |
| One prompt per user turn (turn-resolved prefixes) | Lu et al. [1] turn-resolved drift curves |
| Several stochastic rollouts per prompt, T=1.0 | resampling-rollout methodology [2, 10] |
| Persona target = mean answer-token activation @ layer 32 · unit axis | upstream projection convention [1] |
| Persona coordinates via PCA on centered role centroids | Lu et al. [1]; RepE reading vectors [6] |
| Axis/role directions as difference-in-means | [1, 7, 8, 14] |
| Sentence = CoT step; probe end-of-sentence tokens | Kortukov et al. [2]; Thought Anchors [10] |
| Punkt sentence segmentation | Kiss & Strunk [19] |
| Ridge probes for continuous targets, layer sweep | Gurnee & Tegmark [5]; Alain & Bengio [3]; Hoerl & Kennard [21] |
| Shuffled-target control probes (conversation-block permutation) | Hewitt & Liang control tasks [4]; block permutation for clustered data [26] |
| Context-only (prompt-end) baseline probe | incremental-validity control; prompt-driven vs. emergent persona shifts [14] |
| Within-prompt centering of X and y by training prompt means | within/fixed-effects transformation [25]; per-prompt normalization in CCS [23] and Cluster-Norm [24] |
| Joint multi-output solve (targets + controls as columns) | per-target independence of multi-output ridge [27]; voxelwise encoding models [28] |
| Conversation-grouped held-out splits; leakage caveats | Belinkov [9] |
| Within-prompt residual evaluation (predicting a single trajectory's outcome) | self-verification / temporal-outcome probing [11, 12] |
| Score only the public answer; thinking stays private | upstream convention [2]; Qwen3 usage [17] |

**A deliberate deviation:** the Assistant Axis repository recommends disabling thinking for Qwen3 in its own drift experiments [1]. We enable thinking on purpose — the CoT is the object of study — and keep the persona *targets* on non-thinking-style answer tokens, close to the distribution the artifacts were extracted from.

## Limitations

- **The persona target is an activation-space proxy.** Final answers are scored by the same model's layer-32 activations projected into the Assistant Axis space. The axis itself is behaviorally validated upstream (role-play adherence judging, capping evaluations) [1], but this pipeline includes no independent text-side persona judge of *our* answers; PC2+ coordinates are exploratory. Adding an LLM-judge scoring of answer text is the natural validation step.
- **Prediction is not causation, and CoT text may be unfaithful.** A probe reading persona from CoT states shows the information is linearly present — not that the reasoning causes the persona, nor that the CoT text reflects the true process [22]. Causal claims would need interventions (e.g., steering or resampling at boundaries [2, 10]).
- **Dependence structure.** Rollouts within a prompt and prefixes within a conversation are dependent; pooled metrics and repeated-split standard deviations are not cluster-level uncertainty estimates (nested dependence inflates apparent precision [30]). With the 4 public conversations, cross-conversation cells rest on very few independent units — treat them as pilot results until the prompt set is regenerated at scale.
- **Reasoning-time cells are per-bin cohorts by default.** Excluding rollouts with no boundary by a checkpoint prevents future-state leakage but changes the evaluated cohort across bins (`n_excluded_no_boundary` is reported per cell; `--cohort fixed` holds the cohort constant).
- **Report grids, not maxima.** The console prints a best-cell convenience line; it is post-selection over layers × time × splits. Conclusions should come from the full grid in the JSON report against the matched baselines and controls.
- **Foreign-model histories.** See the estimand note in Stage 1: the public prefixes were recorded with Llama/Gemma targets.

## Tests

```bash
uv run python -m unittest discover -s tests
```

covers: span/boundary parsing and token alignment with controlled fake tokenizers, including split multibyte characters (`tests/test_gather_persona_drift_activations.py`); integration tests against the **real Qwen3 tokenizer** — fast-path/fallback equivalence on emoji and CJK rollouts — skipped automatically when the tokenizer is not cached locally (`tests/test_real_tokenizer_integration.py`); the Assistant Axis scorer against manual PCA/cosine computations (`tests/test_persona_axis.py`); prefix construction and metadata isolation of the persona-drift dataset (`tests/test_persona_drift_dataset.py`); and the probe analysis end-to-end on synthetic data where the signal is known (`tests/test_analyze_persona_drift_probes.py`) — including that the probe beats baselines and shuffled controls when signal exists, that early checkpoints exclude rollouts rather than leak future states, that the cross-prompt control is a conversation-block permutation, and that a predictive context beats a noise CoT in the context cell.

## Repository structure

```
src/custom_datasets/persona_drift.py      # Assistant Axis transcripts -> per-user-turn conversation prefixes
src/interp/persona_axis.py                # persona space: axis projection + PCA / role-cosine coordinates
src/interp/activations_nnsight.py         # extract_boundary_and_span_activations (single-trace extraction)
unsteered_generation.py                   # rollout generation (upstream) + token-id capture, label-free datasets
gather_persona_drift_activations.py       # rollout replay -> boundary activations + persona targets (.pt v1)
analyze_persona_drift_probes.py           # ridge probes across layers x reasoning time, baselines, controls
tests/                                    # unit tests for all of the above
data/persona_drift/                       # upstream persona-drift transcripts (4 domains)
```

Other top-level scripts (`behavior_distribution_analysis.py`, `gather_activations.py`, `train_probe.py`, `evaluate_probe.py`, `future_probe_controlled_generation.py`, `activation_steering.py`, `compute_steering_vector.py`) belong to the upstream future-probes experiments — see the [upstream repository](https://github.com/kortukov/future_probes) for their documentation.

## References

[1] Christina Lu, Jack Gallagher, Jonathan Michala, Kyle Fish, Jack Lindsey. **The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models.** arXiv:2601.10387, 2026. [paper](https://arxiv.org/abs/2601.10387) · [code](https://github.com/safety-research/assistant-axis) · [vectors](https://huggingface.co/datasets/lu-christina/assistant-axis-vectors) · [blog](https://www.anthropic.com/research/assistant-axis)

[2] Evgenii Kortukov, Piotr Komorowski, Florian Klein, Paula Engl, Gabriele Sarti, Seong Joon Oh, Sebastian Lapuschkin, Wojciech Samek. **Predicting Future Behaviors in Reasoning Models Enables Better Steering.** Mechanistic Interpretability Workshop at ICML 2026; arXiv:2606.11172. [paper](https://openreview.net/forum?id=48NnVTsirb) · [code](https://github.com/kortukov/future_probes)

[3] Guillaume Alain, Yoshua Bengio. **Understanding intermediate layers using linear classifier probes.** arXiv:1610.01644, 2016. [paper](https://arxiv.org/abs/1610.01644)

[4] John Hewitt, Percy Liang. **Designing and Interpreting Probes with Control Tasks.** EMNLP 2019. [paper](https://arxiv.org/abs/1909.03368)

[5] Wes Gurnee, Max Tegmark. **Language Models Represent Space and Time.** ICLR 2024. [paper](https://arxiv.org/abs/2310.02207)

[6] Andy Zou et al. **Representation Engineering: A Top-Down Approach to AI Transparency.** arXiv:2310.01405, 2023. [paper](https://arxiv.org/abs/2310.01405)

[7] Samuel Marks, Max Tegmark. **The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets.** COLM 2024. [paper](https://arxiv.org/abs/2310.06824)

[8] Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, Neel Nanda. **Refusal in Language Models Is Mediated by a Single Direction.** NeurIPS 2024. [paper](https://arxiv.org/abs/2406.11717)

[9] Yonatan Belinkov. **Probing Classifiers: Promises, Shortcomings, and Advances.** Computational Linguistics 48(1), 2022. [paper](https://aclanthology.org/2022.cl-1.7/)

[10] Paul C. Bogdan, Uzay Macar, Neel Nanda, Arthur Conmy. **Thought Anchors: Which LLM Reasoning Steps Matter?** arXiv:2506.19143, 2025. [paper](https://arxiv.org/abs/2506.19143)

[11] Anqi Zhang, Yulin Chen, Jane Pan, Chen Zhao, Aurojit Panda, Jinyang Li, He He. **Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification.** arXiv:2504.05419, 2025. [paper](https://arxiv.org/abs/2504.05419)

[12] Joey David. **Temporal Predictors of Outcome in Reasoning Language Models.** arXiv:2511.14773, 2025. [paper](https://arxiv.org/abs/2511.14773)

[13] Liyan Xu, Mo Yu, Fandong Meng, Jie Zhou. **How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning.** arXiv:2602.02103, 2026. [paper](https://arxiv.org/abs/2602.02103)

[14] Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, Jack Lindsey. **Persona Vectors: Monitoring and Controlling Character Traits in Language Models.** arXiv:2507.21509, 2025. [paper](https://arxiv.org/abs/2507.21509)

[15] Kenneth Li, Tianle Liu, Naomi Bashkansky, David Bau, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg. **Measuring and Controlling Instruction (In)Stability in Language Model Dialogs.** COLM 2024. [paper](https://arxiv.org/abs/2402.10962)

[16] Murray Shanahan, Kyle McDonell, Laria Reynolds. **Role play with large language models.** Nature 623:493–498, 2023. [paper](https://www.nature.com/articles/s41586-023-06647-8)

[17] Qwen Team (An Yang et al.). **Qwen3 Technical Report.** arXiv:2505.09388, 2025. [paper](https://arxiv.org/abs/2505.09388)

[18] Woosuk Kwon et al. **Efficient Memory Management for Large Language Model Serving with PagedAttention.** SOSP 2023. [paper](https://arxiv.org/abs/2309.06180)

[19] Tibor Kiss, Jan Strunk. **Unsupervised Multilingual Sentence Boundary Detection.** Computational Linguistics 32(4):485–525, 2006. [paper](https://aclanthology.org/J06-4003/)

[20] Jaden Fiotto-Kaufman et al. **NNsight and NDIF: Democratizing Access to Open-Weight Foundation Model Internals.** ICLR 2025. [paper](https://arxiv.org/abs/2407.14561)

[21] Arthur E. Hoerl, Robert W. Kennard. **Ridge Regression: Biased Estimation for Nonorthogonal Problems.** Technometrics 12(1):55–67, 1970.

[22] Miles Turpin, Julian Michael, Ethan Perez, Samuel R. Bowman. **Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting.** NeurIPS 2023. [paper](https://arxiv.org/abs/2305.04388)

[23] Collin Burns, Haotian Ye, Dan Klein, Jacob Steinhardt. **Discovering Latent Knowledge in Language Models Without Supervision.** ICLR 2023. [paper](https://arxiv.org/abs/2212.03827)

[24] Walter Laurito, Sharan Maiya, Grégoire Dhimoïla, Owen Ho Wan Yeung, Kaarel Hänni. **Cluster-Norm for Unsupervised Probing of Knowledge.** EMNLP 2024. [paper](https://aclanthology.org/2024.emnlp-main.780/)

[25] Yair Mundlak. **On the Pooling of Time Series and Cross Section Data.** Econometrica 46(1):69–85, 1978.

[26] Anderson M. Winkler, Matthew A. Webster, Diego Vidaurre, Thomas E. Nichols, Stephen M. Smith. **Multi-level block permutation.** NeuroImage 123:253–268, 2015. (Framework: Winkler et al., *Permutation inference for the general linear model*, NeuroImage 92:381–397, 2014.)

[27] Trevor Hastie, Robert Tibshirani, Jerome Friedman. **The Elements of Statistical Learning** (2nd ed.), §3.2.4 "Multiple Outputs". Springer, 2009.

[28] Tom Dupré la Tour, Michael Eickenberg, Anwar O. Nunez-Elizalde, Jack L. Gallant. **Feature-space selection with banded ridge regression.** NeuroImage 264:119728, 2022.

[29] Sebastian Farquhar, Vikrant Varma, Zachary Kenton, Johannes Gasteiger, Vladimir Mikulik, Rohin Shah. **Challenges with unsupervised LLM knowledge discovery.** arXiv:2312.10029, 2023. [paper](https://arxiv.org/abs/2312.10029)

[30] Emmeke Aarts, Matthijs Verhage, Jesse V. Veenvliet, Conor V. Dolan, Sophie van der Sluis. **A solution to dependency: using multilevel analysis to accommodate nested data.** Nature Neuroscience 17(4):491–496, 2014.

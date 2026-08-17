# Exploratory run: Qwen3-32B on the 58 public persona-drift prefixes

**Run** (2026-08-17): 58 prefixes (all 4 public conversations) × 8 rollouts, `--max_new_tokens 4096 --temperature 1.0 --max_model_len 12288`; 459/464 rollouts usable (5 skipped: truncated or unclosed thinking); activations at layers [0, 9, 18, 27, 32, 36, 45, 54, 63]; analysis with `--ridge-alphas 1,10,100,1000,10000 --cross-splits logo`, per-bin and fixed-cohort variants (`probe_analysis_*.json` in this directory).

## Findings (exploratory; all Limitations in the main README apply)

1. **Answer persona is strongly predictable from the context alone.** Leave-one-conversation-out, the prompt-end (pre-CoT) probe recovers a held-out conversation's persona *ordering* at Pearson r up to **0.79 (layer 45)**, far above the block-derangement control (r ≈ 0.2–0.4). Mid-depth layers (27–45) carry the most signal. Absolute levels do not transfer (negative R² — rank information only).
2. **The CoT adds no incremental persona information.** The context cell matches or beats the final CoT boundary cell at every layer (e.g. layer 45: 0.79 vs 0.70), and the reasoning-time trajectory is flat. The persona of the eventual answer is present in the residual stream *before the first reasoning token* and is not further developed during thinking, in a linearly decodable sense.
3. **Within-prompt null.** Rollout-to-rollout persona deviations (prompt-residual space) are not linearly readable from CoT boundary states: primary endpoint (axis, layer 32, final boundary) r ≈ 0.04, R² < 0, indistinguishable from shuffled controls across all cells. (Within-prompt *context* cells are degenerate by construction — identical context per prompt — and should be ignored.)
4. **Exploratory lead:** persona PCs (beyond the axis) show weak late-CoT within-prompt predictability (macro r ≈ 0.10–0.15, layers 32–45) where context shows ≈ 0 — the only cells where CoT out-predicts context. Underpowered at 8 rollouts/prompt; would need a larger within-prompt design to test.

**One-line summary:** in naturally drifting conversations, Qwen3-32B's answer persona is context-driven — fully present at the prompt's last token — and the chain-of-thought neither adds to it nor detectably deviates from it.

## Files
- `probe_analysis_perbin.json` — 90 cells (9 layers × [context + 4 time bins] × 2 splits), per-bin cohorts
- `probe_analysis_fixedcohort.json` — same grid, constant cohort across bins
- `generation_config.json` — full generation configuration and per-prompt behavior summary

The 1.7 GB activation payload and 464-rollout outputs are not checked in (see `results/…/nfull_nsamp8_l4096_gumbel_s43_ss42_t1.0_*` on the machine that ran the experiment); regenerate with the commands in the main README.

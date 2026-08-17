"""Gather CoT boundary activations and final-answer persona targets per rollout.

Consumes the ``*_results.json``/``*_outputs.json`` pair written by
``unsteered_generation.py`` for a persona-drift dataset and produces the
version-1 activation tensor file read by ``analyze_persona_drift_probes.py``.

For every stochastic rollout this script:

1. replays the exact saved prompt + generated token ids through the model
   (no re-tokenization, so BPE boundaries match generation),
2. locates the end-of-sentence token of every CoT sentence inside the
   ``<think>`` ... ``</think>`` block and stores the residual-stream state at
   those tokens for the requested layers, plus the prompt-end state (last
   prompt token) as the context-only baseline input,
3. computes the mean residual-stream state over the public final-answer tokens
   at the Assistant Axis target layer, and
4. projects that mean state into the Assistant Axis persona space
   (axis scalar + persona coordinates) as the rollout's prediction target.

Rollouts whose thinking block never closes, or whose final answer or thinking
content is empty, are skipped and recorded with a reason in the output config.
"""

from __future__ import annotations

import argparse
import json
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch


THINK_START_TAG = "<think>"
THINK_END_TAG = "</think>"


@dataclass(frozen=True)
class CharSpan:
    """Half-open character span inside the decoded rollout text."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.end:
            raise ValueError(f"Invalid character span: ({self.start}, {self.end})")


class RolloutSkipped(Exception):
    """A rollout that cannot contribute a (CoT, final answer) pair."""


def trim_trailing_special_tokens(token_ids: Sequence[int], special_ids: set[int]) -> list[int]:
    """Drop end-of-turn/eos padding so the answer span covers visible text only."""

    trimmed = [int(token_id) for token_id in token_ids]
    while trimmed and trimmed[-1] in special_ids:
        trimmed.pop()
    return trimmed


def parse_thinking_and_answer_spans(text: str) -> tuple[CharSpan, CharSpan]:
    """Split decoded rollout text into thinking and final-answer character spans.

    Follows the Qwen3 thinking format: an optional ``<think>`` opener, thinking
    content, a mandatory ``</think>`` closer, then the public final answer.
    """

    # Qwen3's reference parsing locates the LAST </think> token; a spurious
    # early close would otherwise leak the remaining CoT into the answer span.
    think_end_index = text.rfind(THINK_END_TAG)
    if think_end_index < 0:
        raise RolloutSkipped("thinking never closed (no </think> in rollout)")

    think_start_index = text.find(THINK_START_TAG)
    thinking_start = (
        think_start_index + len(THINK_START_TAG)
        if 0 <= think_start_index < think_end_index
        else 0
    )
    thinking_end = think_end_index
    while thinking_start < thinking_end and text[thinking_start].isspace():
        thinking_start += 1
    while thinking_end > thinking_start and text[thinking_end - 1].isspace():
        thinking_end -= 1
    if thinking_start >= thinking_end:
        raise RolloutSkipped("empty thinking content")

    answer_start = think_end_index + len(THINK_END_TAG)
    answer_end = len(text)
    while answer_start < answer_end and text[answer_start].isspace():
        answer_start += 1
    while answer_end > answer_start and text[answer_end - 1].isspace():
        answer_end -= 1
    if answer_start >= answer_end:
        raise RolloutSkipped("empty final answer")

    return CharSpan(thinking_start, thinking_end), CharSpan(answer_start, answer_end)


def _punkt_sentence_tokenizer():
    import nltk

    try:
        from nltk.tokenize import PunktTokenizer
    except ImportError as error:  # pragma: no cover - nltk>=3.9 is a declared dependency
        raise ImportError("nltk>=3.9 with PunktTokenizer is required") from error

    try:
        return PunktTokenizer("english")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        return PunktTokenizer("english")


def sentence_end_offsets(
    text: str,
    span: CharSpan,
    sentence_spans_fn: Callable[[str], Sequence[tuple[int, int]]] | None = None,
) -> list[int]:
    """Absolute character offsets at which each sentence of ``span`` ends.

    Sentences are the same NLTK Punkt units used throughout the repository for
    per-sentence CoT analysis.  The final offset always equals ``span.end`` so
    the last boundary coincides with the end of the thinking content.
    """

    if sentence_spans_fn is None:
        tokenizer = _punkt_sentence_tokenizer()
        sentence_spans_fn = lambda segment: list(tokenizer.span_tokenize(segment))

    segment = text[span.start : span.end]
    offsets = [
        span.start + sentence_end
        for _, sentence_end in sentence_spans_fn(segment)
        if sentence_end > 0
    ]
    if not offsets:
        raise RolloutSkipped("no sentences found in thinking content")
    if offsets != sorted(offsets):
        raise RolloutSkipped("sentence spans are not monotonically ordered")
    if offsets[-1] != span.end:
        # Punkt never drops trailing text, but keep the invariant explicit: the
        # last CoT boundary must be the final thinking token.
        offsets[-1] = span.end
    return offsets


class TokenTextAligner:
    """Map character offsets in the decoded rollout back to generated tokens.

    The fast path re-encodes the decoded text and uses the fast tokenizer's
    offset mapping, accepted only when it reproduces the generated ids exactly.
    Otherwise a binary search over prefix decodes is used, where a prefix's
    coverage is the length of its longest exact match with ``text``.  Comparing
    against ``text`` (rather than trusting ``len(decode(prefix))``) matters for
    multibyte characters split across tokens: a dangling-byte prefix decodes to
    U+FFFD replacement characters that inflate the raw length.
    """

    def __init__(self, tokenizer, token_ids: Sequence[int], text: str) -> None:
        self.tokenizer = tokenizer
        self.token_ids = [int(token_id) for token_id in token_ids]
        self.text = text
        if not self.token_ids:
            raise ValueError("TokenTextAligner requires at least one token")
        self._token_start_offsets: list[int] | None = None
        self._prefix_length_cache: dict[int, int] = {0: 0, len(self.token_ids): len(text)}

        try:
            encoding = tokenizer(
                text, add_special_tokens=False, return_offsets_mapping=True
            )
            if list(encoding["input_ids"]) == self.token_ids:
                self._token_start_offsets = [
                    int(start) for start, _ in encoding["offset_mapping"]
                ]
        except (TypeError, ValueError, NotImplementedError):
            self._token_start_offsets = None

    def _prefix_length(self, n_tokens: int) -> int:
        """Number of leading ``text`` characters fully produced by ``n_tokens``."""

        if n_tokens not in self._prefix_length_cache:
            decoded = self.tokenizer.decode(
                self.token_ids[:n_tokens], clean_up_tokenization_spaces=False
            )
            if self.text.startswith(decoded):
                covered = len(decoded)
            else:
                # Dangling UTF-8 bytes decode to U+FFFD; count only the exactly
                # matching prefix so incomplete characters are not credited.
                limit = min(len(decoded), len(self.text))
                covered = 0
                while covered < limit and decoded[covered] == self.text[covered]:
                    covered += 1
            self._prefix_length_cache[n_tokens] = covered
        return self._prefix_length_cache[n_tokens]

    def token_index_covering(self, char_end: int) -> int:
        """Index of the generated token that completes ``text[:char_end]``."""

        if not 0 < char_end <= len(self.text):
            raise ValueError(
                f"char_end must lie in (0, {len(self.text)}], got {char_end}"
            )
        if self._token_start_offsets is not None:
            index = bisect_right(self._token_start_offsets, char_end - 1) - 1
            return max(index, 0)

        low, high = 1, len(self.token_ids)
        while low < high:
            mid = (low + high) // 2
            if self._prefix_length(mid) >= char_end:
                high = mid
            else:
                low = mid + 1
        return low - 1


@dataclass(frozen=True)
class RolloutGeometry:
    """Token-level layout of one rollout, in generated-token coordinates."""

    boundary_token_indices: list[int]  # CoT sentence-end tokens
    cot_progress: list[float]  # fraction of thinking tokens completed, in (0, 1]
    answer_token_span: tuple[int, int]  # half-open span of final-answer tokens
    n_thinking_tokens: int


def compute_rollout_geometry(
    aligner: TokenTextAligner,
    thinking_span: CharSpan,
    answer_span: CharSpan,
    sentence_offsets: Sequence[int],
) -> RolloutGeometry:
    thinking_start_token = aligner.token_index_covering(thinking_span.start + 1)
    thinking_end_token = aligner.token_index_covering(thinking_span.end)

    boundary_token_indices: list[int] = []
    for offset in sentence_offsets:
        token_index = aligner.token_index_covering(offset)
        # Very short sentences can end inside the same token; keep one boundary.
        if boundary_token_indices and token_index <= boundary_token_indices[-1]:
            continue
        boundary_token_indices.append(token_index)
    if not boundary_token_indices:
        raise RolloutSkipped("no distinct CoT boundary tokens")

    thinking_token_count = thinking_end_token - thinking_start_token + 1
    cot_progress = [
        (token_index - thinking_start_token + 1) / thinking_token_count
        for token_index in boundary_token_indices
    ]

    answer_start_token = aligner.token_index_covering(answer_span.start + 1)
    answer_end_token = aligner.token_index_covering(answer_span.end) + 1
    if answer_start_token >= answer_end_token:
        raise RolloutSkipped("empty final-answer token span")

    return RolloutGeometry(
        boundary_token_indices=boundary_token_indices,
        cot_progress=cot_progress,
        answer_token_span=(answer_start_token, answer_end_token),
        n_thinking_tokens=thinking_token_count,
    )


def build_rollout_record(
    tokenizer,
    prompt_token_ids: Sequence[int],
    response_token_ids: Sequence[int],
    rollout_index: int,
    *,
    layers: Sequence[int],
    scorer,
    extract_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    sentence_spans_fn: Callable[[str], Sequence[tuple[int, int]]] | None = None,
) -> dict[str, Any]:
    """Build one v1 rollout entry (activations + persona targets) or skip."""

    special_ids = set(int(token_id) for token_id in tokenizer.all_special_ids)
    visible_ids = trim_trailing_special_tokens(response_token_ids, special_ids)
    if not visible_ids:
        raise RolloutSkipped("rollout contains no visible tokens")

    text = tokenizer.decode(visible_ids, clean_up_tokenization_spaces=False)
    thinking_span, answer_span = parse_thinking_and_answer_spans(text)
    offsets = sentence_end_offsets(text, thinking_span, sentence_spans_fn)
    aligner = TokenTextAligner(tokenizer, visible_ids, text)
    geometry = compute_rollout_geometry(aligner, thinking_span, answer_span, offsets)

    prompt_length = len(prompt_token_ids)
    full_ids = torch.tensor(
        [[int(token_id) for token_id in prompt_token_ids] + visible_ids],
        dtype=torch.long,
    )
    boundary_positions = [
        prompt_length + token_index for token_index in geometry.boundary_token_indices
    ]
    answer_span_positions = (
        prompt_length + geometry.answer_token_span[0],
        prompt_length + geometry.answer_token_span[1],
    )

    # The prompt-end state (last prompt token, before any generated token) is
    # captured in the same trace and serves downstream as the context-only
    # baseline probe input.
    extracted_activations, answer_mean_activation = extract_fn(
        input_ids=full_ids,
        boundary_token_positions=[prompt_length - 1, *boundary_positions],
        layer_indices=list(layers),
        mean_span=answer_span_positions,
        mean_layer=scorer.target_layer,
    )
    context_activations = extracted_activations[0]
    cot_activations = extracted_activations[1:]
    persona_score = scorer.score(answer_mean_activation)

    return {
        "rollout_index": int(rollout_index),
        "cot_activations": cot_activations,  # [boundaries, layers, hidden]
        "context_activations": context_activations,  # [layers, hidden]
        "cot_progress": torch.tensor(geometry.cot_progress, dtype=torch.float32),
        "assistant_axis_score": float(persona_score.assistant_axis_score),
        "persona_coordinates": persona_score.persona_coordinates,
        "boundary_token_positions": boundary_positions,
        "answer_token_span": list(answer_span_positions),
        "n_thinking_tokens": int(geometry.n_thinking_tokens),
        "n_answer_tokens": int(
            geometry.answer_token_span[1] - geometry.answer_token_span[0]
        ),
    }


def resolve_layers(spec: str, num_layers: int, axis_layer: int) -> list[int]:
    """Resolve --layers into a sorted, de-duplicated list of decoder layers.

    ``auto`` spreads eight layers evenly over depth and always includes the
    Assistant Axis target layer so CoT states can also be probed there.
    """

    spec = spec.strip().lower()
    if spec == "all":
        layers = list(range(num_layers))
    elif spec == "auto":
        n_auto = min(8, num_layers)
        layers = sorted(
            {
                round(index * (num_layers - 1) / max(n_auto - 1, 1))
                for index in range(n_auto)
            }
            | {int(axis_layer)}
        )
    else:
        layers = sorted({int(item) for item in spec.split(",") if item.strip()})
        if not layers:
            raise ValueError("At least one layer must be requested")
    if any(layer < 0 or layer >= num_layers for layer in layers):
        raise ValueError(
            f"Layers must lie in [0, {num_layers}), got {layers}"
        )
    return layers


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def gather_persona_drift_activations(
    results_file: Path,
    *,
    layers_spec: str,
    assistant_axis_path: str | None,
    axis_layer: int,
    coordinate_mode: str,
    max_pca_components: int,
    output: Path | None,
    max_prompts: int | None,
    max_rollouts: int | None,
    device: str,
) -> Path:
    import nnsight
    from tqdm import tqdm

    from src.interp.activations_nnsight import extract_boundary_and_span_activations
    from src.interp.persona_axis import AssistantAxisScorer
    from src.model_utils import get_model_and_tokenizer

    results_file = Path(results_file)
    outputs_file = Path(str(results_file).replace("_results.json", "_outputs.json"))
    results = _load_json(results_file)
    outputs = _load_json(outputs_file)
    model_name = results["model_name"]

    scorer = AssistantAxisScorer(
        assistant_axis_path,
        target_layer=axis_layer,
        coordinate_mode=coordinate_mode,
        max_pca_components=max_pca_components,
    )

    print(f"Loading model {model_name}")
    model, tokenizer = get_model_and_tokenizer(
        model_name, device=device, half_precision=True, multi_gpu=device != "cpu"
    )
    wrapped_model = nnsight.LanguageModel(model, tokenizer=tokenizer, dispatch=True)

    num_layers = int(model.config.num_hidden_layers)
    layers = resolve_layers(layers_spec, num_layers, scorer.target_layer)
    hidden_size = int(model.config.hidden_size)
    if hidden_size != scorer.hidden_size:
        raise ValueError(
            f"Assistant Axis artifacts have hidden size {scorer.hidden_size} but "
            f"{model_name} has {hidden_size}; the artifacts belong to another model."
        )

    def extract_fn(**kwargs):
        return extract_boundary_and_span_activations(wrapped_model, **kwargs)

    if max_prompts is not None:
        outputs = outputs[:max_prompts]

    prompts: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for prompt_index, prompt_output in enumerate(
        tqdm(outputs, desc="Gathering persona-drift activations")
    ):
        prompt_token_ids = prompt_output.get("prompt_token_ids")
        if not prompt_token_ids:
            raise ValueError(
                f"{outputs_file} lacks prompt_token_ids (prompt {prompt_index}); "
                "re-run unsteered_generation.py with the current code to record "
                "exact tokenizations."
            )
        metadata = prompt_output.get("metadata", {})
        responses = prompt_output.get("responses", [])
        if max_rollouts is not None:
            responses = responses[:max_rollouts]

        rollouts = []
        for rollout_index, response in enumerate(responses):
            token_ids = response.get("token_ids")
            if not token_ids:
                raise ValueError(
                    f"{outputs_file} lacks response token_ids (prompt {prompt_index}, "
                    f"rollout {rollout_index}); re-run unsteered_generation.py."
                )
            # A rollout cut off by the token budget has a truncated final
            # answer; scoring partial answer text would corrupt the persona
            # target.  finish_reason is authoritative when present; the length
            # comparison covers outputs written before it was recorded.
            finish_reason = response.get("finish_reason")
            generation_cap = results.get("max_new_tokens")
            if finish_reason == "length" or (
                finish_reason is None
                and generation_cap
                and len(token_ids) >= int(generation_cap)
            ):
                skipped.append(
                    {
                        "prompt_index": prompt_index,
                        "conversation_id": metadata.get("conversation_id"),
                        "turn_index": metadata.get("turn_index"),
                        "rollout_index": rollout_index,
                        "reason": "generation hit the token budget (truncated answer)",
                    }
                )
                continue
            try:
                rollouts.append(
                    build_rollout_record(
                        tokenizer,
                        prompt_token_ids,
                        token_ids,
                        rollout_index,
                        layers=layers,
                        scorer=scorer,
                        extract_fn=extract_fn,
                    )
                )
            except RolloutSkipped as reason:
                skipped.append(
                    {
                        "prompt_index": prompt_index,
                        "conversation_id": metadata.get("conversation_id"),
                        "turn_index": metadata.get("turn_index"),
                        "rollout_index": rollout_index,
                        "reason": str(reason),
                    }
                )

        if not rollouts:
            continue
        prompts.append(
            {
                "conversation_id": metadata.get("conversation_id"),
                "turn_index": metadata.get("turn_index"),
                "metadata": metadata,
                "rollouts": rollouts,
            }
        )

    if not prompts:
        raise ValueError(
            f"No usable rollouts found in {outputs_file}; "
            f"skipped {len(skipped)} rollouts."
        )

    payload = {
        "format_version": 1,
        "layers": torch.tensor(layers, dtype=torch.long),
        "persona_coordinate_names": list(scorer.coordinate_names),
        "prompts": prompts,
        "config": {
            "model_name": model_name,
            "results_file": str(results_file.resolve()),
            "layers": layers,
            "num_hidden_layers": num_layers,
            "boundary_unit": "nltk_punkt_sentence_end_tokens",
            "answer_target": "mean_answer_token_activation",
            "assistant_axis": scorer.metadata(),
            "generation": {
                key: results.get(key)
                for key in (
                    "dataset_name",
                    "dataset_path",
                    "subset",
                    "num_samples",
                    "max_new_tokens",
                    "seed",
                    "sampling_seed",
                    "temperature",
                    "max_model_len",
                )
            },
            "n_skipped_rollouts": len(skipped),
            "skipped_rollouts": skipped,
        },
    }

    if output is None:
        output = Path(
            str(results_file).replace("_results.json", "_persona_activations.pt")
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    n_rollouts = sum(len(prompt["rollouts"]) for prompt in prompts)
    print(
        f"Saved {n_rollouts} rollouts from {len(prompts)} prompts "
        f"({len(skipped)} skipped) at layers {layers} to {output}"
    )
    return output


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gather CoT sentence-boundary activations and Assistant Axis "
            "final-answer persona targets from persona-drift rollouts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_file",
        type=Path,
        required=True,
        help="A *_results.json written by unsteered_generation.py",
    )
    parser.add_argument(
        "--layers",
        default="auto",
        help="Comma-separated decoder layers, 'all', or 'auto' (8 evenly spaced + axis layer)",
    )
    parser.add_argument(
        "--assistant_axis_path",
        default=None,
        help=(
            "Local Assistant Axis artifact directory (containing assistant_axis.pt "
            "and role_vectors/). Downloads the official Qwen3-32B artifacts from "
            "HuggingFace when omitted."
        ),
    )
    parser.add_argument(
        "--axis_layer",
        type=int,
        default=32,
        help="Decoder layer of the Assistant Axis projection (upstream uses mid-depth)",
    )
    parser.add_argument(
        "--coordinate_mode",
        choices=("pca", "role_cosine", "both"),
        default="pca",
        help="Persona coordinate construction in src/interp/persona_axis.py",
    )
    parser.add_argument(
        "--max_pca_components",
        type=int,
        default=8,
        help="PCA components fitted on centered role centroids",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--max_prompts", type=int, default=None, help="Debug cap on prompts"
    )
    parser.add_argument(
        "--max_rollouts", type=int, default=None, help="Debug cap on rollouts per prompt"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    gather_persona_drift_activations(
        args.results_file,
        layers_spec=args.layers,
        assistant_axis_path=args.assistant_axis_path,
        axis_layer=args.axis_layer,
        coordinate_mode=args.coordinate_mode,
        max_pca_components=args.max_pca_components,
        output=args.output,
        max_prompts=args.max_prompts,
        max_rollouts=args.max_rollouts,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fit linear probes from CoT states to final-answer persona coordinates.

The native input is the version-1 tensor file produced by
``gather_persona_drift_activations.py``::

    {
        "format_version": 1,
        "layers": LongTensor[num_layers],
        "persona_coordinate_names": [str, ...],
        "prompts": [
            {
                "conversation_id": str,
                "turn_index": int,
                "rollouts": [
                    {
                        "rollout_index": int,
                        "cot_activations": Tensor[num_boundaries, num_layers, hidden],
                        "cot_progress": Tensor[num_boundaries],
                        "assistant_axis_score": float,
                        "persona_coordinates": Tensor[num_coordinates],
                    },
                    ...
                ],
            },
            ...
        ],
    }

For every layer and normalized reasoning checkpoint, a multi-output ridge (or
ordinary linear) regressor predicts the final answer's Assistant Axis scalar and
selected persona coordinates.  Two complementary evaluations are reported:

* ``cross_prompt`` holds out complete prompt groups.  ``--cross-group
  conversation`` is stricter and holds out every prefix from a conversation.
* ``within_prompt`` holds out stochastic rollouts inside every prompt.  Both
  activations and targets are centered using *training rollouts only*, so this
  measures whether activation deviations predict persona deviations rather than
  merely recovering prompt identity.  Its reported metrics are computed in this
  prompt-residual target space.

The prompt-mean baseline only uses training labels (and falls back to the global
training mean for unseen prompts).  The shuffled baseline refits the same probe
after shuffling targets globally for cross-prompt evaluation and within each
prompt for within-prompt evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class RolloutTrajectory:
    """One stochastic rollout and its eventual final-answer persona target."""

    prompt_id: str
    conversation_id: str
    rollout_id: str
    layer_activations: dict[int, np.ndarray]  # layer -> (boundaries, hidden)
    progress: np.ndarray  # (boundaries,), normalized to (0, 1]
    assistant_axis: float
    persona_coordinates: np.ndarray  # (coordinates,)


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    evaluation: np.ndarray


def _to_numpy(value: Any, *, dtype: np.dtype | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().float().numpy()
    array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    return [int(item) for item in _to_numpy(value).reshape(-1).tolist()]


def _mapping_value(mappings: Iterable[Mapping[str, Any]], names: Sequence[str]) -> Any:
    for mapping in mappings:
        for name in names:
            if name in mapping and mapping[name] is not None:
                return mapping[name]
    return None


def _score_mappings(rollout: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    mappings: list[Mapping[str, Any]] = [rollout]
    for key in ("final_answer_scores", "answer_persona", "persona_scores", "targets"):
        value = rollout.get(key)
        if isinstance(value, Mapping):
            mappings.append(value)
    return mappings


def _layer_number(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    match = re.search(r"-?\d+", str(value))
    if match is None:
        raise ValueError(f"Cannot parse layer number from {value!r}")
    return int(match.group())


def _parse_layer_activations(
    value: Any,
    layer_ids: Sequence[int],
) -> dict[int, np.ndarray]:
    """Normalize dict, [B,L,H], or [L,B,H] activations to a layer mapping."""

    if isinstance(value, Mapping):
        parsed: dict[int, np.ndarray] = {}
        for layer, activations in value.items():
            array = _to_numpy(activations, dtype=np.float32)
            if array.ndim != 2:
                raise ValueError(
                    f"Layer {layer!r} activations must be [boundaries, hidden], got {array.shape}"
                )
            parsed[_layer_number(layer)] = array
        return parsed

    array = _to_numpy(value, dtype=np.float32)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 2:
        if len(layer_ids) != 1:
            raise ValueError(
                "Two-dimensional CoT activations require exactly one layer id; "
                f"got shape {array.shape} and layers {list(layer_ids)}"
            )
        return {int(layer_ids[0]): array}
    if array.ndim != 3:
        raise ValueError(
            "CoT activations must be [boundaries, layers, hidden], "
            f"[layers, boundaries, hidden], or a layer mapping; got {array.shape}"
        )

    # Prefer the native v1 [boundaries, layers, hidden] layout.  The second
    # branch accepts early experimental files that stored layers first.
    if layer_ids and array.shape[1] == len(layer_ids):
        return {int(layer): array[:, index, :] for index, layer in enumerate(layer_ids)}
    if layer_ids and array.shape[0] == len(layer_ids):
        return {int(layer): array[index, :, :] for index, layer in enumerate(layer_ids)}
    if not layer_ids:
        return {index: array[:, index, :] for index in range(array.shape[1])}
    raise ValueError(
        f"Activation shape {array.shape} is incompatible with layers {list(layer_ids)}"
    )


def _normalize_progress(value: Any, n_boundaries: int) -> np.ndarray:
    if value is None:
        return np.arange(1, n_boundaries + 1, dtype=np.float64) / n_boundaries
    progress = _to_numpy(value, dtype=np.float64).reshape(-1)
    if len(progress) != n_boundaries:
        raise ValueError(
            f"cot_progress has {len(progress)} entries for {n_boundaries} boundaries"
        )
    if not np.all(np.isfinite(progress)):
        raise ValueError("cot_progress contains non-finite values")
    if np.any(np.diff(progress) < 0):
        raise ValueError("cot_progress must be non-decreasing")
    if progress[-1] <= 0:
        raise ValueError("cot_progress must end above zero")
    if progress[-1] > 1.0 + 1e-6 or progress[0] < -1e-6:
        # Defensive support for files containing token positions instead of
        # fractions.  Preserve spacing while mapping the last boundary to one.
        progress = progress / progress[-1]
    return np.clip(progress, 0.0, 1.0)


def _prompt_identity(prompt: Mapping[str, Any], prompt_index: int) -> tuple[str, str]:
    metadata = prompt.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    conversation = str(
        prompt.get("conversation_id")
        or metadata.get("conversation_id", "")
    )
    turn = prompt.get("turn_index")
    if turn is None:
        turn = metadata.get("turn_index")

    # Prefix prompts from one transcript overlap heavily.  Keep a stable
    # conversation id separately so --cross-group conversation can hold out all
    # prefixes, while prompt_id identifies the precise conversation + user turn.
    if conversation and turn is not None:
        prompt_id = f"{conversation}::turn_{int(turn)}"
    else:
        prompt_id = str(prompt.get("prompt_id") or prompt.get("sample_idx") or prompt_index)
    return prompt_id, conversation or prompt_id


def load_persona_activation_dataset(
    path: str | Path,
) -> tuple[list[RolloutTrajectory], dict[str, Any]]:
    """Load and validate a native persona-drift activation tensor dataset."""

    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Activation dataset root must be a mapping")

    raw_layers = payload.get("layers")
    if raw_layers is None and isinstance(payload.get("config"), Mapping):
        raw_layers = payload["config"].get("layers")
    layer_ids = _as_int_list(raw_layers)

    coordinate_names = payload.get("persona_coordinate_names")
    if coordinate_names is None and isinstance(payload.get("config"), Mapping):
        coordinate_names = payload["config"].get("persona_coordinate_names")
    if coordinate_names is not None:
        coordinate_names = [str(name) for name in coordinate_names]

    raw_prompts = payload.get("prompts")
    if raw_prompts is None:
        # A small compatibility path for early files that named this collection
        # activation_data.  Each element must still contain responses/rollouts.
        raw_prompts = payload.get("activation_data")
    if not isinstance(raw_prompts, Sequence):
        raise ValueError("Activation dataset must contain a 'prompts' list")

    trajectories: list[RolloutTrajectory] = []
    skipped: list[dict[str, Any]] = []
    inferred_coordinate_count: int | None = None

    for prompt_index, prompt_value in enumerate(raw_prompts):
        if not isinstance(prompt_value, Mapping):
            skipped.append({"prompt_index": prompt_index, "reason": "prompt is not a mapping"})
            continue
        prompt = prompt_value
        prompt_id, conversation_id = _prompt_identity(prompt, prompt_index)
        raw_rollouts = prompt.get("rollouts", prompt.get("responses", []))
        if not isinstance(raw_rollouts, Sequence):
            skipped.append({"prompt_id": prompt_id, "reason": "rollouts is not a list"})
            continue

        for fallback_rollout_index, rollout_value in enumerate(raw_rollouts):
            rollout_ref = {
                "prompt_id": prompt_id,
                "rollout_index": fallback_rollout_index,
            }
            try:
                if not isinstance(rollout_value, Mapping):
                    raise ValueError("rollout is not a mapping")
                rollout = rollout_value
                rollout_index = rollout.get(
                    "rollout_index", rollout.get("response_idx", fallback_rollout_index)
                )
                rollout_ref["rollout_index"] = rollout_index

                activations_value = _mapping_value(
                    [rollout],
                    (
                        "cot_activations",
                        "thinking_activations",
                        "reasoning_activations",
                        "cot_boundary_activations",
                    ),
                )
                if activations_value is None:
                    raise ValueError("missing cot_activations")
                layer_activations = _parse_layer_activations(activations_value, layer_ids)
                if not layer_activations:
                    raise ValueError("no CoT activation layers")
                boundary_counts = {array.shape[0] for array in layer_activations.values()}
                hidden_sizes = {array.shape[1] for array in layer_activations.values()}
                if len(boundary_counts) != 1 or len(hidden_sizes) != 1:
                    raise ValueError("all layers must share boundary count and hidden size")
                n_boundaries = next(iter(boundary_counts))
                if n_boundaries == 0:
                    raise ValueError("no CoT boundaries")
                if any(not np.all(np.isfinite(array)) for array in layer_activations.values()):
                    raise ValueError("CoT activations contain non-finite values")

                progress_value = _mapping_value(
                    [rollout], ("cot_progress", "boundary_progress", "reasoning_progress")
                )
                progress = _normalize_progress(progress_value, n_boundaries)

                mappings = _score_mappings(rollout)
                assistant_axis = _mapping_value(
                    mappings,
                    ("assistant_axis_score", "assistant_axis", "assistant_score"),
                )
                coordinates_value = _mapping_value(
                    mappings,
                    (
                        "persona_coordinates",
                        "persona_coords",
                        "role_similarities",
                        "role_coordinates",
                    ),
                )
                if assistant_axis is None or coordinates_value is None:
                    raise ValueError("missing final-answer persona scores")
                assistant_axis = float(assistant_axis)

                if isinstance(coordinates_value, Mapping):
                    local_names = [str(name) for name in coordinates_value]
                    coordinates = np.asarray(
                        [coordinates_value[name] for name in coordinates_value], dtype=np.float64
                    )
                    if coordinate_names is None:
                        coordinate_names = local_names
                    elif set(local_names) == set(coordinate_names):
                        local_lookup = dict(zip(local_names, coordinates.tolist()))
                        coordinates = np.asarray(
                            [local_lookup[name] for name in coordinate_names], dtype=np.float64
                        )
                    else:
                        raise ValueError("persona coordinate names are inconsistent")
                else:
                    coordinates = _to_numpy(coordinates_value, dtype=np.float64).reshape(-1)

                if not math.isfinite(assistant_axis) or not np.all(np.isfinite(coordinates)):
                    raise ValueError("final-answer persona scores contain non-finite values")
                if inferred_coordinate_count is None:
                    inferred_coordinate_count = len(coordinates)
                elif len(coordinates) != inferred_coordinate_count:
                    raise ValueError("persona coordinate dimensions are inconsistent")

                trajectories.append(
                    RolloutTrajectory(
                        prompt_id=prompt_id,
                        conversation_id=conversation_id,
                        rollout_id=f"{prompt_id}::rollout_{rollout_index}",
                        layer_activations=layer_activations,
                        progress=progress,
                        assistant_axis=assistant_axis,
                        persona_coordinates=coordinates,
                    )
                )
            except (TypeError, ValueError, IndexError) as error:
                skipped.append({**rollout_ref, "reason": str(error)})

    if not trajectories:
        first_errors = "; ".join(item["reason"] for item in skipped[:3])
        raise ValueError(f"No usable rollout trajectories found. {first_errors}")

    coordinate_count = len(trajectories[0].persona_coordinates)
    if coordinate_names is None:
        coordinate_names = [f"persona_{index + 1}" for index in range(coordinate_count)]
    if len(coordinate_names) != coordinate_count:
        raise ValueError(
            f"Found {coordinate_count} persona values but {len(coordinate_names)} names"
        )

    available_layers = sorted(
        set.intersection(*(set(trajectory.layer_activations) for trajectory in trajectories))
    )
    info = {
        "format_version": payload.get("format_version"),
        "coordinate_names": coordinate_names,
        "available_layers": available_layers,
        "n_prompts": len({trajectory.prompt_id for trajectory in trajectories}),
        "n_conversations": len({trajectory.conversation_id for trajectory in trajectories}),
        "n_rollouts": len(trajectories),
        "n_skipped_rollouts": len(skipped),
        "skipped_rollouts": skipped,
        "source_config": dict(payload.get("config", {})),
    }
    return trajectories, info


def checkpoint_activations(
    trajectory: RolloutTrajectory,
    layer: int,
    checkpoints: np.ndarray,
    representation: str,
) -> np.ndarray:
    """Sample a variable-length CoT trajectory at normalized checkpoints."""

    activations = trajectory.layer_activations[layer]
    progress = trajectory.progress
    sampled: list[np.ndarray] = []
    for checkpoint in checkpoints:
        eligible = np.flatnonzero(progress <= checkpoint + 1e-9)
        boundary_index = int(eligible[-1]) if len(eligible) else 0
        if representation == "latest":
            sampled.append(activations[boundary_index])
        elif representation == "cumulative_mean":
            sampled.append(activations[: boundary_index + 1].mean(axis=0))
        else:
            raise ValueError(f"Unknown trajectory representation: {representation}")
    return np.stack(sampled)


def select_persona_coordinates(
    trajectories: Sequence[RolloutTrajectory],
    coordinate_names: Sequence[str],
    top_k: int,
    requested_names: Sequence[str] | None,
    ranking: str,
) -> tuple[list[int], list[str], str]:
    if requested_names:
        name_to_index = {name: index for index, name in enumerate(coordinate_names)}
        missing = [name for name in requested_names if name not in name_to_index]
        if missing:
            raise ValueError(f"Unknown persona coordinate names: {missing}")
        indices = [name_to_index[name] for name in requested_names]
        return indices, list(requested_names), "explicit"
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if top_k == 0:
        return [], [], "none"
    if ranking == "stored":
        # PCA coordinates are stored in explained-variance order (pc_1, pc_2,
        # ...), so this avoids using evaluation labels for target selection.
        indices = np.arange(min(top_k, len(coordinate_names)))
        selection = "first_stored"
    elif ranking == "variance":
        coordinates = np.stack([item.persona_coordinates for item in trajectories])
        variances = np.var(coordinates, axis=0)
        # Stable ordering makes ties follow the stored coordinate order.  This
        # mode is exploratory because it examines all rollout targets.
        indices = np.argsort(-variances, kind="stable")[: min(top_k, len(coordinate_names))]
        selection = "top_variance_over_rollouts"
    else:
        raise ValueError(f"Unknown coordinate ranking: {ranking}")
    return (
        indices.astype(int).tolist(),
        [coordinate_names[index] for index in indices],
        selection,
    )


def _split_group_names(
    trajectories: Sequence[RolloutTrajectory], cross_group: str
) -> np.ndarray:
    if cross_group == "prompt":
        return np.asarray([item.prompt_id for item in trajectories], dtype=object)
    if cross_group == "conversation":
        return np.asarray([item.conversation_id for item in trajectories], dtype=object)
    raise ValueError(f"Unknown cross_group: {cross_group}")


def make_cross_prompt_split(
    trajectories: Sequence[RolloutTrajectory],
    evaluation_fraction: float,
    seed: int,
    cross_group: str,
) -> SplitIndices:
    groups = _split_group_names(trajectories, cross_group)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError(f"cross_{cross_group} evaluation requires at least two groups")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_groups)
    n_evaluation = int(round(len(unique_groups) * evaluation_fraction))
    n_evaluation = min(max(n_evaluation, 1), len(unique_groups) - 1)
    evaluation_groups = set(shuffled[:n_evaluation].tolist())
    evaluation = np.asarray(
        [index for index, group in enumerate(groups) if group in evaluation_groups], dtype=int
    )
    train = np.asarray(
        [index for index, group in enumerate(groups) if group not in evaluation_groups], dtype=int
    )
    return SplitIndices(train=train, evaluation=evaluation)


def make_within_prompt_split(
    trajectories: Sequence[RolloutTrajectory],
    evaluation_fraction: float,
    seed: int,
) -> SplitIndices:
    """Hold out rollouts independently within prompts, dropping singleton prompts."""

    prompt_to_indices: dict[str, list[int]] = {}
    for index, trajectory in enumerate(trajectories):
        prompt_to_indices.setdefault(trajectory.prompt_id, []).append(index)
    rng = np.random.default_rng(seed)
    train: list[int] = []
    evaluation: list[int] = []
    for prompt_id in sorted(prompt_to_indices):
        indices = np.asarray(prompt_to_indices[prompt_id], dtype=int)
        if len(indices) < 2:
            continue
        indices = rng.permutation(indices)
        n_evaluation = int(round(len(indices) * evaluation_fraction))
        n_evaluation = min(max(n_evaluation, 1), len(indices) - 1)
        evaluation.extend(indices[:n_evaluation].tolist())
        train.extend(indices[n_evaluation:].tolist())
    if not train or not evaluation:
        raise ValueError("within_prompt evaluation requires at least one prompt with two rollouts")
    return SplitIndices(
        train=np.asarray(sorted(train), dtype=int),
        evaluation=np.asarray(sorted(evaluation), dtype=int),
    )


def _make_regressor(kind: str, ridge_alpha: float):
    if kind == "ridge":
        estimator = Ridge(alpha=ridge_alpha, solver="lsqr", fit_intercept=True)
    elif kind == "linear":
        estimator = LinearRegression(fit_intercept=True)
    else:
        raise ValueError(f"Unknown regressor: {kind}")
    return make_pipeline(StandardScaler(), estimator)


def _training_prompt_means(
    values: np.ndarray,
    prompt_ids: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    global_mean = values[train_indices].mean(axis=0)
    means = {
        prompt_id: values[train_indices[prompt_ids[train_indices] == prompt_id]].mean(axis=0)
        for prompt_id in np.unique(prompt_ids[train_indices])
    }
    return means, global_mean


def _rows_from_prompt_means(
    prompt_ids: np.ndarray,
    indices: np.ndarray,
    means: Mapping[str, np.ndarray],
    fallback: np.ndarray,
) -> tuple[np.ndarray, float]:
    rows = []
    covered = 0
    for index in indices:
        prompt_id = str(prompt_ids[index])
        if prompt_id in means:
            rows.append(means[prompt_id])
            covered += 1
        else:
            rows.append(fallback)
    return np.stack(rows), covered / len(indices)


def _shuffle_rows(
    targets: np.ndarray,
    prompt_ids: np.ndarray,
    rng: np.random.Generator,
    within_prompt: bool,
) -> np.ndarray:
    shuffled = targets.copy()
    if within_prompt:
        for prompt_id in np.unique(prompt_ids):
            group = np.flatnonzero(prompt_ids == prompt_id)
            shuffled[group] = targets[rng.permutation(group)]
    else:
        shuffled = targets[rng.permutation(len(targets))]
    return shuffled


def fit_predict_with_controls(
    X: np.ndarray,
    y: np.ndarray,
    prompt_ids: np.ndarray,
    split: SplitIndices,
    *,
    split_kind: str,
    regressor: str,
    ridge_alpha: float,
    shuffle_repeats: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], float]:
    """Fit one probe plus target-shuffled controls in a single multi-output solve."""

    train, evaluation = split.train, split.evaluation
    X_prompt_means, X_global_mean = _training_prompt_means(X, prompt_ids, train)
    y_prompt_means, y_global_mean = _training_prompt_means(y, prompt_ids, train)
    prompt_baseline, prompt_coverage = _rows_from_prompt_means(
        prompt_ids, evaluation, y_prompt_means, y_global_mean
    )
    global_baseline = np.repeat(y_global_mean[None, :], len(evaluation), axis=0)

    if split_kind == "within_prompt":
        X_train_centers, _ = _rows_from_prompt_means(
            prompt_ids, train, X_prompt_means, X_global_mean
        )
        X_eval_centers, coverage = _rows_from_prompt_means(
            prompt_ids, evaluation, X_prompt_means, X_global_mean
        )
        if coverage < 1.0 or prompt_coverage < 1.0:
            raise ValueError("within_prompt split contains a prompt absent from training")
        y_train_centers, _ = _rows_from_prompt_means(
            prompt_ids, train, y_prompt_means, y_global_mean
        )
        y_eval_offset = prompt_baseline
        X_train = X[train] - X_train_centers
        X_evaluation = X[evaluation] - X_eval_centers
        y_train = y[train] - y_train_centers
        shuffle_within_prompt = True
    else:
        X_train = X[train]
        X_evaluation = X[evaluation]
        y_train = y[train]
        y_eval_offset = np.zeros_like(prompt_baseline)
        shuffle_within_prompt = False

    rng = np.random.default_rng(seed)
    target_blocks = [y_train]
    train_prompt_ids = prompt_ids[train]
    for _ in range(shuffle_repeats):
        target_blocks.append(
            _shuffle_rows(y_train, train_prompt_ids, rng, shuffle_within_prompt)
        )
    combined_targets = np.concatenate(target_blocks, axis=1)

    model = _make_regressor(regressor, ridge_alpha)
    model.fit(X_train, combined_targets)
    combined_predictions = np.asarray(model.predict(X_evaluation))
    if combined_predictions.ndim == 1:
        combined_predictions = combined_predictions[:, None]
    target_width = y.shape[1]
    predictions = combined_predictions[:, :target_width] + y_eval_offset
    shuffled_predictions = [
        combined_predictions[:, start : start + target_width] + y_eval_offset
        for start in range(target_width, combined_predictions.shape[1], target_width)
    ]
    return (
        predictions,
        prompt_baseline,
        global_baseline,
        shuffled_predictions,
        prompt_coverage,
    )


def _metric_vector(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    centered = y_true - y_true.mean()
    denominator = float(np.sum(centered**2))
    r2 = None
    if len(y_true) >= 2 and denominator > 1e-15:
        r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / denominator)
    pearson = None
    pred_centered = y_pred - y_pred.mean()
    correlation_denominator = float(
        np.sqrt(np.sum(centered**2) * np.sum(pred_centered**2))
    )
    if len(y_true) >= 2 and correlation_denominator > 1e-15:
        pearson = float(np.sum(centered * pred_centered) / correlation_denominator)
    return {"r2": r2, "mae": mae, "pearson_r": pearson}


def _mean_optional(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    coordinate_names: Sequence[str],
) -> dict[str, Any]:
    per_target = [
        _metric_vector(y_true[:, index], y_pred[:, index])
        for index in range(y_true.shape[1])
    ]
    persona = {
        name: per_target[index + 1] for index, name in enumerate(coordinate_names)
    }
    metric_names = ("r2", "mae", "pearson_r")
    persona_macro = {
        metric: _mean_optional(values[metric] for values in persona.values())
        for metric in metric_names
    }
    all_macro = {
        metric: _mean_optional(values[metric] for values in per_target)
        for metric in metric_names
    }
    return {
        "n_evaluation": int(len(y_true)),
        "assistant_axis": per_target[0],
        "persona_coordinates": persona,
        "persona_macro": persona_macro,
        "all_target_macro": all_macro,
    }


def summarize_metric_runs(runs: Sequence[Any]) -> Any:
    """Recursively report mean/std for repeated numeric metric dictionaries."""

    if not runs:
        return None
    if all(isinstance(run, Mapping) for run in runs):
        keys = runs[0].keys()
        return {key: summarize_metric_runs([run[key] for run in runs]) for key in keys}
    numeric = [
        float(run)
        for run in runs
        if run is not None and isinstance(run, (int, float, np.number)) and math.isfinite(run)
    ]
    if not numeric:
        return {"mean": None, "std": None}
    return {
        "mean": float(np.mean(numeric)),
        "std": float(np.std(numeric, ddof=0)),
    }


def run_analysis(
    trajectories: Sequence[RolloutTrajectory],
    coordinate_names: Sequence[str],
    *,
    layers: Sequence[int],
    time_bins: int,
    trajectory_representation: str,
    split_kinds: Sequence[str],
    cross_group: str,
    evaluation_fraction: float,
    split_repeats: int,
    regressor: str,
    ridge_alpha: float,
    shuffle_repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    if time_bins < 1:
        raise ValueError("time_bins must be at least one")
    if not 0.0 < evaluation_fraction < 1.0:
        raise ValueError("evaluation_fraction must be between zero and one")
    if split_repeats < 1 or shuffle_repeats < 1:
        raise ValueError("split_repeats and shuffle_repeats must be at least one")

    prompt_ids = np.asarray([item.prompt_id for item in trajectories], dtype=object)
    y = np.column_stack(
        [
            np.asarray([item.assistant_axis for item in trajectories]),
            np.stack([item.persona_coordinates for item in trajectories]),
        ]
    ).astype(np.float64)
    checkpoints = np.arange(1, time_bins + 1, dtype=np.float64) / time_bins

    split_plans: dict[str, list[SplitIndices]] = {}
    for split_offset, split_kind in enumerate(split_kinds):
        plans = []
        for repeat in range(split_repeats):
            split_seed = seed + 10_007 * repeat + 1_000_003 * split_offset
            if split_kind == "cross_prompt":
                plans.append(
                    make_cross_prompt_split(
                        trajectories, evaluation_fraction, split_seed, cross_group
                    )
                )
            elif split_kind == "within_prompt":
                plans.append(
                    make_within_prompt_split(trajectories, evaluation_fraction, split_seed)
                )
            else:
                raise ValueError(f"Unknown split kind: {split_kind}")
        split_plans[split_kind] = plans

    results: list[dict[str, Any]] = []
    for layer in layers:
        if any(layer not in trajectory.layer_activations for trajectory in trajectories):
            raise ValueError(f"Layer {layer} is not present for every usable rollout")
        sampled = np.stack(
            [
                checkpoint_activations(
                    trajectory, layer, checkpoints, trajectory_representation
                )
                for trajectory in trajectories
            ]
        )  # (rollouts, time_bins, hidden)
        for time_bin, checkpoint in enumerate(checkpoints):
            X = sampled[:, time_bin, :].astype(np.float64, copy=False)
            for split_offset, split_kind in enumerate(split_kinds):
                probe_runs: list[dict[str, Any]] = []
                prompt_mean_runs: list[dict[str, Any]] = []
                global_mean_runs: list[dict[str, Any]] = []
                shuffled_runs: list[dict[str, Any]] = []
                diagnostics = []
                for repeat, split in enumerate(split_plans[split_kind]):
                    (
                        predictions,
                        prompt_baseline,
                        global_baseline,
                        shuffled_predictions,
                        coverage,
                    ) = fit_predict_with_controls(
                        X,
                        y,
                        prompt_ids,
                        split,
                        split_kind=split_kind,
                        regressor=regressor,
                        ridge_alpha=ridge_alpha,
                        shuffle_repeats=shuffle_repeats,
                        seed=seed
                        + 97 * repeat
                        + 1009 * layer
                        + 7919 * time_bin
                        + 104_729 * split_offset,
                    )
                    y_evaluation = y[split.evaluation]
                    if split_kind == "within_prompt":
                        # Score persona deviations directly.  Adding prompt means
                        # before scoring would let between-prompt variance inflate
                        # R2 even though the fit itself was centered correctly.
                        metric_targets = y_evaluation - prompt_baseline
                        metric_predictions = predictions - prompt_baseline
                        metric_prompt_baseline = np.zeros_like(prompt_baseline)
                        metric_global_baseline = global_baseline - prompt_baseline
                        metric_shuffled = [
                            shuffled - prompt_baseline
                            for shuffled in shuffled_predictions
                        ]
                    else:
                        metric_targets = y_evaluation
                        metric_predictions = predictions
                        metric_prompt_baseline = prompt_baseline
                        metric_global_baseline = global_baseline
                        metric_shuffled = shuffled_predictions
                    probe_runs.append(
                        evaluate_predictions(
                            metric_targets, metric_predictions, coordinate_names
                        )
                    )
                    prompt_mean_runs.append(
                        evaluate_predictions(
                            metric_targets, metric_prompt_baseline, coordinate_names
                        )
                    )
                    global_mean_runs.append(
                        evaluate_predictions(
                            metric_targets, metric_global_baseline, coordinate_names
                        )
                    )
                    shuffled_runs.extend(
                        evaluate_predictions(metric_targets, shuffled, coordinate_names)
                        for shuffled in metric_shuffled
                    )
                    diagnostics.append(
                        {
                            "repeat": repeat,
                            "n_train": int(len(split.train)),
                            "n_evaluation": int(len(split.evaluation)),
                            "n_train_prompts": int(len(np.unique(prompt_ids[split.train]))),
                            "n_evaluation_prompts": int(
                                len(np.unique(prompt_ids[split.evaluation]))
                            ),
                            "prompt_mean_training_coverage": float(coverage),
                        }
                    )
                results.append(
                    {
                        "split": split_kind,
                        "cross_group": cross_group if split_kind == "cross_prompt" else None,
                        "layer": int(layer),
                        "time_bin": time_bin,
                        "checkpoint_fraction": float(checkpoint),
                        "target_space": (
                            "prompt_residual"
                            if split_kind == "within_prompt"
                            else "absolute"
                        ),
                        "metrics": {
                            "probe": summarize_metric_runs(probe_runs),
                            "prompt_mean": summarize_metric_runs(prompt_mean_runs),
                            "global_mean": summarize_metric_runs(global_mean_runs),
                            "target_shuffled_probe": summarize_metric_runs(shuffled_runs),
                        },
                        "split_diagnostics": diagnostics,
                        "n_probe_fits": split_repeats,
                        "n_shuffled_fits": split_repeats * shuffle_repeats,
                    }
                )
    return results


def _comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _json_ready(value: Any) -> Any:
    """Convert config metadata to strict JSON without changing analysis values."""

    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _json_ready(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _parse_layers(value: str | None, available_layers: Sequence[int]) -> list[int]:
    if value is None or value.strip().lower() == "all":
        return list(available_layers)
    requested = [int(item) for item in _comma_list(value)]
    unavailable = sorted(set(requested) - set(available_layers))
    if unavailable:
        raise ValueError(f"Requested unavailable layers: {unavailable}")
    return requested


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit per-layer, per-reasoning-time vector regressors from persona-drift "
            "CoT activations to final-answer persona scores."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--activations", type=Path, required=True, help="Input .pt dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON (defaults beside the input with _probe_analysis.json suffix)",
    )
    parser.add_argument("--layers", default="all", help="Comma-separated layers or 'all'")
    parser.add_argument("--time-bins", type=int, default=4)
    parser.add_argument(
        "--trajectory-representation",
        choices=("latest", "cumulative_mean"),
        default="latest",
        help="CoT state at each normalized checkpoint",
    )
    parser.add_argument(
        "--splits",
        default="cross_prompt,within_prompt",
        help="Comma-separated subset of cross_prompt,within_prompt",
    )
    parser.add_argument(
        "--cross-group",
        choices=("prompt", "conversation"),
        default="prompt",
        help="Grouping unit held out by cross_prompt evaluation",
    )
    parser.add_argument("--evaluation-fraction", type=float, default=0.2)
    parser.add_argument("--split-repeats", type=int, default=1)
    parser.add_argument("--regressor", choices=("ridge", "linear"), default="ridge")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--shuffle-repeats", type=int, default=1)
    parser.add_argument(
        "--top-k-coordinates",
        type=int,
        default=8,
        help="Number of persona coordinates to predict (0 keeps axis only)",
    )
    parser.add_argument(
        "--coordinate-ranking",
        choices=("stored", "variance"),
        default="stored",
        help="How --top-k-coordinates are selected; stored avoids target-selection leakage",
    )
    parser.add_argument(
        "--coordinate-names",
        default=None,
        help="Explicit comma-separated persona coordinates; overrides --top-k-coordinates",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    trajectories, dataset_info = load_persona_activation_dataset(args.activations)
    layers = _parse_layers(args.layers, dataset_info["available_layers"])
    split_kinds = _comma_list(args.splits)
    invalid_splits = sorted(set(split_kinds) - {"cross_prompt", "within_prompt"})
    if invalid_splits:
        raise ValueError(f"Unknown splits: {invalid_splits}")
    requested_names = _comma_list(args.coordinate_names) if args.coordinate_names else None
    indices, selected_names, selection = select_persona_coordinates(
        trajectories,
        dataset_info["coordinate_names"],
        args.top_k_coordinates,
        requested_names,
        args.coordinate_ranking,
    )
    selected_trajectories = [
        RolloutTrajectory(
            prompt_id=item.prompt_id,
            conversation_id=item.conversation_id,
            rollout_id=item.rollout_id,
            layer_activations=item.layer_activations,
            progress=item.progress,
            assistant_axis=item.assistant_axis,
            persona_coordinates=item.persona_coordinates[indices],
        )
        for item in trajectories
    ]

    results = run_analysis(
        selected_trajectories,
        selected_names,
        layers=layers,
        time_bins=args.time_bins,
        trajectory_representation=args.trajectory_representation,
        split_kinds=split_kinds,
        cross_group=args.cross_group,
        evaluation_fraction=args.evaluation_fraction,
        split_repeats=args.split_repeats,
        regressor=args.regressor,
        ridge_alpha=args.ridge_alpha,
        shuffle_repeats=args.shuffle_repeats,
        seed=args.seed,
    )

    output = args.output or args.activations.with_name(
        f"{args.activations.stem}_probe_analysis.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "format_version": 1,
        "input_file": str(args.activations.resolve()),
        "dataset": dataset_info,
        "analysis": {
            "layers": layers,
            "time_bins": args.time_bins,
            "trajectory_representation": args.trajectory_representation,
            "splits": split_kinds,
            "cross_group": args.cross_group,
            "evaluation_fraction": args.evaluation_fraction,
            "split_repeats": args.split_repeats,
            "regressor": args.regressor,
            "ridge_alpha": args.ridge_alpha,
            "shuffle_repeats": args.shuffle_repeats,
            "seed": args.seed,
            "target_names": ["assistant_axis", *selected_names],
            "persona_coordinate_selection": selection,
            "coordinate_ranking": args.coordinate_ranking,
        },
        "results": results,
    }
    with output.open("w", encoding="utf-8") as file:
        json.dump(_json_ready(report), file, indent=2, allow_nan=False)

    print(
        f"Analyzed {dataset_info['n_rollouts']} rollouts from "
        f"{dataset_info['n_prompts']} prompts; wrote {len(results)} layer/time/split "
        f"results to {output}"
    )
    for split_kind in split_kinds:
        candidates = [item for item in results if item["split"] == split_kind]
        best = max(
            candidates,
            key=lambda item: item["metrics"]["probe"]["assistant_axis"]["r2"]["mean"]
            if item["metrics"]["probe"]["assistant_axis"]["r2"]["mean"] is not None
            else -math.inf,
        )
        r2 = best["metrics"]["probe"]["assistant_axis"]["r2"]["mean"]
        print(
            f"  {split_kind}: best Assistant Axis R2={r2} at layer "
            f"{best['layer']}, t={best['checkpoint_fraction']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

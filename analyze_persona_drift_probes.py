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
                        "context_activations": Tensor[num_layers, hidden],  # optional
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

* ``cross_prompt`` holds out complete groups.  The default ``--cross-group
  conversation`` holds out every prefix of a conversation, since prefixes of
  one transcript overlap heavily; ``--cross-group prompt`` is a deliberately
  weaker variant that only holds out individual prompts.
* ``within_prompt`` holds out stochastic rollouts inside every prompt.  Both
  activations and targets are centered using *training rollouts only*, so this
  measures whether activation deviations predict persona deviations rather than
  merely recovering prompt identity.  Its reported metrics are computed in this
  prompt-residual target space.

The prompt-mean baseline only uses training labels (and falls back to the global
training mean for unseen prompts; under cross-prompt holdout every evaluation
prompt is unseen, so prompt-mean coincides with global-mean there — the
context-only probe below is the meaningful context baseline).  The shuffled
control refits the same probe on permuted targets: a *block* permutation across
groups for cross-prompt evaluation (preserving within-group target structure)
and a within-prompt permutation for within-prompt evaluation.  When the dataset
stores prompt-end activations, a ``context`` cell per layer probes the state
before any reasoning token — the context-only baseline that CoT cells must beat
to demonstrate incremental predictive power.

Rollouts whose first CoT boundary lies after a checkpoint are excluded from that
checkpoint's cell (never represented by a later state), so early-time results
cannot leak future reasoning states.
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
    context_activations: dict[int, np.ndarray] | None = None  # layer -> (hidden,)


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


def _parse_context_activations(
    value: Any, layer_ids: Sequence[int]
) -> dict[int, np.ndarray] | None:
    """Parse the optional prompt-end (context) activation as a layer mapping."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        parsed = {
            _layer_number(layer): _to_numpy(item, dtype=np.float32).reshape(-1)
            for layer, item in value.items()
        }
    else:
        array = _to_numpy(value, dtype=np.float32)
        array = array.reshape(-1) if array.ndim == 1 else array.squeeze()
        if array.ndim == 1 and len(layer_ids) == 1:
            parsed = {int(layer_ids[0]): array}
        elif array.ndim == 2 and array.shape[0] == len(layer_ids):
            parsed = {int(layer): array[index] for index, layer in enumerate(layer_ids)}
        else:
            raise ValueError(
                f"context_activations shape {array.shape} is incompatible with "
                f"layers {list(layer_ids)}"
            )
    if any(not np.all(np.isfinite(item)) for item in parsed.values()):
        raise ValueError("context activations contain non-finite values")
    return parsed


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
                        context_activations=_parse_context_activations(
                            rollout.get("context_activations"), layer_ids
                        ),
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
        "has_context": all(
            trajectory.context_activations is not None for trajectory in trajectories
        ),
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
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a variable-length CoT trajectory at normalized checkpoints.

    Returns ``(states, valid)``.  ``valid[j]`` is False when the rollout has no
    boundary at or before checkpoint ``j`` — such cells must be excluded rather
    than filled with a later state, which would leak future reasoning into an
    earlier checkpoint.  Invalid rows of ``states`` are zero-filled.
    """

    activations = trajectory.layer_activations[layer]
    progress = trajectory.progress
    sampled: list[np.ndarray] = []
    valid: list[bool] = []
    for checkpoint in checkpoints:
        eligible = np.flatnonzero(progress <= checkpoint + 1e-9)
        if not len(eligible):
            sampled.append(np.zeros_like(activations[0]))
            valid.append(False)
            continue
        boundary_index = int(eligible[-1])
        valid.append(True)
        if representation == "latest":
            sampled.append(activations[boundary_index])
        elif representation == "cumulative_mean":
            sampled.append(activations[: boundary_index + 1].mean(axis=0))
        else:
            raise ValueError(f"Unknown trajectory representation: {representation}")
    return np.stack(sampled), np.asarray(valid, dtype=bool)


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
    group_ids: np.ndarray,
    rng: np.random.Generator,
    within_group: bool,
) -> np.ndarray:
    """Shuffle targets for the control probe.

    ``within_group=True`` permutes rows inside every group (the within-prompt
    control).  ``within_group=False`` performs a *block* permutation that
    preserves within-group target correlation.  A plain row-level permutation
    would destroy that correlation and make the control probe artificially easy
    to beat when targets are group-correlated.
    """

    if within_group:
        shuffled = targets.copy()
        for group in np.unique(group_ids):
            rows = np.flatnonzero(group_ids == group)
            shuffled[rows] = targets[rng.permutation(rows)]
        return shuffled

    # True permutation for unequal block sizes: concatenate intact group blocks
    # in a permuted order over the grouped row layout.  Every target appears
    # exactly once (marginals preserved), within-group runs stay contiguous,
    # and group boundaries misalign with the destination groups — the intended
    # null under cluster dependence.
    shuffled = np.empty_like(targets)
    unique_groups = np.unique(group_ids)
    rows_by_group = {
        group: np.flatnonzero(group_ids == group) for group in unique_groups
    }
    grouped_row_order = np.concatenate(
        [rows_by_group[group] for group in unique_groups]
    )
    permuted_blocks = np.concatenate(
        [targets[rows_by_group[group]] for group in rng.permutation(unique_groups)]
    )
    shuffled[grouped_row_order] = permuted_blocks
    return shuffled


def _select_ridge_alpha(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_groups: np.ndarray,
    alphas: Sequence[float],
    seed: int,
) -> float:
    """Pick ridge alpha by grouped cross-validation on training data only."""

    if len(alphas) == 1:
        return float(alphas[0])
    from sklearn.model_selection import GroupKFold

    n_splits = min(3, len(np.unique(train_groups)))
    if n_splits < 2:
        return float(alphas[0])
    folds = list(GroupKFold(n_splits=n_splits).split(X_train, y_train, train_groups))
    best_alpha, best_error = float(alphas[0]), math.inf
    for alpha in alphas:
        errors = []
        for fit_rows, validation_rows in folds:
            model = _make_regressor("ridge", float(alpha))
            model.fit(X_train[fit_rows], y_train[fit_rows])
            predictions = np.asarray(model.predict(X_train[validation_rows]))
            if predictions.ndim == 1:
                predictions = predictions[:, None]
            errors.append(float(np.mean((predictions - y_train[validation_rows]) ** 2)))
        error = float(np.mean(errors))
        if error < best_error:
            best_error, best_alpha = error, float(alpha)
    return best_alpha


def fit_predict_with_controls(
    X: np.ndarray,
    y: np.ndarray,
    prompt_ids: np.ndarray,
    split: SplitIndices,
    *,
    split_kind: str,
    control_group_ids: np.ndarray,
    regressor: str,
    ridge_alpha: float,
    ridge_alphas: Sequence[float] | None,
    shuffle_repeats: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], float, float]:
    """Fit one probe plus target-shuffled controls in a single multi-output solve.

    ``control_group_ids`` carries the dependence structure: the split's grouping
    unit (conversation or prompt for cross-prompt; prompt for within-prompt).
    It drives both the block-permutation shuffled control and, when
    ``ridge_alphas`` is given, grouped training-only alpha selection.
    """

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
    train_control_groups = control_group_ids[train]
    for _ in range(shuffle_repeats):
        target_blocks.append(
            _shuffle_rows(y_train, train_control_groups, rng, shuffle_within_prompt)
        )
    combined_targets = np.concatenate(target_blocks, axis=1)

    if regressor == "ridge" and ridge_alphas:
        ridge_alpha = _select_ridge_alpha(
            X_train, y_train, train_control_groups, ridge_alphas, seed
        )
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
        float(ridge_alpha),
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
    ridge_alphas: Sequence[float] | None = None,
    cohort: str = "per_bin",
    include_context: bool = False,
) -> list[dict[str, Any]]:
    if time_bins < 1:
        raise ValueError("time_bins must be at least one")
    if not 0.0 < evaluation_fraction < 1.0:
        raise ValueError("evaluation_fraction must be between zero and one")
    if split_repeats < 1 or shuffle_repeats < 1:
        raise ValueError("split_repeats and shuffle_repeats must be at least one")
    if cohort not in {"per_bin", "fixed"}:
        raise ValueError(f"Unknown cohort mode: {cohort}")
    if include_context and any(
        trajectory.context_activations is None for trajectory in trajectories
    ):
        raise ValueError(
            "include_context requires context activations for every rollout; "
            "regenerate the dataset with the current gather script"
        )

    prompt_ids = np.asarray([item.prompt_id for item in trajectories], dtype=object)
    # Dependence structure per split kind, used for the shuffled control's block
    # permutation and grouped ridge-alpha selection.
    control_groups_by_split = {
        "cross_prompt": _split_group_names(trajectories, cross_group),
        "within_prompt": prompt_ids,
    }
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

    def evaluate_cell(
        layer: int,
        input_kind: str,
        time_bin: int,
        checkpoint_fraction: float,
        X: np.ndarray,
        cell_valid: np.ndarray,
        seed_offset: int,
    ) -> None:
        for split_offset, split_kind in enumerate(split_kinds):
            control_group_ids = control_groups_by_split[split_kind]
            probe_runs: list[dict[str, Any]] = []
            prompt_mean_runs: list[dict[str, Any]] = []
            global_mean_runs: list[dict[str, Any]] = []
            shuffled_runs: list[dict[str, Any]] = []
            diagnostics = []
            for repeat, full_split in enumerate(split_plans[split_kind]):
                # Rollouts with no CoT boundary by this checkpoint are excluded
                # from the cell instead of being represented by a future state.
                train = full_split.train[cell_valid[full_split.train]]
                evaluation = full_split.evaluation[
                    cell_valid[full_split.evaluation]
                ]
                if split_kind == "within_prompt" and len(train):
                    train_prompts = set(prompt_ids[train].tolist())
                    evaluation = np.asarray(
                        [
                            index
                            for index in evaluation
                            if prompt_ids[index] in train_prompts
                        ],
                        dtype=int,
                    )
                if not len(train) or not len(evaluation):
                    diagnostics.append(
                        {
                            "repeat": repeat,
                            "skipped": "no valid rollouts at this checkpoint",
                            "n_excluded_no_boundary": int(np.sum(~cell_valid)),
                        }
                    )
                    continue
                split = SplitIndices(train=train, evaluation=evaluation)
                (
                    predictions,
                    prompt_baseline,
                    global_baseline,
                    shuffled_predictions,
                    coverage,
                    chosen_alpha,
                ) = fit_predict_with_controls(
                    X,
                    y,
                    prompt_ids,
                    split,
                    split_kind=split_kind,
                    control_group_ids=control_group_ids,
                    regressor=regressor,
                    ridge_alpha=ridge_alpha,
                    ridge_alphas=ridge_alphas,
                    shuffle_repeats=shuffle_repeats,
                    seed=seed
                    + 97 * repeat
                    + 1009 * layer
                    + 7919 * seed_offset
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
                        "n_excluded_no_boundary": int(np.sum(~cell_valid)),
                        "prompt_mean_training_coverage": float(coverage),
                        "ridge_alpha": chosen_alpha,
                    }
                )
            results.append(
                {
                    "split": split_kind,
                    "cross_group": cross_group if split_kind == "cross_prompt" else None,
                    "layer": int(layer),
                    "input": input_kind,
                    "time_bin": time_bin,
                    "checkpoint_fraction": checkpoint_fraction,
                    "cohort": cohort,
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
                    "n_probe_fits": len(probe_runs),
                    "n_shuffled_fits": len(shuffled_runs),
                }
            )

    for layer in layers:
        if any(layer not in trajectory.layer_activations for trajectory in trajectories):
            raise ValueError(f"Layer {layer} is not present for every usable rollout")
        sampled_with_validity = [
            checkpoint_activations(
                trajectory, layer, checkpoints, trajectory_representation
            )
            for trajectory in trajectories
        ]
        sampled = np.stack(
            [states for states, _ in sampled_with_validity]
        )  # (rollouts, time_bins, hidden)
        valid = np.stack(
            [validity for _, validity in sampled_with_validity]
        )  # (rollouts, time_bins)
        fixed_cohort_valid = valid.all(axis=1)

        if include_context:
            missing = [
                trajectory.rollout_id
                for trajectory in trajectories
                if layer not in (trajectory.context_activations or {})
            ]
            if missing:
                raise ValueError(
                    f"Layer {layer} context activations missing for {missing[:3]}"
                )
            X_context = np.stack(
                [trajectory.context_activations[layer] for trajectory in trajectories]
            ).astype(np.float64)
            # The context (prompt-end) probe is the context-only baseline: it
            # measures how much of the persona target is already predictable
            # before any reasoning token is generated.  Under --cohort fixed it
            # uses the same cohort as every CoT bin, making all incremental
            # comparisons cohort-matched; under per_bin only the final bin
            # (where every rollout has a boundary) is cohort-matched to it.
            context_valid = (
                fixed_cohort_valid
                if cohort == "fixed"
                else np.ones(len(trajectories), dtype=bool)
            )
            evaluate_cell(
                layer,
                "context",
                -1,
                0.0,
                X_context,
                context_valid,
                seed_offset=time_bins,
            )

        for time_bin, checkpoint in enumerate(checkpoints):
            X = sampled[:, time_bin, :].astype(np.float64, copy=False)
            cell_valid = (
                fixed_cohort_valid if cohort == "fixed" else valid[:, time_bin]
            )
            evaluate_cell(
                layer, "cot", time_bin, float(checkpoint), X, cell_valid, time_bin
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
        default="conversation",
        help=(
            "Grouping unit held out by cross_prompt evaluation. Persona-drift "
            "prompts are nested prefixes of one conversation, so 'conversation' "
            "(default) prevents the probe from exploiting conversation identity; "
            "'prompt' is a weaker, deliberately leaky variant."
        ),
    )
    parser.add_argument("--evaluation-fraction", type=float, default=0.2)
    parser.add_argument("--split-repeats", type=int, default=1)
    parser.add_argument("--regressor", choices=("ridge", "linear"), default="ridge")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--ridge-alphas",
        default=None,
        help=(
            "Comma-separated alpha grid; selected per fit by grouped "
            "cross-validation on training data only (overrides --ridge-alpha)"
        ),
    )
    parser.add_argument("--shuffle-repeats", type=int, default=1)
    parser.add_argument(
        "--cohort",
        choices=("per_bin", "fixed"),
        default="per_bin",
        help=(
            "per_bin excludes rollouts with no boundary at each checkpoint "
            "(cohort may change over time bins); fixed restricts every bin to "
            "rollouts valid at all checkpoints (constant cohort)"
        ),
    )
    parser.add_argument(
        "--include-context",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Add a prompt-end (pre-CoT) probe cell per layer as the "
            "context-only baseline; auto enables it when the dataset stores "
            "context activations"
        ),
    )
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
            context_activations=item.context_activations,
        )
        for item in trajectories
    ]

    if args.include_context == "auto":
        include_context = bool(dataset_info["has_context"])
    else:
        include_context = args.include_context == "on"
    ridge_alphas = (
        [float(item) for item in _comma_list(args.ridge_alphas)]
        if args.ridge_alphas
        else None
    )

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
        ridge_alphas=ridge_alphas,
        shuffle_repeats=args.shuffle_repeats,
        seed=args.seed,
        cohort=args.cohort,
        include_context=include_context,
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
            "ridge_alphas": ridge_alphas,
            "shuffle_repeats": args.shuffle_repeats,
            "cohort": args.cohort,
            "include_context": include_context,
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
        candidates = [
            item
            for item in results
            if item["split"] == split_kind and item["metrics"]["probe"] is not None
        ]
        if not candidates:
            print(f"  {split_kind}: no cells with successful probe fits")
            continue
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

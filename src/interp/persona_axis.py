"""Load and score final-answer activations in the Assistant Axis persona space.

The precomputed Qwen3-32B artifacts are published in
``lu-christina/assistant-axis-vectors/qwen-3-32b``.  This module deliberately
keeps artifact loading separate from model tracing so it can also be used to
rescore previously gathered answer activations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


HF_REPO_ID = "lu-christina/assistant-axis-vectors"
DEFAULT_MODEL_SUBDIR = "qwen-3-32b"


def _unwrap_tensor(value: Any, source: Path) -> torch.Tensor:
    """Accept raw tensors and the small wrapper dictionaries used by some runs."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        for key in ("vector", "tensor", "activation", "activations", "mean"):
            candidate = value.get(key)
            if isinstance(candidate, torch.Tensor):
                return candidate.detach().cpu()
        tensors = [candidate for candidate in value.values() if isinstance(candidate, torch.Tensor)]
        if len(tensors) == 1:
            return tensors[0].detach().cpu()
    raise TypeError(f"Expected a tensor artifact in {source}, got {type(value).__name__}")


def _load_tensor(path: Path) -> torch.Tensor:
    return _unwrap_tensor(torch.load(path, map_location="cpu", weights_only=True), path)


def _select_layer(vector: torch.Tensor, layer: int, source: Path) -> torch.Tensor:
    vector = vector.squeeze()
    if vector.ndim == 1:
        selected = vector
    elif vector.ndim == 2:
        if not 0 <= layer < vector.shape[0]:
            raise IndexError(
                f"Layer {layer} is outside {source}'s first dimension {vector.shape[0]}"
            )
        selected = vector[layer]
    else:
        raise ValueError(
            f"Expected a [hidden] or [layers, hidden] tensor in {source}, "
            f"got shape {tuple(vector.shape)}"
        )
    return selected.to(dtype=torch.float32).contiguous()


def _find_local_model_dir(path: str | Path, model_subdir: str) -> Path:
    supplied = Path(path).expanduser()
    if supplied.is_file():
        supplied = supplied.parent
    candidates = (
        supplied,
        supplied / model_subdir,
        supplied / "assistant-axis-vectors" / model_subdir,
    )
    for candidate in candidates:
        if (candidate / "assistant_axis.pt").is_file():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find assistant_axis.pt under the supplied artifact path. Searched: {searched}"
    )


def resolve_assistant_axis_dir(
    artifact_path: str | Path | None = None,
    *,
    model_subdir: str = DEFAULT_MODEL_SUBDIR,
    local_files_only: bool = False,
) -> Path:
    """Resolve a local model directory, downloading the official HF snapshot if needed."""

    if artifact_path is not None:
        return _find_local_model_dir(artifact_path, model_subdir)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - depends on runtime extras
        raise ImportError(
            "huggingface-hub is required when --assistant-axis-path is omitted. "
            "Install project dependencies or pass a local artifact directory."
        ) from error

    snapshot = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        allow_patterns=(
            f"{model_subdir}/assistant_axis.pt",
            f"{model_subdir}/role_vectors/*.pt",
        ),
        local_files_only=local_files_only,
    )
    return _find_local_model_dir(snapshot, model_subdir)


@dataclass(frozen=True)
class PersonaScore:
    assistant_axis_score: float
    persona_coordinates: torch.Tensor


class AssistantAxisScorer:
    """Project final-answer mean activations into Assistant Axis coordinates.

    ``coordinate_mode='pca'`` reproduces the upstream role-centroid construction:
    role vectors are mean-centered, PCA is fit on those centered centroids, and the
    same centering/transform is applied to an answer activation.  ``role_cosine``
    instead returns cosine similarities to every centered role centroid.
    """

    VALID_COORDINATE_MODES = {"pca", "role_cosine", "both"}

    def __init__(
        self,
        artifact_path: str | Path | None = None,
        *,
        target_layer: int = 32,
        coordinate_mode: str = "pca",
        max_pca_components: int = 8,
        model_subdir: str = DEFAULT_MODEL_SUBDIR,
        local_files_only: bool = False,
    ) -> None:
        if coordinate_mode not in self.VALID_COORDINATE_MODES:
            raise ValueError(
                f"coordinate_mode must be one of {sorted(self.VALID_COORDINATE_MODES)}, "
                f"got {coordinate_mode!r}"
            )
        if max_pca_components < 1 and coordinate_mode in {"pca", "both"}:
            raise ValueError("max_pca_components must be positive when PCA is requested")

        self.target_layer = int(target_layer)
        self.coordinate_mode = coordinate_mode
        self.artifact_dir = resolve_assistant_axis_dir(
            artifact_path,
            model_subdir=model_subdir,
            local_files_only=local_files_only,
        )

        axis_path = self.artifact_dir / "assistant_axis.pt"
        self.assistant_axis = _select_layer(
            _load_tensor(axis_path), self.target_layer, axis_path
        )
        axis_norm = torch.linalg.vector_norm(self.assistant_axis)
        if not torch.isfinite(axis_norm) or axis_norm <= 0:
            raise ValueError(f"Assistant Axis vector at layer {target_layer} has zero/invalid norm")
        self.assistant_axis_unit = self.assistant_axis / axis_norm

        role_paths = sorted((self.artifact_dir / "role_vectors").glob("*.pt"))
        if not role_paths:
            raise FileNotFoundError(
                f"No role vector .pt files found in {self.artifact_dir / 'role_vectors'}"
            )
        self.role_names = [path.stem for path in role_paths]
        self.role_vectors = torch.stack(
            [_select_layer(_load_tensor(path), self.target_layer, path) for path in role_paths]
        )
        if self.role_vectors.shape[1] != self.assistant_axis.shape[0]:
            raise ValueError(
                "Assistant Axis and role-vector hidden sizes differ: "
                f"{self.assistant_axis.shape[0]} vs {self.role_vectors.shape[1]}"
            )
        self.role_mean = self.role_vectors.mean(dim=0)
        self.centered_role_vectors = self.role_vectors - self.role_mean

        self._pca = None
        self.pca_explained_variance_ratio: list[float] = []
        pca_names: list[str] = []
        if coordinate_mode in {"pca", "both"}:
            if len(role_paths) < 2:
                raise ValueError("At least two role vectors are required to fit persona PCA")
            try:
                from sklearn.decomposition import PCA
            except ImportError as error:  # pragma: no cover - declared project dependency
                raise ImportError("scikit-learn is required for Assistant Axis PCA") from error
            n_components = min(
                int(max_pca_components),
                len(role_paths) - 1,
                self.centered_role_vectors.shape[1],
            )
            self._pca = PCA(n_components=n_components)
            self._pca.fit(self.centered_role_vectors.numpy())
            self.pca_explained_variance_ratio = [
                float(value) for value in self._pca.explained_variance_ratio_
            ]
            pca_names = [f"pc_{index + 1}" for index in range(n_components)]

        cosine_names = (
            [f"role_cosine:{name}" for name in self.role_names]
            if coordinate_mode in {"role_cosine", "both"}
            else []
        )
        self.coordinate_names = [*pca_names, *cosine_names]

    @property
    def hidden_size(self) -> int:
        return int(self.assistant_axis.shape[0])

    def _validate_activation(self, activation: torch.Tensor | np.ndarray | Sequence[float]) -> torch.Tensor:
        value = torch.as_tensor(activation, dtype=torch.float32).detach().cpu().reshape(-1)
        if value.numel() != self.hidden_size:
            raise ValueError(
                f"Expected answer activation with hidden size {self.hidden_size}, "
                f"got {value.numel()}"
            )
        if not torch.all(torch.isfinite(value)):
            raise ValueError("Answer activation contains non-finite values")
        return value

    def score(self, activation: torch.Tensor | np.ndarray | Sequence[float]) -> PersonaScore:
        value = self._validate_activation(activation)
        assistant_score = float(torch.dot(value, self.assistant_axis_unit))
        centered = value - self.role_mean

        coordinates: list[torch.Tensor] = []
        if self._pca is not None:
            transformed = self._pca.transform(centered.reshape(1, -1).numpy())[0]
            coordinates.append(torch.from_numpy(transformed).to(dtype=torch.float32))
        if self.coordinate_mode in {"role_cosine", "both"}:
            normalized_roles = F.normalize(self.centered_role_vectors, dim=1, eps=1e-12)
            normalized_value = F.normalize(centered, dim=0, eps=1e-12)
            coordinates.append((normalized_roles @ normalized_value).to(dtype=torch.float32))

        persona_coordinates = (
            torch.cat(coordinates).contiguous()
            if coordinates
            else torch.empty(0, dtype=torch.float32)
        )
        return PersonaScore(assistant_score, persona_coordinates)

    def metadata(self) -> dict[str, Any]:
        return {
            "artifact_dir": str(self.artifact_dir),
            "target_layer": self.target_layer,
            "hidden_size": self.hidden_size,
            "coordinate_mode": self.coordinate_mode,
            "coordinate_names": list(self.coordinate_names),
            "role_names": list(self.role_names),
            "pca_explained_variance_ratio": list(self.pca_explained_variance_ratio),
            "assistant_axis_definition": "dot(answer_mean_activation, normalized_assistant_axis)",
        }

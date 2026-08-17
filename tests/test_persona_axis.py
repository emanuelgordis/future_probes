import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.interp.persona_axis import AssistantAxisScorer, resolve_assistant_axis_dir


def _write_artifacts(root: Path, num_layers: int = 4, hidden: int = 6) -> dict:
    rng = np.random.default_rng(3)
    axis = torch.tensor(rng.normal(size=(num_layers, hidden)), dtype=torch.float32)
    roles = {
        name: torch.tensor(rng.normal(size=(num_layers, hidden)), dtype=torch.float32)
        for name in ("alpha", "beta", "gamma")
    }
    root.mkdir(parents=True, exist_ok=True)
    torch.save(axis, root / "assistant_axis.pt")
    (root / "role_vectors").mkdir()
    for name, tensor in roles.items():
        torch.save(tensor, root / "role_vectors" / f"{name}.pt")
    return {"axis": axis, "roles": roles}


class AssistantAxisScorerTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "qwen-test"
        self.artifacts = _write_artifacts(self.root)
        self.layer = 2

    def tearDown(self):
        self.temp_dir.cleanup()

    def _scorer(self, **kwargs):
        return AssistantAxisScorer(self.root, target_layer=self.layer, **kwargs)

    def test_axis_score_is_projection_on_unit_axis(self):
        scorer = self._scorer(coordinate_mode="pca", max_pca_components=2)
        value = torch.arange(6, dtype=torch.float32)
        axis = self.artifacts["axis"][self.layer]
        expected = float(torch.dot(value, axis / axis.norm()))
        self.assertAlmostEqual(
            scorer.score(value).assistant_axis_score, expected, places=5
        )

    def test_pca_coordinates_match_manual_sklearn_transform(self):
        from sklearn.decomposition import PCA

        scorer = self._scorer(coordinate_mode="pca", max_pca_components=2)
        self.assertEqual(scorer.coordinate_names, ["pc_1", "pc_2"])

        roles = torch.stack(
            [self.artifacts["roles"][name][self.layer] for name in ("alpha", "beta", "gamma")]
        )
        centered = roles - roles.mean(dim=0)
        pca = PCA(n_components=2)
        pca.fit(centered.numpy())

        value = torch.arange(6, dtype=torch.float32)
        expected = pca.transform((value - roles.mean(dim=0)).reshape(1, -1).numpy())[0]
        result = scorer.score(value).persona_coordinates.numpy()
        np.testing.assert_allclose(np.abs(result), np.abs(expected), rtol=1e-4)

    def test_role_cosine_coordinates(self):
        scorer = self._scorer(coordinate_mode="role_cosine")
        self.assertEqual(
            scorer.coordinate_names,
            ["role_cosine:alpha", "role_cosine:beta", "role_cosine:gamma"],
        )
        value = torch.arange(6, dtype=torch.float32)
        coordinates = scorer.score(value).persona_coordinates
        roles = torch.stack(
            [self.artifacts["roles"][name][self.layer] for name in ("alpha", "beta", "gamma")]
        )
        centered_roles = roles - roles.mean(dim=0)
        centered_value = value - roles.mean(dim=0)
        for index in range(3):
            expected = float(
                torch.dot(centered_roles[index], centered_value)
                / (centered_roles[index].norm() * centered_value.norm())
            )
            self.assertAlmostEqual(float(coordinates[index]), expected, places=5)

    def test_rejects_wrong_hidden_size_and_non_finite(self):
        scorer = self._scorer(coordinate_mode="role_cosine")
        with self.assertRaises(ValueError):
            scorer.score(torch.zeros(5))
        with self.assertRaises(ValueError):
            scorer.score(torch.tensor([1.0, 2.0, math.nan, 0.0, 0.0, 0.0]))

    def test_resolve_accepts_parent_directory(self):
        resolved = resolve_assistant_axis_dir(
            self.temp_dir.name, model_subdir="qwen-test"
        )
        self.assertEqual(resolved, self.root.resolve())

    def test_metadata_reports_coordinate_construction(self):
        scorer = self._scorer(coordinate_mode="both", max_pca_components=2)
        metadata = scorer.metadata()
        self.assertEqual(metadata["target_layer"], self.layer)
        self.assertEqual(len(metadata["coordinate_names"]), 5)
        self.assertEqual(len(metadata["pca_explained_variance_ratio"]), 2)


if __name__ == "__main__":
    unittest.main()

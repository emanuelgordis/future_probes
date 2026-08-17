import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from analyze_persona_drift_probes import (
    load_persona_activation_dataset,
    main,
    run_analysis,
)


def _synthetic_payload() -> dict:
    rng = np.random.default_rng(7)
    prompts = []
    for prompt_index in range(8):
        conversation_id = f"conversation_{prompt_index // 2}"
        turn_index = prompt_index % 2
        prompt_effect = (prompt_index - 3.5) / 3.5
        rollouts = []
        for rollout_index in range(5):
            stochastic_effect = (rollout_index - 2.0) / 2.0 + rng.normal(scale=0.03)
            assistant_axis = 0.7 * prompt_effect + stochastic_effect
            coordinates = np.asarray(
                [
                    -0.2 * prompt_effect + 1.5 * stochastic_effect,
                    0.5 * prompt_effect - 0.4 * stochastic_effect,
                ],
                dtype=np.float32,
            )
            boundaries = []
            for boundary_index in range(4):
                signal = np.asarray(
                    [assistant_axis, coordinates[0], coordinates[1], prompt_effect, 1.0],
                    dtype=np.float32,
                )
                informative_layer = signal + rng.normal(scale=0.001, size=5)
                noise_layer = rng.normal(size=5)
                boundaries.append(np.stack([informative_layer, noise_layer]))
            rollouts.append(
                {
                    "rollout_index": rollout_index,
                    "cot_activations": torch.tensor(np.stack(boundaries)),  # [B,L,H]
                    "cot_progress": torch.tensor([0.25, 0.5, 0.75, 1.0]),
                    "assistant_axis_score": assistant_axis,
                    "persona_coordinates": torch.tensor(coordinates),
                }
            )
        prompts.append(
            {
                # This intentionally conflicts with the derived identity.  The
                # loader must group by the exact conversation prefix + user turn.
                "prompt_id": f"legacy_prompt_{prompt_index}",
                "conversation_id": conversation_id,
                "turn_index": turn_index,
                "rollouts": rollouts,
            }
        )
    return {
        "format_version": 1,
        "config": {"model_name": "synthetic"},
        "layers": torch.tensor([0, 1]),
        "persona_coordinate_names": ["role_alpha", "role_beta"],
        "prompts": prompts,
    }


class PersonaDriftProbeAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "persona_activations.pt"
        torch.save(_synthetic_payload(), self.path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_loads_native_boundary_layer_layout_and_derives_prompt_identity(self):
        trajectories, info = load_persona_activation_dataset(self.path)

        self.assertEqual(info["available_layers"], [0, 1])
        self.assertEqual(info["n_rollouts"], 40)
        self.assertEqual(info["n_prompts"], 8)
        self.assertEqual(info["n_conversations"], 4)
        self.assertEqual(trajectories[0].prompt_id, "conversation_0::turn_0")
        self.assertEqual(trajectories[0].conversation_id, "conversation_0")
        self.assertEqual(trajectories[0].layer_activations[0].shape, (4, 5))

    def test_signal_beats_prompt_mean_and_within_prompt_shuffle(self):
        trajectories, info = load_persona_activation_dataset(self.path)
        results = run_analysis(
            trajectories,
            info["coordinate_names"],
            layers=[0],
            time_bins=2,
            trajectory_representation="latest",
            split_kinds=["cross_prompt", "within_prompt"],
            cross_group="prompt",
            evaluation_fraction=0.4,
            split_repeats=1,
            regressor="ridge",
            ridge_alpha=1e-6,
            shuffle_repeats=5,
            seed=11,
        )

        self.assertEqual(len(results), 4)
        for result in results:
            probe = result["metrics"]["probe"]["assistant_axis"]
            prompt_mean = result["metrics"]["prompt_mean"]["assistant_axis"]
            shuffled = result["metrics"]["target_shuffled_probe"]["assistant_axis"]
            self.assertGreater(probe["r2"]["mean"], 0.95)
            self.assertLess(probe["mae"]["mean"], prompt_mean["mae"]["mean"])
            self.assertLess(probe["mae"]["mean"], shuffled["mae"]["mean"])

    def test_cli_writes_strict_json_report(self):
        output = Path(self.temp_dir.name) / "analysis.json"
        exit_code = main(
            [
                "--activations",
                str(self.path),
                "--output",
                str(output),
                "--layers",
                "0",
                "--time-bins",
                "1",
                "--splits",
                "within_prompt",
                "--top-k-coordinates",
                "1",
                "--ridge-alpha",
                "1e-6",
            ]
        )

        self.assertEqual(exit_code, 0)
        with output.open(encoding="utf-8") as file:
            report = json.load(file)
        self.assertEqual(report["analysis"]["target_names"][0], "assistant_axis")
        self.assertEqual(len(report["results"]), 1)
        self.assertIn("target_shuffled_probe", report["results"][0]["metrics"])


if __name__ == "__main__":
    unittest.main()

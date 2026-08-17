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


class CheckpointExclusionTest(unittest.TestCase):
    """Early checkpoints must exclude rollouts, never borrow future states."""

    def test_checkpoint_activations_reports_validity(self):
        from analyze_persona_drift_probes import checkpoint_activations
        from analyze_persona_drift_probes import RolloutTrajectory

        trajectory = RolloutTrajectory(
            prompt_id="p", conversation_id="c", rollout_id="r",
            layer_activations={0: np.arange(6, dtype=np.float64).reshape(2, 3)},
            progress=np.asarray([0.6, 1.0]),
            assistant_axis=0.0,
            persona_coordinates=np.zeros(1),
        )
        states, valid = checkpoint_activations(
            trajectory, 0, np.asarray([0.5, 1.0]), "latest"
        )
        self.assertEqual(valid.tolist(), [False, True])
        self.assertTrue((states[0] == 0).all())  # no future state leaked
        np.testing.assert_array_equal(states[1], [3.0, 4.0, 5.0])

    def test_late_first_boundary_rollouts_are_excluded_from_early_bins(self):
        from analyze_persona_drift_probes import run_analysis

        rng = np.random.default_rng(0)
        trajectories = []
        for prompt_index in range(4):
            for rollout_index in range(3):
                target = float(prompt_index) + 0.1 * rollout_index
                # First boundary only at 60% of thinking: bin t=0.5 has no state.
                activations = np.stack([
                    np.asarray([target, 1.0, rng.normal()]),
                    np.asarray([target, 2.0, rng.normal()]),
                ])
                trajectories.append(
                    __import__("analyze_persona_drift_probes").RolloutTrajectory(
                        prompt_id=f"p{prompt_index}",
                        conversation_id=f"c{prompt_index % 2}",
                        rollout_id=f"p{prompt_index}r{rollout_index}",
                        layer_activations={0: activations.astype(np.float64)},
                        progress=np.asarray([0.6, 1.0]),
                        assistant_axis=target,
                        persona_coordinates=np.asarray([target]),
                    )
                )
        results = run_analysis(
            trajectories, ["coordinate"], layers=[0], time_bins=2,
            trajectory_representation="latest",
            split_kinds=["cross_prompt"], cross_group="prompt",
            evaluation_fraction=0.25, split_repeats=1, regressor="ridge",
            ridge_alpha=1e-6, shuffle_repeats=1, seed=0,
        )
        early = next(r for r in results if r["time_bin"] == 0)
        late = next(r for r in results if r["time_bin"] == 1)
        # All rollouts lack a boundary at t=0.5 -> the early cell has no fits.
        self.assertEqual(early["n_probe_fits"], 0)
        self.assertEqual(
            early["split_diagnostics"][0]["skipped"],
            "no valid rollouts at this checkpoint",
        )
        self.assertEqual(late["n_probe_fits"], 1)
        self.assertEqual(late["split_diagnostics"][0]["n_excluded_no_boundary"], 0)
        self.assertGreater(late["metrics"]["probe"]["assistant_axis"]["r2"]["mean"], 0.9)

    def test_cross_group_default_is_conversation(self):
        from analyze_persona_drift_probes import build_argument_parser

        args = build_argument_parser().parse_args(["--activations", "x.pt"])
        self.assertEqual(args.cross_group, "conversation")


class ControlsAndBaselinesTest(unittest.TestCase):
    def test_cross_shuffle_is_a_marginal_preserving_block_permutation(self):
        from analyze_persona_drift_probes import _shuffle_rows

        groups = np.asarray(["a", "a", "a", "b", "b", "c"], dtype=object)
        targets = np.asarray([[1.0], [2.0], [3.0], [10.0], [20.0], [100.0]])
        blocks = {"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0], "c": [100.0]}
        for seed in range(6):
            rng = np.random.default_rng(seed)
            shuffled = _shuffle_rows(targets, groups, rng, within_group=False)
            # A true permutation: every target appears exactly once.
            self.assertEqual(
                sorted(shuffled.reshape(-1).tolist()),
                sorted(targets.reshape(-1).tolist()),
            )
            # In grouped row order, the sequence is intact blocks in some order.
            sequence = shuffled.reshape(-1).tolist()  # rows already group-ordered
            remaining, order = sequence, []
            while remaining:
                for name, block in blocks.items():
                    if remaining[: len(block)] == block and name not in order:
                        order.append(name)
                        remaining = remaining[len(block) :]
                        break
                else:
                    self.fail(f"sequence {sequence} is not a block concatenation")
            self.assertEqual(sorted(order), ["a", "b", "c"])

    def test_context_cell_probes_prompt_end_state(self):
        from analyze_persona_drift_probes import RolloutTrajectory, run_analysis

        rng = np.random.default_rng(1)
        trajectories = []
        for prompt_index in range(6):
            for rollout_index in range(3):
                target = float(prompt_index) + 0.05 * rollout_index
                cot = rng.normal(size=(2, 3)).astype(np.float64)  # pure noise CoT
                context = np.asarray([target, 1.0, rng.normal()])  # predictive context
                trajectories.append(
                    RolloutTrajectory(
                        prompt_id=f"p{prompt_index}",
                        conversation_id=f"c{prompt_index % 3}",
                        rollout_id=f"p{prompt_index}r{rollout_index}",
                        layer_activations={0: cot},
                        progress=np.asarray([0.5, 1.0]),
                        assistant_axis=target,
                        persona_coordinates=np.asarray([target]),
                        context_activations={0: context},
                    )
                )
        results = run_analysis(
            trajectories, ["coordinate"], layers=[0], time_bins=1,
            trajectory_representation="latest",
            split_kinds=["cross_prompt"], cross_group="conversation",
            evaluation_fraction=0.34, split_repeats=2, regressor="ridge",
            ridge_alpha=1e-6, shuffle_repeats=1, seed=3, include_context=True,
        )
        context_cell = next(r for r in results if r["input"] == "context")
        cot_cell = next(r for r in results if r["input"] == "cot")
        self.assertEqual(context_cell["checkpoint_fraction"], 0.0)
        self.assertGreater(
            context_cell["metrics"]["probe"]["assistant_axis"]["r2"]["mean"], 0.9
        )
        # Noise CoT cannot beat the predictive context.
        self.assertLess(
            cot_cell["metrics"]["probe"]["assistant_axis"]["r2"]["mean"],
            context_cell["metrics"]["probe"]["assistant_axis"]["r2"]["mean"],
        )

    def test_fixed_cohort_matches_context_cell_cohort(self):
        from analyze_persona_drift_probes import RolloutTrajectory, run_analysis

        rng = np.random.default_rng(5)
        trajectories = []
        for prompt_index in range(4):
            for rollout_index in range(3):
                late = rollout_index == 0
                progress = np.asarray([0.7, 1.0]) if late else np.asarray([0.3, 1.0])
                target = float(prompt_index)
                acts = rng.normal(size=(2, 3))
                trajectories.append(
                    RolloutTrajectory(
                        prompt_id=f"p{prompt_index}",
                        conversation_id=f"p{prompt_index}",
                        rollout_id=f"p{prompt_index}r{rollout_index}",
                        layer_activations={0: acts},
                        progress=progress,
                        assistant_axis=target,
                        persona_coordinates=np.asarray([target]),
                        context_activations={0: rng.normal(size=3)},
                    )
                )
        results = run_analysis(
            trajectories, ["coordinate"], layers=[0], time_bins=2,
            trajectory_representation="latest",
            split_kinds=["cross_prompt"], cross_group="prompt",
            evaluation_fraction=0.25, split_repeats=1, regressor="ridge",
            ridge_alpha=1.0, shuffle_repeats=1, seed=0,
            cohort="fixed", include_context=True,
        )
        excluded = {
            (cell["input"], cell["time_bin"]): cell["split_diagnostics"][0][
                "n_excluded_no_boundary"
            ]
            for cell in results
        }
        # Context and every CoT bin share the fixed cohort (4 late rollouts out).
        self.assertEqual(set(excluded.values()), {4})

    def test_fixed_cohort_restricts_all_bins(self):
        from analyze_persona_drift_probes import RolloutTrajectory, run_analysis

        rng = np.random.default_rng(2)
        trajectories = []
        for prompt_index in range(4):
            for rollout_index in range(3):
                # Half the rollouts have a late first boundary.
                late = rollout_index == 0
                progress = np.asarray([0.7, 1.0]) if late else np.asarray([0.3, 1.0])
                target = float(prompt_index)
                acts = np.stack([
                    np.asarray([target, rng.normal()]),
                    np.asarray([target, rng.normal()]),
                ])
                trajectories.append(
                    RolloutTrajectory(
                        prompt_id=f"p{prompt_index}",
                        conversation_id=f"p{prompt_index}",
                        rollout_id=f"p{prompt_index}r{rollout_index}",
                        layer_activations={0: acts},
                        progress=progress,
                        assistant_axis=target,
                        persona_coordinates=np.asarray([target]),
                    )
                )
        results = run_analysis(
            trajectories, ["coordinate"], layers=[0], time_bins=2,
            trajectory_representation="latest",
            split_kinds=["cross_prompt"], cross_group="prompt",
            evaluation_fraction=0.25, split_repeats=1, regressor="ridge",
            ridge_alpha=1.0, shuffle_repeats=1, seed=0, cohort="fixed",
        )
        for cell in results:
            for diag in cell["split_diagnostics"]:
                # The late-boundary rollouts are excluded from EVERY bin.
                self.assertEqual(diag["n_excluded_no_boundary"], 4)

    def test_ridge_alpha_grid_records_chosen_alpha(self):
        from analyze_persona_drift_probes import RolloutTrajectory, run_analysis

        rng = np.random.default_rng(4)
        trajectories = []
        for prompt_index in range(6):
            for rollout_index in range(2):
                target = float(prompt_index)
                acts = np.asarray([[target, rng.normal(), 1.0]])
                trajectories.append(
                    RolloutTrajectory(
                        prompt_id=f"p{prompt_index}",
                        conversation_id=f"c{prompt_index % 3}",
                        rollout_id=f"p{prompt_index}r{rollout_index}",
                        layer_activations={0: acts},
                        progress=np.asarray([1.0]),
                        assistant_axis=target,
                        persona_coordinates=np.asarray([target]),
                    )
                )
        results = run_analysis(
            trajectories, ["coordinate"], layers=[0], time_bins=1,
            trajectory_representation="latest",
            split_kinds=["cross_prompt"], cross_group="conversation",
            evaluation_fraction=0.34, split_repeats=1, regressor="ridge",
            ridge_alpha=1.0, ridge_alphas=[1e-6, 1.0, 1e6],
            shuffle_repeats=1, seed=0,
        )
        chosen = results[0]["split_diagnostics"][0]["ridge_alpha"]
        self.assertIn(chosen, (1e-6, 1.0, 1e6))
        self.assertLess(chosen, 1e6)  # strong signal -> small alpha wins

    def test_summary_survives_cells_without_fits(self):
        # Regression: empty early-bin cells must not crash the CLI summary.
        import tempfile
        from pathlib import Path
        from analyze_persona_drift_probes import main as analyze_main

        rng = np.random.default_rng(0)
        prompts = []
        for p in range(4):
            prompts.append({
                "conversation_id": f"c{p}", "turn_index": 0,
                "rollouts": [{
                    "rollout_index": r,
                    "cot_activations": torch.tensor(
                        rng.normal(size=(2, 1, 4)).astype(np.float32)
                    ),
                    "cot_progress": torch.tensor([0.6, 1.0]),
                    "assistant_axis_score": float(p + 0.1 * r),
                    "persona_coordinates": torch.tensor([float(p)]),
                } for r in range(3)],
            })
        payload = {"format_version": 1, "layers": torch.tensor([0]),
                   "persona_coordinate_names": ["x"], "prompts": prompts}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.pt"
            torch.save(payload, path)
            exit_code = analyze_main([
                "--activations", str(path), "--layers", "0", "--time-bins", "2",
                "--splits", "cross_prompt", "--cross-group", "conversation",
            ])
        self.assertEqual(exit_code, 0)

    def test_cross_shuffle_never_returns_identity(self):
        from analyze_persona_drift_probes import _shuffle_rows

        groups = np.asarray(["a", "a", "b"], dtype=object)
        targets = np.asarray([[1.0], [2.0], [9.0]])
        for seed in range(30):  # identity would appear ~50% of the time
            rng = np.random.default_rng(seed)
            shuffled = _shuffle_rows(targets, groups, rng, within_group=False)
            self.assertFalse(
                np.array_equal(shuffled, targets), f"identity at seed {seed}"
            )

    def test_alpha_selection_invariant_to_target_rescaling(self):
        from analyze_persona_drift_probes import _select_ridge_alpha

        rng = np.random.default_rng(7)
        n, d = 60, 8
        X = rng.normal(size=(n, d))
        groups = np.asarray([f"g{i % 3}" for i in range(n)], dtype=object)
        # Target 0: strong linear signal; target 1: pure noise.
        y = np.column_stack([X[:, 0] + 0.05 * rng.normal(size=n), rng.normal(size=n)])
        alphas = [1e-3, 1.0, 1e3]
        base = _select_ridge_alpha(X, y, groups, alphas, seed=0)
        scaled = y.copy()
        scaled[:, 1] *= 1000.0  # inflate the noise target's scale
        rescaled = _select_ridge_alpha(X, scaled, groups, alphas, seed=0)
        self.assertEqual(base, rescaled)

    def test_cross_shuffle_is_a_derangement(self):
        from analyze_persona_drift_probes import _shuffle_rows

        groups = np.asarray(["a", "b", "b", "c"], dtype=object)
        targets = np.asarray([[1.0], [10.0], [20.0], [100.0]])
        for seed in range(25):
            rng = np.random.default_rng(seed)
            shuffled = _shuffle_rows(targets, groups, rng, within_group=False)
            # No group may keep its own block (fixed points rejected).
            self.assertNotEqual(shuffled[0, 0], 1.0, f"seed {seed}: a kept its block")
            self.assertNotEqual(
                set(shuffled[1:3, 0]), {10.0, 20.0}, f"seed {seed}: b kept its block"
            )
            self.assertNotEqual(shuffled[3, 0], 100.0, f"seed {seed}: c kept its block")

    def test_logo_cross_splits_hold_out_every_conversation_once(self):
        from analyze_persona_drift_probes import RolloutTrajectory, run_analysis

        rng = np.random.default_rng(9)
        trajectories = []
        for conversation in range(4):
            for prompt in range(2):
                for rollout in range(2):
                    target = float(conversation)
                    trajectories.append(
                        RolloutTrajectory(
                            prompt_id=f"c{conversation}p{prompt}",
                            conversation_id=f"c{conversation}",
                            rollout_id=f"c{conversation}p{prompt}r{rollout}",
                            layer_activations={0: rng.normal(size=(1, 3))},
                            progress=np.asarray([1.0]),
                            assistant_axis=target,
                            persona_coordinates=np.asarray([target]),
                        )
                    )
        results = run_analysis(
            trajectories, ["coordinate"], layers=[0], time_bins=1,
            trajectory_representation="latest",
            split_kinds=["cross_prompt"], cross_group="conversation",
            evaluation_fraction=0.2, split_repeats=5, regressor="ridge",
            ridge_alpha=1.0, shuffle_repeats=1, seed=0, cross_splits="logo",
        )
        cell = results[0]
        # One deterministic split per conversation, not split_repeats draws.
        self.assertEqual(cell["n_probe_fits"], 4)
        self.assertEqual(
            [diag["n_evaluation"] for diag in cell["split_diagnostics"]], [4, 4, 4, 4]
        )

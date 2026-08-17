import json
import tempfile
import unittest
from pathlib import Path

try:
    from src.custom_datasets.persona_drift import PersonaDriftDataset
    IMPORT_ERROR = None
except ImportError as error:  # pragma: no cover - requires full environment
    PersonaDriftDataset = None
    IMPORT_ERROR = error


def _upstream_style_record(domain: str, n_user_turns: int, conversation_id=None) -> dict:
    conversation = []
    for turn in range(n_user_turns):
        conversation.append({"role": "user", "content": f"{domain} user message {turn}"})
        conversation.append(
            {"role": "assistant", "content": f"{domain} assistant reply {turn}"}
        )
    record = {
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "auditor_model": "anthropic/claude-sonnet-4.5",
        "domain": domain,
        "persona_id": 0,
        "persona": f"A {domain} persona",
        "topic_id": 5,
        "topic": f"A {domain} topic",
        "turns": 2 * n_user_turns,
        "conversation": conversation,
    }
    if conversation_id is not None:
        record["conversation_id"] = conversation_id
    return record


@unittest.skipIf(PersonaDriftDataset is None, f"missing dependencies: {IMPORT_ERROR}")
class PersonaDriftDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        for domain, n_turns, conversation_id in (
            ("coding", 3, "coding_p0_t5_5"),
            ("therapy", 2, None),
        ):
            with (self.data_dir / f"{domain}.json").open("w") as handle:
                json.dump(_upstream_style_record(domain, n_turns, conversation_id), handle)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_every_user_turn_becomes_a_prefix_example(self):
        dataset = PersonaDriftDataset("persona_drift", data_path=self.data_dir)
        examples = list(dataset)
        self.assertEqual(len(examples), 5)  # 3 coding + 2 therapy user turns

        coding = sorted(
            (example for example in examples if example["domain"] == "coding"),
            key=lambda example: example["turn_index"],
        )
        # Prefix through user turn t has 2t+1 messages and ends with a user turn.
        for turn_index, example in enumerate(coding):
            self.assertEqual(example["turn_index"], turn_index)
            self.assertEqual(len(example["messages"]), 2 * turn_index + 1)
            self.assertEqual(example["messages"][-1]["role"], "user")
            self.assertEqual(example["conversation_id"], "coding_p0_t5_5")
            # Auditor-side persona metadata is kept for analysis only.
            self.assertEqual(example["persona"], "A coding persona")
            self.assertNotIn("persona", str(example["messages"]))

    def test_domain_filtering_by_dataset_name(self):
        dataset = PersonaDriftDataset("persona_drift_therapy", data_path=self.data_dir)
        examples = list(dataset)
        self.assertEqual(len(examples), 2)
        self.assertTrue(all(example["domain"] == "therapy" for example in examples))
        # A conversation_id is derived when the record has none.
        self.assertTrue(examples[0]["conversation_id"])

    def test_no_behavior_labels(self):
        dataset = PersonaDriftDataset("persona_drift", data_path=self.data_dir)
        self.assertFalse(dataset.supports_behavior_scoring)
        self.assertIsNone(dataset.detect_behavior(next(iter(dataset)), "reply"))


if __name__ == "__main__":
    unittest.main()

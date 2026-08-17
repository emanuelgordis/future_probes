"""Assistant Axis persona-drift conversation prefixes.

The source transcripts contain auditor-side persona/topic metadata.  That metadata is
kept for analysis, but is deliberately never added to the model-facing messages: the
drift should come from the conversation itself rather than an explicit role prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import datasets
import torch
import transformers

from src.custom_datasets.custom_dataset import CustomDataset


DEFAULT_DATA_PATHS = (
    Path("data/persona_drift"),
    Path("data/assistant_axis/transcripts/persona_drift"),
    Path("assistant-axis/transcripts/persona_drift"),
    Path("../assistant-axis/transcripts/persona_drift"),
)

METADATA_KEYS = (
    "conversation_id",
    "domain",
    "turn_index",
    "message_index",
    "persona_id",
    "persona",
    "topic_id",
    "topic",
    "source_file",
)


class PersonaDriftDataset(CustomDataset):
    """Multi-turn prefixes from Assistant Axis persona-drift transcripts.

    Every user turn becomes one example whose ``messages`` are the full conversation
    prefix through that turn.  Existing assistant turns therefore provide the natural
    drift-inducing history, while the assistant response immediately following the
    prefix is excluded from model input. ``turn_index`` is the zero-based user-turn
    ordinal and ``message_index`` is the zero-based index in the source conversation.
    """

    supports_behavior_scoring = False
    requires_long_context = True
    output_metadata_keys = METADATA_KEYS

    def __init__(
        self,
        dataset_name: str,
        subset: int | None = None,
        seed: int | None = None,
        disable_reasoning: bool = False,
        data_path: str | Path | None = None,
    ):
        self.dataset_name = dataset_name
        self.disable_reasoning = disable_reasoning
        self.subset = subset
        self.seed = seed

        requested_domain = self._domain_from_dataset_name(dataset_name)
        source_files = self._find_source_files(data_path)
        examples: list[dict[str, Any]] = []

        for source_file in source_files:
            for record_index, record in enumerate(self._load_records(source_file)):
                domain = str(record.get("domain") or source_file.stem)
                if requested_domain is not None and domain != requested_domain:
                    continue
                examples.extend(
                    self._conversation_prefixes(record, source_file, record_index)
                )

        if not examples:
            domain_suffix = (
                f" for domain {requested_domain!r}" if requested_domain else ""
            )
            raise ValueError(
                f"No persona-drift conversation prefixes found{domain_suffix} in "
                f"{', '.join(str(path) for path in source_files)}."
            )

        self.data = datasets.Dataset.from_list(examples)
        self.data = self.take_subset_of_data(self.data, self.subset, self.seed)

    @staticmethod
    def _domain_from_dataset_name(dataset_name: str) -> str | None:
        if dataset_name == "persona_drift":
            return None
        prefix = "persona_drift_"
        if not dataset_name.startswith(prefix) or not dataset_name[len(prefix) :]:
            raise ValueError(
                "Persona-drift dataset names must be 'persona_drift' or "
                "'persona_drift_<domain>' (for example, 'persona_drift_coding')."
            )
        return dataset_name[len(prefix) :]

    @staticmethod
    def _find_source_files(data_path: str | Path | None) -> list[Path]:
        if data_path is None:
            source_path = next(
                (path for path in DEFAULT_DATA_PATHS if path.exists()), None
            )
            if source_path is None:
                searched = "\n  - ".join(str(path) for path in DEFAULT_DATA_PATHS)
                raise FileNotFoundError(
                    "Assistant Axis persona-drift transcripts were not found. "
                    "Pass --dataset_path pointing to a transcript JSON file or directory, "
                    "or place the upstream JSON files in one of:\n  - "
                    f"{searched}"
                )
        else:
            source_path = Path(data_path).expanduser()

        if not source_path.exists():
            raise FileNotFoundError(
                f"Persona-drift dataset path does not exist: {source_path}"
            )

        # Also accept the root of a checkout of safety-research/assistant-axis.
        if source_path.is_dir():
            upstream_subdir = source_path / "transcripts" / "persona_drift"
            transcripts_subdir = source_path / "persona_drift"
            if upstream_subdir.is_dir():
                source_path = upstream_subdir
            elif source_path.name == "transcripts" and transcripts_subdir.is_dir():
                source_path = transcripts_subdir

        if source_path.is_file():
            if source_path.suffix.lower() not in {".json", ".jsonl"}:
                raise ValueError(
                    f"Persona-drift source must be JSON or JSONL: {source_path}"
                )
            return [source_path]

        source_files = sorted(
            path
            for path in source_path.iterdir()
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
        )
        if not source_files:
            raise FileNotFoundError(
                f"No JSON or JSONL transcript files found in {source_path}"
            )
        return source_files

    @staticmethod
    def _load_records(source_file: Path) -> list[dict[str, Any]]:
        if source_file.suffix.lower() == ".jsonl":
            with source_file.open("r", encoding="utf-8") as handle:
                loaded: Any = [json.loads(line) for line in handle if line.strip()]
        else:
            with source_file.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)

        if isinstance(loaded, dict) and "conversation" in loaded:
            records = [loaded]
        elif isinstance(loaded, dict) and isinstance(loaded.get("conversations"), list):
            records = loaded["conversations"]
        elif isinstance(loaded, list):
            records = loaded
        else:
            raise ValueError(
                f"Expected a transcript object or list of transcript objects in "
                f"{source_file}"
            )

        if not all(isinstance(record, dict) for record in records):
            raise ValueError(f"All transcript records must be objects in {source_file}")
        return records

    @staticmethod
    def _conversation_prefixes(
        record: dict[str, Any],
        source_file: Path,
        record_index: int,
    ) -> list[dict[str, Any]]:
        conversation = record.get("conversation")
        if not isinstance(conversation, list) or not conversation:
            raise ValueError(
                f"Transcript record {record_index} in {source_file} has no conversation"
            )

        messages: list[dict[str, str]] = []
        for message_index, message in enumerate(conversation):
            if not isinstance(message, dict):
                raise ValueError(
                    f"Message {message_index} in {source_file} must be an object"
                )
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError(
                    f"Message {message_index} in {source_file} must have string "
                    "'role' and 'content' fields"
                )
            messages.append({"role": role, "content": content})

        conversation_id = record.get("conversation_id")
        if conversation_id is None:
            domain = str(record.get("domain") or source_file.stem)
            persona_id = record.get("persona_id", "unknown")
            topic_id = record.get("topic_id", "unknown")
            conversation_id = (
                f"{domain}_p{persona_id}_t{topic_id}_{source_file.stem}_r{record_index}"
            )
        conversation_id = str(conversation_id)

        shared_metadata = {
            key: record[key]
            for key in ("persona_id", "persona", "topic_id", "topic")
            if key in record
        }
        shared_metadata.update(
            {
                "conversation_id": conversation_id,
                "domain": str(record.get("domain") or source_file.stem),
                "source_file": str(source_file),
            }
        )

        examples = []
        user_turn_index = 0
        for message_index, message in enumerate(messages):
            if message["role"] != "user":
                continue

            example = {
                "messages": messages[: message_index + 1],
                **shared_metadata,
                "turn_index": user_turn_index,
                "message_index": message_index,
            }
            examples.append(example)
            user_turn_index += 1

        return examples

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def get_model_input_from_an_example(
        self, example: dict, tokenizer: transformers.AutoTokenizer
    ) -> tuple[torch.Tensor, str]:
        messages = example["messages"]
        template_kwargs = {
            "add_generation_prompt": True,
            "enable_thinking": not self.disable_reasoning,
        }
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", **template_kwargs
        )
        input_as_string = tokenizer.apply_chat_template(
            messages, tokenize=False, **template_kwargs
        )
        return input_ids, input_as_string

    @property
    def behavior_name(self) -> str:
        return "No behavior label; final answers are scored in persona space downstream."

    def detect_behavior(self, example: dict, response: str) -> None:
        """Persona-drift rollouts intentionally have no binary behavior target."""

        return None

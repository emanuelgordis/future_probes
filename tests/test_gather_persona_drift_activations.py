import unittest

import torch

from gather_persona_drift_activations import (
    CharSpan,
    RolloutSkipped,
    TokenTextAligner,
    build_rollout_record,
    compute_rollout_geometry,
    parse_thinking_and_answer_spans,
    resolve_layers,
    sentence_end_offsets,
    trim_trailing_special_tokens,
)


class FakeTokenizer:
    """Token ids are indices into a fixed piece table; decode concatenates."""

    def __init__(self, pieces, special_ids=(), support_offsets=False):
        self.pieces = list(pieces)
        self.all_special_ids = list(special_ids)
        self.support_offsets = support_offsets

    def decode(self, token_ids, **kwargs):
        return "".join(self.pieces[token_id] for token_id in token_ids)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        if not self.support_offsets:
            raise TypeError("offset mapping unsupported")
        token_ids, offsets, cursor = [], [], 0
        by_length = sorted(enumerate(self.pieces), key=lambda item: -len(item[1]))
        while cursor < len(text):
            for token_id, piece in by_length:
                if piece and text.startswith(piece, cursor):
                    token_ids.append(token_id)
                    offsets.append((cursor, cursor + len(piece)))
                    cursor += len(piece)
                    break
            else:
                raise ValueError(f"Cannot tokenize text at position {cursor}")
        return {"input_ids": token_ids, "offset_mapping": offsets}


def _simple_sentence_spans(segment):
    """Sentences end at '.' or '!' followed by whitespace or end of segment."""

    spans, start = [], 0
    for index, char in enumerate(segment):
        at_end = index == len(segment) - 1
        if char in ".!" and (at_end or segment[index + 1].isspace()):
            spans.append((start, index + 1))
            start = index + 1
            while start < len(segment) and segment[start].isspace():
                start += 1
    if start < len(segment):
        spans.append((start, len(segment)))
    return spans


PIECES = [
    "<think>",        # 0
    "\n",             # 1
    "I think",        # 2
    " here.",         # 3
    " Second",        # 4
    " thought!",      # 5
    "\n</think>",     # 6
    "\n\n",           # 7
    "Final",          # 8
    " answer.",       # 9
    "<|im_end|>",     # 10 (special)
]
ROLLOUT_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class ParseSpansTest(unittest.TestCase):
    def test_parses_thinking_and_answer(self):
        text = "<think>\nI think here. Second thought!\n</think>\n\nFinal answer."
        thinking, answer = parse_thinking_and_answer_spans(text)
        self.assertEqual(text[thinking.start : thinking.end], "I think here. Second thought!")
        self.assertEqual(text[answer.start : answer.end], "Final answer.")

    def test_handles_template_injected_think_opener(self):
        # When the chat template already emitted <think>, the rollout starts
        # with bare thinking content.
        text = "Some thinking.\n</think>\n\nAnswer."
        thinking, answer = parse_thinking_and_answer_spans(text)
        self.assertEqual(text[thinking.start : thinking.end], "Some thinking.")
        self.assertEqual(text[answer.start : answer.end], "Answer.")

    def test_skips_unclosed_thinking(self):
        with self.assertRaisesRegex(RolloutSkipped, "never closed"):
            parse_thinking_and_answer_spans("<think>\nStill thinking forever")

    def test_skips_empty_answer_and_empty_thinking(self):
        with self.assertRaisesRegex(RolloutSkipped, "empty final answer"):
            parse_thinking_and_answer_spans("<think>\nThinking.\n</think>\n\n  ")
        with self.assertRaisesRegex(RolloutSkipped, "empty thinking"):
            parse_thinking_and_answer_spans("<think>\n \n</think>\n\nAnswer.")


class SentenceOffsetsTest(unittest.TestCase):
    def test_offsets_are_absolute_and_end_at_span_end(self):
        text = "<think>\nI think here. Second thought!\n</think>\n\nFinal answer."
        thinking, _ = parse_thinking_and_answer_spans(text)
        offsets = sentence_end_offsets(text, thinking, _simple_sentence_spans)
        self.assertEqual(len(offsets), 2)
        self.assertEqual(text[thinking.start : offsets[0]], "I think here.")
        self.assertEqual(offsets[-1], thinking.end)

    def test_rejects_empty_sentence_list(self):
        with self.assertRaisesRegex(RolloutSkipped, "no sentences"):
            sentence_end_offsets("abc", CharSpan(0, 3), lambda segment: [])


class TokenTextAlignerTest(unittest.TestCase):
    def _assert_alignment(self, tokenizer):
        text = tokenizer.decode(ROLLOUT_IDS[:-1])  # visible ids exclude <|im_end|>
        aligner = TokenTextAligner(tokenizer, ROLLOUT_IDS[:-1], text)
        # "I think here." ends inside piece 3 (" here.")
        end_of_first_sentence = text.index(" here.") + len(" here.")
        self.assertEqual(aligner.token_index_covering(end_of_first_sentence), 3)
        # First char of text is inside piece 0
        self.assertEqual(aligner.token_index_covering(1), 0)
        # Last char is inside the final piece
        self.assertEqual(aligner.token_index_covering(len(text)), len(ROLLOUT_IDS) - 2)

    def test_fallback_binary_search_path(self):
        self._assert_alignment(FakeTokenizer(PIECES, support_offsets=False))

    def test_fast_offset_mapping_path(self):
        tokenizer = FakeTokenizer(PIECES, support_offsets=True)
        text = tokenizer.decode(ROLLOUT_IDS[:-1])
        aligner = TokenTextAligner(tokenizer, ROLLOUT_IDS[:-1], text)
        self.assertIsNotNone(aligner._token_start_offsets)
        self._assert_alignment(tokenizer)

    def test_mismatched_reencode_falls_back(self):
        # Piece table that re-tokenizes text differently than the given ids.
        pieces = ["ab", "a", "b"]
        tokenizer = FakeTokenizer(pieces, support_offsets=True)
        aligner = TokenTextAligner(tokenizer, [1, 2], "ab")  # re-encode gives [0]
        self.assertIsNone(aligner._token_start_offsets)
        self.assertEqual(aligner.token_index_covering(1), 0)
        self.assertEqual(aligner.token_index_covering(2), 1)


class RolloutGeometryTest(unittest.TestCase):
    def test_geometry_boundaries_progress_and_answer_span(self):
        tokenizer = FakeTokenizer(PIECES)
        text = tokenizer.decode(ROLLOUT_IDS[:-1])
        thinking, answer = parse_thinking_and_answer_spans(text)
        offsets = sentence_end_offsets(text, thinking, _simple_sentence_spans)
        aligner = TokenTextAligner(tokenizer, ROLLOUT_IDS[:-1], text)
        geometry = compute_rollout_geometry(aligner, thinking, answer, offsets)

        # Sentence ends fall in pieces " here." (3) and " thought!" (5).
        self.assertEqual(geometry.boundary_token_indices, [3, 5])
        # Thinking tokens are pieces 2..5 -> progress (3-2+1)/4 and (5-2+1)/4.
        self.assertEqual(geometry.cot_progress, [0.5, 1.0])
        # Answer tokens are pieces 8..9.
        self.assertEqual(geometry.answer_token_span, (8, 10))

    def test_merges_boundaries_sharing_a_token(self):
        tokenizer = FakeTokenizer(["<think>", "Hi. Ho!", "\n</think>", "\n\nA."])
        text = tokenizer.decode([0, 1, 2, 3])
        thinking, answer = parse_thinking_and_answer_spans(text)
        offsets = sentence_end_offsets(text, thinking, _simple_sentence_spans)
        self.assertEqual(len(offsets), 2)  # two sentences in one token
        aligner = TokenTextAligner(tokenizer, [0, 1, 2, 3], text)
        geometry = compute_rollout_geometry(aligner, thinking, answer, offsets)
        self.assertEqual(geometry.boundary_token_indices, [1])
        self.assertEqual(geometry.cot_progress, [1.0])


class FakeScorer:
    target_layer = 1

    class _Score:
        def __init__(self, activation):
            self.assistant_axis_score = float(activation.sum())
            self.persona_coordinates = activation[:2].clone()

    def score(self, activation):
        return self._Score(torch.as_tensor(activation, dtype=torch.float32))


class BuildRolloutRecordTest(unittest.TestCase):
    def test_builds_record_with_prompt_offset_positions(self):
        tokenizer = FakeTokenizer(PIECES, special_ids=[10])
        prompt_token_ids = [0, 1, 2]  # arbitrary three-token prompt
        captured = {}

        def extract_fn(**kwargs):
            captured.update(kwargs)
            n_boundaries = len(kwargs["boundary_token_positions"])
            n_layers = len(kwargs["layer_indices"])
            return (
                torch.zeros(n_boundaries, n_layers, 4, dtype=torch.float16),
                torch.arange(4, dtype=torch.float32),
            )

        record = build_rollout_record(
            tokenizer,
            prompt_token_ids,
            ROLLOUT_IDS,  # trailing <|im_end|> must be trimmed
            rollout_index=2,
            layers=[0, 1],
            scorer=FakeScorer(),
            extract_fn=extract_fn,
            sentence_spans_fn=_simple_sentence_spans,
        )

        prompt_length = len(prompt_token_ids)
        self.assertEqual(
            captured["boundary_token_positions"], [prompt_length + 3, prompt_length + 5]
        )
        self.assertEqual(
            captured["mean_span"], (prompt_length + 8, prompt_length + 10)
        )
        self.assertEqual(captured["mean_layer"], FakeScorer.target_layer)
        self.assertEqual(
            captured["input_ids"].tolist(),
            [prompt_token_ids + ROLLOUT_IDS[:-1]],
        )
        self.assertEqual(record["rollout_index"], 2)
        self.assertEqual(record["cot_activations"].shape, (2, 2, 4))
        self.assertEqual(record["cot_progress"].tolist(), [0.5, 1.0])
        self.assertEqual(record["assistant_axis_score"], 6.0)  # sum(0..3)
        self.assertEqual(record["persona_coordinates"].tolist(), [0.0, 1.0])
        self.assertEqual(record["n_answer_tokens"], 2)

    def test_skips_rollout_with_only_special_tokens(self):
        tokenizer = FakeTokenizer(PIECES, special_ids=[10])
        with self.assertRaisesRegex(RolloutSkipped, "no visible tokens"):
            build_rollout_record(
                tokenizer,
                [0],
                [10, 10],
                rollout_index=0,
                layers=[0],
                scorer=FakeScorer(),
                extract_fn=lambda **kwargs: None,
                sentence_spans_fn=_simple_sentence_spans,
            )


class TrimSpecialTokensTest(unittest.TestCase):
    def test_trims_only_trailing_special_tokens(self):
        self.assertEqual(
            trim_trailing_special_tokens([5, 10, 6, 10, 10], {10}), [5, 10, 6]
        )


class ResolveLayersTest(unittest.TestCase):
    def test_auto_includes_axis_layer_and_spans_depth(self):
        layers = resolve_layers("auto", 64, axis_layer=32)
        self.assertIn(32, layers)
        self.assertIn(0, layers)
        self.assertIn(63, layers)
        self.assertEqual(layers, sorted(set(layers)))

    def test_all_and_explicit_lists(self):
        self.assertEqual(resolve_layers("all", 4, 1), [0, 1, 2, 3])
        self.assertEqual(resolve_layers("3,1,3", 8, 0), [1, 3])
        with self.assertRaises(ValueError):
            resolve_layers("64", 64, 32)


if __name__ == "__main__":
    unittest.main()

"""Integration tests against the real Qwen3 tokenizer (skipped when unavailable)."""

import unittest

try:
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    _SKIP_REASON = None
except Exception as error:  # pragma: no cover - depends on local cache
    _TOKENIZER = None
    _SKIP_REASON = f"Qwen3 tokenizer unavailable locally: {error}"

from gather_persona_drift_activations import (
    TokenTextAligner,
    parse_thinking_and_answer_spans,
)


def _simple_sentence_spans(segment):
    spans, start = [], 0
    for index, char in enumerate(segment):
        if char in ".!?" and (index == len(segment) - 1 or segment[index + 1].isspace()):
            spans.append((start, index + 1))
            start = index + 1
            while start < len(segment) and segment[start].isspace():
                start += 1
    if start < len(segment):
        spans.append((start, len(segment)))
    return spans


TRICKY_ROLLOUTS = [
    "<think>\nOkay, the user greets me. Let me respond warmly!\n</think>\n\nHello! How can I help? \U0001F60A",
    "<think>\nHmm — em-dashes… and “curly quotes”. Also naïve café.\n</think>\n\nVoilà — done! \U0001F389\U0001F389",
    "<think>\n数学の問題ですね。まず計算します。答えは42です。\n</think>\n\n答えは 42 です。\U0001F914",
    "<think>\nShort.\n</think>\n\nOk.",
]


@unittest.skipIf(_TOKENIZER is None, _SKIP_REASON)
class RealQwenTokenizerAlignmentTest(unittest.TestCase):
    def test_fast_path_and_fallback_agree_on_tricky_unicode(self):
        """The binary-search fallback must match validated offset mapping,
        including multibyte characters (emoji, CJK) split across tokens."""

        for text in TRICKY_ROLLOUTS:
            ids = _TOKENIZER(text, add_special_tokens=False)["input_ids"]
            canonical = _TOKENIZER.decode(ids, clean_up_tokenization_spaces=False)
            fast = TokenTextAligner(_TOKENIZER, ids, canonical)
            self.assertIsNotNone(fast._token_start_offsets, text)
            fallback = TokenTextAligner(_TOKENIZER, ids, canonical)
            fallback._token_start_offsets = None  # force prefix-decode path
            probe_points = sorted(
                {1, len(canonical)} | {i for i in range(1, len(canonical) + 1, 3)}
            )
            for char_end in probe_points:
                self.assertEqual(
                    fast.token_index_covering(char_end),
                    fallback.token_index_covering(char_end),
                    f"mismatch at char {char_end} of {text!r}",
                )

    def test_spans_and_boundaries_on_real_tokenization(self):
        from gather_persona_drift_activations import (
            compute_rollout_geometry,
            sentence_end_offsets,
        )

        text = TRICKY_ROLLOUTS[0]
        ids = _TOKENIZER(text, add_special_tokens=False)["input_ids"]
        canonical = _TOKENIZER.decode(ids, clean_up_tokenization_spaces=False)
        thinking, answer = parse_thinking_and_answer_spans(canonical)
        offsets = sentence_end_offsets(canonical, thinking, _simple_sentence_spans)
        aligner = TokenTextAligner(_TOKENIZER, ids, canonical)
        geometry = compute_rollout_geometry(aligner, thinking, answer, offsets)

        self.assertEqual(len(geometry.boundary_token_indices), 2)
        self.assertEqual(geometry.cot_progress[-1], 1.0)
        answer_text = _TOKENIZER.decode(
            ids[geometry.answer_token_span[0] : geometry.answer_token_span[1]]
        )
        # The final emoji token must be inside the answer span.
        self.assertIn("\U0001F60A", answer_text)
        self.assertIn("Hello", answer_text)


if __name__ == "__main__":
    unittest.main()

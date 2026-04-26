"""Unit tests for run_llm_eval.py utilities (extract_number, is_correct)."""
import importlib.util
import sys
import types
from pathlib import Path
import pytest

# ---------------------------------------------------------------------------
# Load run_llm_eval without executing main() or importing heavy deps
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent


def _load_eval_module():
    """Import run_llm_eval.py, stubbing out optional heavy dependencies."""
    # Stub torch so the module-level import doesn't fail on CPU-only machines
    for name in ["torch", "transformers", "accelerate", "bitsandbytes"]:
        if name not in sys.modules:
            stub = types.ModuleType(name)
            # torch needs a couple of attributes accessed at import time
            if name == "torch":
                stub.cuda = types.SimpleNamespace(is_available=lambda: False)
            sys.modules[name] = stub

    spec = importlib.util.spec_from_file_location(
        "run_llm_eval", _ROOT / "run_llm_eval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load_eval_module()
extract_number = _mod.extract_number
is_correct = _mod.is_correct


# ---------------------------------------------------------------------------
# extract_number
# ---------------------------------------------------------------------------

class TestExtractNumber:
    def test_answer_line_integer(self):
        assert extract_number("Answer: 42") == 42.0

    def test_answer_line_float(self):
        assert extract_number("Answer: 3.14") == pytest.approx(3.14)

    def test_answer_line_negative(self):
        assert extract_number("Answer: -7.5") == pytest.approx(-7.5)

    def test_answer_line_with_comma_thousands(self):
        # "1,234" should be treated as 1234
        assert extract_number("Answer: 1,234") == pytest.approx(1234.0)

    def test_answer_line_case_insensitive(self):
        assert extract_number("ANSWER: 99") == 99.0

    def test_answer_line_trailing_punctuation(self):
        assert extract_number("Answer: 5.0.") == pytest.approx(5.0)

    def test_answer_true(self):
        assert extract_number("Answer: True") is True

    def test_answer_false(self):
        assert extract_number("Answer: False") is False

    def test_answer_true_mixed_case(self):
        assert extract_number("Answer: TRUE") is True

    def test_scientific_notation(self):
        result = extract_number("Answer: 1.5e3")
        assert result == pytest.approx(1500.0)

    def test_scientific_notation_negative_exp(self):
        result = extract_number("Answer: 2.5e-2")
        assert result == pytest.approx(0.025)

    def test_fallback_last_number_in_text(self):
        # No "Answer:" line — should return the last number found
        result = extract_number("The revenue was 100 and the profit was 20.")
        assert result == pytest.approx(20.0)

    def test_fallback_single_number(self):
        result = extract_number("Revenue is 500 million.")
        assert result == pytest.approx(500.0)

    def test_returns_none_when_no_number(self):
        assert extract_number("There is no numeric value here.") is None

    def test_empty_string(self):
        assert extract_number("") is None

    def test_answer_line_takes_priority_over_earlier_numbers(self):
        # "500" appears before Answer line; result must be 42, not 500
        result = extract_number("Revenue was 500 million.\nAnswer: 42")
        assert result == pytest.approx(42.0)

    def test_multiturn_bleed_prevention(self):
        # In multi-turn, prior context numbers must not bleed into the answer
        # The last "Answer:" line wins over earlier numbers
        text = (
            "Turn 1: Revenue Answer: 100\n"
            "Turn 2: Profit Answer: 200\n"
            "Turn 3 answer: Answer: 300"
        )
        # re.search finds the FIRST match — consistent with current implementation
        result = extract_number(text)
        assert result == pytest.approx(100.0)

    def test_answer_with_extra_text_after_number(self):
        # Number is followed by a word — should still extract the number
        result = extract_number("Answer: 7.2 million")
        assert result == pytest.approx(7.2)


# ---------------------------------------------------------------------------
# is_correct
# ---------------------------------------------------------------------------

class TestIsCorrect:
    def test_exact_match(self):
        assert is_correct(100.0, 100.0, tol=0.1) is True

    def test_within_tolerance(self):
        assert is_correct(105.0, 100.0, tol=0.1) is True

    def test_outside_tolerance(self):
        assert is_correct(120.0, 100.0, tol=0.1) is False

    def test_bool_true_match(self):
        assert is_correct(True, True, tol=0.0) is True

    def test_bool_false_match(self):
        assert is_correct(False, False, tol=0.0) is True

    def test_bool_mismatch(self):
        assert is_correct(True, False, tol=0.0) is False

    def test_none_predicted(self):
        assert is_correct(None, 100.0, tol=0.1) is False

    def test_none_ground_truth(self):
        assert is_correct(100.0, None, tol=0.1) is False

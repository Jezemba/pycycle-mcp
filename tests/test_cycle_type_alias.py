"""The model name a response reports must be accepted as input.

B35. `create_cycle_model(cycle_type="turbofan")` returns
`{"model_name": "HBTF", ...}` and then REFUSED "HBTF" as a cycle_type. Observed
live 2026-08-12: the caller read the name the tool had just given it, passed it
back, and was rejected twice in a row -- the identical-retry signature.

Same shape as SU2 naming PHYSICAL_PROBLEM in an error and then refusing that
option: a system whose OUTPUT vocabulary its INPUT does not accept.

This is the remedy class that measurably works -- a VALUE to substitute, not a
different tool to call (recovery from a named value went from a mean 8.8 wrong
guesses to 1), so the error names the accepted value rather than only listing
the vocabulary.
"""

from __future__ import annotations

import pytest

from pycycle_mcp.tools.create_model import _resolve_builtin_cycle


def _resolve(name):
    try:
        return _resolve_builtin_cycle(name)[0]
    except ImportError:
        pytest.skip("pycycle not installed in this environment")


class TestTheModelNameRoundTrips:
    def test_hbtf_is_accepted(self):
        assert _resolve("HBTF") == "turbofan"

    def test_case_insensitive(self):
        assert _resolve("hbtf") == "turbofan"

    def test_long_form_accepted(self):
        assert _resolve("high_bypass_turbofan") == "turbofan"

    def test_canonical_names_unchanged(self):
        assert _resolve("turbofan") == "turbofan"
        assert _resolve("turbojet") == "turbojet"


class TestUnknownTypesStillRejected:
    def test_a_genuine_typo_is_an_error(self):
        with pytest.raises((ValueError, ImportError), match="Unsupported cycle type"):
            _resolve_builtin_cycle("turbofanx")

    def test_the_error_explains_the_name_vs_type_distinction(self):
        """The confusion that caused this: model_name is not a cycle_type."""
        try:
            _resolve_builtin_cycle("nonsense")
        except ImportError:
            pytest.skip("pycycle not installed")
        except ValueError as exc:
            assert "model NAME" in str(exc)
            assert "turbofan" in str(exc)

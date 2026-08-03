"""get_cycle_summary must not die on numpy array values.

Found by mining the MAS-Aviary run logs (2026-08-03): `get_cycle_summary`
failed on EVERY call — 6/6 occurrences — with

    ValueError: The truth value of an array with more than one element is
    ambiguous. Use a.any() or a.all()

and returned empty options/key_inputs/key_outputs, so the propulsion
discipline's summary tool was completely non-functional.

Cause was one line:

    "current_value": meta_entry.get("val") or meta_entry.get("value")

OpenMDAO values are numpy arrays, and `X or Y` evaluates bool(X), which raises
for any array with more than one element.
"""

import json

import numpy as np
import pytest

from pycycle_mcp.utils import jsonable_variable_value, render_variable_entry


class TestMultiElementArrays:
    def test_multi_element_array_does_not_raise(self):
        """THE regression — this is the exact shape that broke the tool."""
        out = jsonable_variable_value({"val": np.array([1.0, 2.0, 3.0])})
        assert out == [1.0, 2.0, 3.0]

    def test_the_old_or_idiom_would_have_raised(self):
        """Pin the reason, so nobody reintroduces `or`."""
        meta = {"val": np.array([1.0, 2.0, 3.0]), "value": None}
        with pytest.raises(ValueError, match="truth value of an array"):
            _ = meta.get("val") or meta.get("value")

    def test_result_is_json_serializable(self):
        out = jsonable_variable_value({"val": np.array([[1.0, 2.0], [3.0, 4.0]])})
        assert json.dumps(out) == "[[1.0, 2.0], [3.0, 4.0]]"


class TestZeroAndFalsyValues:
    def test_zero_array_is_preserved_not_dropped(self):
        """`or` would also have silently swapped a legitimate 0.0 for the
        fallback — a quieter bug than the crash. Single-element arrays do not
        raise on bool(), so this one would have failed SILENTLY."""
        assert jsonable_variable_value({"val": np.array([0.0])}) == [0.0]

    def test_scalar_zero_preserved(self):
        assert jsonable_variable_value({"val": np.float64(0.0)}) == 0.0

    def test_empty_array_preserved(self):
        assert jsonable_variable_value({"val": np.array([])}) == []


class TestScalarTypes:
    @pytest.mark.parametrize(
        "raw,expected,typ",
        [
            (np.float64(2.5), 2.5, float),
            (np.int64(7), 7, int),
            (np.bool_(True), True, bool),
        ],
    )
    def test_numpy_scalars_become_python(self, raw, expected, typ):
        out = jsonable_variable_value({"val": raw})
        assert out == expected and isinstance(out, typ)
        json.dumps(out)  # must not raise

    def test_plain_python_passthrough(self):
        assert jsonable_variable_value({"val": 5900.0}) == 5900.0
        assert jsonable_variable_value({"val": "text"}) == "text"


class TestKeyPreference:
    def test_value_key_wins_when_present(self):
        assert jsonable_variable_value({"value": 1.0, "val": 2.0}) == 1.0

    def test_falls_back_to_val(self):
        assert jsonable_variable_value({"val": 2.0}) == 2.0

    def test_missing_both_is_none(self):
        assert jsonable_variable_value({"units": "lbf"}) is None


class TestRenderVariableEntryStillWorks:
    def test_entry_shape_unchanged(self):
        entry = render_variable_entry("perf.Fn", {"val": np.array([5900.0]), "units": "lbf"}, "outputs")
        assert entry["name"] == "perf.Fn"
        assert entry["io"] == "outputs"
        assert entry["units"] == "lbf"
        assert entry["value"] == [5900.0]
        json.dumps(entry)

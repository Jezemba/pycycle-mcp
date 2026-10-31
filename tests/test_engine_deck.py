"""Avion B3: engine deck builder -- pure helpers (no pyCycle solve)."""

from __future__ import annotations

import pytest

from pycycle_mcp import engine_deck as ed


def _point(m, a, t, fg=1000.0, ram=100.0, wf=500.0):
    return ed.DeckPoint(mach=m, alt_ft=a, throttle=t, gross_thrust_lbf=fg, ram_drag_lbf=ram,
                        fuel_flow_lbh=wf, net_thrust_lbf=fg - ram, t4_degR=2857.0)


def test_format_read_round_trip_sorted_with_header():
    pts = [_point(0.8, 35000.0, 1.0, 9000.0, 3000.0, 3800.0), _point(0.0, 0.0, 1.0, 20300.0, 20.0, 7400.0),
           _point(0.0, 0.0, 0.5, 10000.0, 15.0, 2500.0)]
    text = ed.format_deck(pts, ["made by a test", ""])
    assert text.startswith("# made by a test\n#\n")
    rows = ed.read_deck(text)
    assert [(r["mach_number"], r["altitude"], r["throttle"]) for r in rows] == [
        (0.0, 0.0, 0.5), (0.0, 0.0, 1.0), (0.8, 35000.0, 1.0)]
    assert rows[1]["gross_thrust"] == pytest.approx(20300.0)
    assert rows[2]["fuel_flow"] == pytest.approx(3800.0)
    assert ed.sls_net_thrust(rows) == pytest.approx(20280.0)


def test_deck_columns_are_aviary_aliases():
    # Aviary's EngineDeck maps these names (whitespace -> _, lower case) to its variables.
    names = [c.split(" (")[0].lower() for c in ed.DECK_COLUMNS]
    assert names == ["mach_number", "altitude", "throttle", "gross_thrust", "ram_drag", "fuel_flow"]


def test_cells_with_one_throttle_are_dropped():
    pts = [_point(0.0, 37000.0, 1.0), _point(0.8, 35000.0, 1.0), _point(0.8, 35000.0, 0.5)]
    kept, dropped = ed.usable_points(pts)
    assert {(p.mach, p.alt_ft) for p in kept} == {(0.8, 35000.0)}
    assert dropped == [{"mach": 0.0, "alt_ft": 37000.0, "points": 1}]


def test_sls_none_without_static_row():
    assert ed.sls_net_thrust([{"mach_number": 0.2, "altitude": 0.0, "gross_thrust": 1.0, "ram_drag": 0.0}]) is None


def test_chunk_jobs_cover_grid_once_and_walk_outward():
    machs, alts = (0.0, 0.1, 0.2, 0.3, 0.8, 0.85, 0.9), (0.0, 35000.0)
    jobs = ed.chunk_jobs(machs, alts, design_mach=0.8, chunk=3)
    covered = sorted((a, m) for a, run in jobs for m in run)
    assert covered == sorted((a, m) for a in alts for m in machs)
    for _alt, run in jobs:
        # each run starts at the end nearest the design Mach
        assert abs(run[0] - 0.8) <= abs(run[-1] - 0.8)
    assert len(jobs) == 2 * 3


def test_size_invariant_and_deck_inputs_normalise_thrust():
    inputs = {"Fn_DES": {"value": 12345.0, "units": "lbf"}, "hp_shaft.HPX": {"value": 0.0, "units": "hp"},
              "splitter.BPR": {"value": 8.0, "units": None}}
    assert ed.size_invariant(inputs)
    gen, normalised = ed.deck_inputs(inputs)
    assert normalised and gen["Fn_DES"]["value"] == ed.REFERENCE_FN_DES_LBF
    assert inputs["Fn_DES"]["value"] == 12345.0  # caller's dict untouched
    # two sizes of the same cycle share one deck key
    other = dict(inputs, Fn_DES={"value": 777.0, "units": "lbf"})
    grid = {"mach": [0.8], "altitude_ft": [35000.0], "throttle": [1.0]}
    assert ed.design_key(ed.deck_inputs(other)[0], grid) == ed.design_key(gen, grid)
    # a cycle change does not
    changed = dict(inputs, **{"splitter.BPR": {"value": 9.0, "units": None}})
    assert ed.design_key(ed.deck_inputs(changed)[0], grid) != ed.design_key(gen, grid)


def test_builder_version_is_part_of_the_key(monkeypatch):
    inputs = {"Fn_DES": {"value": 5900.0, "units": "lbf"}}
    grid = {"mach": [0.8], "altitude_ft": [35000.0], "throttle": [1.0]}
    before = ed.design_key(inputs, grid)
    monkeypatch.setattr(ed, "DECK_BUILDER_VERSION", ed.DECK_BUILDER_VERSION + 1)
    assert ed.design_key(inputs, grid) != before


def test_shaft_power_extraction_breaks_size_invariance():
    inputs = {"Fn_DES": {"value": 5000.0, "units": "lbf"}, "hp_shaft.HPX": {"value": 250.0, "units": "hp"}}
    assert not ed.size_invariant(inputs)
    gen, normalised = ed.deck_inputs(inputs)
    assert not normalised and gen["Fn_DES"]["value"] == 5000.0


def test_throttles_must_include_full_power():
    with pytest.raises(ValueError):
        ed.generate_deck({}, throttles=(0.9, 0.5))

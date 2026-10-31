"""Avion B3: generate_engine_deck tool -- MCP surface, cache, refusals (no pyCycle solve)."""

from __future__ import annotations

import json
import os

import anyio

from pycycle_mcp import engine_deck as ed
from pycycle_mcp import fastmcp_server
from pycycle_mcp.tools import deck

INPUTS = {"Fn_DES": {"value": 7000.0, "units": "lbf"}, "hp_shaft.HPX": {"value": 0.0, "units": "hp"},
          "splitter.BPR": {"value": 5.105, "units": None}}
DESIGN = {"Fn_lbf": 5900.0, "Fn_DES": 5900.0, "lpt_PR": 3.0, "TSFC": 0.65, "BPR": 5.105}


def _fake_session(monkeypatch):
    monkeypatch.setattr(deck.session_manager, "get", lambda sid: (object(), {}))
    monkeypatch.setattr(deck.ed, "session_design_inputs", lambda problem: dict(INPUTS))


def _grid():
    return {"mach": sorted(ed.DEFAULT_MACHS), "altitude_ft": sorted(ed.DEFAULT_ALTITUDES_FT),
            "throttle": sorted(ed.DEFAULT_THROTTLES, reverse=True)}


def test_server_exposes_generate_engine_deck(monkeypatch):
    monkeypatch.setattr(fastmcp_server.tools.deck, "generate_engine_deck",
                        lambda payload: {"success": True, "deck_file": "/x/deck_k.deck", "envelope_holes": []})
    server = fastmcp_server.build_server()
    tool = anyio.run(server.get_tool, "generate_engine_deck")
    out = anyio.run(tool.run, {"session_id": "s"}).structured_content
    assert out["success"] is True and out["deck_file"] == "/x/deck_k.deck"


def test_cached_deck_is_returned_without_solving(monkeypatch, tmp_path):
    _fake_session(monkeypatch)
    monkeypatch.setenv("PYCYCLE_DECK_DIR", str(tmp_path))
    monkeypatch.setattr(deck.ed, "solve_design_point",
                        lambda inputs: (_ for _ in ()).throw(AssertionError("must not solve")))
    gen_inputs, _ = ed.deck_inputs(INPUTS)
    key = ed.design_key(gen_inputs, _grid())
    path = tmp_path / f"deck_{key}.deck"
    path.write_text("# deck\n")
    cells = [{"mach": m, "alt_ft": a} for m in _grid()["mach"] for a in _grid()["altitude_ft"]]
    (tmp_path / f"deck_{key}.json").write_text(json.dumps({
        "deck_path": str(path), "design_key": key, "reference_fn_des_lbf": 5900.0, "size_normalised": True,
        "design": DESIGN, "sls_net_thrust_lbf": 20000.0, "n_points": 100, "n_failed": 2,
        "wall_seconds": 1.0, "grid": _grid(), "cells": cells}))
    out = deck.generate_engine_deck({"session_id": "s"})
    assert out["cached"] is True and out["deck_file"] == str(path)
    assert out["reference_fn_des_lbf"] == 5900.0 and out["envelope_holes"] == []


def test_size_only_change_hits_the_same_deck(monkeypatch, tmp_path):
    # Fn_DES is normalised away for a size-invariant cycle: 7,000 and 12,000 lbf share one key.
    a = ed.design_key(ed.deck_inputs(INPUTS)[0], _grid())
    b = ed.design_key(ed.deck_inputs(dict(INPUTS, Fn_DES={"value": 12000.0, "units": "lbf"}))[0], _grid())
    assert a == b


def test_unconverged_design_point_is_refused(monkeypatch, tmp_path):
    _fake_session(monkeypatch)
    monkeypatch.setenv("PYCYCLE_DECK_DIR", str(tmp_path))
    monkeypatch.setattr(deck.ed, "solve_design_point", lambda inputs: dict(DESIGN, Fn_lbf=1.4e24))
    out = deck.generate_engine_deck({"session_id": "s"})
    assert out["error"]["type"] == "DesignNotConverged"
    assert not os.listdir(tmp_path)


def test_missing_session_and_id():
    assert deck.generate_engine_deck({})["error"]["type"] == "ValidationError"
    assert "error" in deck.generate_engine_deck({"session_id": "no-such-session"})


def test_envelope_holes_only_inside_the_mission_envelope():
    grid = {"mach": [0.0, 0.2, 0.8, 0.9], "altitude_ft": [0.0, 35000.0, 41000.0], "throttle": [1.0, 0.5]}
    cells = [{"mach": m, "alt_ft": a} for m in grid["mach"] for a in grid["altitude_ft"]]
    cells.remove({"mach": 0.0, "alt_ft": 41000.0})   # outside: Mach 0
    cells.remove({"mach": 0.8, "alt_ft": 41000.0})   # outside: above 38,000 ft
    cells.remove({"mach": 0.2, "alt_ft": 0.0})       # inside
    assert deck.envelope_holes({"grid": grid, "cells": cells}) == [{"mach": 0.2, "alt_ft": 0.0}]

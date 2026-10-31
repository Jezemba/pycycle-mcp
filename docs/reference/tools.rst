Tool Catalog
============

The current tool catalog is assembled by ``pycycle_mcp.fastmcp_server``.

Available tool groups
---------------------

- Health and lifecycle: ``ping``, ``create_cycle_model``, ``close_cycle_model``,
  ``get_cycle_summary``
- Variables and execution: ``list_variables``, ``set_inputs``, ``get_outputs``,
  ``run_cycle``
- Analysis: ``sweep_inputs``, ``compute_totals``
- Mission engine: ``generate_engine_deck`` -- the session's cycle over Mach x altitude x throttle as an
  Aviary engine deck (off-design pyCycle, cached per cycle; Avion B3)

All tools use Pydantic-backed request validation and return JSON-serializable
payloads suitable for MCP clients.

.. automodule:: pycycle_mcp.fastmcp_server
   :members:

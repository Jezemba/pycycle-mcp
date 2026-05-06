# pycycle-mcp

[![CI](https://github.com/cmudrc/pycycle-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/cmudrc/pycycle-mcp/actions/workflows/ci.yml)
[![Docs](https://github.com/cmudrc/pycycle-mcp/actions/workflows/docs-pages.yml/badge.svg)](https://github.com/cmudrc/pycycle-mcp/actions/workflows/docs-pages.yml)
[![Examples](https://github.com/cmudrc/pycycle-mcp/actions/workflows/examples.yml/badge.svg)](https://github.com/cmudrc/pycycle-mcp/actions/workflows/examples.yml)

`pycycle-mcp` is a lightweight Model Context Protocol server for pyCycle/OpenMDAO
engine-cycle workflows. The repository includes deterministic examples/tests so
local development and CI can validate tooling contracts without requiring a full
runtime installation of pyCycle/OpenMDAO assets.

## Overview

The project currently provides:

- A FastMCP-powered server with stdio and HTTP-compatible transports.
- Tooling for cycle lifecycle, variable inspection/updates, execution, sweeps,
  and total-derivative evaluation.
- Pydantic-backed validation with structured MCP-style error payloads.
- Deterministic examples for repository scaffolding checks and smoke tests.

## Quickstart

Requires Python 3.12+.

```bash
python3 -m venv .venv
source .venv/bin/activate
make dev
make test
make ci
```

Start the server over stdio:

```bash
pycycle-mcp --transport stdio
```

Inspect the non-blocking HTTP transport configuration example:

```bash
PYTHONPATH=src python3 examples/server/http_launch_config.py
```

## Examples

The examples are deterministic and aligned with the current repository
contracts.

- Examples index: [`examples/README.md`](examples/README.md)
- Tool discovery: [`examples/client/tool_discovery.py`](examples/client/tool_discovery.py)
- Session lifecycle: [`examples/cpacs/session_lifecycle.py`](examples/cpacs/session_lifecycle.py)
- Export snapshot: [`examples/cpacs/export_snapshot.py`](examples/cpacs/export_snapshot.py)

## Docs

- Docs source: [`docs/index.rst`](docs/index.rst)
- Published docs (placeholder): <https://cmudrc.github.io/pycycle-mcp/>

Build the docs locally with:

```bash
make docs
```

## Python API Rename

The package import root is now:

- `pycycle_mcp` (new)

Legacy pre-rename import paths and CLI aliases are intentionally
removed.

## Shared-CPACS Integration

This MCP includes a **CPACS adapter** (`src/pycycle_mcp/cpacs_adapter.py`) that
bridges pyCycle to the shared-CPACS aircraft analysis pipeline.

### What it does

The adapter reads engine parameters and aerodynamic drag from CPACS, runs a real
OpenMDAO/pyCycle turbofan cycle analysis, and writes performance results — net
thrust, TSFC, OPR, BPR, fuel flow — into `//mcpResults`.

| Direction | XPath |
|-----------|-------|
| **Reads** | `.//vehicles/engines`, `.//analysisResults/aero/coefficients/CD` |
| **Writes** | `.//vehicles/engines/engine/analysis/mcpResults` (Fn, TSFC, OPR, BPR, fuel flow) |

### Running as part of the pipeline

```bash
python pipeline/shared_cpacs_orchestrator.py D150_v30.xml --mcps tigl su2 pycycle mission
```

See [cmudrc/aircraft-analysis](https://github.com/cmudrc/aircraft-analysis) for
full pipeline documentation, versioning details, and installation instructions.

### Related MCP servers

| MCP | Repository |
|-----|-----------|
| TiGL (geometry) | [cmudrc/tigl-mcp](https://github.com/cmudrc/tigl-mcp) |
| SU2 (CFD aerodynamics) | [cmudrc/su2-mcp](https://github.com/cmudrc/su2-mcp) |
| Mission (trajectory/fuel) | [cmudrc/mission-mcp](https://github.com/cmudrc/mission-mcp) |

## Contributing

Contribution guidelines live in [`CONTRIBUTING.md`](CONTRIBUTING.md).

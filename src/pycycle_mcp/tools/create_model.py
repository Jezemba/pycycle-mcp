"""Tools for creating and summarizing pyCycle/OpenMDAO models."""

from __future__ import annotations

import logging
from collections.abc import Callable

from ..errors import error_response, to_error
from ..session_manager import session_manager
from ..types import CycleProblem
from ..utils import (
    error_on_missing_session,
    load_callable,
    select_interesting_variables,
)

LOGGER = logging.getLogger(__name__)


def _patch_pycycle_numpy2_compat() -> None:
    """Patch pyCycle CEA thermo classes for numpy >= 2.0 compatibility.

    numpy 2.x no longer allows assigning a 1-d array into a scalar slot
    (e.g. ``out[i] = arr`` where ``arr.shape == (1,)``).  pyCycle 4.4
    uses ``inputs['n_moles']`` (shape (1,)) in scalar contexts throughout
    ``PropsRHS.compute`` and ``PropsCalcs.compute / compute_partials``.

    This function monkey-patches both classes so ``n_moles`` is extracted
    as a Python float via ``.item()`` before use.
    """
    try:
        import numpy as np
        from pycycle.thermo.cea.props_rhs import PropsRHS
        from pycycle.thermo.cea.props_calcs import PropsCalcs
    except ImportError:
        return  # pycycle not installed; nothing to patch

    # --- PropsRHS.compute ---
    def _patched_props_rhs_compute(self, inputs, outputs):  # type: ignore[no-untyped-def]
        thermo = self.thermo
        num_element = thermo.num_element
        T = inputs['T']
        n = inputs['n']
        b0 = inputs['composition']

        for i in range(num_element):
            outputs['lhs_TP'][i][:num_element] = np.dot(thermo.aij_prod[i], n)

        outputs['lhs_TP'][num_element, :num_element] = b0
        outputs['lhs_TP'][:num_element, num_element] = b0
        outputs['lhs_TP'][num_element, num_element] = 0

        outputs['rhs_P'][:num_element] = b0
        n_moles = inputs['n_moles']
        outputs['rhs_P'][num_element] = n_moles.item() if hasattr(n_moles, 'item') else n_moles

        self.H0_T = H0_T = thermo.H0(T)
        n_H0 = n * H0_T
        outputs['rhs_T'][:num_element] = np.sum(thermo.aij * n_H0, axis=1)
        outputs['rhs_T'][num_element] = np.sum(n_H0)

    PropsRHS.compute = _patched_props_rhs_compute  # type: ignore[assignment]

    # --- PropsCalcs: rewrite compute and compute_partials to use scalar n_moles ---
    from pycycle.constants import P_REF, R_UNIVERSAL_ENG, R_UNIVERSAL_SI, MIN_VALID_CONCENTRATION

    def _patched_calcs_compute(self, inputs, outputs):  # type: ignore[no-untyped-def]
        thermo = self.options['thermo']
        num_prod = thermo.num_prod
        num_element = thermo.num_element

        T = inputs['T']
        P = inputs['P']
        result_T = inputs['result_T']
        nj = inputs['n'][:num_prod]
        n_moles = inputs['n_moles'].item() if hasattr(inputs['n_moles'], 'item') else inputs['n_moles']

        self.dlnVqdlnP = dlnVqdlnP = -1 + inputs['result_P'][num_element]
        self.dlnVqdlnT = dlnVqdlnT = 1 - result_T[num_element]

        self.Cp0_T = Cp0_T = thermo.Cp0(T)
        Cpf = np.sum(nj * Cp0_T)
        self.H0_T = H0_T = thermo.H0(T)
        self.S0_T = S0_T = thermo.S0(T)
        self.nj_H0 = nj_H0 = nj * H0_T

        Cpe = -np.sum(np.sum(thermo.aij * nj_H0, axis=1) * result_T[:num_element])
        Cpe += np.sum(nj_H0 * H0_T)
        Cpe -= np.sum(nj_H0) * result_T[num_element]

        outputs['h'] = np.sum(nj_H0) * R_UNIVERSAL_ENG * T
        try:
            val = S0_T + np.log(n_moles / nj / (P / P_REF))
        except FloatingPointError:
            P = 1e-5
            val = S0_T + np.log(n_moles / nj / (P / P_REF))

        outputs['S'] = R_UNIVERSAL_ENG * np.sum(nj * val)
        outputs['Cp'] = Cp = (Cpe + Cpf) * R_UNIVERSAL_ENG
        outputs['Cv'] = Cv = Cp + n_moles * R_UNIVERSAL_ENG * dlnVqdlnT ** 2 / dlnVqdlnP
        outputs['gamma'] = -1 * Cp / Cv / dlnVqdlnP
        outputs['rho'] = P / (n_moles * R_UNIVERSAL_SI * T) * 100
        outputs['R'] = R_UNIVERSAL_SI * n_moles

    def _patched_calcs_partials(self, inputs, J):  # type: ignore[no-untyped-def]
        thermo = self.options['thermo']
        num_prod = thermo.num_prod
        num_element = thermo.num_element

        T = inputs['T']
        P = inputs['P']
        nj = inputs['n']
        n_moles = inputs['n_moles'].item() if hasattr(inputs['n_moles'], 'item') else inputs['n_moles']
        result_T = inputs['result_T']
        result_T_last = result_T[num_element]
        result_T_rest = result_T[:num_element]

        dlnVqdlnP = -1 + inputs['result_P'][num_element]
        dlnVqdlnT = 1 - result_T_last

        Cp0_T = thermo.Cp0(T)
        Cpf = np.sum(nj * Cp0_T)
        H0_T = thermo.H0(T)
        S0_T = thermo.S0(T)
        nj_H0 = nj * H0_T

        Cpe = -np.sum(np.sum(thermo.aij * nj_H0, axis=1) * result_T_rest)
        Cpe += np.sum(nj_H0 * H0_T)
        Cpe -= np.sum(nj_H0) * result_T_last

        Cp = (Cpe + Cpf) * R_UNIVERSAL_ENG
        Cv = Cp + n_moles * R_UNIVERSAL_ENG * dlnVqdlnT ** 2 / dlnVqdlnP

        dH0_dT = thermo.H0_applyJ(T, 1.)
        dS0_dT = thermo.S0_applyJ(T, 1.)
        dCp0_dT = thermo.Cp0_applyJ(T, 1.)
        sum_nj_R = n_moles * R_UNIVERSAL_SI

        dCpe_dT = 2 * np.sum(nj * H0_T * dH0_dT)
        dCpe_dT -= np.sum(np.sum(thermo.aij * nj * dH0_dT, axis=1) * result_T_rest)
        dCpe_dT -= np.sum(nj * dH0_dT) * result_T_last

        dCpf_dT = np.sum(nj * dCp0_dT)

        J['h', 'T'] = R_UNIVERSAL_ENG * (np.sum(nj * dH0_dT) * T + np.sum(nj * H0_T))
        J['h', 'n'] = R_UNIVERSAL_ENG * T * H0_T

        J['S', 'n'] = R_UNIVERSAL_ENG * (S0_T + np.log(n_moles) - np.log(P / P_REF) - np.log(nj) - 1)
        _trace = np.where(nj <= MIN_VALID_CONCENTRATION + 1e-20)
        J['S', 'n'][0, _trace] = 0
        J['S', 'T'] = R_UNIVERSAL_ENG * np.sum(nj * dS0_dT)
        J['S', 'P'] = -R_UNIVERSAL_ENG * np.sum(nj / P)
        J['S', 'n_moles'] = R_UNIVERSAL_ENG * np.sum(nj) / n_moles
        J['rho', 'T'] = -P / (sum_nj_R * T ** 2) * 100
        J['rho', 'n_moles'] = -P / (n_moles ** 2 * R_UNIVERSAL_SI * T) * 100
        J['rho', 'P'] = 1 / (sum_nj_R * T) * 100

        dCp_dnj = R_UNIVERSAL_ENG * (Cp0_T + H0_T ** 2)
        for j in range(num_prod):
            for i in range(num_element):
                dCp_dnj[j] -= R_UNIVERSAL_ENG * thermo.aij[i][j] * H0_T[j] * result_T[i]
        dCp_dnj -= R_UNIVERSAL_ENG * H0_T * result_T_last
        J['Cp', 'n'] = dCp_dnj

        dCp_dresultT = np.zeros(num_element + 1)
        dCp_dresultT[:num_element] = -R_UNIVERSAL_ENG * np.sum(thermo.aij * nj_H0, axis=1)
        dCp_dresultT[num_element] = -R_UNIVERSAL_ENG * np.sum(nj_H0)
        J['Cp', 'result_T'] = dCp_dresultT

        dCp_dT = (dCpe_dT + dCpf_dT) * R_UNIVERSAL_ENG
        J['Cp', 'T'] = dCp_dT

        J['Cv', 'n'] = dCp_dnj

        dCv_dnmoles = R_UNIVERSAL_ENG * dlnVqdlnT ** 2 / dlnVqdlnP
        J['Cv', 'n_moles'] = dCv_dnmoles
        J['Cv', 'T'] = dCp_dT

        dCv_dresultP = np.zeros((1, num_element + 1))
        dCv_dresultP[0, -1] = -R_UNIVERSAL_ENG * n_moles * (dlnVqdlnT / dlnVqdlnP) ** 2
        J['Cv', 'result_P'] = dCv_dresultP

        J['Cv', 'result_T'] = dCp_dresultT
        J['Cv', 'result_T'][0, -1] -= n_moles * R_UNIVERSAL_ENG / dlnVqdlnP * (2 * dlnVqdlnT)
        dCv_dresultT_last = J['Cv', 'result_T'][0, -1]

        J['gamma', 'n'] = dCp_dnj * (Cp / Cv - 1) / (dlnVqdlnP * Cv)
        J['gamma', 'n_moles'] = Cp / dlnVqdlnP / Cv ** 2 * dCv_dnmoles
        J['gamma', 'T'] = dCp_dT / dlnVqdlnP / Cv * (Cp / Cv - 1)

        dgamma_dresultT = np.zeros((1, num_element + 1))
        dgamma_dresultT[0, :num_element] = 1 / Cv / dlnVqdlnP * dCp_dresultT[:num_element] * (Cp / Cv - 1)
        dgamma_dresultT[0, -1] = (-dCp_dresultT[-1] / Cv + Cp / Cv ** 2 * dCv_dresultT_last) / dlnVqdlnP
        J['gamma', 'result_T'] = dgamma_dresultT

        gamma_dresultP = np.zeros((1, num_element + 1))
        gamma_dresultP[0, num_element] = Cp / Cv / dlnVqdlnP * (dCv_dresultP[0, -1] / Cv + 1 / dlnVqdlnP)
        J['gamma', 'result_P'] = gamma_dresultP

    PropsCalcs.compute = _patched_calcs_compute  # type: ignore[assignment]
    PropsCalcs.compute_partials = _patched_calcs_partials  # type: ignore[assignment]

    LOGGER.debug("Patched PropsRHS and PropsCalcs for numpy 2.x compatibility")


# Apply patch on import
_patch_pycycle_numpy2_compat()

INTERESTING_INPUT_KEYWORDS = ["mach", "alt", "pr", "turbine", "throttle"]
INTERESTING_OUTPUT_KEYWORDS = ["fn", "fnet", "thrust", "tsfc", "power", "eff"]


CycleBuilder = Callable[[], object]


def _resolve_builtin_cycle(cycle_type: str) -> tuple[str, CycleBuilder]:
    """Resolve a built-in cycle type to a callable builder.

    NASA pyCycle provides engine components (Compressor, Turbine, etc.)
    rather than ready-made engine classes. We ship bundled cycle definitions
    that wire those components into complete engine models.
    """

    try:
        import pycycle.api as pyc_api  # noqa: F401 – verify pycycle is installed
    except Exception as exc:
        raise ImportError("pycycle (om-pycycle) is required for built-in cycle types") from exc

    from ..cycles.high_bypass_turbofan import HBTF
    from ..cycles.simple_turbojet import Turbojet

    mapping: dict[str, CycleBuilder] = {
        "turbofan": lambda: HBTF(thermo_method="CEA"),
        "turbojet": lambda: Turbojet(),
    }
    builder = mapping.get(cycle_type)
    if builder is None:
        supported = ", ".join(sorted(mapping.keys()))
        raise ValueError(f"Unsupported cycle type: {cycle_type}. Supported: {supported}")
    return cycle_type, builder


def _build_problem(builder: CycleBuilder, mode: str, options: dict[str, object]) -> tuple[CycleProblem, str]:
    try:
        from openmdao.api import Problem
    except Exception as exc:
        raise ImportError("openmdao is required to build cycle models") from exc

    problem: CycleProblem = Problem()
    model = builder()
    problem.model = model  # type: ignore[assignment]

    problem.setup(check=False)

    # Apply default design-point values for built-in cycles
    _apply_design_defaults(problem, model, mode)

    for key, value in options.items():
        try:
            problem.set_val(key, value)
        except Exception as exc:
            LOGGER.debug("Could not set %s=%s: %s", key, value, exc)

    return problem, type(model).__name__


def _apply_design_defaults(problem: CycleProblem, model: object, mode: str) -> None:
    """Set sensible design-point values so the model can run out of the box."""
    from ..cycles.high_bypass_turbofan import HBTF
    from ..cycles.simple_turbojet import Turbojet

    if isinstance(model, HBTF):
        # Station Mach number defaults (must match MPhbtf.setup)
        problem.set_val("inlet.MN", 0.751)
        problem.set_val("fan.MN", 0.4578)
        problem.set_val("splitter.BPR", 5.105)
        problem.set_val("splitter.MN1", 0.3104)
        problem.set_val("splitter.MN2", 0.4518)
        problem.set_val("duct4.MN", 0.3121)
        problem.set_val("lpc.MN", 0.3059)
        problem.set_val("duct6.MN", 0.3563)
        problem.set_val("hpc.MN", 0.2442)
        problem.set_val("bld3.MN", 0.3000)
        problem.set_val("burner.MN", 0.1025)
        problem.set_val("hpt.MN", 0.3650)
        problem.set_val("duct11.MN", 0.3063)
        problem.set_val("lpt.MN", 0.4127)
        problem.set_val("duct13.MN", 0.4463)
        problem.set_val("byp_bld.MN", 0.4489)
        problem.set_val("duct15.MN", 0.4589)
        problem.set_val("LP_Nmech", 4666.1, units="rpm")
        problem.set_val("HP_Nmech", 14705.7, units="rpm")

        # Component performance
        problem.set_val("fan.PR", 1.685)
        problem.set_val("fan.eff", 0.8948)
        problem.set_val("lpc.PR", 1.935)
        problem.set_val("lpc.eff", 0.9243)
        problem.set_val("hpc.PR", 9.369)
        problem.set_val("hpc.eff", 0.8707)
        problem.set_val("hpt.eff", 0.8888)
        problem.set_val("lpt.eff", 0.8996)
        problem.set_val("fc.alt", 35000.0, units="ft")
        problem.set_val("fc.MN", 0.8)
        problem.set_val("T4_MAX", 2857.0, units="degR")
        problem.set_val("Fn_DES", 5900.0, units="lbf")
        # Initial guesses for Newton solver
        problem["balance.FAR"] = 0.025
        problem["balance.W"] = 100.0
        problem["balance.lpt_PR"] = 4.0
        problem["balance.hpt_PR"] = 3.0
        problem["fc.balance.Pt"] = 5.2
        problem["fc.balance.Tt"] = 440.0

    elif isinstance(model, Turbojet):
        problem.set_val("fc.alt", 0.0, units="ft")
        problem.set_val("fc.MN", 0.000001)
        problem.set_val("balance.Fn_target", 11800.0, units="lbf")
        problem.set_val("balance.T4_target", 2370.0, units="degR")
        problem.set_val("comp.PR", 13.5)
        problem.set_val("comp.eff", 0.83)
        problem.set_val("turb.eff", 0.86)
        problem["balance.FAR"] = 0.0175
        problem["balance.W"] = 168.0
        problem["balance.turb_PR"] = 4.46
        problem["fc.balance.Pt"] = 14.696
        problem["fc.balance.Tt"] = 518.67


def _extract_variables(problem: CycleProblem, target: str) -> list[tuple[str, dict[str, object]]]:
    if target == "inputs":
        items = problem.model.list_inputs(prom_name=True, out_stream=None)
    else:
        items = problem.model.list_outputs(prom_name=True, out_stream=None)
    return [(name, meta) for name, meta in items]


def _summarize_variables(
    problem: CycleProblem,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    inputs = _extract_variables(problem, "inputs")
    outputs = _extract_variables(problem, "outputs")

    interesting_inputs = select_interesting_variables(inputs, INTERESTING_INPUT_KEYWORDS)
    interesting_outputs = select_interesting_variables(outputs, INTERESTING_OUTPUT_KEYWORDS)

    def _render(names: list[str], source: list[tuple[str, dict[str, object]]]) -> list[dict[str, object]]:
        rendered: list[dict[str, object]] = []
        metadata_map = {name: meta for name, meta in source}
        for name in names:
            meta = metadata_map.get(name, {})
            rendered.append({"name": name, "units": meta.get("units"), "desc": meta.get("desc")})
        return rendered

    return _render(interesting_inputs, inputs), _render(interesting_outputs, outputs)


def create_cycle_model(payload: dict[str, object]) -> dict[str, object]:
    """Instantiate a pyCycle/OpenMDAO Problem for a specified engine cycle."""

    cycle_type = payload.get("cycle_type")
    mode = payload.get("mode")
    options: dict[str, object] = payload.get("options", {})  # type: ignore[assignment]
    cycle_module_path = payload.get("cycle_module_path")

    if not cycle_type or not mode:
        return error_response("ValidationError", "cycle_type and mode are required")

    try:
        if cycle_type == "custom":
            if not cycle_module_path:
                raise ValueError("cycle_module_path is required for custom cycles")
            builder = load_callable(str(cycle_module_path))
            model_name = getattr(builder, "__name__", cycle_module_path)
        else:
            model_name, builder = _resolve_builtin_cycle(str(cycle_type))

        problem, resolved_name = _build_problem(builder=builder, mode=str(mode), options=options)
        session_id = session_manager.create_session(problem=problem, meta={"mode": mode, "options": options})
        top_inputs, top_outputs = _summarize_variables(problem)

        return {
            "session_id": session_id,
            "model_name": resolved_name or model_name,
            "top_promoted_inputs": top_inputs,
            "top_promoted_outputs": top_outputs,
        }
    except Exception as exc:  # pragma: no cover - handled in tests via fake errors
        LOGGER.error("Failed to create cycle model: %s", exc)
        return to_error(exc)


def close_cycle_model(payload: dict[str, object]) -> dict[str, object]:
    """Close a pyCycle session and free resources."""

    session_id = payload.get("session_id")
    if not session_id:
        return error_response("ValidationError", "session_id is required")

    try:
        session_manager.close(str(session_id))
        return {"success": True}
    except Exception as exc:  # pragma: no cover
        return to_error(exc)


def get_cycle_summary(payload: dict[str, object]) -> dict[str, object]:
    """Return a succinct summary of the current cycle model."""

    session_id = payload.get("session_id")
    if not session_id:
        return error_response("ValidationError", "session_id is required")

    try:
        problem, meta = session_manager.get(str(session_id))
        mode = meta.get("mode")
        options = meta.get("options", {})
        inputs = _extract_variables(problem, "inputs")
        outputs = _extract_variables(problem, "outputs")

        def _populate(
            entries: list[tuple[str, dict[str, object]]],
        ) -> list[dict[str, object]]:
            rendered: list[dict[str, object]] = []
            for name, meta_entry in entries:
                rendered.append(
                    {
                        "name": name,
                        "units": meta_entry.get("units"),
                        "desc": meta_entry.get("desc"),
                        "current_value": meta_entry.get("val") or meta_entry.get("value"),
                    }
                )
            return rendered

        return {
            "model_name": getattr(problem.model, "name", "cycle"),
            "mode": mode,
            "options": options,
            "key_inputs": _populate(inputs),
            "key_outputs": _populate(outputs),
        }
    except KeyError as exc:
        return error_on_missing_session(str(session_id), exc)
    except Exception as exc:  # pragma: no cover
        return to_error(exc)

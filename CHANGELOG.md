# Changelog

## [0.2.1] - 2026-03-30

### Fixed

- **list_variables numpy serialization**: `render_variable_entry` now converts
  `numpy.ndarray` values to Python lists (and numpy scalars to `int`/`float`)
  before returning, preventing `Unable to serialize unknown type: <class 'numpy.ndarray'>`
  errors during JSON serialization.

- **run_cycle fails on freshly created turbofan**: Two root causes addressed:
  1. Missing station Mach number defaults (`inlet.MN`, `fan.MN`, etc.) and
     shaft speeds (`LP_Nmech`, `HP_Nmech`) when building a standalone `HBTF`
     cycle. These values were only set in the `MPhbtf` multi-point wrapper but
     are required for the Newton solver to converge.
  2. numpy 2.x compatibility in pyCycle 4.4.0 CEA thermo classes
     (`PropsRHS.compute`, `PropsCalcs.compute`, `PropsCalcs.compute_partials`).
     numpy >= 2.0 rejects assigning a shape-(1,) array into a scalar slot.
     A monkey-patch converts `inputs['n_moles']` to a Python float via
     `.item()` at the top of each affected method.

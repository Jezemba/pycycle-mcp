# Changelog

## [0.2.2] - 2026-03-30

### Changed

- **Pin numpy < 2.0**: pyCycle 4.4.0 has pervasive numpy 2.x incompatibilities
  across `ThermoAdd`, `PropsRHS`, `PropsCalcs`, and other classes. The monkey-patch
  approach only covers a few sites. Pinning `numpy>=1.26,<2.0` is the reliable fix.

- **Add om-pycycle to `[full]` extra**: `pip install -e .[full]` now installs
  `om-pycycle>=4.0.0` alongside `openmdao>=3.27.0`, giving a complete environment
  for real cycle computation.

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

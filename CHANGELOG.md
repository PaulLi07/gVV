# GVV release changelog

## gVV v1.0.0-rc1 — 2026-08-16

### Physics scope

- Kept the active `include/`, `src/`, and `tests/` physics implementation
  identical to the reconstructed and previously fitted GVV nominal baseline.
- Kept the default Fit submission at 10 starts with base seed `20260815`.
- Kept truth MC outside `Fit.exe`; it remains an input only to `PostFit.exe`.

### Runtime and layout

- Fixed the Fit and PostFit Slurm workers so they receive the real project root
  through `GVV_PROJECT_ROOT` and `--chdir`, instead of deriving it from the
  scheduler-spooled `/var/spool/.../slurm_script` path.
- Made project-environment loading a checked failure point in both workers.
- Moved the five ROOT plotting macros from `nominal/plot/` to
  `nominal/scripts/plot/`.
- Routed all generated PDF/EPS plot products to `nominal/results/plot/`; Fit,
  covariance, projection, and PostFit numerical outputs remain directly under
  `nominal/results/`.
- Kept historical gKK/K-matrix utilities excluded from the release.

### Documentation and validation

- Updated the release README, build guide, and detailed architecture HTML for
  the normalized directory structure, the 10-start default, Slurm path
  propagation, test-directory contract, and plot locations.
- Shell syntax, production build state, unit tests, PostFit algebra self-test,
  local-link coverage, and non-submitting Slurm path simulation are validated
  as part of the release update.
- A real GPU Fit and truth-MC PostFit remain release-gating runtime checks.

# GVV release changelog

## gVV v1.1.0-dev — modular architecture refactor

- Omit Resonances referenced only by inactive Terms from the compiled GPU model
  and Minuit parameter layout, so `active: false` is independent of propagator
  type and free propagator parameters.
- Isolated the projection ROOT schema and serialization in
  `process/ProjectionWriter`, leaving `FitLikelihood` responsible only for
  sample/GPU orchestration and likelihood evaluation.
- Moved GVV photon projection, Wave contraction, and F-matrix assembly from
  `WaveRegistry` to the process-specific `ProcessAmplitude.cuh` boundary.
- Restored one-way `tensors -> dynamics -> process` dependencies, merged all
  Minuit/model state translation into `ParameterMapping.cu`, and removed
  repeated internal checks after model compilation or sample preparation.
- Renamed the canonical project directory from `gVV_v1` to `gVV` without
  creating a second repository or changing Git history.
- Completed a production-source audit and added file-level/critical-path
  comments throughout the fit system; downstream `postfit/` remains deferred.
- Moved GVV fit-policy flags out of the reusable propagator descriptor, made
  process parameter/dynamics contracts strict, and kept validation at the
  configuration, I/O, and numerical boundaries without changing the nominal
  physics model.
- Enabled generated header dependencies and compiler warnings in the Makefile;
  corrected project-local ROOT data and CUDA `lib64` environment paths.
- Reorganized the canonical repository into reusable `framework/` and
  GVV-specific `process/` layers while preserving the current physics model.
- Made `model.json` the only resonance/Term model description and `fit.json`
  the only run/output configuration.
- Added a process-neutral multistart fit engine, strict configuration loader,
  output writer, propagator/tensor libraries, and likelihood arithmetic.
- Split the three existing complete GVV waves into independently registered
  files under `process/waves/`; no new `2++` wave was introduced.
- Removed legacy fit-result reading compatibility and configuration snapshots.
- Replaced the old submission script set with one root `submit.sh`.
- Preserved downstream plotting sources and user presentation changes under
  `postfit/`; their migration is intentionally deferred.

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

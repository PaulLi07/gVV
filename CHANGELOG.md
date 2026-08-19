# GVV release changelog

## gVV v1.1.0-dev — modular architecture refactor

- Reworked the documentation into a user-facing installation/physics README,
  a complete architecture guide, separate Fit/Post workflow manual, per-file
  code reference, and detailed Resonance/Term and Wave extension guides.
- Aggregated fully evaluated Term coefficients by exact Wave slot in the Fit
  hot path, reducing its contraction from `O(T^2)` to `O(T + W^2)` without
  dropping any coherent cross-Wave term.
- Replaced repeated masked-coupling projection calculations with bounded-batch
  direct Term-pair components, and replaced Post's event-by-pair matrix with
  GPU-integrated components.
- Generalized `two_body_running_bw` to carry explicit `orbital_l=0,1,2`
  independently of the registered Wave ID.
- Upgraded Projection to schema version 2 with dynamic background metadata,
  corrected Wave labels, and no fixed SB1/SB2 fields.
- Added explicit complete-Wave and intensity-equivalence CUDA regression
  targets while keeping GPU execution out of the ordinary test target.
- Split downstream work into independent `post/calculation` and
  `post/plotting` modules. `make` builds only Fit and `make post` builds the
  numerical Post executable.
- Split batch submission into `submit_fit.sh` and `submit_post.sh`; each script
  now owns only its corresponding executable and Slurm worker invocation.
- Replaced the separate covariance file with `fit_state-<tag>.json`, the
  machine-readable Fit-to-Post contract. The user-facing fit report now
  includes multistart diagnostics, the complete active physical model,
  covariance, and correlation matrices.
- Made projection group/component metadata fully dynamic and removed the
  hard-coded `0++` and `0-+` weight branches from the new schema.
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
- Completed production-source and Post-system audits and added
  file-level/critical-path comments throughout the Fit and Post code.
- Moved GVV fit-policy flags out of the reusable propagator descriptor, made
  process parameter/dynamics contracts strict, and kept validation at the
  configuration, I/O, and numerical boundaries without changing the nominal
  physics model.
- Enabled generated header dependencies in the Makefile and corrected
  project-local ROOT data and CUDA `lib64` environment paths.
- Reorganized the canonical repository into reusable `framework/` and
  GVV-specific `process/` layers while preserving the current physics model.
- Made `model.json` the only resonance/Term model description and `fit.json`
  the only run/output configuration.
- Added a process-neutral multistart fit engine, strict configuration loader,
  output writer, propagator/tensor libraries, and likelihood arithmetic.
- Split the three existing complete GVV waves into independently registered
  files under `process/waves/`; no new `2++` wave was introduced.
- Removed legacy fit-result reading compatibility and configuration snapshots.
- Replaced the old submission script set with dedicated root-level Fit and
  Post Calculation submission entry points.

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

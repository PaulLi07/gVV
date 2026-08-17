# Modular architecture refactor log

Branch: `refactor/modular-architecture`

Scope: architecture-only refactor of the existing GVV model.  No new
resonance, no new propagator physics, and no `2++` wave are introduced.  The
nominal event intensity, normalization convention, sideband likelihood, and
fit parameterization are regression invariants unless an entry explicitly
states otherwise.

## 2026-08-17 — baseline and isolation

- Selected fixed host `lxlogin005.ihep.ac.cn` and reused tmux session
  `gvv_general`.
- Audited repository `/besfs10/groups/psip/psipgroup/user/liyuhong/GVV/analysis/pwa/ctpwa/Release/gVV_v1`.
- Confirmed `main` and `origin/main` at `a29dfdd` (`gVV v1.0.0-rc1`).
- Found pre-existing uncommitted user changes in
  `nominal/include/GVVPlotUtils.h` and `nominal/scripts/subgpu.sh`.
- Preserved those changes in the original worktree and created an isolated
  worktree at `../gVV_v1_refactor_modular`.
- Created branch `refactor/modular-architecture` from `main`.
- Confirmed CUDA 12.9, ROOT 6.32.02, C++17, and availability of the system
  `nlohmann/json.hpp` header.
- Baseline verification passed:
  - `test_dynamics.exe`
  - `test_gvv_amplitude.exe`
  - `test_gvv_model.exe`
  - `test_gvv_fit_parameters.exe`
  - `PostFit.exe --self-test`

## Planned stages

1. Document boundaries and establish the tracked work log.
2. Add the directly edited JSON model schema, loader, validator, and nominal
   runtime model with stable string ids.
3. Replace fixed resonance/term/active-wave storage with runtime vectors and
   dense GPU indices while preserving the nominal calculation.
4. Generate the parameter layout once from model metadata and reuse it in
   Minuit, result I/O, covariance handling, and PostFit.
5. Separate intensity, normalization, and likelihood composition; move GVV
   event/wave/term responsibilities behind the process boundary.
6. Replace fixed projection/component arrays and hard-coded JPC groups with
   runtime model metadata.
7. Update user documentation and run unit, build, self-test, and numerical
   regression checks before the final branch audit.

## Change record

- 2026-08-17: Added the approved framework/process boundary and this log.  No
  production code changed in this stage.
- 2026-08-17: Added the framework-level `ModelDefinition`, strict JSON parser
  and validator, schema v1, directly edited nominal `config/model.json`, and
  model-definition unit tests.  The new layer is host-only and does not yet
  replace the legacy hard-coded GPU model.  New and pre-existing unit tests
  passed after the change.
- 2026-08-17: Added the process-level GVV wave registry and model compiler.
  Stable resonance, wave, and term ids from `model.json` are now validated and
  compiled to dense runtime vectors used at the CUDA boundary.  The migration
  regression test confirms that the nominal JSON compiles to exactly the same
  seven resonance descriptors, seven Term descriptors, initial couplings, and
  active wave set as the legacy hard-coded model.  No new wave or physics
  formula was introduced.
- 2026-08-17: Switched the production likelihood path to runtime model sizes.
  The sample cache now stores only active Wave Gram-matrix entries and a
  model-sized Term-coefficient workspace.  CUDA evaluation was separated into
  process-specific GVV Term construction followed by a coherent contraction
  whose loop bounds are supplied at runtime.  `NLL_estimator` now owns the
  JSON-compiled model and uploads vectors instead of fixed-size arrays.  A
  temporary compatibility overload remains only for PostFit and is scheduled
  for removal in the next stage.  Full build and all tests passed.
- 2026-08-17: Added a model-generated `GVVFitParameterSpec` layout and switched
  Fit/Minuit to it.  Parameter names, order, transformed initial values,
  steps, bounds, and mutation targets now come from one compiled-model view;
  multistart randomization and boundary checks consume the same descriptors.
  Fit accepts an optional model JSON path and writes the exact canonical model
  beside the result as `<fit-result>.model.json`.  Legacy result-reading APIs
  remain temporarily available until PostFit is migrated.  Build and tests,
  including a runtime parameter round trip, passed.
- 2026-08-17: Migrated PostFit to the compiled runtime model and removed the
  fixed nominal fit-state, Resonance/Term enums, default-model factories, and
  fixed component-pair APIs. PostFit now obtains Term labels, `J^PC` grouping,
  fit parameters, pair counts, covariance dimensions, and model selection from
  the result model snapshot. Historical nominal result rows remain readable by
  resolving stable ids against the supplied snapshot; no hard-coded default
  model is reconstructed.
- 2026-08-17: Replaced projection's fixed component matrix and fixed `0++`/
  `0-+` group construction with runtime vectors and metadata-generated groups.
  Projection files now include `component_map`, `group_map`, `n_terms`, and
  `n_groups`; nominal `weight_0pp`/`weight_0mp` branches remain presentation
  aliases only. Plotting validates the runtime component-vector length.
- 2026-08-17: Added process-neutral MC normalization and signed likelihood
  functions under `framework/Likelihood.h`, with independent unit tests. The
  GVV NLL layer now supplies event intensities and sample coefficients to this
  module instead of owning the likelihood mathematics inline.
- 2026-08-17: Exposed an optional model JSON through the Slurm submission
  wrapper, tightened the C++ JSON loader to reject unknown fields, documented
  all model editing/extension contracts, and added regression cases that
  compile both a reduced 6-Term model and an expanded 8-Resonance/8-Term model.
  This explicitly tests that nominal 7/7 counts are not production constants.

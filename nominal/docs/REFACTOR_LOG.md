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

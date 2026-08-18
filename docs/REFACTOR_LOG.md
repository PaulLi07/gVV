# Modular refactor work log

Canonical repository:
`/besfs10/groups/psip/psipgroup/user/liyuhong/GVV/analysis/pwa/ctpwa/Release/gVV`

## Scope and invariants

The work is an architecture-equivalent refactor. It does not add or remove a
nominal physics contribution, introduce a `2++` Wave, or change propagator,
Wave, omega-width, normalization, or sideband formulae. Old fit-result parsing
compatibility and configuration snapshots were intentionally removed.

All active work uses one Git repository. An early isolated worktree was merged
back and removed. The project directory was renamed from `gVV_v1` to `gVV`
without creating a second repository or changing history. User-adjusted plot
styles were retained during the downstream migration.

## Core architecture (2026-08-16 to 2026-08-17)

- Established `framework/`, `process/`, `app/`, `config/`, `tests/`, and
  downstream boundaries.
- Extracted reusable math, tensor, kinematics, propagator, likelihood, and
  multistart fit components.
- Made `model.json` the only Resonance/Term description and `fit.json` the only
  run/output description.
- Split the three existing complete Waves into `process/waves/` and created one
  host registry plus one device dispatch.
- Removed fixed Resonance, Term, Wave, and parameter counts from production
  paths.
- Centralized model-to-Minuit translation in `ParameterMapping`.
- Restored one-way `math -> tensors -> dynamics -> process` dependencies.
- Moved common GVV polarization/contraction logic to `ProcessAmplitude`.
- Isolated the process-specific ROOT projection schema in `ProjectionWriter`.
- Replaced multiple submission helpers with one root `submit.sh`.
- Added English architecture, configuration, Wave-development, and README
  documentation and explanatory source comments.

The refactor was built from a clean tree with CUDA 12 / ROOT 6.32.02 and its
unit suite passed. Earlier user-run GPU regression selected the same start and
NLL within Minuit numerical tolerance. Those historical checks predate the
Post contract update below; no cluster job was submitted by Codex during the
current work.

## Inactive-Term robustness (2026-08-18)

Model compilation now determines referenced Resonances only after filtering
inactive Terms. A free propagator parameter belonging only to inactive Terms
does not enter Minuit. The behavior no longer depends on propagator type.

## Fit output and Post system (2026-08-18)

Working branch: `refactor/post-system`.

- Replaced `Cova_matrix-<tag>.dat` with `fit_state-<tag>.json`.
- Expanded `fit_result-<tag>.txt` into a user-only report containing run
  provenance, samples, minimizer policy, every start, selected diagnostics,
  ordered free parameters, active Waves/Terms/Resonances, all fixed/free active
  physical parameters, covariance, and correlation.
- Added a deterministic signature of canonical `model.json`. Post Calculation
  checks both this signature and exact parameter ordering before applying a fit
  state.
- Split the old PostFit source into `post/calculation/` and `post/plotting/`.
  Post Calculation reads state, model, truth MC, and selected normalization MC;
  Post Plotting reads only the fit projection ROOT file.
- Routed numerical downstream products to `post/calculation/results/` and
  figures to `post/plotting/results/`.
- Removed hard-coded `0++`/`0-+` projection branches. Plotting reads dynamic
  group/component maps and model-provided Term labels.
- Kept `make` limited to `Fit.exe`; added explicit `make post` for `Post.exe`.
- Extended the single `submit.sh` with explicit fit and Post Calculation modes.

Verification completed on the fixed `lxlogin005` node without submitting a
cluster job:

- `make clean && make -j2` built `Fit.exe` and confirmed that `Post.exe` was
  absent from the default target;
- `make -j2 post` built `Post.exe` independently;
- `make check` passed all 11 unit/contract tests, including report and fitted
  state round trips;
- ROOT 6.32.02 loaded all five plotting macros without parser errors;
- shell syntax, JSON syntax, patch whitespace, English-only documentation, and
  the `framework/` to `process/` dependency direction were audited.

Only existing external ROOT `TStorage.h` and CUDA sm70 deprecation warnings
appeared; no project-source compiler error or warning was introduced. Runtime
Fit and Post Calculation jobs are intentionally left to the user. The focused
commit identifier is available in the Git history and final work report.

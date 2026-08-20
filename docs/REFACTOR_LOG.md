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
- Replaced obsolete submission helpers with a dedicated root
  `submit_fit.sh`.
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
- Added independent `submit_fit.sh` and `submit_post.sh` entry points. Each
  script submits and runs only its own executable; neither performs mode
  dispatch for the other system.

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

## Extensibility closeout (2026-08-19)

Working branch: `refactor/extensibility-closeout`.

This phase remains physics-equivalent and does not add a `2++` Wave or change
the nominal active model. It closes the scaling and contract issues found in
the post-refactor architecture audit:

- unified the reusable two-body running-width propagator behind one device
  model with explicit `orbital_l`; the GVV compiler accepts the barrier
  library's implemented range `L=0,1,2` without coupling the propagator to a
  registered Wave ID;
- rejected a zero fixed scale-and-phase reference while leaving the fitted
  phase-reference convention unchanged;
- replaced the Fit's Term-squared contraction with an exact aggregation of
  fully evaluated Term coefficients by dense Wave slot, reducing its hot path
  from `O(T^2)` to `O(T + W^2)` and its sample workspace from `N*T` to `N*W`;
- replaced Projection's repeated coupling masking with one packed direct
  Term-pair calculation in bounded batches, preserving signed interference,
  group definitions, full-model normalization, and the complete external
  component matrix;
- changed Post Calculation to reduce packed Term-pair integrals on the GPU in
  bounded batches instead of retaining an `N_event*N_pair` matrix;
- completed selected-sample cross-group interference output and made bounded
  finite differences prefer a resolvable one-sided step near a fit boundary;
- upgraded Projection to schema version 2 with dynamic `background_map`
  metadata and removed every fixed SB1/SB2 field;
- corrected `component_map.wave_label` to use the registered Wave label;
- added structural fitted-state validation and log-space likelihood arithmetic
  at their owning framework boundaries;
- added explicit runtime CUDA regressions for complete Wave identities and
  Term/Wave/component equivalence. These tests are separate from ordinary
  login-node-safe checks and require an allocated GPU for execution.

Build and login-node-safe verification on the fixed `lxlogin005` node
completed without running Fit or Post Calculation:

- `make clean`, `make -j2`, and `make -j2 post` built `Fit.exe` and `Post.exe`;
- `make -j2 tests` built the ordinary test suite and `make check` passed all
  11 login-node-safe tests;
- `make -j2 gpu-tests` compiled both new CUDA runtime regressions;
- all five plotting macros passed a ROOT-aware C++ syntax check;
- shell syntax, JSON syntax, and patch whitespace checks passed.

At the user's request, the two CUDA runtime regressions were subsequently
submitted as background Slurm job `23444` (`gvv-gpu-tests`), with output
routed to `runlog/gpu-tests-23444.log`. On 2026-08-20 the user reported
completion; Slurm recorded the job, batch shell, and `make` step as
`COMPLETED` with exit code `0:0`. Both runtime markers were present:
`GVV complete-Wave GPU numerical tests passed` and
`GVV Term/Wave/component GPU equivalence tests passed`.

### Deferred improvements

The following supported-scope limitations are recorded rather than expanded
in this closeout:

- make the reusable human Fit report heading process-neutral instead of using
  the GVV title;
- further separate generic propagator-configuration compilation from the GVV
  host model compiler beyond the two-body descriptor generalized here;
- expose omega-width-table range, resolution, endpoint policy, and related
  constants for systematic studies;
- add optional full physical four-momentum validation at the external sample
  boundary;
- define an explicit Post policy for an active ordinary Term whose fitted
  complex coupling is exactly zero: its fit fraction is zero, but its current
  selected/truth component-efficiency ratio is mathematically `0/0`; the
  present implementation rejects that undefined observable rather than
  silently assigning a value;
- continue Plotting cleanup beyond the schema-v2 reader update, including
  ROOT-only environment loading, axis ranges that include diagnostic curves,
  high-multiplicity styles, and goodness-of-fit presentation;
- perform a broader historical changelog/documentation cleanup without mixing
  it into the amplitude and fit implementation changes.

No Fit or Post Calculation job was submitted. Apart from the explicitly
user-authorized GPU-regression job recorded above, production runtime
validation remains user-controlled.

## Documentation overhaul (2026-08-19)

The user and developer documentation was rewritten as one consistent English
manual set without changing source code, configuration semantics, physics
formulae, or output schemas:

- `README.md` now provides the normal project landing page: physics scope,
  architecture, installation, input contract, build, Fit/Post usage, outputs,
  and concise extension entry points;
- `docs/ARCHITECTURE.md` traces the complete model, event, amplitude,
  likelihood, fit-output, Projection, and Post data flow and states the
  framework/process replacement boundary for another final state;
- `docs/WORKFLOW.md` separates Amplitude Fit from Post Calculation and Post
  Plotting and documents the login-node/background-Slurm operating model;
- `docs/CODE_REFERENCE.md` explains every production/configuration/submission
  file, important internal code blocks, all plotting entry points, and the
  ordinary versus GPU verification inventory;
- `docs/MODEL_CONFIGURATION.md` provides the authoritative user guide for
  adding, disabling, re-enabling, and permanently deleting Resonance Terms;
- `docs/WAVE_DEVELOPMENT.md` provides the physics, implementation,
  registration, coherence, and numerical-validation procedure for a new
  complete GVV Wave;
- `post/README.md` documents the independent numerical and plotting modules,
  their exact inputs, outputs, algorithms, and acceptance checks.

The documentation was checked for valid local links, balanced Markdown code
fences, parseability of every complete embedded JSON example, trailing
whitespace, and English-only prose. No Fit, Post, CUDA executable, or new
Slurm job was run for this documentation-only phase.

## Standalone Plotting macros (2026-08-20)

Working branch: `refactor/plotting-modules`.

- Kept `post/plotting/draw.sh` as the one-command driver for all five figures.
- Converted every plotting `.cxx` from a thin wrapper into an independently
  executable ROOT macro with no-argument defaults.
- Moved each figure's observable or moment list, binning, axes, canvas layout,
  curve styles, legend, annotations, and output defaults into that figure's
  `.cxx` file.
- Reduced `GVVPlotUtils.h` to the shared Projection schema reader, dynamic map
  and branch handling, exchange-symmetric filling, unstyled histogram
  construction, Pearson diagnostic, project-root path helper, and common base
  style.
- Reduced `GVVAngularMoments.h` to Legendre/moment arithmetic, unstyled moment
  construction, and the moment chi-square diagnostic.
- Preserved the Projection schema, dynamic Term/JPC/background discovery,
  physics weights, binning, draw order, styles, and output products of the
  previous macros.

Verification was performed with ROOT 6.32.02 on the fixed `lxlogin005` node,
without running Fit, Post Calculation, CUDA code, or a Slurm job:

- all five macros passed ROOT parsing;
- each macro was invoked directly by basename with no arguments from its
  `macros/` directory against an isolated synthetic schema-v2 Projection;
- the main projection macro was also invoked by its repository-relative path
  from the isolated project root;
- the five direct invocations produced the expected five PDF and five EPS
  files;
- the copied `draw.sh` wrapper processed the same Projection under a second
  tag and produced the expected five PDF and five EPS files;
- ROOT logs contained no fatal/error/abort markers, and representative
  detailed-projection and angular-moment PDFs were rendered for visual
  inspection.

The repository's pre-existing `results/projection-initial.root` is schema
version 1 and is intentionally rejected by the schema-v2 reader. It was not
modified. A new Fit result produced by the current Projection writer is needed
before the no-argument project default can draw a physical result.

Follow-up readability work collected every figure-specific setting into a
clearly delimited `User configuration` preamble in each macro. Default input
and output paths now appear directly in each ROOT-callable function signature.
Relative defaults and runtime overrides such as `results/projection-TAG.root`
and `post/plotting/results/projection-TAG` are resolved from the project root;
absolute paths remain unchanged. The drawing implementation below each
preamble contains no duplicated default-path literals.

The follow-up was verified with ROOT 6.32.02 on `lxlogin005` using an isolated
schema-v2 Projection fixture. All five macros parsed and ran with no arguments
from the isolated project root, producing ten default PDF/EPS files. The main
macro also ran with no arguments from `post/plotting/macros/`, and the explicit
two-argument form produced the requested custom output prefix. The unchanged
`draw.sh` driver produced its ten tagged files, all ROOT logs passed the error
scan, and no Fit, Post Calculation, CUDA executable, or Slurm job was run.

## Projection polarization observables (2026-08-21)

Working branch: `refactor/plotting-modules`.

- Upgraded the fitted Projection contract from schema version 2 to version 3.
- Completed the omega1 direction with `cos_theta_omega1` and `phi_omega1` in
  the X helicity frame defined by the beam-X production plane.
- Replaced the ambiguous omega decay-plane branch names with
  `phi_decay_plane_omega1/2` and added the matching
  `cos_theta_decay_plane_omega1/2` polar cosines.
- Added all six charged/neutral pion polar cosines in their parent omega
  helicity frames and the two unnormalized `p(pi+) cross p(pi-)` magnitudes.
- Kept all existing pion-pair invariant masses and the wrapped signed
  difference of the two local decay-plane azimuths.
- Updated the shared plotting reader, angular-moment helper, standard macro
  identifiers/axis label, and current schema documentation without adding new
  standard figure panels.

Verification was performed with ROOT 6.32.02 and CUDA 12 on the fixed
`lxlogin005` node, without running Fit, Post Calculation, or a Slurm job:

- the changed Projection writer compiled and `make check` passed all 11
  login-node tests;
- a focused host-side kinematics test checked the decay-plane polar cosine,
  decay-plane azimuth, representative pion polar cosines, and unnormalized
  plane-normal magnitude for a constructed boosted omega decay;
- a branch-contract comparison found exactly the 14 intended additions and
  the three intended branch renames, with every other branch preserved;
- an isolated schema-v3 ROOT fixture was accepted by all five plotting macros;
  each macro produced its PDF and EPS outputs, and the ROOT logs contained no
  error, fatal, abort, segmentation-fault, missing-branch, or unsupported-
  schema diagnostics.

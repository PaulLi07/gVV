# Modular refactor work log

Canonical repository:
`/besfs10/groups/psip/psipgroup/user/liyuhong/GVV/analysis/pwa/ctpwa/Release/gVV`

## Initial modular-refactor scope and invariants

The initial architecture migration was an amplitude-equivalent refactor. It
did not add or remove a nominal physics contribution, introduce a `2++` Wave,
or change propagator, Wave, omega-width, normalization, or sideband formulae.
Old fit-result parsing compatibility and configuration snapshots were
intentionally removed.

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

## Plot catalogue update (2026-08-21)

Working branch: `refactor/plotting-modules`.

- Renamed the main 3x2 entry point from `Draw_projection_2_3.cxx` to
  `Draw_projection.cxx` and removed the separate detailed 4x2 figure.
- Defined the main six panels as `M(omega omega)`, candidate-combined
  `M(gamma omega)`, `cos(theta_gamma)`, exchange-symmetric
  `cos(theta_omega)` and `phi_omega`, and candidate-combined
  `M(pi+ pi- pi0)`.
- Synchronized the diagonal-Term component figure to the same six variables.
- Added a standalone three-angle polarization figure and a standalone 3x2
  omega-decay check figure for the three pion helicity cosines and three
  pion-pair invariant masses.
- Used half-weight candidate merging for local omega-decay variables. No
  additional pion-cosine reflection was introduced.
- Updated `draw.sh` to run the six independent ROOT macros.

Verification used an isolated schema-v3 Projection fixture and did not run
Fit, Post Calculation, CUDA kernels, or a Slurm job:

- all six macros passed ROOT-aware C++ syntax checks and direct ROOT execution;
- focused histogram checks confirmed the `+pi` omega-azimuth exchange image,
  the `+/-cos(theta_omega)` exchange pair, and half-weight candidate merging
  for pion angles and pion-pair masses;
- `draw.sh` completed successfully on `lxlogin005`, produced the expected six
  PDF and six EPS files, and its ROOT log passed the error-marker scan;
- the main, polarization, and omega-decay check PDFs were rendered and
  inspected for panel content, labels, and layout.

## Unified omega and rho line-shape building blocks (2026-08-22)

Working branch: `refactor/unified-omega-lineshape`.

- Added the process-independent `BW_from_width` denominator and routed fixed,
  analytic-running, scalar-S+D, nominal two-body, and tabulated omega widths
  through the same sign convention.
- Added a reusable host/device uniform-table view while preserving the existing
  clamped interpolation behavior exactly.
- Extracted `process/OmegaDecayModel.cuh` as the single definition of particle
  constants, rho parameters, both P-wave barriers, the analytic rho BWR, and
  the coherent three-rho sum.
- Removed the duplicate rho-isobar expression from `OmegaWidthTable.cu`; the
  event current and the three-body width integration now call the same
  host/device function.
- Kept the rho and omega as fixed process daughter/subchannel dynamics rather
  than adding particle-specific entries to the configurable X Resonance
  registry.

The refactor preserves all masses, widths, radii, table dimensions, Dalitz
binning, interpolation boundaries, amplitude factorization, model JSON, fit
parameters, and output contracts. Verification was performed with ROOT 6.32.02
and CUDA 12 on the fixed `lxlogin005` node:

- `make check` passed all 11 login-node tests;
- the focused dynamics regression independently reconstructed the previous
  three-rho expression and matched the shared omega-decay helper;
- the propagator regression matched the tabulated omega wrapper to the shared
  `BW_from_width` denominator;
- `make`, `make post`, and `make gpu-tests` compiled and linked Fit, Post
  Calculation, and both GPU regression executables.

Fit, Post Calculation, and the GPU regression executables were not run, and no
Slurm job was submitted during this architecture-only verification.

## Propagator compiler boundary and nominal pion-mass policy (2026-08-22)

Working branch: `refactor/propagator-compiler`.

- Reduced `WaveRegistry` to the complete-Wave catalogue and device dispatch.
- Added `ProcessModel` for compiled runtime records, `PropagatorCompiler` for
  exact Resonance JSON/channel compilation, and `ModelCompiler` for active
  Term dependency resolution and dense model assembly.
- Made every top-level propagator descriptor self-contained with its nominal
  daughter masses and barrier radius. Fit, Projection, and Post now call
  `evaluate_propagator(s, descriptor)` without rebuilding channel context.
- Replaced propagator-specific flags and cases in `ParameterMapping` and the
  Fit model summary with compiler-produced generic parameter/report metadata.
- Unified the analytic two-body running width on one nominal-daughter-mass
  implementation shared by the rho and configurable Resonance paths.
- Made rho0/rho+/rho- particle identities select nominal daughter masses and
  the nominal bachelor pion. Event `s_omega` and `s_pipi` remain dynamic, but
  reconstructed single-pion `p_i^2` values no longer enter scalar isobar
  dynamics.
- Added necessary compiler validation for positive resonant widths and open
  pole-normalized omega-omega thresholds.
- Centralized omega table resolution, range, Dalitz binning, and clamped
  extrapolation in `OmegaWidthTableConfig`.
- Added a dedicated propagator-compiler regression and extended dynamics,
  table-stability, active-model, and parameter-binding coverage.

Compiler separation, binding generation, and the common running-width formula
are algebraically equivalent for the nominal model. The nominal pion-mass
policy is an intentional physics-definition correction and may cause small
event-level differences if an input pion four-vector has non-nominal
reconstructed `p_i^2`; it makes event evaluation identical to the mass policy
already used by the omega-width integration.

Verification on the fixed `lxlogin005` node:

- Fit, Post Calculation, all 12 login-node tests, and both GPU runtime test
  executables compiled and linked with CUDA 12 and ROOT 6.32.02;
- `make check` passed all 12 login-node-safe tests;
- no Fit, Post Calculation, GPU executable, or Slurm job was run.

## Process-neutral high-spin tensor building blocks (2026-08-23)

Working branch: `feature/tensor-building-blocks`.

- Added `TensorContraction.cuh` with named, metric-explicit rank-two
  contractions instead of an ambiguous tensor multiplication operator.
- Added direct application of the spin-two polarization projector to a
  rank-two source without allocating a rank-four projector.
- Added the reduced bare G-wave contraction
  `t^(4)^{mu nu lambda tau} source_{lambda tau}` without introducing a
  256-component per-thread tensor type. The result is already symmetric,
  transverse, and traceless, so no outer spin-two projector is required.
- Completed the existing Blatt-Weisskopf implementation consecutively through
  `L=4`, preserving the original `L=0,1,2` formulae and units.
- Kept every orbital helper as bare STF geometry. Barrier factors are still
  multiplied exactly once in a process Wave; normalized orbital and LS
  Clebsch-Gordan factors remain process-level responsibilities.
- Fixed the future high-spin convention to Condon-Shortley spherical vectors
  and Racah-normalized harmonics, including the reviewed conversion between
  bare STF and normalized `02`, `20`, `22`, and `42` decay tensors.
- Added a focused CUDA regression that compares the efficient projector and
  G-wave path with independent explicit-index references, plus host checks for
  the new higher-L barrier polynomials.
- Updated the README, architecture guide, Wave-development guide, code
  reference, changelog, and the dedicated tensor-convention document. No
  spin-two Wave, Wave registry entry, Resonance, Term, or model field was
  added.

Verification was performed with CUDA 12 and ROOT 6.32.02 on the fixed
`lxlogin005` node. No Slurm job was submitted and no GPU executable, Fit, or
Post Calculation was run:

- all 12 login-node test executables and all three CUDA runtime test
  executables compiled;
- the finalized tensor CUDA regression compiled after the independent raw-
  source reference and metric-sign checks were added;
- `make fit post` compiled and linked both production executables;
- 10 of the 12 login-node tests passed when run against the user's current
  working model;
- `test_fit_parameters` and `test_wave_registry` stopped on their nominal
  hard-coded parameter/resonance counts because the user's uncommitted
  `config/model.json` deliberately sets `X_2370_11` inactive. That model edit
  was preserved and excluded from this commit; neither failure enters the new
  process-neutral tensor code.

The new CUDA tensor identities still require their first runtime execution in
the user-controlled Slurm GPU environment before the subsequent spin-two Wave
implementation is release-gated.

## Spin-two Waves and phase-reference class names (2026-08-23)

Working branch: `feature/tensor-building-blocks`.

- Added twelve complete `2++` Waves under `process/waves/`, combining
  `LS=02,20,22,42` with the three independent radiative production covariants
  `U1,U2,U3` under the documented normalized-CG convention.
- Added `f2(1565)` and `f2(1810)` to the nominal model with only their three
  `LS=02` Waves active. Each state shares one Resonance propagator across its
  three Terms and fits three independent complex couplings.
- Renamed the registered phase-reference tokens from `scalar` and
  `pseudoscalar` to `positive_parity` and `negative_parity`. The former now
  clearly covers every registered `0++` and `2++` Wave, while the latter
  covers the registered `0-+` Wave.
- Kept stable Wave IDs, JPC labels, the `f0_1710_00` positive-real reference,
  the `eta_1760_11` global fixed reference, all numerical amplitudes, and every
  Gram-matrix cross term unchanged. Existing `model.json` files require no
  class-name edit because the tokens are owned by `WaveRegistry`.
- Updated the README, architecture, model-configuration, Wave-development,
  code-reference, and tensor-convention documents, including the rule that a
  future Wave is classified by event-level Gram-matrix cross terms rather
  than by parity alone.

Verification was performed on the fixed `lxlogin005` node without submitting
a cluster job:

- `make -j2 check` rebuilt the affected registry/model tests and all 12
  login-node-safe tests passed;
- the registry regression explicitly checked scalar and tensor Waves in
  `positive_parity` and the pseudoscalar Wave in `negative_parity`;
- patch whitespace and tracked-text searches found no remaining use of the
  retired tokens as coherence-class values or descriptions.

Only the existing CUDA warning for pre-sm75 offline compilation appeared. No
Fit, Post Calculation, GPU executable, or Slurm job was run.

## Configurable Resonance parameters and self-contained FitState (2026-08-24)

Working branch: `feature/configurable-propagator-parameters`.

- Extended the propagator compiler so supported Resonance `mass` and `width`
  parameters may be fixed or released entirely from `model.json`. Their free
  Minuit coordinates use the physical identity transform, explicit finite
  bounds with a positive lower limit, and the stable names
  `mass_<resonance-id>` and `width_<resonance-id>`.
- Kept positive S/D and effective omega-omega ratios in log coordinates, and
  kept `orbital_l` as a fixed integer model choice. The full free mass range of
  `two_body_running_bw` and `scalar_sd_running_bw` must remain strictly above
  nominal omega-omega threshold; the subtracted effective Flatte continues to
  support the intended subthreshold-pole use case.
- Made the ownership rule explicit in code coverage and documentation: one
  Resonance ID is one propagator instance shared by all Terms that reference
  it. Reusing a propagator type string for a different Resonance ID does not
  share parameters.
- Upgraded the machine Fit-to-Post contract to FitState schema version 2. It
  embeds the complete canonical model as a structured JSON object together
  with the fitted vector, bounds, covariance, diagnostics, provenance, and
  compatibility signatures. Schema version 1 is deliberately rejected and
  requires a new Fit.
- Removed the separate model-file input from Post Calculation. `Post.exe` and
  `submit_post.sh` now accept only the fitted state, generated truth MC, and
  selected normalization MC; the stored source model path is provenance only.
- Split compatibility into the deterministic canonical-definition signature,
  an explicit `gvv-amplitude-contract-vN` implementation signature, and their
  combined identifier. Post checks all three before applying parameters. The
  implementation version must be manually reviewed and bumped whenever an
  unchanged model document could acquire different numerical amplitude or
  parameter semantics.
- Added focused coverage for mass/width policies and application, same-ID
  sharing versus distinct-ID independence, schema-v2 embedded-model round
  trips and schema-v1 rejection, and definition/implementation signature
  behavior.

The nominal configuration keeps its existing mass and width values fixed, so
this change does not alter the nominal amplitude. No Slurm or GPU job was
submitted as part of this configuration and output-contract update.

Final integration verification was performed with CUDA 12 and ROOT 6.32.02
on the fixed `lxlogin005` node:

- `make -j2 check` passed all 13 login-node-safe tests, including the new
  propagator-parameter, sharing/independence, FitState-v2, and implementation-
  signature regressions;
- `make -j2` and `make -j2 post` confirmed the current Fit and Post binaries;
- `make -j2 gpu-tests` compiled all three CUDA runtime test executables without
  executing them;
- `bash -n submit_post.sh`, JSON syntax checks, and `git diff --check` passed;
- the nominal `config/model.json`, `WaveRegistry`, and the user's
  `positive_parity`/`negative_parity` definitions were unchanged.

No Fit, numerical Post Calculation, GPU runtime test, or Slurm job was run.

# gVV code reference

## 1. How to use this reference

This guide explains the responsibility of every production source, header,
configuration file, shell entry point, and test in the repository. Files are
grouped by subsystem so that the list also describes the control flow.

Ownership labels used below are:

- **Fit**: used only while constructing or running the amplitude fit;
- **Calculation**: used by numerical Post Calculation;
- **Plotting**: used by projection plotting;
- **Shared**: used by more than one of those workflows;
- **Infrastructure**: build, environment, submission, or verification.

The architectural boundary is more important than the file extension:
`framework/` is process-independent, `process/` is the complete GVV physics
implementation, and `app/`/`post/` assemble user workflows.

## 2. Top-level project controls

### `Makefile` — Infrastructure, all compiled workflows

The Makefile makes the product split explicit:

- default `all` resolves to `fit` and builds only `bin/Fit.exe`;
- `post` independently builds `bin/Post.exe`;
- `tests`/`check` build and run the login-node-safe suite;
- `gpu-tests`/`check-gpu` build and run the three CUDA-device regressions;
- `clean` removes only generated objects, binaries, tests, and dependency
  files listed by the build graph.

`FRAMEWORK_OBJECTS` and `PROCESS_OBJECTS` keep the source boundary visible at
link time. `POST_OBJECTS` deliberately reuses the generic model/state code and
the GVV compiler/evaluator, but not the Fit engine, Fit likelihood, Fit report,
or Projection writer. All compilation uses `nvcc` with C++17 and generated
`.d` dependencies, so editing a nested Wave or tensor header triggers the
necessary rebuild.

The default toolchain paths and CUDA architectures are project defaults, not
hidden system discovery. `GVV_CUDA_ROOT`, `ROOT_PREFIX`, `BUILD_DIR`,
`OBJ_DIR`, `BIN_DIR`, and `TEST_BIN_DIR` may be overridden by the caller.

### `submit_fit.sh` — Fit, Infrastructure

This is both the Fit Slurm submission front end and the Fit worker entry point.
It has one mode visible to users and an internal `--worker` mode used by
`sbatch`.

Important blocks are:

- **Slurm header**: requests the `gpupwa` partition/QOS/account, one A100, one
  task, one CPU, and the configured memory.
- **Path canonicalization**: derives the real repository root before Slurm
  spools the script and resolves relative configuration paths against it.
- **`read_fit_fields`**: uses a small read-only Python JSON parser to obtain the
  tag, output locations, model, and every input sample. This is submission
  preflight, not a second model layer.
- **`require_files`**: rejects missing immutable inputs before submission and
  again inside the worker.
- **`load_environment`**: sources `config/gvv_env.sh`, records host/job/GPU
  context, and runs `nvidia-smi` on the assigned worker.
- **`run_worker`**: invokes `Fit.exe` through `srun` and verifies that the
  report, fitted state, and projection were produced.
- **submission block**: sets the working directory, exports the canonical
  project root, truncates the tag-specific log, and prints the Slurm job ID.

It does not build the executable, run Post Calculation, plot results, or wait
for the submitted job to complete.

### `submit_post.sh` — Calculation, Infrastructure

This script mirrors the submission/worker structure of `submit_fit.sh`, but it
owns only Post Calculation. It requires exactly four inputs: fitted state,
model JSON, generated truth MC, and selected normalization MC.

The submission side reads only `output_tag` from the fitted-state JSON to name
`runlog/post-<tag>.log`. The worker loads the project environment, invokes
`Post.exe` through `srun`, and checks the tagged Post TXT and ROOT results. It
does not infer truth MC from the fit, submit a Fit job, or invoke plotting.

### `.gitignore` and `VERSION` — Infrastructure

`.gitignore` excludes ROOT inputs, generated ROOT files, binaries, build
products, Fit/Post results, logs, and figures while retaining placeholder
result directories. `VERSION` contains the human release identifier; no
runtime numerical code branches on it.

## 3. Configuration layer

### `config/gvv_env.sh` — Infrastructure

This sourceable shell file establishes the project-local CUDA and ROOT
toolchain without editing a user shell profile. It:

- derives and exports `GVV_PROJECT_ROOT`;
- accepts caller overrides for `GVV_CUDA_ROOT`, `GVV_ROOTSYS`, and the
  convenience `GVV_DATA_DIR`;
- verifies `nvcc` and ROOT's `thisroot.sh` before changing the environment;
- exports the compiler and library variables consumed by the Makefile;
- sources ROOT, updates `PATH`/`LD_LIBRARY_PATH`, and prints version evidence
  unless `GVV_ENV_QUIET=1`.

It does not choose a model, load data, allocate a GPU, or submit a job.

### `config/model.json` — Shared model description

This is the only user-facing description of Resonances and Terms. It contains:

- the process identifier and human metadata;
- named propagator instances with physical parameters;
- active or inactive Terms;
- the registered Wave ID selected by each Term;
- coupling parameterization and reference policy;
- the process dynamics link from a Term to its Resonance.

Array order is meaningful because it determines deterministic active Term and
Resonance ordering after inactive entries are removed. The file contains no
fit sample paths, minimizer policy, output tag, or hard-coded runtime count.

### `config/model.schema.json` — Shared declarative format reference

The JSON Schema documents the generic version-1 surface: legal top-level
fields, stable-ID syntax, parameter fields, coupling modes, reference values,
and Term structure. The C++ loader remains authoritative because it also
enforces generic cross-field rules. `ModelCompiler` and `PropagatorCompiler`
then enforce active reference counts, process dynamics, and exact GVV
propagator contracts that a compact schema does not express.

### `config/fit.json` — Fit run description

This file selects the model, data, accepted normalization MC, a dynamic list of
signed background samples, multistart/Minuit settings, result/log directories,
and one output tag. It does not describe Resonances or truth MC. Reusing the
tag intentionally overwrites the previous tagged Fit products.

## 4. Fit application assembly

### `app/Fit.cu` — Fit

`Fit.cu` is the top-level glue executable. Its important blocks are:

1. **CLI and input contract**: accepts zero or one `fit.json` path and defines
   the GVV `Pwa` branch mapping in `gvv_input_contract()`.
2. **Configuration/model assembly**: loads `FitRunConfig`, creates output
   directories, compiles the GVV model, and constructs `FitLikelihood`.
3. **Sample preparation**: loads data, normalization MC, and every configured
   signed background, then calls `Prepare()` once.
4. **Parameter binding**: builds the GVV binding layout and extracts the
   process-neutral `FitParameterSpec` vector.
5. **Objective lambda**: applies a flat Minuit vector to the mutable GVV model
   and returns the negative process log likelihood.
6. **Multistart selection**: runs the generic engine, prints an overview, and
   refuses to write final outputs when no attempt passes the convergence
   criteria.
7. **Independent output writers**: reapplies the selected state, writes the
   human report, writes the machine fit state, and asks `ProjectionWriter` to
   create the plotting bridge.

Tensor formulae, GPU kernels, Minuit algorithms, parameter-order logic, and
ROOT projection branches deliberately live below this file.

## 5. Generic framework

### 5.1 Math primitives

### `framework/math/DeviceComplex.cuh` — Shared

Provides a minimal double-precision complex type callable on host and device.
It implements arithmetic, conjugation, magnitude, phase, reciprocal, and
printing without depending on device support for `std::complex`. This type is
used by propagators, couplings, omega factors, and CUDA coefficient buffers.

### `framework/math/FourVector.cuh` — Shared

Defines the device `FV` class with internal order `[E, px, py, pz]` and the
`(+---)` inner product. It supplies only vector algebra and scalar products.
ROOT storage conversion is not its responsibility; `TermEvaluator` performs
that conversion at the process boundary.

### `framework/math/Lorentz.cuh` — Shared

Owns the common metric-sign helper and the convention
`epsilon^{0123}=+1`. `levi_civita` validates indices, rejects repeated indices,
and computes the permutation sign. Keeping this convention in one header
prevents Wave files from carrying private sign tables.

### 5.2 Tensor building blocks

### `framework/tensors/Tensor.cuh` — Shared

Implements the device rank-two Lorentz tensor used by all current Waves. It
provides tensor arithmetic, contraction with an `FV` using the adopted metric,
the normal metric tensor, and a two-vector Levi-Civita tensor construction.
It is low-level algebra only; a complete Wave must not be added here.

### `framework/tensors/TensorContraction.cuh` — Shared

Provides explicitly named Lorentz contractions for rank-two tensors. It
covers contraction of the second index with a vector, contraction of the
second indices of two tensors, a double contraction, the Lorentz trace, and
symmetrization. The functions make every metric sign visible without adding
an ambiguous tensor multiplication operator.

### `framework/tensors/SpinProjector.cuh` — Shared

Provides the spin-one transverse metric `g - p p/(p.p)` and applies the
spin-two polarization projector directly to an arbitrary rank-two source.
The implementation symmetrizes, projects both indices, and removes the trace
without materializing a rank-four projector. Process code supplies the
momentum and owns the physical meaning of the source.

### `framework/tensors/OrbitalTensor.cuh` — Shared

Builds bare covariant P- and D-wave orbital objects from a parent momentum and
relative daughter momentum. It also contracts the bare rank-four G-wave STF
tensor directly with a rank-two source, returning the required rank-two
result without allocating a 256-component object. Barrier factors, orbital
normalization, and complete spin couplings remain outside these functions.

### `framework/tensors/BarrierFactor.cuh` — Shared

Defines the project Blatt-Weisskopf normalization, units, default radius, and
the implemented `L=0,1,2,3,4` functions. It returns zero for an unsupported
orbital momentum or invalid radius. The nominal radius lives here so dynamics
does not create a reverse dependency into process code.

### `docs/TENSOR_CONVENTIONS.md` — Shared physics contract

Records the stored-index and metric conventions, projector and bare-STF
formulae, the direct G-wave contraction, barrier separation, and the selected
Condon-Shortley/Racah normalized-CG boundary for future high-spin process
Waves.

### 5.3 Reusable dynamics

### `framework/dynamics/Kinematics.cuh` — Shared

Provides process-independent two-body breakup momentum `Q^2` and `Q` from
three invariant masses squared. It controls the below-threshold real-momentum
policy used by the reusable running-width functions.

### `framework/dynamics/Propagators.cuh` — Shared

Contains identity-free line-shape formulae:

- one shared relativistic Breit-Wigner denominator for a supplied width;
- generic relativistic running width and Breit-Wigner;
- constant-width Breit-Wigner;
- analytic equal-mass two-body phase space with complex continuation;
- subtracted effective Flatte line shape;
- nominal-mass two-body width shape;
- scalar S+D total running width and propagator.

Every function receives explicit physical inputs and knows no Resonance ID or
registered Wave. In particular, a two-body running width receives `L`
explicitly because `q^(2L+1)` and the barrier factor belong to denominator
physics.

### `framework/dynamics/TabulatedFunction.cuh` — Shared

Defines a lightweight uniform-grid view with host/device clamped linear
interpolation. It owns no memory and contains no particle names. The process
that owns a table remains responsible for building, uploading, and assigning
physical meaning to its sampled values.

### `framework/dynamics/PropagatorRegistry.cuh` — Shared

Defines the compact device enum, `PropagatorParameters`, and
`evaluate_propagator` dispatch used in kernels. It is a numerical registry,
not the user-configuration compiler. Mapping JSON strings and deciding which
fields may float are left to `process/PropagatorCompiler.cu`. Each compiled
descriptor carries the nominal daughter masses and barrier radius required by
its denominator, so Fit, Projection, and Post cannot supply inconsistent
channel arguments.

### 5.4 Generic amplitude and likelihood algebra

### `framework/amplitude/IntensityEngine.cuh` — Shared

Owns representation-independent coherent algebra:

- `component_pair_count` and `component_pair_index` define the common packed
  upper-triangle ordering;
- `coherent_intensity_from_terms` is the direct Term reference used for
  decomposition and regression tests;
- `coherent_intensity_from_waves` contracts already aggregated complete-Wave
  coefficients on the Fit hot path;
- `term_pair_component` returns a diagonal contribution or the complete
  forward-plus-reverse signed interference for one Term pair.

The code knows only dense Term descriptors, complex coefficients, and a Wave
Gram matrix. It does not construct an event or propagator.

### `framework/likelihood/Likelihood.h` — Fit, reusable

Implements accepted-MC normalization and signed unbinned log-likelihood
arithmetic. `monte_carlo_normalization` requires finite nonnegative event
intensities and a positive finite mean. `log_likelihood_contribution` evaluates
`coefficient * sum(log(I)-log(N))` and rejects zero, negative, or non-finite
probability inputs. Sample loading and GPU intensity evaluation stay outside
this header.

### 5.5 Generic model parser

### `framework/model/Model.h` — Shared

Declares the persistent configuration types: parameter, Resonance, coupling,
Term, and complete model definitions. Process dynamics are stored as opaque
canonical JSON text in each Term. The header also exposes file/in-memory
parsing, stable model-signature generation, and coupling/reference names.

### `framework/model/Model.cpp` — Shared

The implementation is a strict boundary parser. Its major blocks are:

- typed required-member helpers and unknown-field rejection;
- stable-ID validation and duplicate-ID detection;
- parameter parsing, including transform, step, and bound semantics;
- coupling parsing, including legal mode/reference combinations, positive
  phase-reference magnitude, and a nonzero fixed scale-and-phase reference;
- complete Resonance/Term parsing while leaving `dynamics` opaque;
- the requirement of exactly one active scale-and-phase reference globally;
- canonical JSON storage and deterministic FNV-1a model signature.

It does not recognize GVV propagator strings, Wave IDs, or dynamics types.

### 5.6 Generic fit services

### `framework/fit/FitConfig.h` — Fit

Declares the typed run configuration: weighted background samples, input
files, minimizer options, output directories/tag, and file-name helpers. The
helpers centralize the four tag-derived Fit product names.

### `framework/fit/FitConfig.cpp` — Fit

Strictly parses `fit.json`, rejects unknown keys and unsafe tags, validates
signed coefficients and minimizer ranges, and applies explicit defaults only
where the format defines them. It performs this work before ROOT files are
opened or GPU memory is allocated.

### `framework/fit/FitEngine.h` — Fit, reusable

Declares the process-neutral fitting API:

- `FitParameterSpec` carries initial value, step, optional bounds, and
  randomized-start policy;
- `FitOptions` carries multistart and Minuit policy;
- `FitAttempt` records a complete start result and diagnostics;
- `FitSummary` records all attempts and the selected best one;
- `FitObjective` is the only connection to a process likelihood.

### `framework/fit/FitEngine.cpp` — Fit, reusable

Implements the multistart TMinuit driver. Important blocks are:

- a narrow global callback bridge required by TMinuit's C-style FCN API;
- input and unique-name validation;
- deterministic random starts for adjacent complex pairs and log magnitudes;
- MIGRAD/HESSE execution and complete state extraction;
- acceptance checks for statuses, covariance quality, EDM, finite state, and
  errors;
- deterministic best-attempt comparison at nearly equal NLL.

The callback bridge makes one `run_multistart_fit` invocation intentionally
single-fit/non-reentrant. No process type enters the implementation.

### `framework/fit/FitOutput.h` — Fit

Declares the human report context and a process detail callback. The callback
lets the reusable writer include active physical GVV values without making the
writer depend on `process/`.

### `framework/fit/FitOutput.cpp` — Fit

Writes the readable report: provenance, sample roles, minimizer configuration,
all starts, selected diagnostics, ordered free parameters, process-supplied
active physical details, covariance, and correlation. It validates dimensions
before truncating the output. This text format is explicitly not a machine
contract. The structure and process-detail callback are reusable, but the
current report heading is still GVV-specific and must be neutralized when the
framework is converted to another final-state project.

### `framework/fit/FitState.h` — Shared between Fit and Calculation

Declares the schema-versioned machine handoff containing the output tag,
configuration provenance, model compatibility fields, the selected
`FitAttempt`, and ordered free-parameter specifications.

### `framework/fit/FitState.cpp` — Shared between Fit and Calculation

Serializes and restores the fitted-state JSON. The reader validates safe tags,
ordered unique names, finite values, positive steps, nonnegative errors,
bounds, value containment, covariance dimensions, nonnegative diagonal, and
symmetry. It does not contain the process model; downstream code must compare
the stored model signature and parameter order to a freshly compiled model.

## 6. GVV process layer

### 6.1 Event and kinematic construction

### `process/OmegaDecayModel.cuh` — Shared Fit/Calculation

Owns the process parameters and scalar dynamics shared by the event current
and omega-width integration:

- nominal omega, rho, and pion masses and widths;
- `RhoBWRParameters` for the rho and two vertex radii;
- the explicit rho0/rho+/rho- charge-channel mapping to nominal daughter and
  bachelor pion masses;
- `omega_rho_isobar_factor`, which calls the reusable two-body P-wave `BWR`
  and applies the two vertex barrier factors;
- `coherent_omega_rho_factor`, the one host/device implementation of the three
  rho-pairing sum.

### `process/ProcessKinematics.cuh` — Shared Fit/Calculation

Constructs the complete event-level `omega -> 3pi` current. Its important
blocks are:

- `omega_geometric_current`, including metric lowering under the common
  Levi-Civita convention;
- `build_omega_decay_current`, which derives the three pair invariants, calls
  the shared coherent rho factor, and keeps real geometry separate from the
  complex dynamics.

This file is process-specific even though it uses reusable dynamics.

### `process/ProcessEvent.cuh` — Shared Fit/Calculation

Defines the device views consumed by every complete Wave:

- `GVVBarrierParameters` carries production and X-decay radii;
- `GVVEventKinematics` builds both omegas, X, psi, relative omega momentum,
  and two omega currents from the seven particles;
- `GVVDeviceMomenta` is the lightweight pointer bundle passed to CUDA kernels.

It intentionally knows nothing about ROOT branch APIs or sample roles.

### 6.2 Complete Wave files

### `process/waves/Scalar00.cuh` — Shared Fit/Calculation

Implements the complete registered `0++(00)` numerator. It contracts the two
omega geometric currents to spin zero and multiplies the result by the metric
production tensor.

### `process/waves/Scalar22.cuh` — Shared Fit/Calculation

Implements the existing scalar `0++(22)` basis. It constructs the X-decay
D-wave orbital tensor, applies the `L=2` barrier factor, contracts it with the
two omega currents, and multiplies by the metric production tensor. The file
does not implement a `2++` Wave.

### `process/waves/Pseudoscalar11.cuh` — Shared Fit/Calculation

Implements the complete `0-+(11)` basis. It builds the radiative P-wave
Levi-Civita production tensor, the X-decay P-wave orbital vector, the
omega-spin-one Levi-Civita tensor, and their scalar decay contraction, with
the production and decay `L=1` barriers.

All three Wave functions are pure device numerator functions. Resonance
propagators, couplings, Term indices, and sample storage are absent by design.

### 6.3 Wave catalogue and process-model compilation

### `process/WaveRegistry.cuh` — Shared Fit/Calculation

This header has two responsibilities:

1. the device enum for registered complete GVV Waves;
2. the single `gvv_wave_tensor` device dispatch into `process/waves/` plus the
   small host metadata catalogue declaration.

It does not implement the common photon projection/contraction; that belongs
to `ProcessAmplitude.cuh`.

### `process/WaveRegistry.cu` — Shared Fit/Calculation

This is only the host registration table: stable Wave ID, JPC, LaTeX label,
coherence class, and device type. Adding a Resonance on an existing Wave must
not change this file. Adding a new Wave requires exactly one host record here
and a matching device dispatch entry in the header.

### `process/ProcessModel.h` — Shared Fit/Calculation

Defines dense `TermSpec`, coupling policy codes, compiled Resonance/Term
metadata, generic propagator parameter bindings, and `GVVCompiledModel`.
Keeping these records outside the registries prevents Wave code from owning
propagator or model-compilation policy.

### `process/PropagatorCompiler.h/.cu` — Shared Fit/Calculation

Owns the exact GVV Resonance contract. It maps one propagator string and its
named JSON parameters to a self-contained framework descriptor, validates
positive widths, running-width pole thresholds, transforms, and supported
orbital momentum, and emits both report metadata and generic fit bindings.
This is the only process file that needs a propagator-specific compiler branch.

### `process/ModelCompiler.h/.cu` — Shared Fit/Calculation

Owns complete active-model assembly: process ID, active Term dynamics,
inactive-only Resonance pruning, independent Resonance compilation, Wave-slot
assignment, coupling policies, and one reference per coherence class. It
contains neither Wave formulae nor propagator formulae.

### 6.4 Common process contraction

### `process/ProcessAmplitude.cuh` — Shared Fit/Calculation

Contains process-wide amplitude rules that apply to every registered Wave:

- `gvv_photon_projector` builds the radiative photon polarization projector;
- `gvv_wave_contraction` applies the adopted psi-polarization average, metric
  factors, and Wave-pair contraction;
- `gvv_cal_F` evaluates two registered Wave tensors and contracts them for one
  event.

A new numerator Wave normally leaves this file unchanged. Modify it only when
the common GVV polarization convention itself changes.

### 6.5 Omega running-width service

### `process/OmegaWidthTable.h` — Shared Fit/Calculation

Declares the omega propagator wrapper and the owning `OmegaWidthTable` class.
The wrapper interpolates a generic `ctpwa::TabulatedFunctionView` and delegates
the denominator to the reusable `BW_from_width` function. Separate host and
device views allow the same table to be inspected on the host and used inside
Term kernels.

### `process/OmegaWidthTable.cu` — Shared Fit/Calculation

Numerically integrates the coherent three-pion rho-isobar model over a Dalitz
grid, normalizes it at the omega pole, builds an invariant-mass-squared lookup
table, and uploads it to managed device memory. It calls the same
`coherent_omega_rho_factor` as the event current rather than maintaining a
second rho-isobar expression. The named `OmegaWidthTableConfig` centralizes the
table range, resolution, Dalitz binning, and explicit clamped extrapolation
policy. These remain process support parameters rather than model.json fields.

### 6.6 CUDA evaluation

### `process/TermEvaluator.cuh` — Shared Fit/Calculation

Declares four numerical entry points:

- `CalGVVFmatrix`: cache all active Wave-pair contractions;
- `CalGVVPDF`: optimized total intensity after exact Wave-slot aggregation;
- `CalGVVComponentBatch`: per-event packed Term pairs for Projection;
- `CalGVVComponentIntegrals`: GPU-reduced packed Term integrals for Post.

The declarations expose runtime sizes and never assume the nominal number of
Terms or Waves.

### `process/TermEvaluator.cu` — Shared Fit/Calculation

Implements the event-to-intensity CUDA path. Important blocks are:

- **event decoding**: converts `[px,py,pz,E]` arrays into device `FV` objects
  and constructs `GVVEventKinematics`;
- **F-matrix kernel**: evaluates all active registered Wave pairs once;
- **common omega factor**: combines both coherent rho factors and both omega
  running-width propagators;
- **Term coefficient**: multiplies coupling, X propagator, and common omega
  factor;
- **fit kernels**: aggregate fully evaluated Terms by exact dense Wave slot and
  call the `O(W^2)` coherent contraction;
- **Projection kernels**: evaluate Term coefficients for a selected event
  slice and emit packed diagonal/interference values;
- **Post reduction kernel**: assigns one block to each packed pair and reduces
  over a bounded event batch, accumulating serialized batches without an
  event-by-pair matrix.

Each public entry point owns launch-error and synchronization diagnostics. The
physics algebra itself remains in framework or process headers.

### 6.7 ROOT sample boundary

### `process/SampleLoader.h` — Shared Fit/Calculation

Declares the fixed seven-particle GVV ROOT contract, optional input component
order, and `GVVSample`. `GVVSample` owns host/device momenta, cached F matrix,
Wave-coefficient workspace, and intensity buffer. It is deliberately neutral
about whether the sample is data, background, accepted MC, or truth MC.

### `process/SampleLoader.cu` — Shared Fit/Calculation

`Load()` opens the ROOT file, binds the configured `Pwa` branches, converts
the component order once, and stores contiguous host arrays. `UploadAndBuildF()`
allocates managed device storage, copies momenta, sizes all workspaces from the
compiled active model, and caches the F matrix. ROOT schema checks occur here
once; inner event loops trust the loaded sample.

### 6.8 Fit parameter binding

### `process/ParameterMapping.h` — Shared Fit/Calculation

Declares the only translation between a generic ordered fit vector and mutable
GVV state. Each binding pairs a generic `FitParameterSpec` with a target type
and a dense Term or Resonance index.

### `process/ParameterMapping.cu` — Shared Fit/Calculation

The implementation has three responsibilities:

- build deterministic coupling bindings and append the generic propagator
  bindings already emitted by `PropagatorCompiler`;
- apply flat values, exponentiating positive physical quantities stored in log
  coordinates;
- write active Waves, Terms, couplings, Resonances, and fixed/free physical
  values into the human Fit report.

No other file should independently reconstruct Minuit ordering. Post
Calculation calls the same layout builder and compares it with `fit_state`.

### 6.9 Fit-time process orchestration

### `process/FitLikelihood.h` — Fit, with a narrow Projection interface

Declares the stateful GVV likelihood service. It owns sample roles, the host
compiled model, omega table, device model arrays, and reusable Projection
component scratch. It exposes a small read/evaluate interface to
`ProjectionWriter` without exposing allocation details.

### `process/FitLikelihood.cu` — Fit

The control flow is:

1. load normalization MC, data, and any number of signed backgrounds;
2. `Prepare()` builds/uploads the omega table and model, uploads samples, and
   caches every F matrix;
3. each `LogLikelihood()` synchronizes mutable model state, computes the
   accepted-MC normalization, and sums data/background log contributions;
4. Projection calls evaluate the fitted normalization-MC total once and fetch
   direct Term-pair components in bounded batches.

This file does not own TMinuit policy, parameter layout construction, or any
ROOT output tree.

### 6.10 Fit-to-plotting serialization

### `process/ProjectionWriter.h` — Fit

Declares one operation, `write_gvv_projection`, with only the fitted
`FitLikelihood`, target file, tag/signature, and selected-fit provenance. The
narrow interface keeps output details out of the likelihood class.

### `process/ProjectionWriter.cu` — Fit

This is the complete schema-version-3 projection writer. Its important blocks
are:

- host reconstruction of final-state/composite four-vectors;
- all pion-pair and composite masses;
- the photon polar angle and complete omega1 production direction in the X
  helicity frame;
- the complete direction and unnormalized magnitude of each oriented
  three-pion decay-plane normal in its omega helicity frame;
- all six pion polar cosines and the wrapped difference of the two local
  decay-plane azimuths;
- bounded-batch direct Term-pair evaluation and event-level closure checking;
- accepted-MC total, JPC-group, and symmetric Term-component weights under one
  full-model normalization;
- selected-data and dynamically indexed signed-background event trees;
- dynamic component, group, and background maps;
- model/fit provenance and aggregate closure metadata.

The scratch target is 64 MiB, converted to an event capacity using the actual
number of Term pairs and coefficient bytes. The writer is GVV-specific and is
part of the replacement boundary for a new final state.

## 7. Post Calculation

### `post/calculation/ComponentEvaluator.h` — Calculation

Declares the GPU-backed evaluator shared by all current Post observables. It
owns generated-truth and selected samples, the compiled model and binding
layout, the omega table, device model arrays, one bounded Term-coefficient
workspace, and one packed integral buffer.

### `post/calculation/ComponentEvaluator.cu` — Calculation

The constructor loads and prepares both MC samples and allocates a batch of at
most 4096 events. `Evaluate()` copies the fitted values into a fresh compiled
state, uploads it, and reduces every packed Term pair separately for truth and
selected MC. `ValidateTotal()` independently evaluates the optimized total PDF
and checks its integral against the pair sum for both samples.

This class reconstructs amplitudes but does not define fit fractions,
efficiencies, covariance propagation, or output formats.

### `post/calculation/PostCalculation.cu` — Calculation

This is the Post numerical application. Its major blocks are:

- **contract validation**: load `FitState`, compile the exact model, compare
  the model signature, and compare every free-parameter name in order;
- **observable construction**: Term fit fractions and efficiencies, all pair
  interference fractions, total efficiency, JPC-group quantities, and
  cross-group interference;
- **closure tests**: reconstruct unity from Term pairs and separately from
  group plus cross-group terms;
- **finite-difference Jacobian**: use central steps away from bounds and
  resolvable one-sided steps near bounds;
- **covariance propagation**: calculate the complete observable covariance and
  per-observable errors as `J V J^T`;
- **output writers**: human TXT, ROOT observable/metadata trees plus covariance
  and correlation matrices, and a LaTeX fit-fraction table.

The output directory is fixed to `post/calculation/results/`; the fit tag names
the products. Fit covariance is propagated, but MC statistical and systematic
uncertainties are outside the current implementation.

## 8. Post Plotting

### `post/plotting/GVVPlotUtils.h` — Plotting

This header owns only common Projection data services:

- the observable enum/specification type supplied by individual macros;
- the common BESIII base ROOT style and project-root path resolution;
- schema-v3 and required-branch validation;
- dynamic component/group map readers;
- event branch binding and exchange-symmetric observable filling;
- construction of data, signed-background, fitted-signal, total, group, and
  optional diagonal-Term histograms;
- simple Pearson chi-square diagnostics.

It reads no `model.json` and has no built-in Resonance list. Component plots
use diagonal `weight_component[i][i]` only, so they are not expected to sum to
the coherent total when interference is present. It deliberately contains no
plot catalogue, binning table, axis formatting, canvas, legend, or output
writer.

### `post/plotting/GVVAngularMoments.h` — Plotting

Provides the Legendre recurrence, even/odd moment weights, mass-binned
data-minus-background and fitted-MC histogram construction, and the moment
chi-square diagnostic. It contains no moment-order list, axes, styles, canvas,
annotations, or output writer; those choices belong to the even and odd
macros.

### `post/plotting/draw.sh` — Plotting, Infrastructure

This is the user plotting entry point. It accepts exactly one projection ROOT
file, loads the project ROOT environment, derives the tag from
`projection-<tag>.root`, creates `post/plotting/results/`, and invokes each of
the six macros in ROOT batch mode. It neither requests a GPU nor submits a
Slurm job.

### Standalone plotting macros

Each macro matches its ROOT-callable filename and can be executed directly.
Its function signature exposes the default Projection and output prefix, and a
clearly delimited `User configuration` preamble owns the complete plot
configuration and presentation:

| File | Figure |
|---|---|
| `post/plotting/macros/Draw_projection.cxx` | main `3 x 2` exchange-symmetric kinematic projections with dynamic coherent JPC groups |
| `post/plotting/macros/Draw_projection_components.cxx` | diagonal Term-component diagnostic, intentionally excluding interference curves |
| `post/plotting/macros/Draw_polarization.cxx` | candidate-combined decay-plane polar/azimuthal angles and exchange-symmetric plane-angle difference |
| `post/plotting/macros/Draw_omega_decay_checks.cxx` | candidate-combined pion helicity cosines and pion-pair invariant-mass checks |
| `post/plotting/macros/draw_angular_moments.cxx` | physical even `P0/P2/P4/P6` omega-angle moments |
| `post/plotting/macros/draw_angular_moments_odd.cxx` | ordered-omega odd `P1/P3/P5` diagnostic |

Change a figure's default paths, variables, bins, axes, colors, canvas, draw
options, legend, or annotations in that preamble. Relative runtime paths are
interpreted from the project root; absolute paths are preserved. Change a
shared header only for a genuinely common Projection contract,
histogram-building rule, moment formula, path rule, or base style.

### `post/README.md` — Calculation and Plotting user contract

This focused guide documents the independence of the two Post modules, their
input/output contracts, build/submission commands, and the meaning of the
selected/truth efficiency samples. It is user documentation rather than a
runtime input.

## 9. Verification inventory

The ordinary test suite is deliberately runnable without a CUDA device. The
three GPU runtime tests are separate because an IHEP login node may provide
`nvcc` while exposing no GPU. Compile on the login node; execute
`make check-gpu` only inside a Slurm GPU job.

### Login-node-safe tests: `make check`

| File | What it guards |
|---|---|
| `tests/test_dynamics.cu` | device-complex phase convention, two-body kinematics, legacy and higher-L barrier normalization, unified running-width equivalence, shared BW denominator, threshold continuation, Flatte subtraction, nominal-mass rho-isobar equivalence, and compilation of the omega device path |
| `tests/test_gvv_amplitude.cu` | compile-time integration of registered Waves with the common GVV amplitude contraction |
| `tests/test_propagator_registry.cu` | propagator device dispatch, nominal line-shape contracts, host omega-width interpolation/configuration/convergence, and the omega wrapper's use of the shared BW denominator |
| `tests/test_propagator_compiler.cu` | all supported GVV propagator JSON contracts, self-contained omega-omega channel context, generic free-ratio bindings, and invalid width/threshold/field rejection |
| `tests/test_fit_parameters.cu` | deterministic GVV free-parameter layout, coupling/reference parameterizations, log-ratio bindings, and state application |
| `tests/test_fit_config.cpp` | strict `fit.json` parsing and tag-derived output naming |
| `tests/test_fit_output.cpp` | presence of the required sections in the complete human Fit report |
| `tests/test_fit_state.cpp` | fitted-state round trip and rejection of unsafe tags, duplicate names, asymmetric covariance, and negative covariance diagonal |
| `tests/test_fit_engine.cpp` | generic multistart Minuit execution, convergence selection, values, errors, and covariance extraction on a small objective |
| `tests/test_model.cpp` | nominal generic model parsing, canonical signature input, coupling/reference rules, and key invalid-model diagnostics |
| `tests/test_wave_registry.cu` | active GVV compilation, dense ordering, inactive-Term Resonance pruning, adding a Term without fixed counts, propagator parameter contracts, orbital-L range, and dynamics validation |
| `tests/test_likelihood.cpp` | accepted-MC normalization and signed log-likelihood arithmetic, including invalid numerical inputs |

Some `.cu` files in this group compile device-callable code but their test
executables do not require an allocated CUDA device at runtime.

### CUDA-device regressions: `make check-gpu`

| File | What it guards |
|---|---|
| `tests/test_tensor_building_blocks.cu` | named Lorentz contractions, explicit spin-two projection, symmetry/transversality/trace/idempotence, reduced versus explicit rank-four G-wave contraction, redundant outer-projector removal, parity and power scaling, and analytic rest-frame probes |
| `tests/test_gvv_wave_numerics.cu` | complete-Wave numerical finiteness, rotations, Bose symmetry, omega-current/projector/orbital/photon transversality, D-wave trace, Gram symmetry/positive semidefiniteness, cross-class orthogonality, and nonzero registered-Wave diagonals on a non-collinear physical fixture |
| `tests/test_intensity_equivalence.cu` | direct Term contraction versus optimized Wave aggregation, packed-component closure, shared Wave slots, near-cancelling coefficients, several Term counts, nonzero batch offsets, and GPU pair reduction versus host sums |

The tensor test protects reusable low-level identities and is extended only
when those building blocks change. The complete-Wave test protects the
physics identities of registered bases, while the intensity test protects the
algebraic equivalence of the Fit, Projection, and Post representations. The
last two must be extended when a new Wave changes the active numerical basis.

## 10. Where a change belongs

| Intended change | Primary files | Files that normally stay unchanged |
|---|---|---|
| Add/remove/disable Resonance on an existing Wave | `config/model.json` | all production C++/CUDA, Projection, Post, plotting |
| Change run samples, sideband prescription, starts, or output tag | `config/fit.json` | model and amplitude code |
| Add a complete GVV Wave | new `process/waves/*.cuh`, `WaveRegistry.cuh/.cu`, tests | generic model, Fit engine, likelihood, parameter counts, plotting maps |
| Add a reusable tensor primitive | `framework/tensors/`, focused tests | process compiler unless the Wave uses it |
| Add a reusable propagator formula | `framework/dynamics/`, `PropagatorCompiler.*`, tests, model documentation | Wave registry, `ParameterMapping`, Fit likelihood, Projection, Post |
| Change the GVV ROOT input schema | `SampleLoader.*`, application branch contract, relevant scripts/docs/tests | generic framework |
| Change process-wide photon/polarization contraction | `ProcessAmplitude.cuh`, GPU physics tests | individual Resonance definitions |
| Change Fit minimizer policy | `FitConfig.*`, `FitEngine.*`, `fit.json`, tests | process tensors and Post plotting |
| Change projection observables/schema | `ProjectionWriter.*`, plotting readers/macros, schema tests/docs | likelihood or Minuit engine |
| Change fit fractions/efficiencies/error propagation | `post/calculation/*` | Fit likelihood and projection plotting |
| Change figure selection/style | `post/plotting/*` | Fit and Post Calculation |
| Convert to another final state | replace `process/`, adapt app/Post/plotting; reuse `framework/` | do not add the second topology as branches throughout GVV code |

## 11. Generated products and source ownership

For clarity, the following paths are outputs, not editable source inputs:

```text
bin/                         Fit.exe and Post.exe
build/                       objects, dependency files, test executables
results/                     Fit report, fit state, projection ROOT
runlog/                      Fit/Post/test Slurm logs
post/calculation/results/    numerical downstream products
post/plotting/results/       PDF/EPS figures
RootSet/                     untracked analysis input ROOT files
```

When debugging a numerical disagreement, start from the bridge appropriate to
the workflow: `fit_state` plus exact `model.json` for Post Calculation, or the
projection ROOT schema for Plotting. Do not parse the human report to recreate
program state.

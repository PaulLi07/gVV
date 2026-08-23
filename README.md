# gVV

`gVV` is a CUDA/ROOT covariant-tensor partial-wave analysis project for

```text
psi(2S) -> gamma X
             X -> omega omega
       omega -> pi+ pi- pi0
```

It builds a coherent amplitude from a JSON model, normalizes the event
intensity with accepted Monte Carlo, performs a multistart Minuit fit, and
provides independent numerical and plotting stages after the fit. The current
development version is `gVV v1.1.0-dev`.

The code is organized around one practical rule:

- adding, disabling, or deleting a Resonance contribution that uses an
  existing Wave is a `config/model.json` operation;
- adding a genuinely new covariant basis requires one new complete Wave under
  `process/waves/` and registration in `WaveRegistry`;
- adapting the project to another final state reuses `framework/` and replaces
  the explicit process layer instead of adding channel switches to gVV.

## Physics model

### Decay and event convention

Each event contains the bachelor photon and two reconstructed omega candidates,
with each omega represented by a `pi+ pi- pi0` triplet. Input four-vectors are
stored as `(px, py, pz, E)` in GeV. The amplitude implementation assumes the
`psi(2S)` or `e+e-` center-of-mass frame with the beam along the `z` axis; the
transverse `psi(2S)` polarizations are the `x` and `y` components used in the
common Wave contraction.

For each omega decay, the process layer constructs the geometric current

```text
E^mu = epsilon^{mu nu lambda sigma}
       p(pi+)_{nu} p(pi-)_{lambda} p(pi0)_{sigma}
```

and multiplies it by the coherent sum of the three rho-isobar configurations,
including their P-wave barrier factors and running rho propagators. The omega
line shape uses the tabulated three-body running width implemented by
`OmegaWidthTable`. The rho calls the reusable analytic two-body `BWR`, while
the omega interpolates `Gamma_omega(s)` and calls the same framework-level
Breit-Wigner denominator used by the other line shapes. One shared host/device
rho-isobar implementation is used by both the event current and the omega-width
integration.

All pion masses used by the scalar rho-isobar dynamics are nominal particle
masses. For `rho0`, `rho+`, and `rho-`, the shared decay helper selects the
corresponding nominal daughter pair and nominal bachelor pion. Only `s_omega`
and the appropriate event `s_pipi` remain event dependent; reconstructed
single-pion `p_i^2` values do not enter the rho line shape or either isobar
barrier factor.

### Resonance, Wave, and Term

The model separates three concepts that change at different rates:

- **Resonance**: one propagator instance and its physical parameters;
- **Wave**: one complete process-specific covariant tensor, including the
  radiative-production and `X -> omega omega` angular numerator;
- **Term**: one Resonance combined with one Wave and one complex coupling.

A Resonance is not a Wave, and a propagator is not embedded in a Wave. Several
Resonances can reuse the same Wave, and one Resonance may be used by several
Terms when the intended line shape is common to those contributions.
The Resonance ID is the propagator-instance identity: Terms that name the same
ID share one mass, width, and any other line-shape parameters. Distinct IDs
remain independent even when they select the same reusable `propagator` type.

For Term `t`, the event-dependent complex scalar coefficient can be written
schematically as

```text
C_t(x; theta) = g_t(theta) D_r(s_X; theta)
                D_omega(s_omega1) D_omega(s_omega2)
                R_rho(omega1) R_rho(omega2),
```

while the complete Wave tensor is `U_{w(t)}(x)`. The common polarization sum
and tensor contraction are cached as a real Wave Gram matrix

```text
F_ab(x) = polarization_sum[ U_a(x) U_b(x)* ].
```

The coherent event intensity is

```text
I(x; theta) = sum_{t,u} C_t(x; theta) C_u(x; theta)*
                         F_{w(t),w(u)}(x).
```

For the Fit hot path, fully evaluated Term coefficients that share the exact
same Wave slot are aggregated as

```text
B_a(x; theta) = sum_{t: w(t)=a} C_t(x; theta),
I(x; theta)   = sum_{a,b} B_a B_b* F_ab.
```

This is an exact algebraic rearrangement, not an incoherent approximation. All
cross-Wave entries remain in the sum. It reduces the contraction cost from
`O(T^2)` to `O(T + W^2)` when many Resonances reuse a small Wave basis.

### Registered Waves

| Stable Wave ID | Meaning | Implementation |
|---|---|---|
| `gvv.scalar_00` | `0++`, `L=S=0` | `process/waves/Scalar00.cuh` |
| `gvv.scalar_22` | `0++`, `L=S=2` | `process/waves/Scalar22.cuh` |
| `gvv.pseudoscalar_11` | `0-+`, `L=S=1` | `process/waves/Pseudoscalar11.cuh` |
| `gvv.tensor_02_u1` | `2++`, `LS=02`, production covariant 1 | `process/waves/Tensor02U1.cuh` |
| `gvv.tensor_02_u2` | `2++`, `LS=02`, production covariant 2 | `process/waves/Tensor02U2.cuh` |
| `gvv.tensor_02_u3` | `2++`, `LS=02`, production covariant 3 | `process/waves/Tensor02U3.cuh` |
| `gvv.tensor_20_u1` | `2++`, `LS=20`, production covariant 1 | `process/waves/Tensor20U1.cuh` |
| `gvv.tensor_20_u2` | `2++`, `LS=20`, production covariant 2 | `process/waves/Tensor20U2.cuh` |
| `gvv.tensor_20_u3` | `2++`, `LS=20`, production covariant 3 | `process/waves/Tensor20U3.cuh` |
| `gvv.tensor_22_u1` | `2++`, `LS=22`, production covariant 1 | `process/waves/Tensor22U1.cuh` |
| `gvv.tensor_22_u2` | `2++`, `LS=22`, production covariant 2 | `process/waves/Tensor22U2.cuh` |
| `gvv.tensor_22_u3` | `2++`, `LS=22`, production covariant 3 | `process/waves/Tensor22U3.cuh` |
| `gvv.tensor_42_u1` | `2++`, `LS=42`, production covariant 1 | `process/waves/Tensor42U1.cuh` |
| `gvv.tensor_42_u2` | `2++`, `LS=42`, production covariant 2 | `process/waves/Tensor42U2.cuh` |
| `gvv.tensor_42_u3` | `2++`, `LS=42`, production covariant 3 | `process/waves/Tensor42U3.cuh` |

The Wave registry currently defines two phase-reference classes:

| `coherence_class` | Registered Waves | Nominal reference |
|---|---|---|
| `positive_parity` | all `0++` and `2++` Waves | `f0_1710_00`, positive real |
| `negative_parity` | the `0-+` Wave | `eta_1760_11`, fixed to the global scale and phase |

These names describe phase-reference bookkeeping for the two numerically
orthogonal parity blocks. They do not switch interference on or off: the
intensity engine still evaluates every cross-Wave Gram-matrix entry. Waves
that can interfere, such as the registered `0++` and `2++` bases, must remain
in the same class and share one arbitrary phase convention. A future Wave must
be classified from its event-level Gram-matrix cross terms, not from parity
alone.

`gvv.scalar_22` remains the scalar `0++(22)` basis; it is not a spin-two
Wave. The nominal model currently activates only the three `LS=02` tensor
covariants, for `f2(1565)` and `f2(1810)`. The higher-orbital tensor Waves are
registered for explicit future model tests but do not enter a Fit until selected
by an active Term in `config/model.json`.

### Accepted-MC normalization and signed likelihood

The accepted normalization-MC sample defines

```text
N(theta) = (1 / N_MC) sum_k I(x_k; theta).
```

For data and configured signed background samples, the effective unbinned
log-likelihood is

```text
ln L_eff(theta)
  = sum_{data} ln[I(x; theta) / N(theta)]
    + sum_b alpha_b sum_{x in b} ln[I(x; theta) / N(theta)].
```

The nominal two-dimensional sideband prescription is represented by
`alpha_SB1 = -0.5` and `alpha_SB2 = +0.25` in `config/fit.json`; the likelihood
code itself does not contain fixed SB1/SB2 logic. `Fit.exe` minimizes
`-ln L_eff`.

## Architecture at a glance

The repository has two intentionally separate systems.

### Amplitude Fit

```text
fit.json + model.json
        |
        v
generic parsers ----> GVV model compiler ----> dense active runtime model
                                                |
ROOT samples -> SampleLoader -> cached F_ab ----+----> FitLikelihood
                                                       |
                                             ParameterMapping + FitEngine
                                                       |
                         +-----------------------------+------------------+
                         |                             |                  |
                 fit report TXT                fit-state JSON     projection ROOT
                 (human reader)               (Post contract)    (Plot contract)
```

### Post-processing

```text
fit-state JSON (embedded model) + truth MC + selected MC
                              |
                              v
                       Post Calculation
                              |
                 TXT + ROOT + LaTeX observables

projection ROOT ------------------------------> Post Plotting
                                                  |
                                           PDF/EPS figures
```

Post Calculation and Post Plotting are independent. Calculation does not read
the human fit report or projection file. Plotting does not read the fit state,
model JSON, truth MC, or normalization MC.

### Layer boundary

- `framework/` contains reusable math, tensor, dynamics, model, likelihood,
  minimization, fit-result, and fitted-state components.
- `process/` contains the `psi(2S) -> gamma omega omega` event definition,
  omega decay model, complete Waves, model compiler, CUDA evaluation, ROOT
  sample mapping, likelihood orchestration, and projection schema.
- `app/` contains only executable-level Fit orchestration.
- `post/` contains the independent numerical and plotting consumers.

The dependency direction is one way: `framework/` never includes `process/`.
See [Architecture](docs/ARCHITECTURE.md) for the full object and call flow and
[Code reference](docs/CODE_REFERENCE.md) for the role of every production file.

## Repository layout

```text
gVV/
├── app/                     Fit executable glue
├── config/                  Runtime environment, fit config, and model config
├── framework/               Process-neutral reusable libraries
│   ├── amplitude/
│   ├── dynamics/
│   ├── fit/
│   ├── likelihood/
│   ├── math/
│   ├── model/
│   └── tensors/
├── process/                 GVV-specific event and amplitude implementation
│   ├── waves/               Complete registered GVV Waves
│   ├── WaveRegistry.*       Wave catalogue and device dispatch only
│   ├── PropagatorCompiler.* Resonance JSON and channel compilation
│   └── ModelCompiler.*      Active Term and dense runtime assembly
├── post/
│   ├── calculation/         Fit fractions, efficiencies, and covariance
│   └── plotting/            Projection and angular-moment figures
├── tests/                   Host and explicit GPU regression tests
├── docs/                    User and developer manuals
├── results/                 Generated Fit products
├── runlog/                  Generated Slurm logs
├── Makefile
├── submit_fit.sh
└── submit_post.sh
```

Generated data, ROOT files, binaries, results, logs, and figures are excluded
from Git by `.gitignore`.

## Installation and environment

### Obtain the source

```bash
git clone git@github.com:PaulLi07/gVV.git
cd gVV
```

The project is built in place; there is no separate install prefix or package
installation step.

### Default IHEP toolchain

The checked-in defaults target the IHEP AlmaLinux environment used by this
analysis:

- CUDA 12 under `/usr/local/cuda-12`;
- ROOT 6.32.02 built with GCC 11.4;
- a C++17-capable `nvcc`;
- the `nlohmann/json.hpp` development header on the compiler include path;
- GNU Make;
- Bash 4 or newer and GNU coreutils for the submission/plotting wrappers;
- Python 3 for JSON preflight in the Slurm wrappers;
- Slurm access to the project-approved `gpupwa` A100 resources.

Load the project environment from the repository root:

```bash
source config/gvv_env.sh
```

To use another compatible CUDA or ROOT installation, set the overrides before
sourcing the file:

```bash
export GVV_CUDA_ROOT=/path/to/cuda
export GVV_ROOTSYS=/path/to/root
source config/gvv_env.sh
```

`config/gvv_env.sh` changes only the current shell environment. It does not
edit a shell profile, install software, or select input data for `Fit.exe`.

## Input ROOT contract

The default `config/fit.json` expects:

```text
RootSet/
├── data.root
├── normalization_mc.root
├── SB1.root
└── SB2.root
```

Every Fit, truth-MC, and selected-MC file consumed by the current process layer
must contain a `Pwa` tree with seven `double[4]` branches:

```text
p4_pip1  p4_pim1  p4_pi01
p4_pip2  p4_pim2  p4_pi02
p4_gam
```

Each array is `(px, py, pz, E)` in GeV. The first and second pion triplets
define the two labeled omega candidates. Paths may be absolute or relative to
the project root when passed through the provided submission scripts.

Post Calculation additionally requires generated truth MC and selected MC.
The two samples must represent the same unweighted production, with the
selected sample being the accepted subset under the intended selection. The
current calculation forms efficiency ratios from their raw integrated sums;
it does not read generator weights or infer production-size corrections.

## Build

From the repository root:

```bash
source config/gvv_env.sh
make -j2
make -j2 post
make check
```

| Target | Product or action |
|---|---|
| `make` or `make fit` | Build `bin/Fit.exe` |
| `make post` | Build `bin/Post.exe` separately |
| `make tests` | Build the ordinary test executables |
| `make check` | Run the 13 login-node-safe tests |
| `make gpu-tests` | Compile the three explicit GPU runtime regressions |
| `make check-gpu` | Execute those regressions on an allocated CUDA device |
| `make clean` | Remove generated objects, executables, and dependency files |

On IHEP, compile `gpu-tests` on the login node if desired, but execute
`make check-gpu` only as the payload of a Slurm GPU job. Do not run GPU tests,
`Fit.exe`, or `Post.exe` directly on a login node. The repository does not
provide a separate GPU-test submission wrapper; use the project-approved Slurm
resource combination when performing developer validation.

## Configure the model and fit

Two handwritten JSON files control a run; no Python model generator is used.

### `config/model.json`

This is the single amplitude-model description. It defines Resonance
instances, propagator parameters, Terms, Wave IDs, coupling modes, reference
roles, and the active/inactive selection. Runtime array sizes and the Minuit
parameter vector are compiled from its active content.

Supported propagator parameters are controlled in the same objects. A mass or
width can be fixed or floated with an initial value, step, and explicit
physical bounds when its propagator contract permits it. Sharing is determined
only by the Resonance ID, never by the propagator-type string.

### `config/fit.json`

This defines one run: the model path, data and accepted-MC paths, any number of
signed background samples, multistart/Minuit policy, output directories, and a
single output tag.

The C++ loaders are the executable contract and reject unknown fields,
duplicate IDs, invalid reference conventions, unsupported propagator
parameters, and missing required inputs before the GPU fit begins.

See [Model configuration](docs/MODEL_CONFIGURATION.md) for the complete field
contract and step-by-step model editing procedures.

## Run the Amplitude Fit

Build `Fit.exe`, review both JSON files, and submit from the repository root:

```bash
./submit_fit.sh
```

The default is `config/fit.json`. To use another run configuration:

```bash
./submit_fit.sh path/to/fit.json
```

`submit_fit.sh` performs input preflight on the login node and then submits one
Slurm job. Its worker loads `config/gvv_env.sh`, records the assigned host/GPU,
and launches exactly one `Fit.exe` process with `srun`.

`output.tag` controls all Fit names. Reusing a tag intentionally overwrites
the existing products:

| Product | Default path | Consumer |
|---|---|---|
| Human fit report | `results/fit_result-<tag>.txt` | User only |
| Fitted state | `results/fit_state-<tag>.json` | Post Calculation |
| Projection | `results/projection-<tag>.root` | Post Plotting |
| Slurm log | `runlog/fit-<tag>.log` | User/debugging |

The report contains all fit attempts, selected convergence diagnostics,
active Waves/Terms/Resonances, free and fixed physical parameters, and full
covariance/correlation matrices. The fitted-state JSON contains the ordered
free-parameter state, parameter metadata, covariance, the complete formatted
model definition, and separate definition/implementation compatibility
signatures. This schema-version-2 file is the self-contained machine-readable
bridge; downstream code does not parse the human report or reopen
`model.json`.

The projection ROOT file uses schema version 3. It stores data, signed
backgrounds, fitted accepted-MC weights, complete Term-pair component weights,
JPC group weights, dynamic component/group/background maps, and provenance.
Its event record includes the complete omega-production direction, both
three-pion decay-plane-normal directions, pion helicity polar angles, and the
decay-plane analyser magnitudes needed for polarization studies. Projection
files from schema versions 1 and 2 are not accepted by the current plotting
reader.

## Run Post-processing

### Post Calculation

Build the numerical executable independently:

```bash
make -j2 post
```

Submit it with the fitted state, generated truth MC, and selected
normalization MC. The exact model is already embedded in the state:

```bash
./submit_post.sh \
  results/fit_state-initial.json \
  RootSet/truth_mc.root \
  RootSet/normalization_mc.root
```

Post recompiles the embedded definition, checks its definition signature, the
current GVV amplitude-implementation signature, and the exact free-parameter
order. It then integrates diagonal Term contributions and signed interference
in bounded GPU batches, builds Term and JPC-group fit fractions and
efficiencies, and propagates the Fit covariance by finite differences.

Products are overwritten for the same tag:

```text
post/calculation/results/post_result-<tag>.txt
post/calculation/results/post_result-<tag>.root
post/calculation/results/fit_fractions-<tag>.tex
runlog/post-<tag>.log
```

The reported observable errors contain propagated Fit-covariance uncertainty.
They do not include finite-MC integration uncertainty or systematic
uncertainties.

### Post Plotting

Plotting is a ROOT task driven only by the projection file:

```bash
post/plotting/draw.sh
post/plotting/draw.sh results/projection-<tag>.root
```

With no argument, the driver reads `results/projection-initial.root`; one
argument overrides that input. The script rejects additional arguments.
The script writes the main and component projections, a dedicated polarization
figure, omega-decay check distributions, and even/odd angular-moment figures
under `post/plotting/results/`. It discovers active Terms, JPC groups, and
background samples from the projection maps instead of using a fixed Resonance
list.

All standard plots follow one visual contract: data are black points,
Background is a gray hatched histogram, the original solid blue Total-fit
style is reserved, all coherent JPC groups use the same dashed line style with
distinct colors and the legend wording `coherent <JPC>`, and Term components
receive deterministic distinct styles from their Projection metadata.
Automatic vertical ranges include data errors and every drawn curve. The
component plot enlarges its `3 x 2` physics area, draws individual Term curves
with thin width-1 lines, and places a compact centered two-column legend in the
external right margin, following a conventional projection-plot layout. Every
other shared legend stays inside the first subplot, whose extra headroom
prevents overlap without compressing the other panels. See the
[plotting style guide](post/plotting/PLOTTING_STYLE.md) for the frame,
candidate-combination, layout, labeling, and review conventions.

Each ROOT macro is also a standalone plotting module. After loading the
environment, it can be executed directly with its default `initial` input and
output paths:

```bash
source config/gvv_env.sh
root post/plotting/macros/Draw_projection.cxx
```

Observable definitions, binning, axes, canvas layout, colors, legends, and
annotations are collected in the clearly marked `User configuration` block at
the top of the corresponding `.cxx` file. Its function signature also shows
the default Projection and output prefix. Runtime relative paths are resolved
from the project root, and absolute paths are accepted unchanged. The shared
headers retain only Projection I/O, histogram construction, moment arithmetic,
and the common base style.

See [Workflow](docs/WORKFLOW.md) for the complete operational sequence,
validation checklists, output contracts, and failure diagnosis.

## Modify the amplitude model

### Add, disable, or delete a Resonance contribution

When the Wave already exists, no C++/CUDA source change is required:

1. add the Resonance propagator instance to `resonances`;
2. add a Term that links it to a registered Wave and coupling;
3. preserve exactly one phase reference in every active coherence class
   (`positive_parity` and `negative_parity` in the current catalogue) and
   exactly one nonzero global `scale_and_phase` reference;
4. run model/unit validation before a production scan.

For a temporary scan, set `"active": false` on the Term. The inactive Term,
its coupling, and a Resonance referenced only by inactive Terms are pruned from
the compiled runtime model, Minuit layout, reports, projection maps, and Post
observables. This behavior does not depend on propagator type.

To remove a contribution permanently, delete its Term first, then delete the
Resonance only if no remaining active or inactive Term needs that ID. If the
Term was a reference, reassign the reference role before disabling or deleting
it.

Complete JSON examples and safe scan checklists are in
[Model configuration](docs/MODEL_CONFIGURATION.md).

### Add a new Wave

A new Wave is process code because it defines a new covariant numerator:

1. construct one complete tensor in `process/waves/<Name>.cuh` from reusable
   `framework/math`, `framework/tensors`, and `framework/dynamics` blocks;
2. add its device enum and dispatch in `process/WaveRegistry.cuh`;
3. add its stable ID, JPC, LaTeX label, physically justified coherence class,
   and enum at the single host registry in `process/WaveRegistry.cu`;
4. extend registry/model tests and the complete-Wave GPU numerical tests;
5. only then reference the new stable Wave ID from `model.json`.

Do not put a Resonance denominator or model-wide Term logic into a Wave. See
[Wave development](docs/WAVE_DEVELOPMENT.md) for physics invariants, code
templates, registration details, and the required validation matrix.
The reusable Lorentz contractions, spin-two projection, bare high-spin
orbital tensors, barrier separation, and normalized-CG convention are defined
in [Tensor conventions](docs/TENSOR_CONVENTIONS.md).

### Add a new propagator

A new line shape normally touches only three focused boundaries:

1. implement its identity-free formula and device fields under
   `framework/dynamics/`;
2. add its exact GVV JSON contract, channel context, metadata, and any generic
   fit binding in `process/PropagatorCompiler.*`;
3. add formula/compiler tests and document the JSON parameters.

Numerical changes to a registered Wave, propagator, Term assembly, or fit
binding must also bump the explicit `gvv-amplitude-contract-vN` implementation
identifier in `process/ModelCompiler.cu`. Fit and Post combine that identifier
with the declarative-model signature, preventing a current executable from
silently interpreting state produced by a different numerical convention.

`WaveRegistry`, `ParameterMapping`, `FitLikelihood`, Projection, and Post do
not require propagator-specific branches. The compiled descriptor already
contains its nominal daughter masses and barrier radius, so every numerical
consumer calls the same `evaluate_propagator(s, descriptor)` interface. See
[Model configuration](docs/MODEL_CONFIGURATION.md#adding-a-new-propagator) for
the complete checklist.

## Adapting the framework to another final state

The intended reuse boundary is not a multi-channel runtime switch. A new
decay-channel project should retain the process-neutral `framework/` libraries
and replace the channel-specific event representation, kinematics/currents,
complete Waves, Wave registry, process compilers, Term evaluator, ROOT sample mapping,
likelihood wrapper, and projection writer. It may reuse the generic Fit engine,
model concepts, propagator library, tensor building blocks, likelihood
arithmetic, report/state infrastructure, and output philosophy. The current
`FitOutput.cpp` still contains a GVV-specific report title, so a different
channel must neutralize or replace that heading even though the report writer,
process-detail callback, and fitted-state machinery remain reusable.

## Documentation map

| Document | Purpose |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | Complete layering, data flow, equations, runtime objects, and replacement boundary |
| [Workflow](docs/WORKFLOW.md) | Installation-to-results operator guide for Fit and Post-processing |
| [Code reference](docs/CODE_REFERENCE.md) | Responsibility of every production file and important internal code blocks |
| [Model configuration](docs/MODEL_CONFIGURATION.md) | Full JSON contract and Resonance/Term add-disable-delete procedures |
| [Wave development](docs/WAVE_DEVELOPMENT.md) | Building, registering, and validating a new complete Wave |
| [Tensor conventions](docs/TENSOR_CONVENTIONS.md) | Lorentz indices, spin projectors, bare orbital tensors, barriers, and normalized-CG boundary |
| [Post README](post/README.md) | Focused Post Calculation and Post Plotting contract |
| [Plotting style](post/plotting/PLOTTING_STYLE.md) | Unified curve roles, layouts, axis/frame naming, automatic ranges, and plot-review checklist |
| [Refactor log](docs/REFACTOR_LOG.md) | Historical implementation and verification record |

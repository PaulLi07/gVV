# gVV

gVV is a CUDA/ROOT implementation of a covariant-tensor partial-wave fit for

```text
psi(2S) -> gamma X
             X -> omega omega
       omega -> pi+ pi- pi0
```

The project evaluates coherent amplitudes event by event, normalizes the
intensity with accepted Monte Carlo, and fits resonance and coupling parameters
with a multistart Minuit workflow. The current development version is
`gVV v1.1.0-dev`.

## Physics model

A fit model is assembled from three concepts:

- **Resonance**: a propagator instance and its physical parameters;
- **Wave**: a complete process-specific covariant-tensor basis;
- **Term**: one Resonance combined with one Wave and one complex coupling.

The active Resonances and Terms are defined only in
[`config/model.json`](config/model.json). Within the registered Wave basis,
adding, removing, or disabling a resonance contribution does not require
changing C++/CUDA array sizes or parameter counts.

The currently registered Waves are:

| Wave ID | Quantum numbers | Implementation |
|---|---|---|
| `gvv.scalar_00` | `0++ (00)` | `process/waves/Scalar00.cuh` |
| `gvv.scalar_22` | `0++ (22)` | `process/waves/Scalar22.cuh` |
| `gvv.pseudoscalar_11` | `0-+ (11)` | `process/waves/Pseudoscalar11.cuh` |

`Scalar22` is the existing `0++ (22)` basis; it is not a `2++` Wave.

## Features

- Runtime model construction from JSON, with no hard-coded total number of
  Resonances, Terms, or fit parameters;
- reusable propagator, kinematics, tensor, likelihood, and fit components under
  `framework/`;
- explicit GVV event kinematics, Wave registration, amplitude contraction, and
  sample loading under `process/`;
- accepted-MC normalization of an unbinned coherent intensity;
- signed background samples for sideband-subtracted likelihoods;
- multistart MIGRAD/HESSE fitting with convergence and covariance selection;
- a complete human-readable fit report, a machine-readable fitted-state JSON,
  ROOT projection trees, and Slurm logs under one configurable output tag;
- independent Post Calculation and Post Plotting modules that consume only
  the documented fit outputs plus the MC samples required by the calculation.

## Requirements

The default build targets the IHEP AlmaLinux environment used by this analysis:

- CUDA 12 at `/usr/local/cuda-12`;
- ROOT 6.32.02 built with GCC 11.4;
- a C++17-capable CUDA compiler;
- GNU Make;
- Python 3 for submission-time JSON preflight;
- Slurm and an A100 allocation for the provided submission script.

The CUDA and ROOT locations can be overridden before loading the project
environment:

```bash
export GVV_CUDA_ROOT=/path/to/cuda
export GVV_ROOTSYS=/path/to/root
source config/gvv_env.sh
```

## Input data

Input ROOT files are intentionally not tracked by Git. The default
[`config/fit.json`](config/fit.json) expects:

```text
RootSet/
├── data.root
├── normalization_mc.root
├── SB1.root
└── SB2.root
```

Each file must contain a `Pwa` tree with the seven four-momentum branches

```text
p4_pip1  p4_pim1  p4_pi01
p4_pip2  p4_pim2  p4_pi02
p4_gam
```

stored in `(px, py, pz, E)` order. Alternative paths can be selected in
`config/fit.json`.

## Build and test

From the repository root:

```bash
source config/gvv_env.sh
make -j2
make check
```

Useful Make targets are:

| Target | Purpose |
|---|---|
| `make` or `make fit` | Build `bin/Fit.exe` |
| `make post` | Build `bin/Post.exe` without changing the default build |
| `make tests` | Build all test executables |
| `make check` | Run the complete unit-test set |
| `make clean` | Remove generated binaries, objects, and dependency files |

Build products are written under `build/` and `bin/` and are ignored by
Git.

## Configure a fit

A run is controlled by two JSON files.

### `config/fit.json`

This file selects:

- the model file;
- data, normalization-MC, and signed background samples;
- multistart and Minuit settings;
- result and log directories;
- the output tag.

Background coefficients enter the effective likelihood as

```text
ln L_eff += coefficient * sum_events ln P(event)
```

so the nominal `SB1 = -0.5` and `SB2 = +0.25` prescription is expressed
entirely in configuration.

### `config/model.json`

This file defines:

- propagator instances and their parameters;
- active Terms;
- registered Wave IDs;
- coupling parameterizations and initial values;
- reference amplitudes within each coherence class.

The loader validates the model before GPU allocation. See
[`docs/MODEL_CONFIGURATION.md`](docs/MODEL_CONFIGURATION.md) for the complete
field contract.

## Run a fit

Build `bin/Fit.exe` and submit from the repository root:

```bash
./submit.sh fit
```

The default configuration is `config/fit.json`. A different run
configuration may be supplied explicitly:

```bash
./submit.sh fit path/to/fit.json
```

`submit.sh` is both the submission entry point and the Slurm worker script. It
validates all immutable inputs before submission, exports the canonical project
root to the worker, requests the project-approved A100 resources, and checks
that the expected numerical outputs were produced.

The executable also accepts the run configuration directly:

```bash
bin/Fit.exe config/fit.json
```

Use direct execution only in a suitable GPU runtime environment. Do not run the
fit on an IHEP login node.

## Outputs

The `output.tag` value in `config/fit.json` controls all output names:

| Product | Default pattern |
|---|---|
| Human-readable fit report | `results/fit_result-<tag>.txt` |
| Machine-readable fitted state | `results/fit_state-<tag>.json` |
| Projection ROOT file | `results/projection-<tag>.root` |
| Slurm log | `runlog/fit-<tag>.log` |

Running again with the same tag overwrites the previous files. The fit does not
create per-run directories or copies of the input configuration.

The text report contains convergence diagnostics for every start, the selected
minimum, active Resonances/Waves/Terms, free and fixed physical parameters,
and the full covariance and correlation matrices. It is not parsed by other
programs. `fit_state-<tag>.json` is the stable Fit-to-Post Calculation contract
and contains the ordered free-parameter state and covariance.

The projection file contains fitted normalization-MC weights, selected data,
combined sideband samples, dynamic Term and JPC maps, and fit provenance in the
`MC`, `data`, `bg`, `component_map`, `group_map`, and `metadata`
trees.

## Post processing

Post processing is intentionally split into two independent modules.

Post Calculation reconstructs the fitted model from `fit_state` and
`model.json`, then integrates it over generated truth MC and selected
normalization MC:

```bash
make post
./submit.sh post \
  results/fit_state-initial.json \
  config/model.json \
  RootSet/truth_mc.root \
  RootSet/normalization_mc.root
```

It checks the model signature and exact free-parameter order before applying
the state. Products are written to `post/calculation/results/` as tagged text,
ROOT, and LaTeX files.

Post Plotting needs only the fit projection ROOT file and does not read the fit
report, fit-state JSON, `model.json`, truth MC, or normalization MC directly:

```bash
post/plotting/draw.sh results/projection-initial.root
```

Figures are written to `post/plotting/results/`. Group and component curves
are discovered from the ROOT maps, so plotting does not encode a fixed list of
resonances or the current `0++`/`0-+` group set.

## Modify the amplitude model

### Add or remove a Resonance

For a Resonance that uses an existing Wave:

1. Add or edit its propagator definition under `resonances` in
   `config/model.json`.
2. Add a Term that references the Resonance and a registered Wave.
3. Choose the coupling mode and maintain exactly one phase reference in each
   coherence class.
4. To disable a contribution temporarily, set `"active": false` on its Term.

Resonances referenced only by inactive Terms are omitted from the runtime GPU
model and the Minuit parameter list. No production source file needs to be
edited for this workflow.

### Add a new Wave

A genuinely new covariant-tensor basis requires process code:

1. implement the complete Wave in a new `process/waves/*.cuh` file using the
   reusable blocks from `framework/math/` and `framework/tensors/`;
2. add its device enum and dispatch in `process/WaveRegistry.cuh`;
3. register its stable ID, JPC label, and coherence class in
   `process/WaveRegistry.cu`;
4. add finite-value, contraction, registry, and model-compilation tests;
5. reference the new Wave ID from `config/model.json`.

`process/ProcessAmplitude.cuh` normally remains unchanged because it applies
the common GVV polarization sum and Wave-pair contraction to every registered
Wave.

## Repository layout

```text
gVV/
├── app/          Fit executable and application-level glue
├── config/       Model, run configuration, and project environment
├── framework/    Process-independent math, tensors, dynamics, model, fit,
│                 likelihood, amplitude, and output components
├── process/      psi(2S) -> gamma omega omega event and amplitude code
│   └── waves/    Complete registered GVV Wave implementations
├── post/
│   ├── calculation/  Fit fractions, efficiencies, and covariance propagation
│   └── plotting/     Projection and angular-moment plotting
├── tests/        Unit and model-contract tests
├── docs/         Architecture, configuration, and Wave-development guides
├── results/      Generated numerical outputs
├── runlog/       Generated Slurm logs
├── Makefile
└── submit.sh
```

The dependency direction is deliberately one way: reusable `framework/`
components do not include GVV-specific `process/` code. A different decay
channel can reuse the framework while replacing its event representation,
complete Waves, Term evaluation, sample loader, and projection writer.

The default `make` dependency graph ends at `Fit.exe`; the numerical downstream
executable is built only by `make post`. ROOT plotting macros are interpreted
by `post/plotting/draw.sh` and are not linked into either executable.

## Documentation

- [Architecture and data flow](docs/ARCHITECTURE.md)
- [Model and fit configuration](docs/MODEL_CONFIGURATION.md)
- [Adding a new Wave](docs/WAVE_DEVELOPMENT.md)
- [Refactor and validation history](docs/REFACTOR_LOG.md)
- [Post system contract](post/README.md)

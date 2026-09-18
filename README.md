# gVV

Covariant-tensor partial-wave analysis of **psi(3686) -> gamma omega omega**,
with both omegas decaying to pi+ pi- pi0. This main branch is dedicated to this
channel. A coherent CUDA amplitude, accepted-MC normalization, and multistart
Minuit fit feed independent plotting and numerical Post calculations.

## Run a fit and draw its projections

Run these commands from the project root on IHEP:

```bash
source config/gvv_env.sh
make -j2 fit post
make check
# Review config/fit.json and config/model.json, then submit the fit:
bash submit_fit.sh config/fit.json
# After the fit finishes, substitute your output.tag below:
bash post/plotting/draw.sh results/projection-omega_res_v1.root
```

`make` builds executables; it does not perform a fit. "Nothing to be done"
means the requested executable is up to date. Fit and Post run on an allocated
GPU through the submission scripts; plotting and host tests can run on a login
node. Reusing an output tag overwrites the corresponding results and log.

- **Run settings:** `config/fit.json` selects samples, signed background
  coefficients, multistart settings, and output directories/tag.
- **Physics settings:** `config/model.json` selects Resonances, active Terms,
  registered Waves, initial couplings, fixed/free parameters, and bounds.
- **Plot settings:** each macro in `post/plotting/macros/` has a user configuration
  block for binning, labels, layout, and style. With no argument, `draw.sh`
  still uses `results/projection-initial.root`; pass your current tag explicitly.

The input tree is `Pwa`. Its seven `double[4]` branches are `p4_pip1`, `p4_pim1`,
`p4_pi01`, `p4_pip2`, `p4_pim2`, `p4_pi02`, `p4_gam`, stored as `(px,py,pz,E)` in
GeV. Input paths are relative to the project root or absolute.

## Structure and where to edit

```text
fit/Fit.cu                Complete nominal fit: sample roles, NLL, multistart, reports
core/Model.*              Model compilation, final parameter layout, numeric state
core/ModelIO.cpp          JSON parsing and original-document signatures
core/Sample.*             ROOT input and sample-owned kinematic/GPU caches
core/Amplitude.*          Shared Fit/Projection/Post amplitude evaluation
core/Minuit.*             Minuit callback, random starts, convergence, best selection
core/IO.*                 Run config and fit-state serialization
core/Projection.cu        Projection ROOT schema and derived plotting observables
core/math/                Complex numbers, four-vectors, tensors, barriers
core/physics/             Event currents, waves, propagators, omega width/resolution
post/Post.cu              Fractions, interference, efficiencies, covariance propagation
post/plotting/            Existing independent ROOT plotting workflow
config/ tests/ docs/      User settings, regression checks, documentation
```

The five public modules are **Model, Sample, Amplitude, Minuit, IO**. Fit policy
stays in one script, with ordinary local helper functions. For a future
mass-bin fit, add a focused `FitBins.cu` when needed and reuse these modules;
there is no task framework or placeholder scan implementation.

## Physics and compatibility

A **Resonance** is one propagator instance, a **Wave** is one complete covariant
tensor, and a **Term** combines the two with a coupling. Terms sharing a
Resonance ID share its line-shape parameters. Inactive-only dependencies are
not compiled into the fit. All cross-Wave Gram-matrix entries are retained.

Each complex omega propagator is convolved in mass with a zero-mean Gaussian.
One sigma is shared by both omegas and all Terms, fitted as `log_sigma_omega`.
X propagators have no detector-resolution convolution. Fixed sigma zero
selects the exact legacy path. See [Omega resolution](docs/OMEGA_RESOLUTION.md).

The concentrated-script refactor preserves model/run JSON schema 1, fit-state
schema 2, projection schema 4, parameter ordering, signatures, numerical
formulas, random-start policy, and existing output paths. Old compatible
fit-state files remain self-contained inputs to Post; Post uses their embedded
model, not today's `config/model.json`.

## Numerical Post calculation

Plotting needs only the projection file. Fractions and efficiencies additionally
need truth and selected MC representing the **same generated exposure**:

```bash
bash submit_post.sh results/fit_state-omega_res_v1.json \
  /path/to/truth_mc.root RootSet/normalization_mc.root
```

This produces `post/calculation/results/post_result-<tag>.{txt,root}` and
`fit_fractions-<tag>.tex`. Errors propagate the fit covariance; they do not
include finite-MC or systematic uncertainty. Supply the actual truth file;
its existence is not implied by the example path.

## Verification and documentation

`make check` runs 15 host tests. `make gpu-tests` compiles four GPU regressions;
`make check-gpu` executes them and requires an allocated CUDA device. Merely
compiling the GPU tests is not runtime validation. Generated ROOT files,
executables, logs, and plots are ignored by Git.

- [Workflow](docs/WORKFLOW.md): commands, outputs, diagnostics, validation.
- [Architecture](docs/ARCHITECTURE.md): ownership, data flow, extension boundaries.
- [Code reference](docs/CODE_REFERENCE.md): file/function navigation.
- [Model configuration](docs/MODEL_CONFIGURATION.md): JSON contracts and edits.
- [Wave development](docs/WAVE_DEVELOPMENT.md) and
  [tensor conventions](docs/TENSOR_CONVENTIONS.md): physics implementation.
- [Post guide](post/README.md) and
  [plotting style](post/plotting/PLOTTING_STYLE.md): downstream usage.
- [Refactor log](docs/REFACTOR_LOG.md): migration and verification record.

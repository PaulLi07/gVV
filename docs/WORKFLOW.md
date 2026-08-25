# Build and analysis workflow

This guide describes how to build, configure, run, inspect, and hand off a
gVV analysis. It is intentionally operational: the design boundaries are
described in [Architecture and data flow](ARCHITECTURE.md), the complete model
fields and routine resonance-editing procedures in [Model
configuration](MODEL_CONFIGURATION.md), and the source-level Wave extension
procedure in [Developing and registering a GVV Wave](WAVE_DEVELOPMENT.md).

The analysis has two top-level parts:

1. **Amplitude Fit** loads the fitted data, accepted normalization MC, and any
   signed background samples; minimizes the unbinned likelihood; and writes
   the fit report, fitted state, projection ROOT file, and Slurm log.
2. **Post processing** is split again into two independent consumers:
   **Post Calculation** derives fit fractions, efficiencies, interference
   fractions, and their fitted-parameter covariance; **Post Plotting** draws
   projections and angular moments.

The two Post modules do not call each other and do not consume each other's
outputs.

```text
config/model.json + config/fit.json + fit samples
                         |
                         v
                  Slurm Amplitude Fit
                         |
          +--------------+----------------+
          |              |                |
 fit_result-TAG.txt  fit_state-TAG.json  projection-TAG.root
   human report          |                |
                         |                |
             + truth/selected MC          |
                         |                |
                         v                v
                Slurm Post Calculation  Post Plotting with ROOT
                         |                |
           post_result / covariance    PDF and EPS figures
```

## 1. Execution model on the IHEP cluster

The `lxlogin` node and Slurm-allocated GPU node have different
responsibilities.

| Location | Appropriate work |
|---|---|
| `lxlogin` | Edit configuration, source the environment, compile, run the ordinary test suite, inspect files, submit/query Slurm jobs, and run ROOT plotting macros |
| Slurm-allocated GPU node | Execute `Fit.exe`, `Post.exe`, or `make check-gpu` only as a background Slurm payload |

Users do not log in to or enter a GPU node interactively. Do not run
`bin/Fit.exe`, `bin/Post.exe`, or `make check-gpu` directly on `lxlogin`.
Compile, preflight, submit, and monitor from `lxlogin`; Slurm then runs the
payload in the background on an allocated GPU node. The repository wrappers
submit Fit and Post Calculation. There is no checked-in wrapper for GPU tests,
so `make check-gpu` must be the payload of a separately submitted background
Slurm job when that validation is required; this guide does not invent a
submission script for it.

The provided Fit and Post wrappers currently request:

- partition `gpupwa`;
- QoS and account `pwadedicate`/`gpupwa`;
- one task and one CPU;
- memory option `--mem-per-cpu=24288`;
- one A100 GPU.

These values live in `submit_fit.sh` and `submit_post.sh`. A change to the
site allocation policy belongs in those two wrappers, not in JSON model files.

## 2. Environment and build

### 2.1 Project environment

From the repository root, load the project-local toolchain:

```bash
source config/gvv_env.sh
```

The default paths are CUDA 12 under `/usr/local/cuda-12` and ROOT 6.32.02
under the configured CVMFS release. Override either path before sourcing the
file when an equivalent installation is required:

```bash
export GVV_CUDA_ROOT=/path/to/cuda
export GVV_ROOTSYS=/path/to/root
source config/gvv_env.sh
```

`config/gvv_env.sh` validates `nvcc` and ROOT's `thisroot.sh`, exports
`GVV_PROJECT_ROOT`, `CUDA_HOME`, `ROOTSYS`, `NVCC`, and `ROOT_PREFIX`, sources
ROOT, and updates `PATH` and `LD_LIBRARY_PATH`. It does not modify shell startup
files and does not select or enter a GPU node.

The project does not have a separate installation step. Binaries are built in
place under `bin/`, objects and dependency files under `build/`, and runtime
products under the configured result/log directories.

### 2.2 Build targets

Run the required build from the repository root:

```bash
make -j2          # builds bin/Fit.exe
make -j2 post     # builds bin/Post.exe
```

The complete target contract is:

| Command | Result |
|---|---|
| `make` or `make fit` | Build `bin/Fit.exe`; this is the default target |
| `make post` | Build `bin/Post.exe` |
| `make tests` | Build the ordinary test executables |
| `make check` | Build and run the ordinary test suite |
| `make gpu-tests` | Build the three explicit CUDA runtime test executables without running them |
| `make check-gpu` | Build and run the tensor-building-block, complete-Wave, and intensity-equivalence GPU tests; use only in a Slurm GPU job |
| `make clean` | Remove generated objects, dependency files, binaries, and test executables |

The Makefile keeps generic framework objects separate from GVV process
objects and links only the objects needed by each executable. It also writes
compiler-generated header dependencies, so editing a nested Wave or tensor
header triggers the required recompilation.

## 3. Shared ROOT input contract

Fit samples and both Post Calculation MC samples use the same process-owned
ROOT boundary. Every input file must contain a non-empty `Pwa` tree with these
seven array branches:

```text
p4_pip1  p4_pim1  p4_pi01
p4_pip2  p4_pim2  p4_pi02
p4_gam
```

Each branch is a four-element `double` array stored as `(px, py, pz, E)`.
The first three pion branches reconstruct the first omega and the next three
reconstruct the second omega. Branch names, tree name, and component order are
currently defined by the GVV process glue in `app/Fit.cu` and by the default
`GVVBranchConfig` used by Post Calculation.

ROOT files and generated outputs are not source-controlled. The default fit
configuration expects the input files under `RootSet/`, but every fit sample
path may be changed in `config/fit.json`.

## Part I: Amplitude Fit

### 4. Fit configuration

One Fit run reads exactly one run configuration and one model configuration.

#### 4.1 `config/fit.json`: run policy

`fit.json` owns four concerns:

- `model`: path to the model JSON;
- `inputs`: data, accepted normalization MC, and zero or more signed
  background samples;
- `minimizer`: multistart and Minuit policy;
- `output`: result directory, log directory, and one common output tag.

Paths in the supplied configuration are interpreted from the repository root
by the provided submission workflow. The background coefficient implements

```text
ln L_eff += coefficient * sum_background_events ln P(event).
```

The data contribution has coefficient `+1`. A conventional subtracted
sideband therefore uses a negative coefficient; a correction sample may have
a positive coefficient. Background labels are metadata and are propagated
into the projection ROOT file.

Start zero uses the nominal model values. Later starts randomize only the free
coupling magnitudes and phases according to `random_magnitude`; fixed physical
parameters and non-randomized coordinates retain their configured values.

| Minimizer field | Meaning |
|---|---|
| `n_starts` | Total number of attempts, including nominal start zero |
| `base_seed` | Deterministic seed base; attempt `i` uses `base_seed + i` |
| `maximum_edm` | Largest EDM accepted for a final attempt |
| `maximum_calls` | MIGRAD call limit passed to TMinuit |
| `tolerance` | MIGRAD tolerance passed to TMinuit |
| `error_definition` | TMinuit `SET ERR` value; `0.5` is the nominal NLL convention |
| `random_magnitude` | Positive `[minimum, maximum]` range sampled uniformly in log magnitude for randomized couplings |

The `output.tag` must start with a letter or digit and may then contain only
letters, digits, `.`, `_`, or `-`. Reusing a tag deliberately overwrites the
previous products with that tag. Never run two Fit jobs with the same output
directory and tag concurrently, because both jobs would target the same log
and numerical files.

The Fit submission wrapper passes file paths; it does not snapshot `fit.json`,
`model.json`, or ROOT inputs before the queued Fit starts. Treat those files as
immutable from Fit submission until the background job completes. A successful
Fit then embeds its complete model definition in the fitted-state output, so a
later Post job no longer depends on that source `model.json` path. For
concurrent hypotheses, still use separate configuration files and unique tags
instead of editing one queued Fit's files in place.

#### 4.2 `config/model.json`: physical model

`model.json` is the only user model-description layer. It defines propagator
instances, registered Wave IDs, Terms, coupling policies, and reference
amplitudes. The generic parser and GVV compiler validate it before sample GPU
allocation.

The Resonance ID is the instance boundary. Several Terms naming one Resonance
share its mass, width, and other propagator parameters; different Resonance
IDs remain independent even if they select the same propagator formula. For
supported line shapes, mass and width can be fixed or floated directly through
their parameter objects. Free identity-coordinate mass/width parameters require
explicit finite positive physical bounds; discrete `orbital_l` remains fixed.

For routine resonance scans using existing Waves, edit only this file. See
`MODEL_CONFIGURATION.md` for supported propagators and the add/disable/remove
procedure. When the desired covariant basis is not registered, follow
`WAVE_DEVELOPMENT.md`.

`config/model.schema.json` documents the generic JSON shape for editors and
external validators. The C++ loader owns the runtime validation and does not
shell out to the schema file.

### 5. What one Fit run does

`app/Fit.cu` is the executable glue. Its end-to-end sequence is:

1. parse `fit.json` and create the configured result/log directories;
2. parse `model.json`, remove inactive Terms, and compile stable Resonance and
   Wave IDs into dense runtime arrays;
3. load data, normalization MC, and every configured background ROOT tree;
4. build the process omega-width table and parameter-independent Wave Gram
   matrices on the GPU;
5. create the ordered free-parameter mapping used by Minuit;
6. on every likelihood call, apply the trial parameters and evaluate the
   coherent intensity for accepted normalization MC;
7. compute the MC normalization
   `N = sum_MC I(event) / N_MC`;
8. evaluate `sum[log I(event) - log N]` for data and add every signed
   background contribution;
9. run the configured nominal and randomized MIGRAD/HESSE starts;
10. accept only starts that satisfy the fit-engine convergence contract and
    select the accepted start with the lowest NLL;
11. apply the selected state and serialize the human report and the
    schema-version-2 machine state, including its formatted model definition
    and GVV implementation contract;
12. evaluate the selected model on normalization MC and serialize the
    process-specific projection ROOT contract.

Complete Term coefficients include both propagator dynamics and fitted
couplings before Terms sharing the same exact Wave slot are aggregated. The
final contraction preserves all inter-Wave interference. Projection writing
uses a separate Term-pair path because downstream component plots require the
individual diagonal and signed interference contributions.

If no multistart attempt passes the convergence criteria, Fit exits with an
error and does not write a final report, state, or projection.

An attempt is accepted only when MIGRAD and HESSE both return status zero, the
covariance status is at least 2, EDM is finite and no larger than
`maximum_edm`, and the minimum, fitted values, errors, and covariance entries
are finite. A boundary flag is recorded but does not by itself reject an
otherwise converged attempt. Accepted attempts are ranked by NLL, with
covariance quality, EDM, and start index used as deterministic near-tie
breakers.

### 6. Submit a Fit job

Build first, then submit from the repository root:

```bash
source config/gvv_env.sh
make -j2
./submit_fit.sh
```

The default argument is `config/fit.json`. To use another run configuration:

```bash
./submit_fit.sh path/to/fit.json
```

`submit_fit.sh` performs the following work on `lxlogin` before calling
`sbatch`:

- resolves the repository and configuration paths;
- parses the output tag/directories, model path, and every input path with
  Python;
- rejects an unsafe tag or unreadable immutable input;
- verifies that `bin/Fit.exe` exists and is executable;
- creates the result and log directories;
- submits itself in `--worker` mode with the canonical project root exported;
- directs Slurm stdout/stderr to `fit-<tag>.log` with truncate mode.

Inside the background Slurm allocation, the worker mode sources
`config/gvv_env.sh`, prints the job/host/GPU context, runs `nvidia-smi`, and
launches

```text
srun --ntasks=1 bin/Fit.exe <absolute-fit-config>
```

After the executable returns, the wrapper requires non-empty report, state,
and projection files before declaring the job successful.

Use normal Slurm tools from `lxlogin` to inspect a submitted job. Do not enter
the allocated GPU node:

```bash
squeue -j <job-id>
sacct -j <job-id> --format=JobID,State,ExitCode,Elapsed
tail -f runlog/fit-<tag>.log
```

The exact log path follows `output.log_directory` in the selected `fit.json`.

`bin/Fit.exe config/fit.json` is the worker executable interface, but it
requires a CUDA device and must be launched as a background Slurm payload.
Users do not enter the GPU node to invoke it interactively.

### 7. Fit outputs and their consumers

With output directory `results`, log directory `runlog`, and tag `TAG`, a
successful run creates:

| Product | Purpose | Intended consumer |
|---|---|---|
| `results/fit_result-TAG.txt` | Complete readable diagnostics | Analyst |
| `results/fit_state-TAG.json` | Stable fitted-state handoff | Post Calculation and other numerical tools |
| `results/projection-TAG.root` | Fitted event weights and process observables | Post Plotting |
| `runlog/fit-TAG.log` | Full Slurm/executable stdout and stderr | Analyst and job diagnosis |

All four names are controlled by the same tag. Text/JSON files are opened with
truncate semantics, the ROOT projection is created with ROOT `RECREATE`, and
the Slurm wrapper uses `--open-mode=truncate`. A repeated completed run with
the same tag therefore replaces the previous set.

The three numerical products are written sequentially rather than as one
atomic transaction. A failed job can therefore leave an earlier report or
state file beside a missing/incomplete projection. Treat the set as valid only
when Slurm reports success and the submission wrapper reaches its final
three-product check.

#### 7.1 Human report

`fit_result-TAG.txt` is deliberately presentation-oriented and is never
parsed by Post. It contains:

- configuration and model provenance;
- sample paths, entry counts, and likelihood coefficients;
- minimizer settings and every multistart result;
- the selected NLL, EDM, MIGRAD/HESSE/covariance statuses, boundary flag, and
  elapsed time;
- initial, final, error, step, and bounds for every free coordinate;
- the active Wave registry, every active Term and its final physical coupling,
  and every active Resonance including fixed and fitted physical parameters;
- full free-parameter covariance and correlation matrices.

#### 7.2 Machine fitted state

`fit_state-TAG.json` has schema version 2. It stores the output tag, source
configuration paths as provenance, selected-fit diagnostics, ordered free
coordinates, full covariance matrix, and the complete formatted model
definition. The `model` object carries separate definition and GVV
implementation signatures plus their combined compatibility key. Fixed
parameters remain ordinary fields in the embedded definition rather than
Minuit coordinates.

Its model block has this shape:

```json
"model": {
  "source_file": "config/model.json",
  "name": "nominal",
  "definition": {"schema_version": 1, "process": "..."},
  "definition_signature": "fnv1a64:...",
  "implementation_signature": "gvv-amplitude-contract-v1",
  "signature": "gvv-amplitude-contract-v1:fnv1a64:..."
}
```

`source_file` is provenance only. `definition` is the authoritative Post input;
the three signatures separate a changed JSON model from a changed numerical
GVV implementation.

Post Calculation parses and recompiles that embedded definition, validates all
three signatures, then validates the free-parameter count, names, and order
before applying the state. Schema-version-1 states are intentionally rejected;
rerun Fit to produce a self-contained state. Do not edit generated fitted-state
JSON manually.

#### 7.3 Projection ROOT schema

`projection-TAG.root` uses projection schema version 4. It contains:

| ROOT object | Content |
|---|---|
| `MC` tree | Accepted normalization-MC events, derived GVV observables, total fitted `weight`, dynamic `weight_group`, and symmetric `weight_component` vectors |
| `Data` tree | Selected data events and the same kinematic observables, without model weights |
| `bg` tree | All configured background events plus zero-based `background_index` and plotting weight `weight_bg = -likelihood_coefficient` |
| `component_map` tree | Dynamic Term index, ID, label, Resonance, Wave ID/label, registered device type, and JPC |
| `group_map` tree | Dynamic JPC group index and display label |
| `background_map` tree | Dynamic background index, label, event count, likelihood coefficient, and projection weight |
| `metadata` tree | Schema/tag/model provenance, sample and model sizes, background method, effective signal yield, selected-fit identity, and component-closure diagnostic |

Each event tree contains the seven input four-vectors, reconstructed
`p4_omega1`, `p4_omega2`, and `p4_X`, and all two- and three-body invariant
masses used by the projection plots. The polarization-oriented branches are:

- `cos_theta_gamma` in the psi rest frame;
- `cos_theta_omega1` and `phi_omega1`, which give the complete omega1
  direction in the X helicity frame;
- `cos_theta_decay_plane_omega1`, `phi_decay_plane_omega1`,
  `cos_theta_decay_plane_omega2`, and `phi_decay_plane_omega2`, which give the
  complete direction of each oriented `p(pi+) cross p(pi-)` normal in its
  parent omega helicity frame;
- `delta_phi_decay_planes`, the wrapped difference of those two local normal
  azimuths;
- `cos_theta_pip_omega1`, `cos_theta_pim_omega1`,
  `cos_theta_pi0_omega1`, `cos_theta_pip_omega2`,
  `cos_theta_pim_omega2`, and `cos_theta_pi0_omega2`, the six pion polar
  cosines in their parent omega helicity frames;
- `decay_plane_normal_magnitude_omega1` and
  `decay_plane_normal_magnitude_omega2`, the unnormalized analyser magnitudes
  in the two omega rest frames.

The X-frame axes use `z_X = -unit(p_gamma)`,
`y_X = unit(z_beam cross z_X)`, and `x_X = unit(y_X cross z_X)`. For omega_i,
`z_i` follows its X-frame flight direction, `y_i = unit(z_X cross z_i)`, and
`x_i = unit(y_i cross z_i)`. Azimuths are in radians. The omega2 production
angles are not stored because its X-frame direction is back-to-back with
omega1.

The total normalization-MC weights sum to the effective fitted signal yield

```text
N_data + sum_background coefficient * N_background.
```

`weight_group[g]` includes all diagonal and interference pairs internal to
that JPC group; cross-group interference remains only in the total weight.
`weight_component` is stored as a full symmetric `N_term x N_term` vector.
Each off-diagonal cell already represents the complete signed pair
interference and is mirrored into both symmetric cells for lookup. Do not sum
the entire full matrix as if both cells were independent contributions.

Before closing the ROOT file, the writer checks that the packed Term-pair sum
reconstructs the total intensity to the writer tolerance, currently `1e-7`.

### 8. Fit validation checklist

Before accepting a Fit result, check at least:

1. `sacct` reports a successful job and the log ends with all three output
   paths;
2. the report contains an accepted best start and acceptable MIGRAD, HESSE,
   covariance, and EDM diagnostics;
3. `at_parameter_boundary` and the physical parameter values are understood;
4. the multistart table does not show an unexplained competing minimum;
5. the active Term/Resonance section matches the intended `model.json`;
6. the covariance/correlation structure is numerically and physically
   reasonable;
7. the projection log reports a small component closure residual and the
   fitted weight sum matches its target;
8. Post Plotting gives reasonable distributions and angular-moment checks.

The framework reports numerical diagnostics; it does not decide whether a
model is a scientifically adequate description of the data.

## Part II: Post processing

### 9. The two independent Post modules

Post processing is intentionally modular:

| Module | Fit output read | Additional input | Runs where | Products |
|---|---|---|---|---|
| Post Calculation | `fit_state-TAG.json` (including its embedded model) | generated truth MC, selected normalization MC | Background Slurm GPU job | tagged TXT, ROOT, and LaTeX numerical results |
| Post Plotting | `projection-TAG.root` | none | `lxlogin` with ROOT | tagged PDF and EPS figures |

Post Calculation does not read the projection ROOT file. Post Plotting does
not read the fit state, model JSON, fit report, truth MC, or Post Calculation
results.

### 10. Post Calculation

#### 10.1 Required inputs

The command-line order is fixed:

```text
Post.exe fit_state.json truth_mc.root normalization_mc.root
```

The three inputs mean:

1. the machine state from the accepted Fit, including its exact model
   definition and compatibility signatures;
2. generated truth MC before event selection;
3. the selected normalization-MC sample from the same unweighted production.

Both MC files obey the shared `Pwa` branch contract. They are integrated as
unweighted event samples. The selected/truth integral ratio is an efficiency
only when the selected file is the selected subset of the same generated
production, with compatible generator normalization. Passing unrelated or
independently normalized MC samples produces a meaningless efficiency even if
the files satisfy the branch schema.

#### 10.2 Numerical work

Post Calculation:

1. reads and validates the fit-state JSON;
2. parses and compiles the embedded model and rejects a definition,
   implementation, combined-signature, or free-parameter-order mismatch;
3. restores the selected fitted coordinates;
4. loads generated truth and selected MC and builds the same registered-Wave
   contractions used by Fit;
5. integrates every diagonal Term and packed signed interference pair on the
   GPU in bounded event batches;
6. checks that the component sum equals the direct coherent intensity for both
   samples;
7. constructs Term and JPC-group fit fractions, component/group/total
   efficiencies, and Term/group interference fractions;
8. checks both fraction closure decompositions;
9. propagates the complete fitted-parameter covariance with central or
   bound-aware one-sided finite differences;
10. writes human-readable, ROOT, and LaTeX products.

Reported errors include only propagation of the fitted-parameter covariance.
They do not include MC-integration statistics, efficiency-systematic
uncertainties, model systematics, or external branching-fraction inputs.

An active Term whose fitted diagonal truth integral is exactly non-positive
cannot define a component efficiency and is rejected explicitly. Reassign or
remove a physically zero contribution rather than interpreting an undefined
`0/0` efficiency.

#### 10.3 Build and submit

From the repository root:

```bash
source config/gvv_env.sh
make -j2 post
./submit_post.sh \
  results/fit_state-TAG.json \
  RootSet/truth_mc.root \
  RootSet/normalization_mc.root
```

All three arguments are required. `submit_post.sh` resolves and checks them,
verifies `bin/Post.exe`, derives `TAG` from `fit_state.output_tag`, creates the
fixed Post result and log directories, and submits itself in worker mode.
The paths are passed to the queued job without copying their contents; do not
edit or replace the state, truth MC, or selected MC before completion.
Inside the background Slurm allocation, worker mode loads the environment and
runs:

```text
srun --ntasks=1 bin/Post.exe <state> <truth> <selected>
```

The Post log is always `runlog/post-TAG.log`; unlike the Fit log, its directory
is not configured by `fit.json`. The wrapper truncates an existing same-tag
log and checks the final TXT and ROOT products before reporting success.

#### 10.4 Calculation products

Outputs are fixed under `post/calculation/results/`:

| Product | Content |
|---|---|
| `post_result-TAG.txt` | Readable rows containing category, name, value, propagated error, truth integral, and selected integral, plus closure metadata |
| `post_result-TAG.root` | `observables` and `metadata` trees plus `observable_covariance` and `observable_correlation` symmetric matrices |
| `fit_fractions-TAG.tex` | A compact LaTeX table of per-Term fit fractions |
| `runlog/post-TAG.log` | Slurm and executable diagnostics |

The numerical result files use truncate/ROOT `RECREATE` semantics. Reusing a
tag replaces them. As with Fit, do not submit concurrent Post jobs with the
same tag.

Calculation products are also written sequentially. If the executable fails
after opening one product, partial same-tag files may remain; use them only
after the Slurm job succeeds and the log reaches the normal completion lines.

The result categories are `fit_fraction`, `interference`,
`efficiency_component`, `efficiency_total`, `fit_fraction_group`,
`efficiency_group`, and `interference_group`. The ROOT `observables` tree also
stores Term-pair indices for Term-level rows. The covariance and correlation
matrix order is exactly the `index` order of that tree.

### 11. Post Plotting

Post Plotting is an interpreted ROOT workflow and does not require
`bin/Post.exe`. From the repository root, run:

```bash
post/plotting/draw.sh
post/plotting/draw.sh results/projection-TAG.root
```

The zero-argument form uses `results/projection-initial.root`; one optional
argument overrides that input, and additional arguments are rejected.
`draw.sh` sources the project environment, verifies the input and ROOT binary,
derives `TAG` from the projection basename, creates
`post/plotting/results/`, and runs six independently executable macros in ROOT
batch mode.

When calling the driver from another directory, invoke the script by an
absolute path and pass an absolute Projection path, or a Projection path
relative to that caller's current directory. The script resolves its own
project location, but it deliberately resolves a relative input argument from
the caller's `$PWD`.

The plotting kernels are CPU-side ROOT code and need no GPU allocation.
However, the current driver sources the full `config/gvv_env.sh`, which also
checks that the configured `nvcc` exists. Until a ROOT-only environment loader
is introduced, the plotting shell therefore still requires the project CUDA
installation to be readable even though it does not execute CUDA code.

For focused development, run any macro directly. With no arguments it uses
`results/projection-initial.root` and writes its `initial` PDF/EPS pair under
`post/plotting/results/`:

```bash
source config/gvv_env.sh
root post/plotting/macros/Draw_projection.cxx
```

The same command works from the macro directory:

```bash
cd post/plotting/macros
root Draw_projection.cxx
```

For another input or output prefix, pass both arguments explicitly and use
ROOT batch mode when no interactive session is needed:

```bash
root -l -b -q \
  'post/plotting/macros/Draw_projection.cxx("results/projection-TAG.root","post/plotting/results/projection-TAG")'
```

The two function arguments override the defaults written in the macro. An
absolute path is used unchanged; a relative path is interpreted from the
project root regardless of the caller's current directory.

Every plot-specific setting is collected in the `User configuration` block at
the top of the selected `.cxx`: default paths, variables or moment orders,
binning, axes, canvas geometry, colors, line/marker styles, draw options,
legend, and annotations. The implementation follows below that block. Shared
headers contain only common Projection reading, histogram/moment construction,
diagnostics, path resolution, and base style.
The visual, layout, frame-naming, and review contract is documented in
[`post/plotting/PLOTTING_STYLE.md`](../post/plotting/PLOTTING_STYLE.md).

| Macro | Output prefix | Purpose |
|---|---|---|
| `Draw_projection.cxx` | `projection-TAG` | Main exchange-symmetric 3x2 kinematic projection |
| `Draw_projection_components.cxx` | `projection_components-TAG` | Enlarged 3x2 kinematic area and a compact centered two-column legend in the external right margin for thin diagonal `|A_i|^2` curves; interference is intentionally omitted |
| `Draw_polarization.cxx` | `polarization-TAG` | Three horizontal omega decay-plane-normal projections with the shared legend inside the first subplot |
| `Draw_omega_decay_checks.cxx` | `omega_decay_checks-TAG` | Six candidate-combined pion-angle and pion-pair-mass checks |
| `draw_angular_moments.cxx` | `angular_moments-TAG` | Exchange-symmetrized even Legendre moments `P0`, `P2`, `P4`, and `P6` |
| `draw_angular_moments_odd.cxx` | `angular_moments_odd_diagnostic-TAG` | Ordered-omega odd moments `P1`, `P3`, and `P5` for pairing/order-bias diagnosis |

Each macro writes both `.pdf` and `.eps`, yielding twelve tagged figure files.
Existing same-name figures are replaced by ROOT's print operation.

The main projection contains `M(omega omega)`, candidate-combined
`M(gamma omega_i)`, `cos(theta_gamma)` in the `psi(2S)` rest frame,
exchange-symmetric `cos(theta_omega)` and `phi_omega` in the X helicity frame,
and the candidate-combined `M(pi+ pi- pi0)` distribution.
For `phi_omega`, the omega2 exchange image is obtained by wrapping
`phi_omega1 + pi` into `(-pi, pi]`.

The polarization figure contains candidate-combined
`cos(theta_n_omega)` and `phi_n_omega` for the oriented
`n_i = unit[p(pi+_i) cross p(pi-_i)]` analyzer in each omega helicity frame,
plus the exchange-symmetrized signed, wrapped `Delta phi(n_1,n_2)`. The
omega-decay check figure combines the two candidates
with half weight each for the three pion helicity cosines and for
`M(pi+ pi-)`, `M(pi+ pi0)`, and `M(pi- pi0)`. It does not add an artificial
`+/-cos(theta_pi)` reflection.

The projection utilities require projection schema version 4 and discover
Terms, JPC groups, and background samples dynamically from `component_map`,
`group_map`, and `background_map`. The main plots compare data with fitted
signal plus signed background. Data are black markers, Background is a gray
hatched histogram, Total fit keeps its reserved solid blue appearance, and
every coherent JPC-group curve uses the same dashed line style with a distinct
color and the legend wording `coherent <JPC>`. Diagonal component styles are
deterministic and distinct under model reordering, and each curve uses a thin
width-1 line. The component figure enlarges its `3 x 2` physics area and puts a
compact centered two-column legend block in the external right margin, modeled
on a conventional projection-plot legend. Every other shared legend is inside
the first subplot, whose extra headroom is applied only to that panel. The
automatic vertical envelope covers data errors and every drawn histogram. The
diagonal-component plot is not expected to sum to the coherent total because
it omits Term interference.

Angular-moment panels show unnormalized binwise Legendre sums. Their mass-bin
width labels are derived from the configured range and bin count, and every
signed nonzero moment includes a gray zero reference. Odd moments are
diagnostics tied to the input-labelled omega ordering and are not
label-independent observables of the identical-omega final state.

### 12. Complete operational recipes

#### 12.1 Fit and inspect only

```bash
source config/gvv_env.sh
make -j2
make check
./submit_fit.sh config/fit.json
squeue -j <job-id>
tail -f runlog/fit-TAG.log
```

After completion, inspect `fit_result-TAG.txt` and the full log before using
the result downstream.

#### 12.2 Fit followed by numerical Post Calculation

```bash
make -j2 post
./submit_post.sh \
  results/fit_state-TAG.json \
  RootSet/truth_mc.root \
  RootSet/normalization_mc.root
```

This path needs no projection file.

#### 12.3 Fit followed by plots

```bash
post/plotting/draw.sh results/projection-TAG.root
```

This path needs no truth MC, model JSON, fit-state JSON, or Post Calculation
result.

#### 12.4 Repeated model scan

For each hypothesis:

1. edit or select a model JSON;
2. point a fit JSON at that model;
3. assign a unique, descriptive output tag;
4. submit the Fit and inspect convergence;
5. run either downstream branch only after accepting the Fit;
6. keep the fitted-state JSON and the matching truth/selected MC available for
   Post Calculation.

The schema-version-2 machine state embeds the exact formatted `model.json`
definition used by Fit. Keep the source configuration under normal version
control for provenance and future edits, but Post reconstructs its model from
the state rather than reopening that path.

### 13. Entry points and operational files

| File | Responsibility |
|---|---|
| `Makefile` | Defines Fit/Post/test compilation, one-way object grouping, CUDA architectures, ROOT linking, generated dependencies, and cleanup |
| `config/gvv_env.sh` | Loads the project-local CUDA/ROOT build and runtime environment |
| `config/fit.json` | User-owned sample, minimizer, output-directory, and output-tag configuration |
| `config/model.json` | User-owned Resonance/Wave/Term model description |
| `config/model.schema.json` | Editor/external-tool description of the generic model JSON shape |
| `submit_fit.sh` | Fit-only Slurm submitter and worker entry point |
| `submit_post.sh` | Post-Calculation-only Slurm submitter and worker entry point |
| `app/Fit.cu` | Fit executable glue from configurations through likelihood, minimizer, and all Fit outputs |
| `framework/fit/FitConfig.*` | Strict `fit.json` loader and tagged output-name construction |
| `framework/fit/FitEngine.*` | Process-neutral multistart Minuit driver and convergence selection |
| `framework/fit/FitOutput.*` | Human fit-report serialization |
| `framework/fit/FitState.*` | Machine fitted-state JSON serialization and validation |
| `process/FitLikelihood.*` | GVV sample preparation, GPU synchronization, MC normalization, and signed likelihood |
| `process/ProjectionWriter.*` | GVV projection observables, weights, maps, metadata, and ROOT schema |
| `post/calculation/PostCalculation.cu` | Calculation executable glue, observable construction, covariance propagation, and result serialization |
| `post/calculation/ComponentEvaluator.*` | Batched GPU integration and coherent/component closure checks |
| `post/plotting/draw.sh` | Plot-only ROOT driver and tagged output routing |
| `post/plotting/GVVPlotUtils.h` | Dynamic Projection reader, common histogram construction, diagnostics, path helper, and base style |
| `post/plotting/GVVAngularMoments.h` | Shared Legendre/moment arithmetic and unstyled histogram construction |
| `post/plotting/PLOTTING_STYLE.md` | Unified visual roles, layouts, frame/candidate naming, automatic ranges, and plot-review checklist |
| `post/plotting/macros/*.cxx` | Independently executable, fully configured figure modules |

### 14. Common failures and their meaning

| Symptom | Likely cause or action |
|---|---|
| Submission says an input is missing | A configured path is wrong or unreadable from `lxlogin`; paths should be checked before submission |
| Submission says `Fit.exe` or `Post.exe` is missing | Run `make` or `make post` after sourcing the environment |
| `nvcc` or ROOT setup is unavailable | Correct `GVV_CUDA_ROOT`/`GVV_ROOTSYS` before sourcing `gvv_env.sh` |
| ROOT tree or branch is missing | Regenerate or convert the sample to the exact `Pwa` seven-branch contract |
| Fit configuration reports an unknown field | The loader is strict; correct the field rather than relying on it being ignored |
| No accepted multistart result | Inspect MIGRAD/HESSE/EDM output, parameterization, initialization, boundaries, and model identifiability |
| Likelihood reports invalid intensity or normalization | The active model produced a non-finite, negative, or zero event intensity, or a non-positive MC normalization |
| Projection reports non-positive effective yield | Signed background coefficients and sample sizes imply an invalid fitted signal-yield target |
| Post rejects schema version 1 | Rerun Fit with the current executable to create a schema-version-2 state containing the model definition |
| Post says definition/implementation/signature/order mismatch | The state was edited, is internally inconsistent, or was produced by a different GVV numerical implementation; use the matching current Fit output |
| Post reports component or fraction closure failure | Treat it as a numerical/implementation failure; do not use partial outputs |
| Post reports non-positive Term truth integral | An active fitted contribution is exactly zero/undefined for component efficiency; revisit the active model or reference choice |
| Efficiency is implausible but the job succeeds | Verify truth and selected MC are the same unweighted production before/after selection |
| Plotting rejects the projection schema | Use a projection written by the current schema-version-3 writer |
| Component curves do not add to total | Expected: the component diagnostic shows diagonals only and omits signed interference |
| Odd angular moments are nonzero | Investigate omega assignment/order bias; these are intentionally ordered-omega diagnostics |
| Same-tag outputs change unexpectedly | A repeated or concurrent job used the same directory and tag; use unique tags for simultaneous jobs |
| A queued job used unexpected configuration | Fit submission passes configuration paths and Post passes state/MC paths; do not edit the files used by a queued job until it finishes |
| CUDA runtime failure on `lxlogin` | Submit the relevant executable/test as a background Slurm payload instead of running it directly or entering a GPU node |

## Shared responsibility boundary

The repository automates deterministic configuration loading, GPU numerical
evaluation, fit diagnostics, output contracts, and Post handoff. The analyst
remains responsible for the physical model, reference-amplitude choice,
background prescription, MC provenance, job acceptance, systematic studies,
and scientific interpretation.

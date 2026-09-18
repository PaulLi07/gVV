# Operator workflow

## 1. Environment and build

Work from the repository root. `config/gvv_env.sh` selects CUDA 12 and ROOT
6.32.02 by default; set `GVV_CUDA_ROOT` and `GVV_ROOTSYS` before sourcing to
use another compatible installation. C++17, nlohmann/json.hpp, Python 3, Bash,
and GNU Make are required. This project does not need a BOSS environment.

```bash
source config/gvv_env.sh
make -j2 fit post
make check
make -j2 gpu-tests
```

The first two products are `bin/Fit.exe` and `bin/Post.exe`. The build generates
header dependencies and mirrored object paths under `build/obj/`. An unchanged
build reports "Nothing to be done". Do not interpret that as running the fit.

`make check` runs 15 host tests. `make gpu-tests` only compiles the four CUDA
runtime checks. On an allocated GPU, run `make check-gpu`; require all four
executables to return zero. No GPU job is submitted by any Make target.

## 2. Configure and submit

`config/fit.json` owns sample paths, background coefficients, minimizer settings,
and output tag/directories. `config/model.json` owns model definitions, active
Terms, coupling references, and fixed/free physical parameters. Edit JSON to
change these settings; no recompile is needed for configuration-only changes.
See [Model configuration](MODEL_CONFIGURATION.md).

All input files use tree `Pwa` with seven double[4] four-vectors in (px,py,pz,E)
order. The branch list and conversion live in `core/Sample.cu`. The nominal
signed sideband coefficients are -0.5 and +0.25. MC normalization uses the
mean intensity of the accepted normalization sample.

```bash
bash submit_fit.sh config/fit.json
```

The wrapper checks inputs and executable existence, submits one Slurm GPU job,
and records its job ID/log. Scheduler resources are grouped in the `#SBATCH`
block at the top of the wrapper. Fit policy lives in `fit/Fit.cu`, not in the
submission script. Reusing the tag overwrites the fit products and log.

## 3. Inspect the result

| Product | Purpose |
|---|---|
| `results/fit_result-<tag>.txt` | Attempts, selected status, parameters, covariance/correlation |
| `results/fit_state-<tag>.json` | Self-contained fitted state and embedded model (schema 2) |
| `results/projection-<tag>.root` | Plotting data, MC, backgrounds, maps, provenance (schema 4) |
| `runlog/fit-<tag>.log` | Environment, progress, errors, final output locations |

Configured output directories can differ. The selected attempt must pass
MIGRAD/HESSE, covariance, and EDM acceptance. Boundary flags are reported but
do not alone invalidate an attempt. If no attempt passes, no final output is
written; an existing file from an older run can still exist, so check the new
log and timestamps. Confirm fitted sigma and any other parameter on a bound
before drawing physics conclusions.

## 4. Plot

Plotting is independent of numerical Post and needs only a projection file:

```bash
source config/gvv_env.sh
bash post/plotting/draw.sh results/projection-omega_res_v1.root
```

Replace the tag with the one in your run. With no argument, `draw.sh` retains
its historical `initial` default. Outputs go to `post/plotting/results/`.
For an interactive single macro:

```bash
root 'post/plotting/macros/Draw_projection.cxx("results/projection-omega_res_v1.root")'
```

Edit the macro's user configuration block for binning/style. The migration
preserves the existing macros and output locations. Pair-component matrices
store each complete interference twice symmetrically; sum one triangle only.

## 5. Fractions and efficiencies

Supply a real generated-truth file and its corresponding accepted sample:

```bash
bash submit_post.sh results/fit_state-omega_res_v1.json \
  /path/to/truth_mc.root RootSet/normalization_mc.root
```

Post uses the embedded model and checks signatures and exact parameter order.
It integrates the same amplitude as Fit, checks component/PDF closure, and
propagates covariance through fractions, signed interference, and efficiencies.
Truth and selected MC must have the same generated exposure; the calculation
uses weighted sums. Output paths remain:

```text
post/calculation/results/post_result-<tag>.txt
post/calculation/results/post_result-<tag>.root
post/calculation/results/fit_fractions-<tag>.tex
runlog/post-<tag>.log
```

## Diagnosing failures

- Missing ROOT/CUDA commands: source the environment in the current shell.
- Build already up to date: submit the executable separately.
- Missing input/branch: check JSON paths and the `Pwa` branch contract.
- Unknown model field or reference: inspect the loader's ID/path diagnostic.
- No valid fit: inspect every attempt and limits in the fit log; do not treat
  old products with the same tag as new successful output.
- Post signature/layout mismatch: use the producing numerical implementation;
  do not bypass compatibility checks or replace the embedded model.
- GPU unavailable: run numerical fits/checks in an allocated GPU job.

## Refactor validation

Use a distinct output tag for any comparison run. Host regression coverage
includes dynamics, omega convolution, model/parameter compilation, inactive
pruning, state/signatures, Minuit policy, configuration, reports, and likelihood
arithmetic. GPU coverage includes tensor building blocks, registered Wave
numerics, total/component equivalence, and sigma-cache invalidation across
likelihood/projection/Post paths. GPU compilation alone does not establish
numerical equivalence. Record runtime check results and a same-configuration
fit comparison before treating the migrated build as production-validated.

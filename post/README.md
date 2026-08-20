# Post processing

The Post system contains two independent modules with different inputs and
execution requirements:

- **Post Calculation** restores the accepted fitted model, integrates
  Term-pair components over generated and selected MC, and propagates the fit
  covariance. It is a compiled CUDA program run as a background Slurm job.
- **Post Plotting** reads the Fit projection ROOT file and produces kinematic
  projections and angular-moment figures. It is an interpreted ROOT workflow
  normally run on an `lxlogin` node.

Neither module invokes the other. Calculation products are not inputs to the
plotting macros, and the projection ROOT file is not an input to Calculation.

```text
Fit
 +-- fit_state-<tag>.json + model.json + truth/selected MC
 |                              |
 |                              v
 |                     Post Calculation (Slurm GPU job)
 |                              |
 |                     TXT + ROOT + LaTeX
 |
 +-- projection-<tag>.root
                                |
                                v
                       Post Plotting (ROOT)
                                |
                           PDF + EPS
```

See the [build and analysis workflow](../docs/WORKFLOW.md) for the complete
Fit-to-Post operational sequence.

## Directory layout

```text
post/
├── README.md
├── calculation/
│   ├── ComponentEvaluator.cu
│   ├── ComponentEvaluator.h
│   ├── PostCalculation.cu
│   └── results/                 generated, not source-controlled
└── plotting/
    ├── GVVAngularMoments.h
    ├── GVVPlotUtils.h
    ├── draw.sh
    ├── macros/
    │   ├── Draw_projection_2_3.cxx
    │   ├── Draw_projection.cxx
    │   ├── Draw_projection_components.cxx
    │   ├── draw_angular_moments.cxx
    │   └── draw_angular_moments_odd.cxx
    └── results/                 generated, not source-controlled
```

## Part I: Post Calculation

### Purpose

Post Calculation answers numerical questions that should not be part of the
Minuit objective:

- What is each active Term's truth-level fit fraction?
- What are the signed interference fractions between Terms?
- What are the corresponding coherent JPC-group fractions and interference?
- What is the selection efficiency of each diagonal Term, each JPC group, and
  the complete coherent model?
- How does the fitted free-parameter covariance propagate to these
  observables?

Keeping this work outside Fit avoids repeatedly integrating truth MC during
minimization and gives the calculation its own explicit input/output contract.

### Build target and execution location

From the repository root on an `lxlogin` node:

```bash
source config/gvv_env.sh
make -j2 post
```

This builds `bin/Post.exe`. It is not part of the default `make` target.
`Post.exe` evaluates CUDA kernels and must run as the payload of a background
Slurm GPU job. Users do not enter the GPU node interactively and must not run
the executable directly on `lxlogin`. Use `submit_post.sh` for the normal
submission workflow.

### Four required inputs

The executable and submission wrapper use the same positional contract:

```text
fit_state.json model.json truth_mc.root normalization_mc.root
```

1. **Fit state** is `results/fit_state-<tag>.json` from an accepted Fit. It
   supplies the ordered best-fit free coordinates, complete covariance,
   output tag, and model signature.
2. **Model** is the exact `model.json` used by that Fit. It supplies all fixed
   physical parameters and reconstructs the active Resonance/Wave/Term model.
3. **Truth MC** is generated MC before analysis selection.
4. **Normalization MC** is the selected subset of that same unweighted
   production. It is the accepted sample used as the efficiency numerator.

Both MC files must contain a non-empty `Pwa` tree with

```text
p4_pip1  p4_pim1  p4_pi01
p4_pip2  p4_pim2  p4_pi02
p4_gam
```

as four-element `double` arrays in `(px, py, pz, E)` order. No event-weight
branch is read. Consequently, `selected_integral / truth_integral` has the
meaning of an efficiency only when the two files have compatible generation
normalization and truly represent before/after selection samples from the same
production.

The calculation intentionally does not read:

- `fit_result-<tag>.txt`, because that format is for humans;
- `projection-<tag>.root`, because it contains accepted-MC plot weights rather
  than the generated truth integration sample;
- `config/fit.json`, because run-time sample/minimizer policy is no longer
  needed after the fitted state has been selected.

### Contract validation

Before using fitted parameters, `Post.exe`:

- validates fit-state schema version 1 and its numerical dimensions;
- compiles the supplied `model.json` with the current GVV Wave registry;
- recomputes the deterministic model signature and compares it with the state;
- rebuilds the free-parameter layout and compares every name and index.

A mismatch is a hard error. Do not rename Terms, change active flags, reorder
or alter model definitions, or combine a state and model from different scan
points. The model path recorded inside the state is provenance; the model
passed on the command line is the one that is actually compiled and checked.

### Numerical algorithm

`post/calculation/ComponentEvaluator.*` owns the GPU-backed integration:

1. load truth and selected samples;
2. cache their registered-Wave Gram matrices and the omega-width lookup table;
3. apply the requested fitted coordinate vector to a compiled model copy;
4. evaluate all packed upper-triangle Term pairs in batches of at most 4096
   events;
5. reduce each pair directly on the GPU so no event-by-pair matrix is retained;
6. return one truth and one selected integral per pair;
7. independently evaluate the complete coherent intensity and require it to
   close to the pair sum for each sample.

For `T` active Terms, the packed pair order contains `T(T+1)/2` entries. A
diagonal is an individual `|A_i|^2` integral. An off-diagonal is the complete
signed `A_i A_j* + A_j A_i*` interference and appears once in the packed sum.

`post/calculation/PostCalculation.cu` converts those integrals into:

| Category | Definition |
|---|---|
| `fit_fraction` | One diagonal truth integral divided by the full coherent truth integral |
| `interference` | One signed Term-pair truth integral divided by the full coherent truth integral |
| `efficiency_component` | One diagonal selected integral divided by its diagonal truth integral |
| `efficiency_total` | Full coherent selected integral divided by the full coherent truth integral |
| `fit_fraction_group` | Sum of all pairs internal to one JPC group, divided by the full truth integral |
| `efficiency_group` | Selected internal-group sum divided by its truth internal-group sum |
| `interference_group` | Signed cross-JPC-group pair sum divided by the full truth integral |

The Term fractions plus all Term interferences must close to the coherent
total. The JPC-group fractions plus cross-group interferences must close
independently. Both residuals must have absolute value at most `1e-9`.

An active Term requires a positive diagonal truth integral to define its
component efficiency. A Term fitted exactly to zero produces an undefined
`0/0` efficiency and is rejected rather than assigned an artificial value.

### Covariance propagation

After the central calculation, Post evaluates numerical derivatives with
respect to every free coordinate stored in the fit state. The requested step
is the larger of a coordinate-scaled `1e-5` step and 5% of the fitted standard
deviation. It uses a central difference when the full step fits on both sides
and a one-sided, bound-aware difference otherwise.

The observable covariance is

```text
V_observable = J V_fit J^T,
```

where `J` is the finite-difference Jacobian and `V_fit` is the complete
free-parameter covariance. This accounts for fitted-parameter correlations.
It does not account for MC-integration statistics, detector/systematic
uncertainties, model alternatives, or external branching fractions.

### Submit the calculation

After accepting the Fit, run from the repository root on `lxlogin`:

```bash
./submit_post.sh \
  results/fit_state-<tag>.json \
  config/model.json \
  RootSet/truth_mc.root \
  RootSet/normalization_mc.root
```

All four arguments are required. Relative arguments are resolved against the
repository root. Before submission, `submit_post.sh` verifies all inputs and
`bin/Post.exe`, extracts `output_tag` from the fit state, creates
`runlog/` and `post/calculation/results/`, and submits one A100 Slurm job.
It passes absolute file paths rather than snapshots. Do not edit or replace the
state, model, truth MC, or selected MC while the job is queued or running.

The background worker sources `config/gvv_env.sh`, prints job/host/GPU
information, runs `nvidia-smi`, and launches:

```text
srun --ntasks=1 bin/Post.exe <state> <model> <truth> <selected>
```

The Post log is `runlog/post-<tag>.log`. The wrapper opens it with truncate
semantics and requires the final TXT and ROOT products to be non-empty.
`Post.exe` also writes the LaTeX table; the current wrapper does not include
that third file in its final existence check.

Users monitor the background job from `lxlogin`; they do not enter the GPU
node:

```bash
squeue -j <job-id>
sacct -j <job-id> --format=JobID,State,ExitCode,Elapsed
tail -f runlog/post-<tag>.log
```

### Outputs

All calculation products are tagged with the Fit state's `output_tag` and are
written under `post/calculation/results/`.

#### `post_result-<tag>.txt`

This readable table contains model/tag provenance, total truth and selected
integrals, both closure residuals, and one row per observable:

```text
category name value error truth_integral selected_integral
```

#### `post_result-<tag>.root`

The ROOT file has schema version 1 and contains:

- `observables` tree: ordered index, category, name, optional Term-pair
  indices, value, error, truth integral, and selected integral;
- `metadata` tree: output tag, model signature, truth/selected entry counts,
  observable count, and both closure residuals;
- `observable_covariance`: full `TMatrixDSym` in observable-index order;
- `observable_correlation`: the corresponding correlation matrix.

#### `fit_fractions-<tag>.tex`

This file is a compact two-column LaTeX table containing per-Term fit fractions
and propagated errors in percent. It intentionally does not contain the full
interference or efficiency result set; use TXT/ROOT for those values.

All three products replace existing same-tag files. Do not run two same-tag
Post jobs concurrently.

The files are written sequentially, not as an atomic set. A failed job may
leave one or more partial same-tag files. Accept the set only when Slurm
reports success and the log reaches the normal completion summary.

### Calculation acceptance checklist

Before using Post numbers:

1. confirm the Slurm job completed successfully;
2. read the log through the final TXT/ROOT/LaTeX paths;
3. require both component/direct-intensity closure checks to pass;
4. require `fraction_closure` and `group_closure` to be negligible;
5. confirm truth and selected entry counts match the intended production;
6. inspect finite-difference steps, especially for parameters near bounds;
7. inspect the observable covariance/correlation rather than quoting only
   diagonal errors;
8. remember that the reported uncertainty is fit-covariance-only.

## Part II: Post Plotting

### Purpose and sole input

Post Plotting visualizes the selected fitted model against data and configured
signed backgrounds. Its sole input is:

```text
results/projection-<tag>.root
```

The Fit projection already contains all event observables, fitted weights,
Term/JPC maps, background maps, and provenance required by the macros. Plotting
therefore does not read `model.json`, FitState, truth MC, or Calculation
products.

### Run the plot driver

From the repository root:

```bash
post/plotting/draw.sh results/projection-<tag>.root
```

The script resolves its own location, sources `config/gvv_env.sh`, verifies
the input and ROOT executable, derives the tag from the input basename,
creates `post/plotting/results/`, and invokes each macro with ROOT in batch
mode. From another directory, invoke the script by an absolute path and pass
an absolute Projection path, or one relative to that caller's current
directory; a relative input argument is resolved from the caller's `$PWD`.

Plotting uses CPU-side ROOT and is normally appropriate on `lxlogin`. It does
not submit a Slurm job and does not require `bin/Post.exe` or an allocated
GPU. The current driver nevertheless sources the full project environment,
which checks for the configured `nvcc`. Until a ROOT-only loader is added, the
CUDA installation must therefore be readable even though plotting executes no
CUDA code.

### Run one macro directly

Each `.cxx` file is a complete ROOT macro whose top-level function matches its
filename. After loading the environment, execute it directly from the project
root:

```bash
source config/gvv_env.sh
root post/plotting/macros/Draw_projection_2_3.cxx
```

No-argument execution uses `results/projection-initial.root` and the macro's
matching `initial` output prefix under `post/plotting/results/`. Both defaults
are visible in the function signature and in the `User configuration` block
near the top of the file. Relative input/output arguments are interpreted from
the project root, so this also works:

```bash
cd post/plotting/macros
root Draw_projection_2_3.cxx
```

Pass an explicit Projection and output prefix for another fit tag:

```bash
root -l -b -q \
  'post/plotting/macros/Draw_projection_2_3.cxx("results/projection-TAG.root","post/plotting/results/projection-TAG")'
```

Use the same calling convention for the other four macros. `draw.sh` remains
the convenient way to derive one tag and run all five together. An absolute
input or output path is accepted unchanged.

### Where to change a figure

Each macro begins with a clearly delimited `User configuration` block. It owns
the complete presentation configuration:

- observable list or Legendre orders;
- bins and numerical axis ranges;
- axis titles and event/bin normalization labels;
- canvas dimensions and pad layout;
- curve colors, line/marker/fill styles, and draw order;
- legends, panel labels, diagnostic text, and output defaults.

`GVVPlotUtils.h` retains only schema-v2 input handling, dynamic maps, branch
binding, exchange-symmetric observable filling, unstyled projection-histogram
construction, Pearson diagnostics, project-root path resolution, and the common
base style. `GVVAngularMoments.h` retains only Legendre/moment
arithmetic, unstyled moment-histogram construction, and the moment diagnostic.
Do not add figure-specific axes, binning, canvas layout, or colors to either
header.

### Projection contract consumed

`GVVPlotUtils.h` requires projection schema version 2. It reads:

- `data`: unweighted selected data observables;
- `MC`: accepted normalization-MC observables, total fitted `weight`, coherent
  `weight_group`, and symmetric `weight_component`;
- `bg`: background observables, zero-based `background_index`, and signed
  plotting `weight_bg`;
- `component_map`: current Term indices and display labels;
- `group_map`: current coherent JPC group indices and labels;
- `background_map`: current background metadata;
- `metadata`: schema version and declared background count.

No nominal Resonance list, fixed Term count, fixed JPC set, or SB1/SB2 field is
compiled into the drawing utilities. Changing active Terms or configured
background samples is discovered from the maps written by Fit.

The main total expectation is fitted signal plus the signed background
histogram. The signed background projection weight is the negative of the
corresponding likelihood coefficient, matching the data-minus-background
visual convention.

### Five plot products

`draw.sh` runs these ROOT entry points:

| Entry point | Interpretation | Output basename |
|---|---|---|
| `Draw_projection_2_3.cxx` | Main 3x2 exchange-symmetric projection | `projection-<tag>` |
| `Draw_projection.cxx` | Detailed 4x2 projection including the exchange-symmetrized omega-candidate mass | `projection_detailed-<tag>` |
| `Draw_projection_components.cxx` | Per-Term diagonal component diagnostic | `projection_components-<tag>` |
| `draw_angular_moments.cxx` | Symmetrized even `P0/P2/P4/P6` moments | `angular_moments-<tag>` |
| `draw_angular_moments_odd.cxx` | Ordered-omega `P1/P3/P5` diagnostic | `angular_moments_odd_diagnostic-<tag>` |

Each basename is written as both PDF and EPS under
`post/plotting/results/`, for ten files in a complete run.

The main observables are `M(omega omega)`, symmetrized `M(gamma omega)`,
`cos(theta_gamma)`, symmetrized `cos(theta_omega)`, the omega decay-plane
angle, and symmetrized decay-plane-angle difference. The detailed projection
also fills the two omega-candidate `M(pi+ pi- pi0)` values with half weight,
forming one exchange-symmetrized candidate-mass distribution.

The component diagnostic draws only diagonal `|A_i|^2` entries. It cannot and
should not close to the total coherent curve when interference is present.
The coherent JPC-group curves include only pairs internal to each group;
cross-group interference remains in the total model.

Even moments are exchange-symmetrized physical diagnostics for the identical
omega pair. Odd moments deliberately retain the input-labelled `omega1`
ordering and should be interpreted only as assignment/order-bias diagnostics.

### Plotting acceptance checklist

1. confirm all ten PDF/EPS products were created;
2. check ROOT printed no missing-tree, missing-branch, schema, or map-size
   exception;
3. verify the legends contain the intended dynamic Terms and JPC groups;
4. inspect signed-background behavior and any bins with a non-positive total
   expectation;
5. compare the displayed Pearson `chi2/Nbin` only as a projection diagnostic,
   not as the unbinned fit objective or a complete global goodness-of-fit;
6. treat odd moments as ordering diagnostics, not physical odd moments of an
   unlabeled identical-omega state.

## Common Post errors

| Error or symptom | Interpretation |
|---|---|
| `model.json does not match the model used by this fit state` | State and model came from different definitions or scan points |
| `fit-state parameter order does not match model` | Active Terms or free-parameter layout changed after Fit |
| Missing `Pwa` tree/branch | MC file does not satisfy the shared GVV sample contract |
| Non-positive Term truth integral | Active component efficiency is undefined, commonly because the fitted coupling is exactly zero |
| Component/PDF or fraction closure failure | Numerical or implementation inconsistency; do not use the partial results |
| No resolvable finite-difference step | A fitted coordinate lies too tightly against its configured bound |
| Implausible total efficiency | Truth and selected files may not be the same unweighted production |
| Plotting rejects schema version | Projection was written by an incompatible/legacy Fit writer |
| Missing map or vector-size mismatch | Projection file is incomplete or does not match its metadata |
| Components do not sum to total | Expected for diagonal-only component plots because interference is omitted |
| Same-tag result disappeared | A later calculation or plotting run overwrote it by design |
| Queued job used unexpected state/model content | The wrapper passes paths without copying files; keep all four inputs immutable through completion |

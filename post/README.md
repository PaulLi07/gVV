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
 +-- fit_state-<tag>.json (embedded model) + truth/selected MC
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
    ├── PLOTTING_STYLE.md
    ├── draw.sh
    ├── macros/
    │   ├── Draw_projection.cxx
    │   ├── Draw_projection_components.cxx
    │   ├── Draw_polarization.cxx
    │   ├── Draw_omega_decay_checks.cxx
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

### Three required inputs

The executable and submission wrapper use the same positional contract:

```text
fit_state.json truth_mc.root normalization_mc.root
```

1. **Fit state** is `results/fit_state-<tag>.json` from an accepted Fit. It
   supplies the ordered best-fit free coordinates, complete covariance,
   output tag, complete formatted model definition, and compatibility
   signatures. The recorded source model path is provenance only.
2. **Truth MC** is generated MC before analysis selection.
3. **Normalization MC** is the selected subset of that same unweighted
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
- external `model.json`, because the exact Fit model is embedded in the state;
- `config/fit.json`, because run-time sample/minimizer policy is no longer
  needed after the fitted state has been selected.

### Contract validation

Before using fitted parameters, `Post.exe`:

- validates fit-state schema version 2 and its numerical dimensions;
- parses and compiles the structured `model.definition` embedded by Fit;
- recomputes the definition signature, compares the explicit GVV numerical
  implementation identifier, and checks their combined compatibility key;
- rebuilds the free-parameter layout and compares every name and index.

A mismatch is a hard error. Do not edit the embedded Terms, active flags,
parameter definitions, signatures, or free-coordinate order. Schema-version-1
states do not contain the model and are intentionally rejected; rerun Fit with
the current executable to produce a schema-version-2 state.

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
  RootSet/truth_mc.root \
  RootSet/normalization_mc.root
```

All three arguments are required. Relative arguments are resolved against the
repository root. Before submission, `submit_post.sh` verifies all inputs and
`bin/Post.exe`, extracts `output_tag` from the fit state, creates
`runlog/` and `post/calculation/results/`, and submits one A100 Slurm job.
It passes absolute file paths rather than snapshots. Do not edit or replace the
state, truth MC, or selected MC while the job is queued or running.

The background worker sources `config/gvv_env.sh`, prints job/host/GPU
information, runs `nvidia-smi`, and launches:

```text
srun --ntasks=1 bin/Post.exe <state> <truth> <selected>
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
post/plotting/draw.sh
post/plotting/draw.sh results/projection-<tag>.root
```

The zero-argument form reads `results/projection-initial.root`. The optional
single argument selects another Projection; additional arguments are rejected.
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
root post/plotting/macros/Draw_projection.cxx
```

No-argument execution uses `results/projection-initial.root` and the macro's
matching `initial` output prefix under `post/plotting/results/`. Both defaults
are visible in the function signature and in the `User configuration` block
near the top of the file. Relative input/output arguments are interpreted from
the project root, so this also works:

```bash
cd post/plotting/macros
root Draw_projection.cxx
```

Pass an explicit Projection and output prefix for another fit tag:

```bash
root -l -b -q \
  'post/plotting/macros/Draw_projection.cxx("results/projection-TAG.root","post/plotting/results/projection-TAG")'
```

Use the same calling convention for the other five macros. `draw.sh` remains
the convenient way to derive one tag and run all six together. An absolute
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

`GVVPlotUtils.h` retains only schema-v3 input handling, dynamic maps, branch
binding, exchange-symmetric observable filling, unstyled projection-histogram
construction, Pearson diagnostics, project-root path resolution, and the common
base style. `GVVAngularMoments.h` retains only Legendre/moment
arithmetic, unstyled moment-histogram construction, and the moment diagnostic.
Do not add figure-specific axes, binning, canvas layout, or colors to either
header.

The [plotting style guide](plotting/PLOTTING_STYLE.md) is the authoritative
human-readable contract for visual roles, layout, frame and candidate naming,
automatic ranges, and review. In particular, data use black markers,
Background uses a gray hatched histogram, the solid blue Total-fit appearance
is reserved, all coherent JPC groups use line style 2 with distinct colors and
the legend wording `coherent <JPC>`, and Term-component styles are
deterministic functions of the component metadata. The vertical envelope
includes data errors, signed Background, Total fit, and every group or
component curve actually drawn. Individual Term components use thin width-1
lines. Their enlarged `3 x 2` physics area leaves a narrow external right
margin for a compact centered two-column legend block, following the
conventional projection-plot layout. Every other shared legend is inside the
first subplot, which alone receives extra vertical headroom.

### Projection contract consumed

`GVVPlotUtils.h` requires projection schema version 3. It reads:

- `data`: unweighted selected data observables;
- `MC`: accepted normalization-MC observables, total fitted `weight`, coherent
  `weight_group`, and symmetric `weight_component`;
- `bg`: background observables, zero-based `background_index`, and signed
  plotting `weight_bg`;
- `component_map`: current Term indices, display labels, Resonance IDs, Wave
  IDs/labels, device types, and JPC values;
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

### Six plot products

`draw.sh` runs these ROOT entry points:

| Entry point | Interpretation | Output basename |
|---|---|---|
| `Draw_projection.cxx` | Main 3x2 exchange-symmetric projection | `projection-<tag>` |
| `Draw_projection_components.cxx` | Enlarged `3x2` kinematic area plus a compact centered two-column legend in the external right margin for thin per-Term diagonal curves | `projection_components-<tag>` |
| `Draw_polarization.cxx` | Three horizontal omega decay-plane-normal projections with the shared legend inside the first subplot | `polarization-<tag>` |
| `Draw_omega_decay_checks.cxx` | Pion-angle and pion-pair-mass checks | `omega_decay_checks-<tag>` |
| `draw_angular_moments.cxx` | Symmetrized even `P0/P2/P4/P6` moments | `angular_moments-<tag>` |
| `draw_angular_moments_odd.cxx` | Ordered-omega `P1/P3/P5` diagnostic | `angular_moments_odd_diagnostic-<tag>` |

Each basename is written as both PDF and EPS under
`post/plotting/results/`, for twelve files in a complete run.

The main observables are `M(omega omega)`, candidate-combined
`M(gamma omega_i)`, `cos(theta_gamma)` in the `psi(2S)` rest frame,
exchange-symmetric `cos(theta_omega)` and `phi_omega` in the X helicity frame,
and the two omega-candidate `M(pi+ pi- pi0)` values combined with half weight
each.
The omega2 azimuthal exchange image is reconstructed as wrapped
`phi_omega1 + pi` because the two omegas are back-to-back in the X frame.

The polarization figure combines both candidates for
`cos(theta_n_omega)` and `phi_n_omega` of the oriented
`n_i = unit[p(pi+_i) cross p(pi-_i)]` analyzer in each omega helicity frame,
and symmetrizes the signed, wrapped `Delta phi(n_1,n_2)`. Its three panels are
horizontal and keep the shared legend inside the first subplot with dedicated
headroom. The omega-decay check figure combines both candidates with
half weight each for `cos_theta_pip_omega`, `cos_theta_pim_omega`,
`cos_theta_pi0_omega`, `M(pi+ pi-)`, `M(pi+ pi0)`, and `M(pi- pi0)`. The pion
cosines are not additionally reflected because each is already defined in its
parent omega helicity frame. The stored unnormalized plane-normal magnitudes
remain available for future dedicated studies but are not included in these
standard figures.

The component diagnostic draws only diagonal `|A_i|^2` entries. It cannot and
should not close to the total coherent curve when interference is present.
It assigns every active Term a stable, distinct color/line-style combination
from its Resonance, Wave, and Term metadata rather than its current list
position. Each component is deliberately a thinner width-1 line so the many
curves remain legible without competing with the width-2 Total fit.
The coherent JPC-group curves include only pairs internal to each group;
cross-group interference remains in the total model.

Even moments are exchange-symmetrized physical diagnostics for the identical
omega pair. Odd moments deliberately retain the input-labelled `omega1`
ordering and should be interpreted only as assignment/order-bias diagnostics.
Both moment figures show the unnormalized binwise sum of Legendre weights, not
an event-normalized average; their mass-bin width is derived from the configured
range and bin count. A gray zero reference is drawn for every signed nonzero
moment, and the legend identifies `Data - signed background` and fitted signal
MC.

### Plotting acceptance checklist

1. confirm all twelve PDF/EPS products were created;
2. check ROOT printed no missing-tree, missing-branch, schema, or map-size
   exception;
3. verify the legends contain the intended dynamic Terms and JPC groups, and
   that no two Terms have the same final appearance;
4. verify Background is gray and hatched, Total fit retains its reserved
   style, and every coherent JPC group uses the common dashed style and the
   label `coherent <JPC>`;
5. inspect signed-background behavior and any bins with a non-positive total
   expectation;
6. confirm the complete vertical envelope includes data errors and every drawn
   curve without clipping;
7. compare the displayed Pearson `chi2/Nbin` only as a projection diagnostic,
   not as the unbinned fit objective or a complete global goodness-of-fit;
8. treat odd moments as ordering diagnostics, not physical odd moments of an
   unlabeled identical-omega state.

## Common Post errors

| Error or symptom | Interpretation |
|---|---|
| `fit-state schema version 1 does not embed the model definition` | Rerun Fit with the current executable before Post Calculation |
| Embedded definition or implementation signature mismatch | The state was edited or the Post executable implements a different numerical GVV amplitude contract |
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
| Queued job used unexpected state/MC content | The wrapper passes paths without copying files; keep all three inputs immutable through completion |

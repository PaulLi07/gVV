# Post system

The Post system has two independent modules. Neither is part of the default
`make` target.

## Calculation

`make post` builds `bin/Post.exe`. Its four inputs are:

1. `results/fit_state-<tag>.json`, the machine-readable fitted state;
2. the exact `model.json` used by the fit;
3. generated truth MC before selection;
4. selected normalization MC from the same production.

Both MC files use the same process input contract as the fit samples: a `Pwa`
tree with `p4_pip1`, `p4_pim1`, `p4_pi01`, `p4_pip2`, `p4_pim2`,
`p4_pi02`, and `p4_gam`, each stored as `(px, py, pz, E)`. The selected file
must represent the selected subset of the same unweighted production as the
truth file; otherwise a raw selected/truth integral ratio is not an efficiency.

The executable validates the model signature and free-parameter order, applies
the fitted state, integrates every diagonal and interference component, and
propagates the fit covariance by finite differences. Tagged text, ROOT, and
LaTeX products are written under `post/calculation/results/`.

Term-pair values are reduced on the GPU in bounded event batches. Post retains
one truth and one selected integral per packed upper-triangle pair rather than
an event-by-pair matrix. Reported observable errors propagate only the fit
covariance; they do not include MC-integration statistics or systematic
uncertainties.

Build and submit Post Calculation from the repository root:

```bash
make post
./submit_post.sh \
  results/fit_state-<tag>.json \
  config/model.json \
  RootSet/truth_mc.root \
  RootSet/normalization_mc.root
```

Fit jobs use the separate `submit_fit.sh` entry point; the two submission
scripts do not dispatch to each other.

The human `fit_result-<tag>.txt` is deliberately not an input. It can change
presentation without breaking Post Calculation.

## Plotting

`post/plotting/draw.sh` reads only `results/projection-<tag>.root`. It creates
the projection and angular-moment figures under `post/plotting/results/`.
Component labels and coherent groups are read from `component_map` and
`group_map`; no resonance list or fixed JPC set is compiled into the macros.
Projection schema version 2 also provides a dynamic `background_map`. The
`bg` tree stores a zero-based `background_index` and its signed `weight_bg`;
there are no fixed SB1/SB2 branches or metadata fields.

These modules may be run independently after a fit. Post Calculation needs MC
integration samples but not the projection file, while Post Plotting needs the
projection file but not the fit-state JSON or MC input files.

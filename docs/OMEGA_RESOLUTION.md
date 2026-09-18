> Current structure: see [Architecture](ARCHITECTURE.md). Model compilation and parameter binding now live together in `core/Model.cu`; Fit policy is local to `fit/Fit.cu`.

# Effective Gaussian convolution of the omega propagator

The nominal `main` model fits one shared omega resolution width. It convolves
**each complex omega propagator in mass**, before the coherent amplitude is
squared. It preserves the existing running omega width and every X propagator,
rho-isobar factor, four-vector, angular tensor and barrier factor.

## Definition and scope

For the measured three-pion invariant mass `m = sqrt(s_omega)`, define

\[
 D_\omega(s)=\frac{1}{m_\omega^2-s-i m_\omega\Gamma_\omega(s)},\qquad
 \widetilde D_\omega(m;\sigma)=\int_0^\infty
 \frac{e^{-(m-m')^2/(2\sigma^2)}}{\sqrt{2\pi}\sigma}
 D_\omega(m'^2)\,dm'.
\]

The two omega candidates share the same constant `sigma`; the Gaussian mean
is fixed to zero. The existing `m_omega = 0.78266 GeV` and
`Gamma_omega = 0.00868 GeV` remain fixed. `OmegaWidthTable` supplies the same
three-pion running width, interpolation, and endpoint-clamping policy as before.
The integration measure is `dm'`, not `ds'`. There is no additional Jacobian,
no Gaussian applied to X, and no shift of other kinematic quantities inside
the integral. There is no renormalization over a selected omega mass window.

For each event the replacement is

\[
 A(x;\sigma)=\widetilde D_\omega(m_1;\sigma)
             \widetilde D_\omega(m_2;\sigma)
             R_{\rho,1}(x)R_{\rho,2}(x)
             \sum_t c_t P_{X,t}(s_X)W_t(x).
\]

The implementation computes the real and imaginary parts of the integral,
then evaluates the coherent intensity. In particular,
`|integral G D|^2` is not `integral G |D|^2`. This is the requested effective
amplitude model; its fitted sigma is not automatically a calibrated detector
resolution. No multidimensional detector response or unfolding is introduced.
Changing the common event weight can change fitted X parameters or projected
X distributions even though the X propagator formulas themselves are unchanged.

## Model configuration and fit coordinates

The optional top-level `process_parameters` object in `config/model.json`
contains process-owned parameters. GVV currently accepts only
`omega_resolution_sigma` in this object. The nominal configuration is:

```json
"process_parameters": {
  "omega_resolution_sigma": {
    "value": 0.005,
    "fixed": false,
    "transform": "log",
    "step": 0.05,
    "bounds": [-9.210340371976182, -3.506557897319982]
  }
}
```

`value` is in GeV. Minuit uses the dimensionless numerical coordinate
`log_sigma_omega = log(sigma / GeV)`, with `step` and both `bounds` expressed
in that coordinate. These bounds correspond to `0.1 <= sigma <= 30 MeV`.
The 5 MeV starting value and search interval are initial analysis settings,
not a measured resolution or an external calibration constraint.

The new coordinate is appended after the existing coupling and X propagator
coordinates, taking the current nominal model from 35 to 36 free parameters.
The minimizer floats it in every start; its initial value is not randomized,
matching the existing line-shape parameter policy. It is not duplicated for
the two omega candidates or for individual Terms. A free sigma requires the
`log` transform, a positive starting value, and finite bounds. The supported
upper sigma is 50 MeV; configurations and restored coordinates beyond that
range are rejected, allowing only floating-point rounding at the boundary.

For a fixed resolution use, for example:

```json
"process_parameters": {
  "omega_resolution_sigma": {"value": 0.005, "fixed": true}
}
```

For the exact legacy calculation either omit `process_parameters` entirely,
or use `{"value": 0.0, "fixed": true}` for its sigma entry. Do not retain
`transform: "log"` for a zero value. Fixed sigma adds no fit coordinate.
Explicit fixed zero uses the original unsmeared kernel path. Extremely small
positive sigma below floating-point mass spacing also uses the zero limit.

## Numerical evaluation and cache lifecycle

`core/physics/OmegaResolution.cuh` implements a host/device quadrature. It truncates
the Gaussian at eight standard deviations (the full Gaussian omitted tail is
less than `1.3e-15`) and retains positive true mass. It uses 16-point
Gauss-Legendre panels of width at most `min(2 sigma, Gamma_omega)`, split at
all running-width interpolation knots. Splitting prevents interpolation slope
changes from creating avoidable quadrature noise during minimization.

`GVVSample` stores one complex common omega factor per event, including the
unchanged rho factors: 16 bytes per event. It recomputes this buffer only when
sigma changes. Changing an X parameter or a coupling reuses the buffer and
the existing geometric F matrix. Switching to zero sigma selects the exact
legacy path; returning to positive sigma refreshes the cache. Each sample
owner uses one immutable omega-width table after preparation. If future work
makes that table mutable, its revision must also invalidate the cache.

Fit data, selected normalization MC, both signed background samples, projection
components and Post integrations receive the same current sigma. In particular,
normalization is recomputed with the updated intensity; sigma cannot be applied
only to data or only to the mass plots. Batched component evaluation indexes
the cache by the full sample event index, including nonzero batch offsets.

Post retains the existing effective-amplitude convention for both truth and
selected MC. Its efficiencies and fractions therefore belong to this chosen
model and MC normalization convention; this change does not introduce a
separate response-corrected physical truth amplitude. Sigma also participates
in the saved covariance and Post's finite-difference uncertainty propagation.

## Saved results and running the analysis

`fit_state-<tag>.json` embeds the complete model, `log_sigma_omega`, and the
full covariance. Post restores sigma from that vector. The text fit report
also prints sigma in GeV and its linearly propagated error
`sigma * error(log_sigma_omega)`. This symmetric error approximation should not
be interpreted as a profile interval when the likelihood is asymmetric or a
boundary is reached.

An explicitly configured sigma selects implementation signature
`gvv-amplitude-omega-gauss-v1`; an old model without the entry retains
`gvv-amplitude-contract-v1`. Combined model signatures include the canonical
model hash. Post checks both so new and legacy model states cannot be silently
mixed. Old saved models can still be loaded in unsmeared mode. Projection
metadata gains `omega_resolution_sigma` and `omega_resolution_mean` branches;
these additive branches preserve the existing schema-4 plotting interface.

The default fit output tag is now `omega_res_v1`, producing:

```text
results/fit_result-omega_res_v1.txt
results/fit_state-omega_res_v1.json
results/projection-omega_res_v1.root
runlog/fit-omega_res_v1.log
```

After the GPU validation below, the existing submission entry point remains
`./submit_fit.sh config/fit.json`. Submission is a separate operational action;
a build or a host test does not run a fit. Plot the new projection explicitly:

```bash
post/plotting/draw.sh results/projection-omega_res_v1.root
```

The drawing script's no-argument legacy default is unchanged. Reusing an output
tag retains the project's overwrite behavior, so use a different tag for each
resolution hypothesis whose results should be kept.

## Regression checks

From the project root on a login node:

```bash
source config/gvv_env.sh
make -j2 fit post tests gpu-tests
make check
```

The 15 host tests include independent dense Simpson integration using
`std::complex` arithmetic, convergence with twice as many integration points,
zero and small-sigma limits, interpolation/end-point coverage, distinction
from intensity convolution, sigma derivative stability, strict model input,
35/36-parameter compatibility, preservation of X parameters, and saved-state
recovery with the full covariance.

Inside an allocated GPU job, from the same project root:

```bash
source config/gvv_env.sh
make check-gpu
```

The four GPU regression executables include
`build/tests/test_omega_resolution_gpu.exe`. Its ROOT fixtures check CPU/GPU
omega reweighting, changes and restoration of sigma, all signed likelihood
samples, normalization, full and offset component batches, Post truth/selected
integration and total closure, and projection metadata. The fixtures are
numerical tests, not detector-smeared toys or a fit-bias study.

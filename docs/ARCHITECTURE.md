# Concentrated fit architecture

This branch implements one channel, psi(3686) -> gamma omega omega. The
architecture separates reusable numerical work from the two analysis scripts.
It does not attempt to be a configurable multi-channel framework.

## End-to-end flow

```text
fit.json + model.json -> Model + IO -> immutable compiled topology + numeric state
ROOT samples -> Sample -> momenta + fixed Wave Gram matrices F_ab
                                     |
Fit.cu: parameter coordinates -> Model -> Amplitude -> I(data), I(MC), I(background)
                                     |                     |
                                  Minuit <---- signed normalized NLL
                                     |
                    best attempt -> TXT + fit-state JSON + projection ROOT
                                             |                   |
Post.cu + truth/selected MC -> Amplitude -> observables       ROOT plotting
```

`fit/Fit.cu` contains the complete nominal workflow: load configuration,
assign sample roles, prepare caches, define the objective, run multistart,
select a valid result, restore its parameters, and write outputs. Its local
`FitLikelihood` is a sample bundle for this script, not a public framework
interface. Likelihood arithmetic and text-report helpers stay in this file.

`post/Post.cu` owns truth/selected sample roles, observable definitions,
finite-difference displacements, covariance propagation, and output tables.
Its local evaluator delegates all amplitude work to `GVVAmplitude`.

## Five public modules

| Module | Owns | Does not decide |
|---|---|---|
| `Model.h` | Parsed definition, active dependency compilation, immutable topology, one final binding layout, numeric parameter transforms | NLL, sample roles, optimizer acceptance |
| `Sample.h` | ROOT branch mapping, host/device momenta, F matrix, intensity buffer, sigma cache | Whether a sample is data, MC, sideband, or truth |
| `Amplitude.h` | Width table, device model buffers, intensity and Term-pair evaluation, bounded scratch storage | Likelihood signs, efficiency definition, covariance steps |
| `Minuit.h` | Parameter specifications, callback bridge, random starts, MIGRAD/HESSE and best-fit policy | Physics model and ROOT event schema |
| `IO.h` | Run config, fitted-state JSON, projection interface | Fit objective or Post observables |

`AmplitudeKernels.cuh/.cu` are internal CUDA entry points, not a sixth public
workflow module. `Projection.cu` implements the IO projection contract and
keeps ROOT geometry/serialization away from the fit script. Mathematical
building blocks and registered physics remain meaningful separate units.

## Model and state lifetime

`GVVCompiledModel` contains the original canonical JSON definition, active
Resonance/Term metadata, dense Terms, active Wave IDs, the initial numeric
state, and **one** final `parameters` layout. The compiler creates that layout
once: free Term couplings, free Resonance parameters, then optional omega
resolution. It compiles only Resonances needed by active Terms; unused
unsupported line shapes do not suddenly become errors.

`GVVParameterState` contains only Resonance numerical descriptors, couplings,
and physical omega sigma. `gvv_apply_fit_parameters` changes this state without
copying JSON, labels, topology, or bindings. Fit and Post each keep a working
state. `GVVAmplitude` borrows the immutable model, which must outlive it.

Device Terms are uploaded once. Numerical Resonance parameters and couplings
are synchronized by `SetParameters`; every sample evaluation uses the current
sigma. Sample caches rebuild omega factors only when physical sigma changes.
The angular F matrix is independent of fitted couplings, line shapes, and
sigma. Projection batches reuse scratch space; Post integrals retain one value
per pair instead of an event-by-pair matrix.

## Numerical contracts

With Term t using Wave w(t), the scalar coefficient is the coupling times the
X propagator and the common two-omega/rho factor. The optimized total groups
these coefficients by exact Wave slot, then contracts the full real Gram
matrix. Complexity remains O(T + W^2). The phase-reference classes do not mask
cross terms. Tensor indices use (+---), epsilon(0123)=+1; input arrays are
(px,py,pz,E), while `FV` stores (E,x,y,z).

The accepted-MC normalization is the mean intensity. The nominal script
minimizes minus the data log likelihood plus configured signed background
log likelihoods; nominal coefficients are -0.5 and +0.25. Arithmetic and
reduction order are retained. Random starts modify only eligible couplings,
with the same seed sequence and convergence/tie-selection rules as before.

Pair components are packed by the upper triangle: diagonal K_ii and complete
signed off-diagonal K_ij + K_ji. Projection writes the existing full symmetric
T-by-T storage, duplicating each complete off-diagonal entry; consumers must
sum only one triangle. Post integrates packed components and validates closure
against the total PDF. Efficiency uses selected/truth **sums** for matched
production exposure, not separate sample means.

Post finite differences retain the original scale-dependent step, central
stencils where bounds allow, and one-sided stencils at bounds. Errors use
J C J^T, including the fitted omega-resolution coordinate.

## Extension guide

- Existing Wave / different resonances or bounds: edit model JSON.
- Different nominal objective or scan policy: edit `fit/Fit.cu`, or add one
  substantial independent fit script when its workflow is actually needed.
- New Wave: implement in its relevant `physics/waves/` group, register once in
  `Waves.cuh/.cu`, and add numerical checks.
- New propagator: update `Propagators.cuh`, its thin numerical descriptor in
  `AmplitudeTypes.h` if needed, and compilation/binding in `Model.cu`.
- New plotting observable: update projection IO and the relevant ROOT macro.

A future model-independent `FitBins.cu` should own bin boundaries, per-bin
sample selection, free Wave coefficients, phase/scale conventions, sigma
policy, same-bin MC normalization, and output provenance. It may reuse Model,
Sample, Amplitude, Minuit, and IO; do not add empty scan classes or a generic
pipeline in advance. A new sample-selection API should be introduced only with
that concrete workflow and its required normalization tests.

## Compatibility

No physics implementation signature is bumped for this structural migration.
Canonical JSON hashing and existing signatures are preserved. Run/model schema
1, fit-state schema 2, projection schema 4 and all operational paths stay the
same. Post reconstructs the embedded model and verifies signatures and ordered
parameters before evaluating it. A future numerical physics change must update
its compatibility signature, not silently reinterpret old fitted states.

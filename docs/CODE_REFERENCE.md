# Source navigation

Start with `fit/Fit.cu` for a complete nominal analysis. It contains four
ordered sections: signed likelihood arithmetic, text report, sample roles,
and the main workflow. The main function loads JSON, prepares samples,
defines the Minuit objective, selects/restores the best attempt, and writes
three outputs. Parameter/sample/minimizer values belong in JSON; its local
helpers contain the analysis policy.

## Core interfaces

| Files | Main symbols and responsibility |
|---|---|
| `core/Model.h`, `Model.cu` | `GVVCompiledModel`, `GVVParameterState`, `gvv_compile_model`, `gvv_fit_parameter_layout`, `gvv_apply_fit_parameters`; active compilation and one final binding layout |
| `core/ModelIO.cpp` | `parse_model_definition`, `load_model_definition`, canonical document signature; parses typed Term dynamics once |
| `core/Sample.h`, `Sample.cu` | `GVVBranchConfig`, `GVVSample`; seven-particle ROOT boundary, F and omega caches |
| `core/Amplitude.h`, `Amplitude.cu` | `GVVAmplitude`; Prepare, SetParameters, EvaluateIntensity, EvaluateComponentBatch, IntegrateComponents |
| `core/Minuit.h`, `Minuit.cpp` | `FitParameterSpec`, `FitOptions`, `FitAttempt`, `FitSummary`, `run_multistart_fit` |
| `core/IO.h`, `IO.cpp` | `FitRunConfig`, `FitState`, run config and fitted-state validation/serialization |
| `core/Projection.cu` | `write_gvv_projection`; ROOT geometry, weights, component/group maps and metadata |
| `core/AmplitudeKernels.cuh`, `.cu` | Internal `CalGVVFmatrix`, `CalGVVOmegaFactors`, `CalGVVPDF`, `CalGVVComponentBatch`, `CalGVVComponentIntegrals` |

`GVVAmplitude` borrows a model and evaluates any prepared `GVVSample`; it does
not own sample roles. `SetParameters` updates numeric device state. Each
sample evaluation refreshes its own omega cache if needed. `Model()` remains
the immutable definition/topology; fitted numbers live in the caller's
`GVVParameterState`.

## Mathematical and physical building blocks

| File/group | Contents |
|---|---|
| `core/math/Complex.cuh` | Host/device complex arithmetic |
| `core/math/FourVector.cuh` | FV, metric and epsilon conventions |
| `core/math/Tensor.cuh` | Rank-two tensor operations |
| `core/math/TensorOps.cuh` | Explicit contractions and transverse/spin-two projection |
| `core/math/OrbitalTensor.cuh` | Bare orbital tensor construction |
| `core/math/BarrierFactor.cuh` | Barrier factors and radius convention |
| `core/physics/AmplitudeTypes.h` | Thin numerical Resonance/Term records, no JSON metadata |
| `core/physics/Event.cuh` | Omega decay current, event and device momentum views |
| `core/physics/Propagators.cuh` | Line-shape formulas, two-body kinematics, enum dispatch |
| `core/physics/OmegaDecay.cuh` | Shared rho-isobar decay factors and particle constants |
| `core/physics/OmegaWidth.h`, `.cu` | Three-body width integration, table view/interpolation/upload |
| `core/physics/OmegaResolution.cuh` | Complex omega convolution, exact zero-sigma path |
| `core/physics/Waves.cuh`, `.cu` | Wave IDs, device dispatch, host metadata, common contraction |
| `core/physics/waves/Scalar.cuh` | Both scalar LS bases |
| `core/physics/waves/Pseudoscalar.cuh` | Pseudoscalar 11 base |
| `core/physics/waves/Tensor{02,20,22,42}.cuh` | Three production covariants per tensor LS group |

There remain 15 registered Waves. Grouping their files does not change stable
IDs, enum values, function names, formulas, or activation by model JSON.

## Post and tests

`post/Post.cu` groups sample integration, observable construction, covariance
finite differences, TXT/ROOT/LaTeX output and its main function. Its local
`GVVComponentEvaluator` owns analysis sample roles but calls `GVVAmplitude`;
there is no duplicate CUDA allocation or propagator evaluation path.

`post/plotting/draw.sh` and its macros are independent consumers of projection
ROOT files. See their user configuration blocks and plotting style guide.

Tests of script-local policy include the actual script with `GVV_FIT_NO_MAIN`
or `GVV_POST_NO_MAIN` defined. These guards avoid introducing public headers
solely for tests. Tests link the same core objects as production; host tests
do not instantiate or execute GPU evaluation.

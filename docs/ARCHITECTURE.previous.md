# CTPWA modular architecture

## Design goal

This project analyses one decay process at a time.  The reusable framework is
kept independent of the active process, while `process/` contains the event,
wave-basis, term-dynamics, and projection definitions that are replaced when a
future project is converted from GVV to another topology such as GPPP.

The dependency direction is strict:

```text
apps -> process -> framework
apps -----------> framework
framework -/----> process
```

Framework headers must not include GVV, omega, or ROOT-branch-specific headers.
The active process may compose the generic Lorentz-tensor and propagator
building blocks but owns all topology-dependent formulae.

## Runtime model

`config/model.json` is the only user-edited model description.  It declares
resonances, coherent terms, coupling policies, and process-specific term
dynamics.  The host-side model loader validates stable string identifiers and
compiles them into dense runtime arrays for the GPU.  Serialized results retain
the canonical model description; numeric array indices are never persistent
physics identities.

There are no source-level constants for the number of resonances, terms,
active waves, fit parameters, or component pairs.  The registered wave
implementations remain compile-time process code because CUDA cannot safely
load arbitrary device functions at runtime.  A model selects a compact subset
of those registered waves.

The common term fields are `id`, `wave`, `coupling`, and `active`.  The
`dynamics` object is parsed by the active process.  This allows a future GPPP
project to replace the GVV single-X-propagator topology without modifying the
model loader, parameter registry, likelihood, or fit driver.

## Physics layers

### Reusable framework

- Lorentz vectors, metric and Levi-Civita tensors, spin projectors, orbital
  tensors, and Blatt-Weisskopf factors.
- Pure propagator and line-shape functions parameterized by invariants and
  numeric parameter blocks, with no resonance-name knowledge.
- Runtime model loading, validation, dense-index compilation, and parameter
  descriptors.
- Coherent intensity/component summation, MC normalization, PDF validation,
  likelihood strategies, optimizer integration, and result metadata.

### Active process

- ROOT event schema and reconstruction into a device event cache.
- Complete covariant wave tensors and the wave registry.
- Parsing and evaluation of process-specific term dynamics, including
  propagator chains, common daughter factors, and symmetrization.
- Process-specific projection variables and presentation labels.

Complete GVV waves are intentionally not advertised as generic waves: their
omega decay currents and vector-polarization structure are process physics.
They are built from reusable tensor primitives.

## Event intensity contract

The coherent kernel consumes a process-provided scalar Term coefficient and an
event Wave-bilinear matrix:

```text
d_i(e; theta) = coupling_i(theta) * process_term_dynamics_i(e; theta)

I(e; theta) = sum_ij Re[d_i d_j* F_(b_i,b_j)(e)]
```

`F` is parameter independent and is built once for the active Wave subset.
The active process controls both the wave tensors used to build `F` and the
dynamics used to build `d_i`.  The framework controls storage, coherent
summation, component closure, and numerical validation.

## Likelihood boundary

Intensity evaluation and likelihood semantics are separate responsibilities:

1. `CalGVVTermCoefficients` is process code that builds `d_i`.
2. `CalCoherentIntensity` contracts `d_i` with the compact `F` matrix using
   runtime dimensions.
3. `ctpwa::monte_carlo_normalization` computes and validates the normalization
   MC mean.
4. `ctpwa::log_likelihood_contribution` validates PDFs and returns one signed
   sample contribution.
5. `NLL_estimator` is the GVV application adapter that supplies data and
   sideband samples and their configured coefficients.

The current sideband-subtracted likelihood is one strategy, not a property of
the GVV amplitude.  Sample paths, coefficients, and optimizer controls belong
in fit configuration rather than `model.json`.

## Extension contracts

Adding a resonance that uses existing propagator and wave implementations must
only require editing `model.json`.  Adding a wave requires one process wave
implementation, one registry entry, its metadata, and tensor/numerical tests.
Adding a propagator requires one pure implementation, one registry entry,
parameter validation, and threshold/pole/CPU-GPU tests.

Converting the project to a new decay process keeps the framework and replaces
the process event, wave registry, term evaluator, projection registry, model
configuration, and process tests.

## Current source map

```text
include/framework/ModelDefinition.h + src/ModelDefinition.cpp
    process-neutral JSON model and validation
include/framework/Likelihood.h
    process-neutral normalization and likelihood mathematics

include/process/GVVProcessModel.h + src/GVVProcessModel.cu
    GVV Wave registry, propagator/dynamics registration, dense model compiler
include/GVVAmplitude.h
    GVV event kinematics and complete covariant Wave tensors
include/GVVModel.h
    device-only GVV Resonance/Term representations and X propagator dispatch
include/kernel.h + src/kernel.cu
    GVV Term builder plus runtime-sized coherent contraction/component matrix
include/GVVSample.h + src/GVVSample.cu
    GVV ROOT event contract and process cache

include/GVVFitParameters.h + src/GVVFitParameters.cu
    compiled-model-derived parameter layout and result/covariance I/O
include/NLL_estimator.h + src/NLL_estimator.cu
    current GVV fit orchestration and process-specific projection output
src/Fit.cu / src/PostFit.cu
    user applications
```

Physical directory names express the intended dependency where practical. A
few established GVV files remain at `include/`/`src/` root to avoid a large
path-only rewrite; their GVV prefix is the process boundary. They are replaced,
not generalized, when cloning the framework for another decay topology.

## Result provenance

Fit writes the canonical model as `<fit-result>.model.json`; PostFit loads this
snapshot by default. Fit text, PostFit ROOT output, and projection mapping trees
record stable ids and runtime ordering. Numeric component indices are therefore
interpreted through recorded metadata instead of a compiled-in nominal model.

For concrete editing procedures and the GVV-to-GPPP replacement boundary, see
`MODEL_CONFIGURATION.md`.

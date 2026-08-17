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

The reusable coherent engine consumes a process-provided scalar term factor and
an event wave-bilinear matrix:

```text
d_i(e; theta) = coupling_i(theta) * process_term_dynamics_i(e; theta)

I(e; theta) = sum_ij Re[d_i d_j* F_(b_i,b_j)(e)]
```

`F` is parameter independent and is built once for the active wave subset.
The active process controls both the wave tensors used to build `F` and the
dynamics used to build `d_i`.  The framework controls storage, coherent
summation, component closure, and numerical validation.

## Likelihood boundary

Intensity evaluation, normalization, PDF evaluation, and likelihood
composition are separate responsibilities:

1. `IntensityEngine` evaluates `I(e; theta)`.
2. `Normalizer` computes the weighted normalization-MC mean.
3. `PdfEvaluator` validates and divides by that normalization.
4. `LikelihoodStrategy` combines named samples with configured coefficients.

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

## Result provenance

Fit and PostFit outputs must record the model schema version, process id,
canonical model JSON, stable resonance/term/wave ids, parameter ordering,
coupling reference policies, and component-pair mapping.  PostFit reads this
metadata instead of guessing the active model from hard-coded indices.

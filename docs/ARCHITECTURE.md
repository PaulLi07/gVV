# Architecture and data flow

## Design boundary

The repository separates three rates of change:

- Resonances and Terms change frequently and are edited only in `model.json`.
- A new covariant basis is implemented as one complete process Wave and
  registered under `process/waves/` and `WaveRegistry`.
- A future project for a different final state reuses `framework/` and replaces
  the explicit process layer.

The goal is not a runtime framework that knows every decay channel. Generic
algorithms remain reusable, while final-state physics remains visible and
testable in `process/`.

## Dependency direction

```text
config/model.json -> framework/model -> process/WaveRegistry
                                           |
ROOT samples -> process/SampleLoader -> ProcessAmplitude/TermEvaluator
                                           |
                         framework/likelihood <- FitLikelihood
                                           |
config/fit.json -> framework/fit --------> app/Fit.cu
                                           |
                     +---------------------+---------------------+
                     |                     |                     |
              fit report TXT       fit_state JSON       projection ROOT
                                            |                    |
                                  Post Calculation       Post Plotting
                                  + truth/selected MC
```

`framework/` must never include `process/`. The internal physics-tool
direction is `math -> tensors -> dynamics -> process`.

## Framework layer

### Math and tensors

`framework/math/` provides device complex numbers, four-vectors, metric and
Levi-Civita conventions. `framework/tensors/` provides tensor algebra, spin
projectors, orbital tensors, contractions, and barrier factors. These modules
contain no omega constants, ROOT branch names, or GVV Wave identifiers.

### Dynamics

`framework/dynamics/` contains reusable two-body kinematics and propagator
formulae. `PropagatorRegistry.cuh` is the compact device dispatch. The mapping
from JSON strings to propagator types remains in the process compiler because
it also defines which parameters a GVV Term may fit.

### Model

`framework/model/Model.*` strictly parses Resonances, Terms, coupling policy,
and opaque process dynamics. It also computes the deterministic model
signature used by the Fit-to-Post contract. It does not register GVV Waves.

### Amplitude and likelihood

`IntensityEngine.cuh` implements the generic coherent contraction, while
`Likelihood.h` implements accepted-MC normalization and signed unbinned
log-likelihood arithmetic. Neither module knows the event topology.

### Fit

- `FitConfig` reads the run inputs, minimizer policy, and output tag.
- `FitEngine` is a process-neutral multistart TMinuit driver.
- `FitOutput` writes the complete human-readable diagnostic report.
- `FitState` writes and reads the machine-readable fitted parameter vector and
  covariance used by downstream numerical tools.

The fit engine receives runtime vectors and contains no fixed Resonance, Term,
Wave, or parameter count.

## Process layer

### Event and sample boundary

`ProcessEvent.cuh` represents the seven final particles on the device.
`ProcessKinematics.cuh` defines omega currents and GVV-specific constants.
`SampleLoader` is the only mapping from the ROOT `Pwa` tree to host/device event
arrays. A different final state replaces these modules.

### Complete Waves

Each file under `process/waves/` is one complete process Wave:

- `Scalar00.cuh`: existing `0++(00)` basis;
- `Scalar22.cuh`: existing `0++(22)` basis;
- `Pseudoscalar11.cuh`: existing `0-+(11)` basis.

`WaveRegistry.cuh` owns the single device dispatch. `WaveRegistry.cu` owns the
single host registration of stable ID, JPC, label, coherence class, and device
type. `ProcessAmplitude.cuh` applies the common photon projector and Wave-pair
contraction. Adding a Wave normally does not modify this common contraction.

### Resonance, Wave, and Term

- A Resonance is a propagator instance and its physical parameters.
- A Wave is a complete covariant-tensor basis for the process.
- A Term joins one Resonance, one Wave, and one complex coupling.

`gvv_compile_model` converts stable IDs into dense active runtime arrays.
Inactive Terms are omitted before Resonance compilation, parameter layout, GPU
allocation, fit reporting, and projection maps. A Resonance referenced only by
inactive Terms therefore contributes no free propagator parameter.

### Fit objective and parameter mapping

`ParameterMapping` is the only conversion between a generic Minuit vector and
the mutable GVV model. It builds the ordered free-parameter layout, applies a
vector to couplings/propagators, and writes the active physical model section
of the human report.

`FitLikelihood` owns sample preparation, cached Wave matrices, the omega width
table, device model synchronization, normalization, and signed likelihood
evaluation. It contains no minimizer policy and no ROOT output schema.

`ProjectionWriter` is the process-specific Fit-to-Plotting bridge. It owns the
projection ROOT trees, derived GVV observables, fitted weights, dynamic
component/group maps, and provenance. A future final state replaces this
writer along with the process layer; generic fit output remains reusable.

## Output contracts

One fit tag creates four products:

- `fit_result-<tag>.txt`: human diagnostics only;
- `fit_state-<tag>.json`: ordered free state, covariance, and model signature;
- `projection-<tag>.root`: selected data/background plus weighted accepted MC;
- `fit-<tag>.log`: complete executable/Slurm output.

The separate covariance text file was removed. Its information appears in both
the human report and the machine state, each for its intended consumer.

Post Calculation loads `fit_state`, the exact `model.json`, generated truth MC,
and selected normalization MC. It rejects a model-signature or parameter-order
mismatch. Post Plotting loads only the projection ROOT file and discovers the
current Terms and coherent groups from its maps.

## End-to-end fit flow

1. `FitConfig` reads `fit.json`; `Model` reads `model.json`.
2. `WaveRegistry` validates and compiles only active Terms and Resonances.
3. `SampleLoader` loads data, accepted normalization MC, and signed backgrounds.
4. `TermEvaluator` caches parameter-independent Wave contractions.
5. `ParameterMapping` generates the exact free Minuit vector.
6. Each objective call applies that vector, integrates accepted MC, and
   evaluates data plus signed sidebands.
7. `FitEngine` runs nominal and randomized starts, applies MIGRAD/HESSE, and
   chooses the lowest accepted NLL.
8. The final state is written independently to the report, state JSON, and
   projection ROOT contracts.

## Reading guide

| File | Primary responsibility |
|---|---|
| `app/Fit.cu` | Application glue for configuration, fit, and outputs |
| `framework/model/Model.*` | Generic model parser and signature |
| `process/WaveRegistry.*` | GVV Wave registry and active model compiler |
| `process/waves/*.cuh` | Complete registered GVV Waves |
| `process/ProcessAmplitude.cuh` | Common GVV polarization and Wave contraction |
| `process/TermEvaluator.*` | CUDA Term coefficients and coherent intensity |
| `process/SampleLoader.*` | ROOT-to-GPU GVV sample boundary |
| `process/ParameterMapping.*` | Minuit-vector to physical-state mapping |
| `process/FitLikelihood.*` | GVV sample/GPU likelihood orchestration |
| `process/ProjectionWriter.*` | Projection ROOT schema and serialization |
| `framework/fit/FitEngine.*` | Generic multistart minimization |
| `framework/fit/FitOutput.*` | Human report |
| `framework/fit/FitState.*` | Machine Fit-to-Post state |
| `post/calculation/*` | Truth/selected MC integration and uncertainty propagation |
| `post/plotting/*` | Projection-only ROOT plotting |

## Reusing the framework for another final state

A future project should reuse `framework/` and replace the process event,
kinematics, sample mapping, complete Waves, Wave registry, Term compiler and
evaluator, process likelihood wrapper, and projection writer. It should not
copy or rewrite the generic tensor blocks, propagators, likelihood arithmetic,
multistart driver, fit report, or fitted-state JSON contract unless their
generic semantics genuinely change.

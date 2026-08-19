# Adding a new Wave

The current refactor migrates the existing Waves and does not implement a new
`2++` basis. This guide defines the extension boundary for future work.

## 1. Define a complete process Wave

A Wave file represents the complete covariant basis for the current process:
radiative production, `X -> omega omega` angular coupling, required projectors,
and barrier factors. A propagator is not part of a Wave; multiple Resonances
may reuse one Wave.

## 2. Reuse the building blocks

- `framework/math/`: four-vectors, metric, Levi-Civita, device complex;
- `framework/tensors/`: spin/orbital projectors, barrier factors, contractions;
- `framework/dynamics/`: reusable kinematics and propagators;
- `process/ProcessKinematics.cuh`: GVV currents, composites, and constants.

Add a missing generally useful block to `framework/` with its own test. Keep a
GVV-only formula under `process/`.

Do not embed a Resonance propagator in the Wave or infer a running-width
orbital momentum from the Wave ID. The Wave defines the numerator tensor;
`model.json` selects the Resonance denominator and its explicit physical
partial-width model.

## 3. Implement one Wave file

Create a descriptive `process/waves/*.cuh` file exposing a pure device
function such as:

```cpp
__device__ inline tensor gvv_example_tensor(
    const GVVEventKinematics& event,
    const GVVBarrierParameters& barrier = GVVBarrierParameters());
```

The function must not depend on mutable global state, Term indices, a
Resonance name, or a model-wide count.

## 4. Register it once

In `WaveRegistry.cuh`, add the device enum and dispatch case. In
`WaveRegistry.cu`, add the stable ID, JPC, display/LaTeX label, coherence class,
and device type. Do not duplicate this mapping in Fit, the model parser,
TermEvaluator, Post Calculation, or plotting.

`ProcessAmplitude.cuh` automatically applies the common polarization sum and
pair contraction. Change it only if those process-wide rules change.

## 5. Test before configuring

Add tests for finite device evaluation, registry lookup, self/cross Wave
contractions, Bose symmetry where required, tensor transversality, and minimal
model compilation. Extend the explicit `make check-gpu` suite so the Wave Gram
matrix remains symmetric and positive semidefinite and the optimized
Wave-aggregated intensity remains identical to the direct Term contraction.
Then reference the new Wave ID in `model.json`.

Before merging, verify that the Wave contains no propagator instance, the
framework has no GVV include, registration exists once, and all tests/builds
pass. GPU job submission and physics-result validation remain user-controlled.

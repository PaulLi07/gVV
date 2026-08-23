# Developing and registering a GVV Wave

This guide covers a source-level extension of the existing
\(\psi(2S)\to\gamma\omega\omega\) process. It explains how to add one new,
complete covariant numerator basis after its physics formula has been derived.

The current catalogue includes twelve \(2^{++}\) Waves: four decay couplings
(`02`, `20`, `22`, and `42`) times three independent production covariants.
The nominal model uses only `gvv.tensor_02_u1/u2/u3`. In contrast,
`gvv.scalar_22` and `process/waves/Scalar22.cuh` are the scalar \(0^{++}\)
basis with \(L=S=2\); the `22` label alone does not mean spin two.

## What a registered Wave represents

A GVV Wave is the complete real rank-two numerator tensor used by the current
process contraction. It includes the chosen radiative-production tensor, the
\(X\to\omega\omega\) angular-spin coupling, required orbital tensors, and
associated Blatt-Weisskopf numerator factors.

A Wave does not contain:

- a Resonance identity, mass, width, or line-shape denominator;
- a fitted complex coupling;
- a Term index or model-wide count;
- the common omega propagators or common rho-isobar dynamic factors;
- Minuit, likelihood, Projection, or Post logic.

Those pieces are composed later. For event \(x\), Term \(t\) has a complex
coefficient

\[
C_t(x)=g_tP_{r(t)}(s_X)
\rho_{\omega_1}(x)\rho_{\omega_2}(x)
BW_{\omega_1}(x)BW_{\omega_2}(x),
\]

while the selected Wave supplies \(W_{w(t)}^{\mu\nu}(x)\). The cached real
Wave Gram matrix is

\[
F_{ab}(x)=\sum_{\mathrm{physical\ pol.}}
W_a(x)W_b(x),
\]

and the intensity is

\[
I(x)=\sum_{t,u}C_t(x)C_u^*(x)F_{w(t),w(u)}(x).
\]

Terms sharing the exact same Wave slot are aggregated only after their
Term-specific propagators and couplings have been evaluated. Therefore a new
Wave must be a complete, unambiguous basis: two implementations that differ
in their tensor formula must have different registered Wave IDs.

## Current process factorization

The following boundaries are part of the present GVV physics model.

### Event and omega-current construction

`process/ProcessEvent.cuh` builds `GVVEventKinematics` from the seven final
four-vectors. It supplies:

- the bachelor photon, both omega candidates, \(X=\omega_1+\omega_2\), and
  \(\psi=X+\gamma\);
- `relative_omega_momentum = omega1 - omega2`;
- one `OmegaDecayCurrent` for each \(\omega\to\pi^+\pi^-\pi^0\) decay.

`process/ProcessKinematics.cuh` separates each omega current into:

- a real Levi-Civita geometric vector used by the Wave tensor; and
- a common complex rho-isobar factor applied in `TermEvaluator.cu`.

This separation keeps the cached `F` matrix real and independent of fitted
parameters. A proposed basis with Wave-specific complex sub-dynamics does not
fit this contract and must not be forced into a `tensor` return value. That
would require an explicit redesign of the process amplitude representation.

### Process-wide polarization contraction

`process/ProcessAmplitude.cuh` owns the photon projector and the common
Wave-pair contraction. The first tensor index is contracted with the two
transverse \(\psi\) polarizations used by the project, and the second index is
contracted through the photon projector. The factor and beam-axis convention
are process-wide, not individual Wave choices.

Adding a numerator Wave normally does not modify `ProcessAmplitude.cuh`. Edit
that file only if the process-wide polarization convention itself changes,
and then revalidate every registered Wave.

### Term and intensity evaluation

`process/TermEvaluator.cu`:

1. constructs `GVVEventKinematics`;
2. caches \(F_{ab}\) for all active Wave slots;
3. evaluates every Term propagator and coupling;
4. aggregates coefficients by exact Wave slot for the Fit intensity;
5. retains Term-level coefficients for Projection and Post components.

It dispatches Waves through `gvv_wave_tensor`; a correctly registered new GVV
Wave requires no special branch in `TermEvaluator`, Fit, Projection, or Post.

## Mathematical conventions

Use the existing conventions consistently:

| Convention | Implementation |
|---|---|
| Four-vector order inside device algebra | `[E, px, py, pz]` in `FV` |
| Input sample order | `[px, py, pz, E]`, converted in `TermEvaluator.cu` |
| Metric | \((+,-,-,-)\) |
| Levi-Civita | \(\epsilon^{0123}=+1\) |
| Relative daughter momentum | \(r=k_1-k_2\), then projected transverse to its parent |
| Pion order in an omega | \(\pi^0,\pi^+,\pi^-\) |
| Default barrier radius | 0.59 fm |
| Identical-vector exchange | exchange the complete two omega decay systems |

`tensor::Epsilon` and `omega_geometric_current` include explicit metric-sign
handling. Do not copy a Euclidean epsilon contraction or silently change
raised/lowered-index conventions inside one new Wave.

`GVVBarrierParameters` carries separate production and \(X\)-decay radii.
Use these values rather than hard-coding a radius in a Wave. The current
runtime calls use their default values; making them user-configurable would be
a separate model-interface change.

## Reusable building blocks

Use the narrowest existing layer that owns the operation.

| Layer | Available building blocks |
|---|---|
| `framework/math/` | `FV`, `DeviceComplex`, metric signs, Levi-Civita convention |
| `framework/tensors/Tensor.cuh` | Rank-two tensor algebra, metric tensor, epsilon tensor |
| `framework/tensors/TensorContraction.cuh` | Named one-index, two-index, double, trace, and symmetrization operations |
| `framework/tensors/SpinProjector.cuh` | Spin-1 transverse metric and direct spin-2 projection of a rank-two source |
| `framework/tensors/OrbitalTensor.cuh` | Bare covariant P/D tensors and direct bare G-wave contraction with a rank-two source |
| `framework/tensors/BarrierFactor.cuh` | Blatt-Weisskopf \(B_L\) for \(L=0,1,2,3,4\) |
| `framework/dynamics/Kinematics.cuh` | Process-independent two-body breakup momentum |
| `framework/dynamics/Propagators.cuh` | Process-independent line shapes; not part of a Wave |
| `process/ProcessKinematics.cuh` | GVV masses and \(\omega\to3\pi\) current model |
| `process/ProcessEvent.cuh` | Complete GVV event view and barrier parameters |

If a missing tensor or kinematic operation is genuinely independent of the
final state, add a small tested building block under `framework/`. Such a file
must not include a GVV header or know an omega, pion ordering, Wave ID, or
Resonance. Keep a GVV-only construction under `process/`.

Never call a Resonance propagator from `process/waves/`. The propagator library
is reusable because it accepts explicit physical parameters; the Wave library
is process-specific because it knows the complete event and polarization
structure.

The orbital helpers deliberately return bare STF geometry and never include a
barrier or normalized-CG coefficient. The exact index definitions, reduced
rank-four contraction, and the Condon-Shortley/Racah normalization selected
for the registered and future high-spin Waves are recorded in
[`TENSOR_CONVENTIONS.md`](TENSOR_CONVENTIONS.md). Do not change the existing
bare D-wave normalization to implement one new basis.

## Step 1: derive the physics basis first

Before writing CUDA code, record the intended convention:

1. the \(J^{PC}\) of \(X\);
2. the allowed \(X\to\omega\omega\) orbital angular momentum \(L\) and
   coupled vector spin \(S\);
3. the radiative-production tensor and any production orbital momentum;
4. the parity and charge-conjugation behavior;
5. the required symmetry under exchange of the two complete omega systems;
6. every projector, trace subtraction, and barrier factor;
7. which indices of the final rank-two tensor correspond to the initial
   \(\psi\) and photon polarization contractions.

For two identical omega mesons, exchanging

```text
(pi01, pip1, pim1, omega1, current1)
<->
(pi02, pip2, pim2, omega2, current2)
```

must give the total Bose symmetry required by the state. An odd orbital tensor
can be valid when another spin tensor is also odd so that the complete Wave is
even. Test the complete tensor rather than one factor in isolation.

Do not infer the Resonance running-width `orbital_l` automatically from this
numerator derivation. The denominator describes the state total width and can
contain several partial waves. Select its physical model explicitly in
`model.json`.

## Step 2: implement one complete Wave file

Create one descriptively named header under `process/waves/`, following the
existing files:

```text
process/waves/Scalar00.cuh
process/waves/Scalar22.cuh
process/waves/Pseudoscalar11.cuh
```

Use an include guard and expose one small pure device function. This skeleton
shows the interface, not a physics formula:

```cpp
#ifndef CTPWA_PROCESS_WAVES_EXAMPLE_CUH
#define CTPWA_PROCESS_WAVES_EXAMPLE_CUH

#include "framework/tensors/BarrierFactor.cuh"
#include "framework/tensors/OrbitalTensor.cuh"
#include "process/ProcessEvent.cuh"

__device__ inline tensor gvv_example_tensor(
    const GVVEventKinematics& event,
    const GVVBarrierParameters& barrier = GVVBarrierParameters())
{
    tensor result;
    // Implement the reviewed covariant production and decay numerator here.
    // Use event.omega_current1.geometry and geometry for current2; their
    // common complex rho factors are applied later by TermEvaluator.
    return result;
}

#endif
```

The actual function must:

- return the complete rank-two tensor expected by `gvv_wave_contraction`;
- depend only on its event and explicit barrier arguments;
- use double-precision existing algebra;
- remain finite on ordinary and near-threshold physical events;
- avoid mutable device globals and hidden model state;
- avoid Resonance IDs, Term IDs, coupling values, and runtime model counts;
- avoid duplicating the common omega rho factors or omega propagators.

Useful patterns in the current code are:

- `Scalar00.cuh`: spin-zero contraction with no orbital tensor;
- `Scalar22.cuh`: traceless transverse D-wave orbital tensor contracted with
  both omega geometric currents;
- `Pseudoscalar11.cuh`: epsilon production tensor, P-wave decay tensor, and
  two spin-1 omega currents.

These are implementation examples, not templates that determine the physics
of a new \(J^{PC}\).

## Step 3: register it at the sole Wave boundary

The exact registration boundary is the `process/WaveRegistry.*` pair. CUDA
requires a device-dispatch half and a host-metadata half, but there is no other
Wave registry elsewhere in the project.

### Device half: `process/WaveRegistry.cuh`

1. Include the new `process/waves/*.cuh` file.
2. Append one unique enum value before `GVV_NBASIS` and update
   `GVV_NBASIS`.
3. Add exactly one dispatch branch to `gvv_wave_tensor`.

Conceptually:

```cpp
enum GVVWaveType {
    GVV_SCALAR_00 = 0,
    GVV_SCALAR_22 = 1,
    GVV_PSEUDOSCALAR_11 = 2,
    GVV_EXAMPLE = 3,
    GVV_NBASIS = 4
};

if (wave_type == GVV_EXAMPLE) {
    return gvv_example_tensor(event, barrier);
}
```

Keep enum values contiguous because device/test indexing uses `GVV_NBASIS` as
the explicit registered-basis count. The host registry itself iterates over
its metadata vector and must contain the matching rows.

### Host half: `process/WaveRegistry.cu`

Append one `GVVWaveMetadata` row to `gvv_wave_registry()`:

```cpp
{"gvv.example_ls", "JPC", "J^{PC}(LS)",
 "coherence_token", GVV_EXAMPLE}
```

Choose each field deliberately:

- `id`: stable JSON-facing identifier obeying the model ID syntax;
- `jpc`: physics grouping label used in fit and Post metadata;
- `latex`: presentation label for reports and projections;
- `coherence_class`: phase-reference block, described below;
- `wave_type`: the unique device enum just added.

Do not add mappings in `FitLikelihood`, `ParameterMapping`, `TermEvaluator`,
Projection, Post, or plotting. They consume the compiled runtime metadata.

## Step 4: choose coherence metadata from the Gram matrix

`jpc` and `coherence_class` have different jobs.

- `jpc` labels physics groups in output.
- `coherence_class` tells the model compiler which Terms share one arbitrary
  phase convention.

Neither field removes a cross term from the intensity. `TermEvaluator` always
evaluates the complete active Wave Gram matrix. Therefore:

- if the new Wave can interfere with an existing Wave, place it in the same
  coherence class;
- assign a new coherence class only if the full cross-Wave contraction is
  identically zero under the process polarization sum;
- prove every claimed zero cross block numerically on several physical
  events, not only after phase-space integration.

The current registry assigns `gvv.scalar_00`, `gvv.scalar_22`, and every
registered `2++` Wave to `positive_parity`; it assigns
`gvv.pseudoscalar_11` to `negative_parity`. The cross block between these two
classes is numerically checked to vanish. These tokens name phase-reference
blocks, not spin categories or intensity masks. Do not classify a future Wave
from its JPC label or parity alone: verify its event-level Gram-matrix cross
terms first.

Every active coherence class must have exactly one reference Term. A new Wave
in an existing class normally uses an ordinary `complex_cartesian` coupling.
The first active Term in a genuinely new class normally uses a
`positive_real` phase reference, unless it is deliberately chosen as the one
global nonzero fixed scale-and-phase reference. See [Model
configuration](MODEL_CONFIGURATION.md) before changing reference roles.

## Step 5: use the registered ID in the model

After registration and tests, a Term can select the Wave without any other
source edit:

```json
{
  "id": "example_resonance_term",
  "label": "example resonance",
  "wave": "gvv.example_ls",
  "active": true,
  "coupling": {
    "mode": "complex_cartesian",
    "initial": [0.1, 0.0]
  },
  "dynamics": {
    "type": "gvv_x_to_omega_omega",
    "resonance": "example_resonance"
  }
}
```

The referenced Resonance must be declared separately with a denominator that
matches the physical width hypothesis. Several Resonances can reuse this Wave,
and one Resonance can appear in several Terms with different registered Waves.

## Required verification

A new Wave is not complete when it compiles. Its covariant formula and its
interaction with the optimized intensity path both require tests.

### Registry and compile tests

Update or extend:

- `tests/test_gvv_amplitude.cu` so the complete device dispatch and
  polarization contraction compile with the new enum;
- `tests/test_wave_registry.cu` with registry lookup, metadata, a minimal
  model using the new ID, dense active-Wave slots, and reference validation;
- a focused building-block test if reusable framework tensor algebra was
  added.

Do not merely increase an expected count. Compile an actual model Term that
selects the new ID and verify its `registered_wave_type`, dense `wave_slot`,
JPC, LaTeX label, and coherence class.

### Complete-Wave GPU numerical tests

Extend `tests/test_gvv_wave_numerics.cu` using deterministic, physically valid
events. Include more than one generic phase-space point and a point near any
relevant threshold. At minimum verify:

1. every tensor component and every new \(F_{ab}\) is finite;
2. the new Wave is not identically zero on generic kinematics;
3. exchange of the two complete omega systems has the required Bose behavior;
4. the omega geometric currents remain transverse to their omega momenta;
5. every new orbital tensor is transverse to its parent;
6. every D- or G-wave rank-two result, if used, is symmetric and traceless;
7. the photon projector is transverse;
8. the complete Gram matrix is symmetric with non-negative diagonal and is
   positive semidefinite within a scale-aware numerical tolerance;
9. \(F_{ab}\) is invariant under a common rotation about the beam \(z\) axis;
10. every cross block assigned to different coherence classes is zero within
    tolerance;
11. an interference expected within one coherence class is not accidentally
    erased by the implementation.

The current polarization sum singles out the beam axis, so the relevant
rotation regression is around \(z\). Do not impose an arbitrary-boost or
arbitrary-rotation test that is not a symmetry of the implemented production
setup.

The existing test contains dimension-specific checks for the present three
Wave basis. When `GVV_NBASIS` changes, generalize those checks rather than
leaving a hard-coded 3-by-3 determinant that silently ignores the new row and
column.

### Intensity and component equivalence

Extend `tests/test_intensity_equivalence.cu` so the new slot participates in
all three representations:

\[
I_{\mathrm{Term\ double\ sum}}
=I_{\mathrm{Wave\ aggregation}}
=\sum_{i\le j}K_{ij}.
\]

The test must cover:

- one Term, the nominal-scale Term count, and a larger dynamic Term count;
- at least two Terms sharing the new exact Wave slot;
- different Resonance propagators on the same Wave;
- nearly cancelling complex coefficients;
- a component batch beginning at a nonzero event offset;
- a short final batch;
- GPU integrated pair components versus a host sum of the same packed values.

These checks protect the algebraic optimization from grouping too early by
JPC, coherence class, or Resonance. Aggregation is valid only by exact
`wave_slot` after each Term coefficient is complete.

### Build and execution boundary

From the repository root:

```bash
source config/gvv_env.sh
make -j2
make check
make gpu-tests
```

`make gpu-tests` compiles the explicit CUDA runtime tests. Execute

```bash
make check-gpu
```

only on a node with an allocated CUDA device. It is deliberately separate
from `make check`, and cluster job submission remains user-controlled.

## Review checklist

Before accepting a new Wave, confirm all of the following:

- the reviewed analytic formula has stated \(J^{PC}\), \(L\), \(S\), parity,
  and index conventions;
- the complete Wave has the correct identical-omega exchange symmetry;
- the source is one pure function under `process/waves/`;
- no Resonance propagator, coupling, Term index, or model count appears in the
  Wave;
- common omega complex dynamics is not double-counted;
- reusable additions under `framework/` contain no GVV dependency;
- the only registration edits are the device and host halves of
  `process/WaveRegistry.*`;
- enum, dispatch, stable ID, JPC, label, coherence class, and type agree;
- coherence-class decisions are supported by event-level cross-Wave tests;
- the model contains exactly one reference in each active class and one global
  scale-and-phase reference;
- Gram-matrix and Term/Wave/component equivalence tests include the new basis;
- Fit, Projection, and Post required no Wave-specific special case;
- all source comments and documentation are in English.

## Boundary for a different decay channel

This guide adds a Wave to the existing GVV process. A future project with a
different final-state topology should not accumulate alternate channels inside
these GVV classes. It should replace or reimplement the process layer:

- its event and sample contract;
- process kinematics and decay currents;
- complete Waves and Wave registry;
- process polarization contraction;
- Term coefficient construction and projection observables.

It can directly reuse the process-neutral math, tensor, propagator, amplitude,
likelihood, fit, and result infrastructure under `framework/`. This is the
intended portability boundary.

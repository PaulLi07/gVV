# gVV architecture and end-to-end data flow

## 1. Purpose and scope

This document explains how the repository turns a JSON amplitude model and
ROOT event samples into a fitted covariant-tensor amplitude, and how the two
independent downstream modules consume the fit products. It is an architecture
reference: field-by-field configuration details are in
[`MODEL_CONFIGURATION.md`](MODEL_CONFIGURATION.md), and the practical recipe
for implementing a new basis is in [`WAVE_DEVELOPMENT.md`](WAVE_DEVELOPMENT.md).
The shared Lorentz/STF and normalized-CG conventions are fixed in
[`TENSOR_CONVENTIONS.md`](TENSOR_CONVENTIONS.md).

The current process is

```text
psi(2S) -> gamma X
             X -> omega omega
       omega -> pi+ pi- pi0
```

The implementation deliberately separates code by how often it should change:

1. **Model configuration changes frequently.** Resonances and Terms are
   added, removed, or disabled in `config/model.json`.
2. **The Wave catalogue changes occasionally.** A new complete GVV covariant
   basis is implemented under `process/waves/` and registered once.
3. **The decay process changes rarely and as a unit.** A future project for a
   different final state reuses the generic `framework/` and replaces the
   process boundary rather than teaching the GVV process code about every
   possible topology.

This is therefore not a universal runtime decay-language interpreter. It is a
small reusable numerical framework with an explicit, inspectable process
implementation.

## 2. Architectural invariants

The following rules define the intended structure.

- `framework/` contains no GVV particle names, ROOT branch names, registered
  GVV Wave IDs, or includes from `process/`.
- `process/` may use `framework/` building blocks and owns all
  `psi(2S) -> gamma omega omega` assumptions.
- `app/Fit.cu` is glue. It selects the process implementation and connects it
  to generic fit/output services; it does not contain tensor formulae or
  minimizer algorithms.
- A Resonance is a propagator instance, a Wave is a complete process numerator
  basis, and a Term joins one Resonance, one Wave, and one coupling. None of
  these concepts is represented by a model-wide compile-time count.
- `model.json` is the single user-edited amplitude-model description. There is
  no generated C++ model table and no Python model layer.
- The total fit intensity is always coherent. JPC labels and coherence classes
  describe metadata and phase conventions; they are not switches that remove
  cross-Wave terms from the numerical contraction.
- Fit, Post Calculation, and Post Plotting are separate programs or workflows.
  Their only coupling is through documented output contracts.
- Validation is performed at the boundary that owns the input: JSON loaders,
  the GVV model compiler, ROOT sample loading, fitted-state loading, and output
  contract readers. Inner numerical kernels rely on those invariants.

## 3. Repository map

```text
gVV/
├── app/
│   └── Fit.cu                  Fit application assembly
├── config/
│   ├── model.json              Resonances, Terms, Waves, couplings
│   ├── model.schema.json       Declarative model format reference
│   ├── fit.json                Samples, minimizer, output tag
│   └── gvv_env.sh              Project-local CUDA/ROOT environment
├── framework/                  Process-independent reusable layer
│   ├── amplitude/              Coherent intensity algebra
│   ├── dynamics/               Propagators, kinematics, tabulated functions
│   ├── fit/                    Run config, Minuit, report, fitted state
│   ├── likelihood/             Normalization and log-likelihood arithmetic
│   ├── math/                   Complex, four-vector, Lorentz conventions
│   ├── model/                  Generic JSON model description
│   └── tensors/                Tensor/projector/orbital/barrier blocks
├── process/                    GVV replacement boundary
│   ├── waves/                  Complete registered GVV numerator bases
│   ├── ProcessEvent.cuh        Device event view
│   ├── OmegaDecayModel.cuh     shared omega -> rho pi -> 3pi dynamics
│   ├── ProcessKinematics.cuh   omega currents from event four-vectors
│   ├── ProcessAmplitude.cuh    common GVV polarization contraction
│   ├── ProcessModel.h          compiled process records and bindings
│   ├── WaveRegistry.*          Wave catalogue and device dispatch
│   ├── PropagatorCompiler.*    Resonance JSON and channel compilation
│   ├── ModelCompiler.*         active Term and dense runtime assembly
│   ├── TermEvaluator.*         CUDA F, coefficient, intensity, component path
│   ├── SampleLoader.*          ROOT-to-device sample boundary
│   ├── OmegaWidthTable.*       omega three-body running-width table
│   ├── ParameterMapping.*      flat fit vector <-> GVV physical state
│   ├── FitLikelihood.*         fit-time process orchestration
│   └── ProjectionWriter.*      process-specific projection ROOT contract
├── post/
│   ├── calculation/            numerical efficiencies/fractions/errors
│   └── plotting/               projection-only ROOT figures and moments
├── tests/                      host/compile checks and explicit GPU regressions
├── Makefile                    Fit-default build graph
├── submit_fit.sh               Fit-only Slurm submission/worker
└── submit_post.sh              Post-Calculation-only Slurm submission/worker
```

Generated inputs, binaries, logs, numerical outputs, and figures are not part
of the source tree contract even though their directories may exist locally.

## 4. Dependency direction and ownership

### 4.1 Source dependency direction

```text
framework/math
      |
      v
framework/tensors
      |
      +-----------> framework/dynamics
      |                       |
      +-----------------------+
                              v
                    process event/Waves/dynamics
                              |
                              v
                    process CUDA evaluation
                         /             \
                        v               v
                 FitLikelihood    Post ComponentEvaluator
                        |               |
                        v               v
                 app/Fit.cu       PostCalculation.cu
```

The generic model, likelihood, amplitude, and fit services sit alongside the
low-level math/dynamics stack and are called by process/application code. The
critical prohibition is the reverse edge: `framework/` must never include or
name `process/`.

### 4.2 Configuration and runtime data flow

```text
config/model.json
        |
        v
ModelDefinition --definition signature--+
        |                                |
        +--> embedded canonical model    +--> combined compatibility key
        |                                ^         ^
        v                                |         |
GVV implementation contract ------------+---------+
        |
        v
GVVCompiledModel <---- ParameterMapping <---- flat Minuit vector
        |
        +---------------> device Resonances/Terms/couplings
                                |
ROOT Pwa trees -> GVVSample -> cached Wave Gram matrices -> intensities
                                                        |
config/fit.json -> FitEngine objective <----------------+
        |
        +--> fit_result-<tag>.txt      human diagnostics
        +--> fit_state-<tag>.json      Post Calculation bridge
        +--> projection-<tag>.root     Post Plotting bridge
        +--> fit-<tag>.log             execution record
```

### 4.3 Fit and downstream ownership

| Concern | Owner | Must not own |
|---|---|---|
| JSON model syntax | `framework/model` | GVV Wave dispatch or propagator string policy |
| GVV Wave semantics | `process/WaveRegistry` | Resonance JSON policy |
| GVV propagator semantics | `process/PropagatorCompiler` | Wave registration or Minuit implementation |
| Active model assembly | `process/ModelCompiler` | propagator formulae or sample loading |
| Flat parameter order | `process/ParameterMapping` | sample loading or output schemas |
| Event/Wave numerical evaluation | `process/TermEvaluator` | model-string parsing |
| Probability arithmetic | `framework/likelihood` | ROOT I/O or GVV kinematics |
| Fit sample orchestration | `process/FitLikelihood` | MIGRAD policy or ROOT serialization |
| Multistart minimization | `framework/fit/FitEngine` | Resonance/Wave/Term types |
| User fit report | `framework/fit/FitOutput` plus a process detail callback | machine consumption |
| Machine fit handoff | `framework/fit/FitState` | compiled GVV arrays or external MC samples |
| Projection ROOT schema | `process/ProjectionWriter` | minimization |
| Post numerical observables | `post/calculation` | plot styling |
| Projection figures | `post/plotting` | model reconstruction or truth-MC integration |

## 5. Model objects and runtime objects

Understanding the distinction between persistent configuration and dense
runtime objects is essential when extending the project.

### 5.1 Persistent model: `ModelDefinition`

`framework/model/Model.*` parses the complete JSON document into generic
objects:

- `ResonanceDefinition`: stable ID, label, propagator string, and named
  parameter definitions;
- `TermDefinition`: stable ID, label, Wave ID, active flag, coupling policy,
  and an opaque canonical JSON `dynamics` object;
- `ModelDefinition`: process ID, metadata, all Resonances, all Terms, and the
  canonicalized document.

At this level, process dynamics are intentionally opaque. The generic parser
can validate IDs, coupling syntax, parameter transforms, and the global
scale-and-phase convention without knowing what `gvv_x_to_omega_omega`
means.

The canonical JSON is hashed with deterministic FNV-1a. This definition
signature is a compatibility key, not a security hash. A separate explicit
GVV implementation signature identifies the numerical meaning supplied by
the compiled Wave, propagator, and process-amplitude code. Fit records both
signatures and their combined compatibility identifier.

### 5.2 Active process model: `GVVCompiledModel`

`gvv_compile_model` converts the persistent model into arrays suitable for the
current process and GPU kernels:

- active propagator descriptors in dense Resonance order;
- active `TermSpec` entries containing a Resonance index and a dense Wave
  slot;
- active complex couplings in Term order;
- a unique list of registered Wave types used by the active Terms;
- host metadata that preserves stable IDs, labels, JPC, coherence class,
  propagator names, and coupling policies.

A Resonance ID identifies one physical propagator instance in one model. If
several Terms reference the same Resonance ID, they intentionally share that
instance and therefore share its mass, width, and any other fitted propagator
parameters. The `propagator` string instead selects a reusable implementation
type. Two different Resonance IDs may select the same propagator string and
still represent independent states with independent parameter bindings. The
compiler keys sharing by Resonance ID, never by propagator type.

The work is deliberately split. `ModelCompiler` resolves active dependencies,
dense slots, Term dynamics, and reference conventions. It delegates each
needed Resonance to `PropagatorCompiler`, which validates the exact JSON
contract and emits a self-contained numerical descriptor, report metadata,
and generic free-parameter bindings. `WaveRegistry` is not part of either
compiler; it only supplies registered complete Wave identities and dispatch.

Inactive Terms are skipped before dependencies are compiled. Consequently, a
Resonance referenced only by inactive Terms is absent from GPU arrays, Minuit
parameters, reports, component maps, and Post Calculation. This makes
`"active": false` independent of propagator type and independent of whether
that unused propagator would have a free parameter.

### 5.3 Dense Wave slots versus registered Wave types

A registered Wave has a stable user ID and a device enum value. A particular
model may use only a subset. The compiler assigns that subset dense slots
`0..W-1`, which define the dimensions of every cached event Gram matrix.

Several Terms may share the same dense Wave slot. Their propagators and
couplings remain distinct, but their fully evaluated coefficients can be
summed before the final coherent contraction. This distinction is the basis of
the optimized fit path.

### 5.4 Flat fit state and physical state

The generic `FitEngine` accepts only ordered `FitParameterSpec` entries and a
callback. `ParameterMapping` creates those entries from the compiled process
model and records a binding back to one of:

- a coupling real part;
- a coupling imaginary part;
- the log magnitude of a positive-real phase reference;
- a Resonance mass in its physical identity coordinate;
- a Resonance width in its physical identity coordinate;
- a log S/D width ratio;
- a log effective omega-omega Flatte ratio.

Fixed parameters remain in `model.json` and in the human report but do not
appear in the flat Minuit vector. FitState schema version 2 embeds the complete
canonical model object together with the best-fit vector and covariance, so
Post Calculation can recreate the amplitude without reading an external
`model.json`. The source model path remains provenance only.

### 5.5 Event sample: `GVVSample`

Each sample owns:

- host momentum arrays for seven final particles;
- corresponding managed device arrays, stored internally as
  `[event][px,py,pz,E]`;
- the parameter-independent Wave Gram matrix
  `[event][active_wave][active_wave]`;
- an `[event][active_wave]` coefficient workspace for the fit hot path;
- an `[event]` intensity buffer.

The sample object does not know whether it represents data, normalization MC,
a signed background sample, generated truth MC, or selected MC. Fit and Post
assign those roles.

## 6. Amplitude Fit system

### 6.1 What enters a fit

`config/fit.json` selects one `model.json`, selected data, accepted
normalization MC, zero or more signed background samples, multistart settings,
and a common output tag. `app/Fit.cu` loads these inputs and supplies the fixed
GVV ROOT branch contract:

```text
tree: Pwa
p4_pip1 p4_pim1 p4_pi01 p4_pip2 p4_pim2 p4_pi02 p4_gam
storage order: (px, py, pz, E)
```

No truth MC enters the likelihood fit.

### 6.2 From seven particles to a process event

For each event, `TermEvaluator` converts the stored ROOT ordering into device
four-vectors with component order `(E, px, py, pz)`. `GVVEventKinematics`
then constructs

```text
omega1 = pi01 + pip1 + pim1
omega2 = pi02 + pip2 + pim2
X      = omega1 + omega2
psi    = X + gamma
q_omega(relative) = omega1 - omega2
```

Each omega decay current is factorized into a real geometric pseudovector and
a complex coherent rho-isobar factor:

```text
E_omega^mu = epsilon^mu_{nu lambda sigma}
             p(pi+)^nu p(pi-)^lambda p(pi0)^sigma

rho_factor = f_rho(pi+ pi-) + f_rho(pi+ pi0) + f_rho(pi- pi0)
```

Each `f_rho` contains the rho running-width propagator and the two P-wave
barrier factors. The same omega decay model is common to every production
Wave, which allows the real geometric tensors to form the cached Gram matrix
while the common complex factor stays in each Term coefficient.

`OmegaDecayModel.cuh` is the single host/device implementation of this
coherent rho-isobar factor. Both event-current construction and the numerical
omega-width integration call it, so the numerator and the width table cannot
silently drift to different rho masses, widths, barriers, or line shapes. The
rho itself calls the process-independent two-body `ctpwa::BWR` function.

The scalar isobar dynamics use nominal pion masses throughout. The rho charge
channel selects the nominal daughter pair and the nominal bachelor pion. Event
four-vectors provide `s_omega` and `s_pipi`, but reconstructed single-pion
virtual masses do not enter the rho running width or either barrier factor.

### 6.3 Complete Waves

A complete Wave function returns the GVV covariant numerator tensor for one
registered basis. The current catalogue is:

| Stable Wave ID | Basis | Source |
|---|---|---|
| `gvv.scalar_00` | `0++(00)` | `process/waves/Scalar00.cuh` |
| `gvv.scalar_22` | `0++(22)` | `process/waves/Scalar22.cuh` |
| `gvv.pseudoscalar_11` | `0-+(11)` | `process/waves/Pseudoscalar11.cuh` |
| `gvv.tensor_02_u1` | `2++`, `LS=02`, `U1` | `process/waves/Tensor02U1.cuh` |
| `gvv.tensor_02_u2` | `2++`, `LS=02`, `U2` | `process/waves/Tensor02U2.cuh` |
| `gvv.tensor_02_u3` | `2++`, `LS=02`, `U3` | `process/waves/Tensor02U3.cuh` |
| `gvv.tensor_20_u1` | `2++`, `LS=20`, `U1` | `process/waves/Tensor20U1.cuh` |
| `gvv.tensor_20_u2` | `2++`, `LS=20`, `U2` | `process/waves/Tensor20U2.cuh` |
| `gvv.tensor_20_u3` | `2++`, `LS=20`, `U3` | `process/waves/Tensor20U3.cuh` |
| `gvv.tensor_22_u1` | `2++`, `LS=22`, `U1` | `process/waves/Tensor22U1.cuh` |
| `gvv.tensor_22_u2` | `2++`, `LS=22`, `U2` | `process/waves/Tensor22U2.cuh` |
| `gvv.tensor_22_u3` | `2++`, `LS=22`, `U3` | `process/waves/Tensor22U3.cuh` |
| `gvv.tensor_42_u1` | `2++`, `LS=42`, `U1` | `process/waves/Tensor42U1.cuh` |
| `gvv.tensor_42_u2` | `2++`, `LS=42`, `U2` | `process/waves/Tensor42U2.cuh` |
| `gvv.tensor_42_u3` | `2++`, `LS=42`, `U3` | `process/waves/Tensor42U3.cuh` |

`Scalar22` is the scalar basis with orbital/spin labels `(22)`; it is not a
spin-two `2++` Wave. The nominal model selects only `tensor_02_u1/u2/u3` for
`f2(1565)` and `f2(1810)`. Registration of the remaining tensor Waves does not
allocate Fit matrices unless an active Term selects them.

The Wave contains production and decay angular tensors and the required
barrier factors. It does **not** contain the Resonance propagator or coupling.
That separation lets many Resonances reuse the same Wave.

### 6.4 Cached Wave Gram matrix

`ProcessAmplitude.cuh` supplies the process-wide photon projector and the
contraction convention. For every event and active Wave pair, preparation
computes

```text
F_wv(event) = polarization contraction of U_w(event) and U_v(event).
```

`F` depends on event kinematics and the registered Wave tensors, but not on
fit couplings or Resonance parameters. It is therefore built once per sample
before Minuit starts and reused in every objective call.

The code evaluates all `w,v` entries. It does not suppress entries because of
JPC or coherence-class labels.

### 6.5 Term coefficient and total intensity

For active Term `t`, let `r(t)` be its Resonance and `w(t)` its dense Wave
slot. The event-dependent complex coefficient is

```text
C_t(event; theta) = c_t(theta)
                    R_r(t)(s_X; theta)
                    rho_factor(omega1)
                    rho_factor(omega2)
                    BW_omega(s_omega1)
                    BW_omega(s_omega2).
```

`R` is selected by the compiled propagator descriptor. The generic
two-body-running-width propagator receives its physical `orbital_l` explicitly;
it does not infer the width power from the Wave ID. The omega propagators use a
prebuilt, interpolated three-pion running-width table normalized at the omega
pole. Their denominator is evaluated by the same framework-level
`BW_from_width` function used by fixed and analytic-running Breit-Wigner
models; only the process-specific construction of `Gamma_omega(s)` remains in
`OmegaWidthTable`.

After every Term coefficient has been fully evaluated, Terms that share the
same complete Wave are aggregated:

```text
B_w(event; theta) = sum over t with w(t)=w of C_t(event; theta).
```

The total intensity is then

```text
I(event; theta) = sum over w,v
                  Re[B_w(event; theta) B_v*(event; theta) F_wv(event)].
```

This is algebraically identical to the direct Term-by-Term contraction

```text
sum over t,u Re[C_t C_u* F_w(t),w(u)],
```

but costs `O(T + W^2)` per event instead of `O(T^2)` when many Resonances
reuse a small Wave basis. Term-level coefficients are still evaluated in the
Projection and Post component paths, where individual diagonal and
interference contributions are physically required.

Small negative intensities within the numerical tolerance of zero are set to
zero by the generic contraction. Strictly negative or non-finite intensities
are rejected by the likelihood boundary.

### 6.6 Accepted-MC normalization and signed likelihood

For `N_MC` accepted normalization-MC events, the Monte Carlo normalization is

```text
N(theta) = (1/N_MC) sum_j I(MC_j; theta).
```

For a sample `s` with configured coefficient `alpha_s`, its contribution is

```text
ln L_s(theta) = alpha_s sum_i [ln I(event_i; theta) - ln N(theta)].
```

Data always uses `alpha_data = +1`. Each configured background sample uses its
own signed coefficient. The nominal sideband prescription, for example, is
represented entirely by `-0.5` and `+0.25` in `fit.json`; there is no
hard-coded SB1/SB2 branch in the likelihood.

The effective log likelihood is

```text
ln L_eff = ln L_data + sum_backgrounds ln L_background,
NLL      = -ln L_eff.
```

`FitLikelihood` synchronizes the current host couplings and propagator
parameters to the device, evaluates normalization MC once per objective call,
then evaluates data and every signed background with that same normalization.

### 6.7 Reference conventions

The amplitude has redundant overall phase and scale directions. The current
contract resolves them in two related layers:

- the generic model requires exactly one active fixed
  `scale_and_phase` reference in the complete model;
- the GVV compiler requires exactly one reference coupling in every registered
  `coherence_class` represented by active Terms.

A fixed scale-and-phase reference must be nonzero. Other coherence classes use
a positive-real coupling with free log magnitude to fix only their phase.

The current GVV registry uses the following explicit names:

| `coherence_class` | Registered JPC values | Nominal reference policy |
|---|---|---|
| `positive_parity` | `0++`, `2++` | `f0_1710_00` is positive real and fixes this block's phase |
| `negative_parity` | `0-+` | `eta_1760_11` is the fixed global scale-and-phase reference |

The names summarize the two blocks verified for the current Wave catalogue;
they are not a rule that parity alone determines the class of a future Wave.

`coherence_class` is a physical phase-convention declaration, not a plotting
group and not a numerical mask. A new Wave belongs to an existing class if it
can physically interfere with members of that class. A distinct class is
appropriate only when the cross terms are structurally zero under the adopted
polarization/tensor construction. The evaluator still calculates the cross
entries, which makes an incorrect assumption visible to numerical tests.

JPC is separate metadata. Projection and Post group Terms by the `jpc` string,
while phase-reference validation uses `coherence_class`.

### 6.8 Parameter layout and multistart Minuit

`ParameterMapping` creates a deterministic ordered free-parameter vector:

1. active Term coupling coordinates in active Term order;
2. supported free propagator coordinates in active Resonance order.

Ordinary complex couplings use adjacent real and imaginary coordinates. A
positive-real phase reference uses a log-magnitude coordinate. A free
Resonance `mass` or `width` uses the physical identity coordinate and is named
`mass_<resonance-id>` or `width_<resonance-id>`; it requires explicit finite
bounds with a positive lower limit, an initial value inside those bounds, and
the `identity` transform. Supported positive propagator ratios continue to use
log coordinates.

For `two_body_running_bw` and `scalar_sd_running_bw`, the complete free mass
interval must lie strictly above the nominal `2 m_omega` threshold because
their pole normalization requires an open omega-omega channel. A
`subtracted_effective_flatte` mass may remain below threshold. `orbital_l` is a
discrete model choice and remains a fixed identity-transformed integer; it is
not a Minuit coordinate.

`FitEngine` is process-neutral. Start zero uses the nominal model values.
Later starts randomize only entries marked by the process mapping: complex
couplings receive a random log-uniform magnitude and uniform phase, and
positive-real reference magnitudes receive a random log magnitude. Minuit runs
MIGRAD and HESSE for each start.

An attempt is accepted only when the MIGRAD/HESSE statuses, covariance status,
EDM, parameter values, errors, and covariance pass the generic fit criteria.
The accepted attempt with the lowest NLL is selected; deterministic covariance,
EDM, and start-index tie breakers are used for numerically equal minima.

### 6.9 Fit output contracts

One output tag names four independent products:

| Product | Intended consumer | Contract |
|---|---|---|
| `results/fit_result-<tag>.txt` | Human analyst | Complete readable diagnostics; never parsed by project code |
| `results/fit_state-<tag>.json` | Post Calculation | Schema-v2 embedded canonical model, ordered free values, errors, bounds, covariance, best-fit diagnostics, and compatibility signatures |
| `results/projection-<tag>.root` | Post Plotting | Selected events, fitted accepted-MC weights, signed backgrounds, dynamic maps, provenance |
| `runlog/fit-<tag>.log` | Human/operator | Full Slurm worker and executable output |

Reusing a tag overwrites the existing products. The fit does not write
separate copies of the input configuration; the canonical model object is
stored inside the machine state by contract.

The text report includes every multistart attempt, best-fit diagnostics, the
ordered free parameters, active fixed and free physical model values, and full
covariance and correlation matrices. Its presentation can evolve without
changing software consumers.

FitState schema version 2 contains the generic fitted state and the complete
canonical model object. It stores the model-definition signature, the explicit
GVV implementation signature, and their combined identifier. Downstream
reconstruction parses the embedded model, checks all three values, recompiles
it with the current executable, and verifies the free-parameter order. Schema
version 1 is intentionally rejected; rerun Fit to create a self-contained
schema-v2 state instead of pairing old state with a mutable external model.

### 6.10 Projection ROOT contract

`ProjectionWriter` is part of the GVV process boundary. It computes derived
GVV observables on the host and serializes projection schema version 3.

| Tree | Contents |
|---|---|
| `MC` | Accepted normalization-MC event kinematics, total fitted `weight`, coherent `weight_group`, full symmetric `weight_component` matrix |
| `data` | Selected data event kinematics and derived observables |
| `bg` | All configured background events with zero-based `background_index` and signed `weight_bg` |
| `component_map` | Term index, IDs/labels, Resonance, Wave, Wave label, JPC |
| `group_map` | Dynamic JPC group index and display label |
| `background_map` | Dynamic background label, size, likelihood coefficient, projection weight |
| `metadata` | Schema, tag, model signature, counts, effective yield, best fit, closure diagnostic |

Every event tree carries the complete schema-v3 polarization coordinates.
In the X rest frame, `z_X` points opposite the radiative photon,
`y_X` follows `z_beam cross z_X`, and `x_X = y_X cross z_X`.
`cos_theta_omega1` and `phi_omega1` locate omega1 in this basis; omega2 is
back-to-back and is not duplicated. In each omega_i helicity frame, `z_i`
follows its X-frame momentum, `y_i` follows `z_X cross z_i`, and
`x_i = y_i cross z_i`. The oriented normal
`unit(p(pi+) cross p(pi-))` supplies `cos_theta_decay_plane_omega1/2` and
`phi_decay_plane_omega1/2`. The tree also stores all six pion polar cosines,
the two unnormalized normal magnitudes, and the wrapped difference of the two
local decay-plane azimuths. All pion-pair masses remain available for direct
Dalitz-plot work.

The fitted accepted-MC scale is

```text
projection_scale = effective_signal_yield / sum_MC I,

effective_signal_yield = N_data + sum_b alpha_b N_b.
```

Background events are plotted with `weight_bg = -alpha_b`. Thus the total fit
curve is the fitted signal projection plus the explicitly plotted background
contribution.

Term-pair components are calculated directly in bounded event batches. A
packed upper triangle stores each diagonal `K_ii` and each complete signed
off-diagonal interference `K_ij + K_ji`. The writer verifies that the packed
upper-triangle sum closes to the total intensity event by event.

For external convenience, `weight_component` is written as a symmetric
`T x T` vector: the complete off-diagonal pair value is mirrored into both
`[i,j]` and `[j,i]`. Reconstruct the total using the upper triangle only;
naively summing the entire symmetric matrix double-counts interference.

`weight_group` contains only Term pairs whose two Terms share the same JPC
group. Cross-group interference remains in the total model and is not assigned
to either individual group curve. All component and group weights use the same
full-model normalization.

### 6.11 Fit execution workflow

```text
1. Load and validate fit.json.
2. Load and generically validate model.json.
3. Compile active GVV Resonances, Terms, Wave slots, and coupling policies.
4. Load data, normalization MC, and configured signed backgrounds.
5. Build the omega width table and upload the compiled model.
6. Upload every sample and cache all active Wave Gram matrices.
7. Build the deterministic free-parameter layout.
8. For every Minuit call:
     a. apply the flat vector to the host process model;
     b. synchronize mutable state to the device;
     c. evaluate normalization MC and its normalization;
     d. evaluate data and signed background contributions;
     e. return the NLL.
9. Run all configured starts and select the best accepted solution.
10. Reapply the selected parameters.
11. Write the human report and self-contained machine fit state independently.
12. Evaluate bounded-batch Term components and write the projection ROOT file.
```

Compilation happens on the login node, but numerical execution requires a GPU
allocation. `submit_fit.sh` validates immutable paths, submits a Slurm job, and
executes `Fit.exe` through `srun` on the assigned GPU node. It does not invoke
Post Calculation.

## 7. Post-processing system

Post processing is split into two modules with different inputs and different
physics responsibilities:

```text
fit_state JSON (embedded model) + truth MC + selected MC
                         |
                         v
                  Post Calculation
                  (GPU integration)
                         |
                         +--> post_result TXT/ROOT + fit-fraction LaTeX

projection ROOT from Fit
         |
         v
    Post Plotting
    (ROOT histograms)
         |
         +--> PDF/EPS figures
```

Neither downstream module reads `fit_result-<tag>.txt`. Post Calculation does
not read the projection file, and Post Plotting does not reconstruct the model
or read truth MC.

### 7.1 Post Calculation inputs and compatibility checks

`Post.exe` requires:

1. `fit_state-<tag>.json` from the selected fit;
2. generated truth MC before selection;
3. selected normalization MC from the same unweighted production.

The original source path recorded in the fit state is provenance only. Post
does not open it and cannot be redirected to a different model file.

Both MC files use the same GVV `Pwa` branch contract as fit samples. The
selected sample must represent the selected subset of the truth production;
otherwise the selected/truth ratios are not efficiencies.

Before numerical work, Post Calculation:

- validates the fitted-state JSON structure, finite values, bounds, and
  covariance symmetry;
- parses and recompiles the embedded canonical GVV model;
- recomputes and compares the definition, implementation, and combined
  compatibility signatures;
- rebuilds the parameter mapping and compares every parameter name in order.

This prevents silently applying a covariance or parameter vector to changed
configuration or to an executable with different amplitude semantics.

### 7.2 Component integration

For each truth and selected event, the Term-level path evaluates all
coefficients and every packed upper-triangle pair. The GPU reduces each pair
directly over bounded event batches. Host memory therefore retains only one
truth and one selected integral per pair, rather than an
`N_event x N_pair` matrix.

The integrated pair sum is independently checked against the optimized total
PDF integral for both samples. This is the numerical bridge between the
component representation and the fit representation.

### 7.3 Post observables

Let `T_i` be a diagonal truth integral, `S_i` its selected integral, `T_ij` a
complete signed interference integral, and `T_total` the coherent truth sum.
The output includes:

- Term fit fraction: `T_i / T_total`;
- Term component efficiency: `S_i / T_i`;
- pair interference fraction: `T_ij / T_total`;
- total coherent efficiency: `S_total / T_total`;
- JPC-group fit fraction and efficiency, including all within-group pairs;
- cross-JPC-group interference fractions.

The sum of all diagonal fractions and all pair interference fractions must
close to one. The sum of all group fractions and cross-group interference
fractions must also close to one.

An active ordinary Term whose fitted coupling is exactly zero has a zero fit
fraction but an undefined component efficiency `0/0`. The current code rejects
that observable instead of assigning an artificial value. Move a reference
before testing it against zero, but note that an ordinary non-reference Term
can still reach this mathematical edge case.

### 7.4 Covariance propagation

Post Calculation reevaluates all observables at finite parameter offsets to
construct a numerical Jacobian `J`. It uses a central step when the requested
step fits on both sides of a bound, otherwise a resolvable one-sided step. The
fit covariance `V` is propagated as

```text
V_observable = J V J^T.
```

The reported errors therefore include fitted-parameter covariance only. They
do not include finite-MC integration uncertainty or systematic uncertainty.

### 7.5 Post Calculation outputs and workflow

`Post.exe` writes:

| Product | Contents |
|---|---|
| `post/calculation/results/post_result-<tag>.txt` | readable observable table, integrals, and closure values |
| `post/calculation/results/post_result-<tag>.root` | observable tree, metadata, covariance, and correlation matrices |
| `post/calculation/results/fit_fractions-<tag>.tex` | compact fit-fraction table |
| `runlog/post-<tag>.log` | Slurm worker and executable output |

`submit_post.sh` owns only this numerical workflow. It checks the three inputs,
submits a separate GPU Slurm job, runs `Post.exe` through `srun`, and verifies
the tagged TXT and ROOT products. It never submits a fit or runs plotting.

### 7.6 Post Plotting

Plotting reads only projection schema version 3. It discovers the active Terms,
JPC groups, and configured backgrounds from the map trees; it does not compile
a nominal resonance list.

The common plotting utilities:

- bind the dynamic ROOT branches;
- build data, signed-background, total-fit, group, and optional diagonal-Term
  histograms;
- use exchange-symmetric filling for observables with interchangeable omegas;
- calculate simple binned diagnostic chi-square values;
- find a complete vertical envelope from data errors and every histogram that
  a panel will draw;
- provide the common base ROOT style and project-root path helper.

Each `.cxx` macro is the complete presentation module for one figure. It owns
its default paths, observable list or moment orders, binning, axis labels and
ranges, canvas layout, curve styles, draw options, legend, and annotations in a
single `User configuration` preamble. The angular-moment header retains only
Legendre arithmetic, histogram filling, and the moment chi-square calculation.
The odd macro keeps the ordered-omega diagnostic separate because it is not a
label-independent observable of two identical omegas.

The standard presentation contract is explicit rather than implicit in ROOT
defaults. Data use black markers, Background is a gray hatched histogram, the
original solid-blue Total-fit style is reserved, every coherence class uses
the same dashed line style with a distinct color, and diagonal Term styles are
deterministic and distinct under model reordering. The polarization figure is
three horizontal panels with a shared legend; the component figure keeps six
`3 x 2` physics panels and gives their dynamic legend a full-width strip.
Angular-moment panels identify `Data - signed background` and fitted signal MC,
show unnormalized binwise Legendre sums with computed mass-bin widths, and draw
a gray zero reference for signed nonzero moments. Frame and candidate labels
state the X or omega helicity convention instead of relying on ambiguous short
names.

`post/plotting/draw.sh` runs the six ROOT macros and writes PDF/EPS files under
`post/plotting/results/`. It accepts no argument for the default
`results/projection-initial.root` or one argument to override that input. The
same macros can be executed directly with ROOT; their no-argument defaults and
runtime relative arguments are resolved from the project root located from the
macro source. The full visual and naming contract is recorded in
[`post/plotting/PLOTTING_STYLE.md`](../post/plotting/PLOTTING_STYLE.md).
Plotting needs no GPU and submits no Slurm job.

## 8. Extension boundaries

### 8.1 Add, remove, or disable a Resonance on an existing Wave

This is a configuration-only operation.

1. Add the Resonance object to `config/model.json` with a supported propagator
   and its exact parameter contract.
   Reuse an existing Resonance ID only when the new Term must share that exact
   physical propagator instance; otherwise create a new ID even when the
   propagator type is the same. Set `fixed`, `value`, `step`, and `bounds` on
   each supported propagator parameter in this object.
2. Add a Term that references the Resonance in `dynamics.resonance` and selects
   an existing registered Wave ID.
3. Select the coupling policy and preserve the reference rules.
4. To remove a contribution temporarily, set its Term to `"active": false`.
5. Run the model/registry tests, then perform the fit comparison appropriate to
   the physics study.

No source array, total count, parameter map, Projection map, Post loop, or plot
legend should be edited. Completely deleting a contribution means deleting
its Term and, if no other active Term uses it, its Resonance definition.

### 8.2 Add a new GVV Wave

A new complete tensor basis is a process-code extension.

1. Derive the complete production-and-decay numerator with explicit Lorentz,
   parity, Bose-symmetry, spin-coupling, and barrier-factor conventions.
2. Add any genuinely reusable missing primitive to `framework/`; keep
   GVV-specific constructions in `process/`.
3. Implement one pure device function in a new
   `process/waves/<DescriptiveName>.cuh` file. It receives
   `GVVEventKinematics` and optional barrier parameters and contains no
   Resonance ID, propagator, coupling, or model-wide index.
4. Add one enum value and one dispatch branch in `WaveRegistry.cuh`.
5. Add one host registry record in `WaveRegistry.cu`: stable ID, JPC, LaTeX
   label, physically justified coherence class, and device type.
6. Extend registry/model compilation tests.
7. Extend GPU numerical tests for finiteness, transversality/projector
   identities, required Bose symmetry, Gram-matrix symmetry and positivity,
   nonzero diagonal support, and optimized/direct intensity equivalence.
8. Only then reference the new Wave ID from `model.json` and add Resonance
   Terms that use it.

Normally `ProcessAmplitude.cuh`, `FitLikelihood`, `ParameterMapping`,
`ProjectionWriter`, Post Calculation, and plotting require no edit. They are
already dimensioned by the compiled model. A change to
`ProcessAmplitude.cuh` is justified only if the process-wide polarization sum
or Wave-pair contraction itself changes.

### 8.3 Add a reusable propagator

Propagator formulae and device dispatch are framework concerns, but the
accepted JSON name and parameter policy remain a process-compiler concern.

1. Implement the identity-free numerical formula in
   `framework/dynamics/Propagators.cuh`.
2. Extend `PropagatorModel`, `PropagatorParameters`, and
   `evaluate_propagator` only with the numerical fields the formula needs. A
   compiled descriptor must contain its nominal daughter masses and barrier
   radius so callers do not reconstruct channel context.
3. Define the exact GVV JSON parameter contract and validation in
   `PropagatorCompiler.cu`.
4. Emit parameter/report metadata and any generic fit binding from that same
   compiler. `ParameterMapping` consumes the binding without a new
   propagator-specific branch.
5. Add formula, compiler, inactive-Term, parameter-order, and device-dispatch
   tests.

The implementation contract in `process/ModelCompiler.cu` is deliberately an
explicit version string rather than a source-tree hash. Review it whenever
changing Wave formulae, propagator formulae or compilation, the common process
contraction, omega substructure dynamics, or parameter interpretation. Bump
the string whenever an unchanged model JSON could acquire different numerical
amplitude or fitted-parameter semantics. Fit and Post must be rebuilt together
after such a bump. This compatibility guard complements, but does not replace,
the physics regression suite.

Do not bind a generic propagator to a Wave ID. Physical quantities such as
orbital angular momentum must be explicit propagator parameters when they
control the denominator.

Fixed daughter and subchannel line shapes are a separate process concern from
the configurable X Resonance registry. They should call the same reusable
framework formulae, but they do not need model JSON entries or device registry
enums unless the analysis intentionally makes those submodels configurable.

### 8.4 Reuse the project for a different final state

Changing from GVV to another topology is a project conversion, not adding a
second channel inside the current runtime model. The expected boundary is:

| Reuse directly | Replace for the new process | Review/adapt at application boundary |
|---|---|---|
| `framework/math` | process event representation | `app/Fit.cu` assembly and branch contract |
| `framework/tensors` | process kinematics/currents/constants | model `process` identifier and dynamics JSON |
| generic parts of `framework/dynamics` | complete Wave files and registry | supported propagator-to-JSON compiler policy |
| `framework/amplitude` pair algebra | Term coefficient/evaluation kernels | process parameter mapping for new free quantities |
| `framework/likelihood` | ROOT sample loader/schema | projection observables and ROOT schema |
| `framework/model` | process model compiler | Post observable definitions |
| `framework/fit` | process likelihood orchestration | submission names/paths and user documentation |
| fit-state JSON semantics | process-specific Post evaluator and plotting | tests and physics validation suite |

The new process should not copy and rename generic four-vector, tensor,
propagator, likelihood, Minuit, report, or fitted-state implementations. It
should supply a new explicit process layer with the same small interfaces. If
the new topology needs a generally useful primitive, that primitive belongs in
`framework/`; a full process amplitude never does.

One current presentation detail still needs adaptation during such a
conversion: `framework/fit/FitOutput.cpp` hard-codes the GVV report heading.
The report structure, process-detail callback, and fitted-state code remain
reusable, but that heading must be made process-neutral or replaced for the
new channel.

## 9. Scaling and memory model

Let `N` be sample events, `T` active Terms, `W` active complete Waves, and
`P=T(T+1)/2` packed Term pairs.

| Operation | Work/memory characteristic |
|---|---|
| F-matrix preparation | `O(N W^2)` once per sample; persistent `N W^2` doubles |
| Fit coefficient/intensity call | `O(N(T + W^2))`; persistent `N W` complex workspace and `N` intensities |
| Projection components | `O(N P)` in bounded batches; external per-event `T x T` ROOT vector |
| Post component integration | `O(N P)` GPU reduction in batches; host retains `O(P)` integrals |
| Post covariance propagation | central/one-sided reevaluations proportional to the number of free parameters |

This design optimizes the repeated Minuit path for the common use case of many
Resonances on relatively few Waves while retaining exact Term decomposition
only in downstream work.

## 10. Maintenance checklist

Before accepting an architecture change, verify:

- no `framework/` file includes `process/`;
- no new fixed model-wide Resonance, Term, Wave, or parameter count appears;
- one user-visible mapping has one owner rather than parallel lists;
- inactive Terms are filtered before their Resonance and parameters are
  compiled;
- complete Waves contain no Resonance propagator or coupling;
- denominator angular-momentum dependence is explicit and physically correct;
- optimized total intensity closes against direct Term-pair components;
- fit-state reconstruction uses only its embedded model and checks definition,
  implementation, combined signature, and parameter order;
- projection maps remain dynamic when model content changes;
- Fit and Post submission scripts remain independent;
- GPU runtime tests are submitted through Slurm on an allocated GPU node, not
  run on an IHEP login node.

## 11. Further reading

- [`WORKFLOW.md`](WORKFLOW.md): build, Fit submission, output inspection, and
  both Post workflows.
- [`CODE_REFERENCE.md`](CODE_REFERENCE.md): every production file and its
  important internal blocks.
- [`MODEL_CONFIGURATION.md`](MODEL_CONFIGURATION.md): exact `model.json`
  contract and Resonance/Term editing procedures.
- [`WAVE_DEVELOPMENT.md`](WAVE_DEVELOPMENT.md): concise new-Wave checklist.
- [`TENSOR_CONVENTIONS.md`](TENSOR_CONVENTIONS.md): Lorentz indices, spin
  projection, bare orbital tensors, barrier separation, and normalized-CG
  conventions.
- [`../post/README.md`](../post/README.md): Post input/output usage.
- [`REFACTOR_LOG.md`](REFACTOR_LOG.md): refactor decisions, verification, and
  deferred support limits.

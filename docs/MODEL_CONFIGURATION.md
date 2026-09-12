# Model configuration

`config/model.json` is the single user-written description of an amplitude
model. It declares propagator instances, selects registered process Waves, and
combines them into active Terms. No generated source file and no hard-coded
model size is involved.

This document describes the contract implemented by
`framework/model/Model.cpp`, `process/ModelCompiler.cu`,
`process/PropagatorCompiler.cu`, and
`process/ParameterMapping.cu`. `config/model.schema.json` is useful for editor
completion and basic JSON validation, but the C++ loaders remain authoritative
for process-specific rules that JSON Schema does not express.

## The three model objects

The separation between Resonance, Wave, and Term is fundamental.

| Object | Owns | Does not own |
|---|---|---|
| Resonance | One propagator instance and its line-shape parameters | Angular tensor, coupling, active/inactive selection |
| Wave | One registered, complete GVV covariant numerator basis | Resonance identity, pole parameters, coupling |
| Term | One Resonance-Wave pairing, one complex coupling, and the active flag | Propagator formula or Wave implementation |

For event kinematics \(x\), the process evaluates a Term coefficient of the
form

\[
C_t(x)=g_t\,P_{r(t)}(s_X)\,D_{\omega_1}(x)D_{\omega_2}(x),
\]

and combines it with the registered Wave tensor \(W_{w(t)}(x)\). The two
omega decay factors are common to all current GVV Waves. A Resonance can be
used by more than one Term, and any number of Resonances can reuse the same
Wave.

The daughter factors are not additional `resonances` entries. The rho
subchannel uses the reusable two-body P-wave BWR, while the omega uses the
shared Breit-Wigner denominator with a process-built tabulated three-pion
running width. Their common `omega -> rho pi -> 3pi` implementation belongs to
the process decay model rather than the configurable X Resonance catalogue.
The rho charge channel selects nominal daughter and bachelor pion masses;
event-by-event reconstructed single-pion virtual masses are not line-shape
inputs.

This separation has two practical consequences:

- adding or removing a Resonance on an already registered Wave is a JSON-only
  operation;
- adding a genuinely new covariant basis first requires the source-level Wave
  registration described in [Wave development](WAVE_DEVELOPMENT.md).

## Top-level document

The accepted top-level fields are:

```json
{
  "schema_version": 1,
  "process": "psi2s_to_gamma_omega_omega",
  "metadata": {
    "name": "nominal",
    "description": "GVV nominal amplitude model"
  },
  "process_parameters": {},
  "resonances": [],
  "terms": []
}
```

`schema_version`, `process`, `resonances`, and `terms` are required. The
current GVV compiler accepts only schema version 1 and process
`psi2s_to_gamma_omega_omega`. Both arrays must be non-empty, and the active
model must contain exactly one global scale-and-phase reference.

Unknown top-level fields are rejected. IDs must start with a letter and then
contain only letters, digits, `_`, `.`, or `-`. Resonance IDs and Term IDs must
each be unique within their own arrays. Treat IDs as stable machine keys;
`label` is the human-readable or LaTeX-style presentation string and defaults
to the ID when omitted.

Array order is meaningful for reproducibility. Active Terms retain their JSON
order in the compiled coupling vector. Free coupling parameters are emitted in
that order, followed by free propagator parameters in active Resonance order,
then any free process parameter. GVV currently supports one optional process
parameter, `omega_resolution_sigma`, shared by both omega propagators.
Omitting it preserves the legacy unsmeared amplitude and parameter layout.
The nominal model floats it as `log_sigma_omega`, giving 36 free coordinates.
Its JSON form, mass convolution, units and saved-state behavior are described
in [Omega resolution](OMEGA_RESOLUTION.md).

## Resonance objects

A Resonance object has an ID, an optional label, one propagator name, and the
exact parameter set required by that propagator:

```json
{
  "id": "f0_1710",
  "label": "f0(1710)",
  "propagator": "two_body_running_bw",
  "parameters": {
    "mass": {"value": 1.723, "fixed": true},
    "width": {"value": 0.149, "fixed": true},
    "orbital_l": {"value": 0, "fixed": true}
  }
}
```

The Resonance `id` identifies the propagator **instance**. It is the sharing
boundary for line-shape parameters. If several Terms name `f2_1810`, they use
one compiled denominator and one fitted `mass_f2_1810`/`width_f2_1810` pair.
If two physically distinct states happen to use `two_body_running_bw`, give
them different Resonance IDs; the common `propagator` string selects a formula
and never causes their parameters to be shared.

### Parameter fields

Every parameter is an object with a required physical `value` and these
optional controls:

| Field | Default | Meaning |
|---|---:|---|
| `fixed` | `true` | Whether the parameter is excluded from Minuit |
| `transform` | `"identity"` | Minuit coordinate: `identity` or `log` |
| `step` | `0.1` | Initial Minuit step in the transformed coordinate |
| `bounds` | none | `[lower, upper]` in the transformed coordinate |

All numbers must be finite, `step` must be positive, and bounds must be finite
and increasing. A log-transformed physical `value` must be positive. For
example, if `value` is `0.5`, `transform` is `log`, and `bounds` are
`[-6.0, 3.0]`, Minuit starts from `log(0.5)` and the bounds apply to that log,
not directly to the physical ratio.

The generic schema permits these fields on any parameter, while the GVV
propagator compiler decides which controls are meaningful for each named
parameter. Mass and width use physical GeV coordinates with the `identity`
transform. When either is free, explicit finite bounds with a strictly
positive lower limit are required. Ratio parameters use the positive `log`
coordinate documented for their specific propagator. Discrete `orbital_l`
always remains fixed.

## Supported propagators

The parameter contract is exact. Missing, extra, or misspelled parameters are
rejected when the Resonance is needed by an active Term.

| `propagator` | Required parameters | Current fit policy |
|---|---|---|
| `nonresonant` | none | Unity, no propagator parameter |
| `fixed_width_bw` | `mass`, `width` | Mass/width fixed or free, identity |
| `two_body_running_bw` | `mass`, `width`, `orbital_l` | Mass/width fixed or free, identity; fixed integer \(L=0,1,2\) |
| `scalar_sd_running_bw` | `mass`, `width`, `sd_ratio` | Mass/width fixed or free, identity; positive log ratio fixed or free |
| `subtracted_effective_flatte` | `mass`, `width`, `omegaomega_ratio` | Mass/width fixed or free, identity; positive log ratio fixed or free |

Mass and width must both be positive for every resonant model. A
pole-normalized running-width model must also have its pole above the nominal
omega-omega threshold; use a physically appropriate sub-threshold line shape
rather than allowing a silently vanishing pole width.

For a free `mass` in `two_body_running_bw` or `scalar_sd_running_bw`, the
complete allowed interval must lie above the nominal omega-omega threshold.
This guarantees that every Minuit trial point has the pole normalization
required by those running-width formulae. The subtracted effective Flatte
continuation explicitly supports a subthreshold pole, so it does not impose
that running-width bound.

### `nonresonant`

This propagator is exactly \(P(s)=1\):

```json
{
  "id": "NR_0mp",
  "label": "nonresonant 0-+",
  "propagator": "nonresonant",
  "parameters": {}
}
```

Any mass, width, or other parameter on a nonresonant object is an error.

### `fixed_width_bw`

The convention is

\[
P(s)=\frac{1}{m_0^2-s-i m_0\Gamma_0}.
\]

Use this when a constant-width hypothesis is intended, including cases where
the nominal pole lies below a two-body threshold and a pole-normalized running
width would not be meaningful.

```json
{
  "id": "example_fixed",
  "label": "example fixed-width state",
  "propagator": "fixed_width_bw",
  "parameters": {
    "mass": {"value": 1.50, "fixed": true},
    "width": {"value": 0.10, "fixed": true}
  }
}
```

To float its pole parameters, change only their parameter objects. For
example, the following uses GeV for values, steps, and bounds:

```json
"mass": {
  "value": 1.81,
  "fixed": false,
  "transform": "identity",
  "step": 0.002,
  "bounds": [1.70, 1.92]
},
"width": {
  "value": 0.20,
  "fixed": false,
  "transform": "identity",
  "step": 0.005,
  "bounds": [0.02, 0.50]
}
```

### `two_body_running_bw`

This is a Wave-ID-independent two-body propagator. Its running width still
depends on the physical orbital angular momentum:

\[
\Gamma(s)=\Gamma_0\frac{m_0}{\sqrt{s}}
\left(\frac{q}{q_0}\right)^{2L+1}
\left(\frac{B_L(q)}{B_L(q_0)}\right)^2.
\]

The GVV compiler passes the nominal omega masses to this denominator. The
explicit `orbital_l` is a fixed integer from 0 through 2, matching the current
Blatt-Weisskopf library. It is not inferred from the selected Wave. This is
intentional: a Resonance propagator describes its total width, whereas a Wave
describes one numerator tensor.

```json
{
  "id": "eta_example",
  "label": "eta example",
  "propagator": "two_body_running_bw",
  "parameters": {
    "mass": {"value": 1.90, "fixed": true},
    "width": {"value": 0.20, "fixed": true},
    "orbital_l": {"value": 1, "fixed": true}
  }
}
```

Do not choose `orbital_l` from a filename alone. Derive it from the physical
partial-width hypothesis. A pole below the nominal omega-omega threshold
requires a different line-shape treatment.

The compiled descriptor stores both nominal daughter masses and the barrier
radius. Fit, Projection, and Post therefore call the same descriptor-only
dispatch and cannot substitute event-dependent daughter masses.

### `scalar_sd_running_bw`

This denominator models a scalar total width containing S- and D-wave
omega-omega contributions:

\[
\Gamma(s)=\Gamma_0\frac{\Phi_0(s)+r_{D/S}\Phi_2(s)}{1+r_{D/S}},
\qquad r_{D/S}=\frac{\Gamma_D}{\Gamma_S}\bigg|_{s=m_0^2}.
\]

Both \(\Phi_L\) factors are pole-normalized, so the total width equals
`width` at the pole. `sd_ratio` must use the `log` transform and have a
strictly positive physical value. Set `fixed` to `false` only when this ratio
is part of the fit.

```json
{
  "id": "f0_sd_example",
  "label": "f0 S+D example",
  "propagator": "scalar_sd_running_bw",
  "parameters": {
    "mass": {"value": 2.02, "fixed": true},
    "width": {"value": 0.20, "fixed": true},
    "sd_ratio": {
      "value": 0.10,
      "fixed": false,
      "transform": "log",
      "step": 0.10,
      "bounds": [-6.0, 3.0]
    }
  }
}
```

This propagator supplies one shared denominator when the same scalar
Resonance is used in both `gvv.scalar_00` and `gvv.scalar_22` Terms. It does
not constrain the two numerator couplings to equal the width ratio; any such
relation would be an additional physics-model constraint not implemented by
the current coupling layer. For a pure S- or pure D-width hypothesis, use
`two_body_running_bw` with `orbital_l` 0 or 2 instead of trying to set the log
ratio exactly to zero.

### `subtracted_effective_flatte`

The effective equal-mass omega-omega threshold correction is

\[
D(s)=m_0^2-s-i\left[m_0\Gamma_{\mathrm{rest}}+
G_{\omega\omega}\left(\rho(s)-\rho(m_0^2)\right)\right],
\]

with

\[
G_{\omega\omega}=R_{\omega\omega}m_0\Gamma_{\mathrm{rest}}.
\]

In JSON, `width` is \(\Gamma_{\mathrm{rest}}\) and
`omegaomega_ratio` is \(R_{\omega\omega}\). The subtraction preserves the
meaning of the supplied pole mass and rest width. The phase-space factor is
analytically continued below threshold; finite-width convolution of the omega
spectral functions is not part of this effective propagator.

`omegaomega_ratio` must be positive and log-transformed. If the exact
zero-coupling limit is required, use `fixed_width_bw`; a log parameter cannot
represent zero exactly.

## Adding a new propagator

Adding a new line shape is intentionally independent of Wave registration:

1. Implement the identity-free host/device formula in
   `framework/dynamics/Propagators.cuh`.
2. Add only the numerical enum/fields and dispatch needed by that formula in
   `framework/dynamics/PropagatorRegistry.cuh`.
3. Add one exact JSON compiler branch in `process/PropagatorCompiler.cu`.
   Compile nominal daughter masses, barrier radius, and every other channel
   constant into the descriptor at this boundary.
4. Emit human-readable parameter metadata and, for each supported free
   parameter, a generic binding containing its Minuit name, coordinate,
   bounds, transform, Resonance index, and target field.
5. Add formula tests, compiler valid/invalid tests, inactive-only pruning, and
   parameter-order/application tests.
6. Document the required JSON fields, units, physical domain, threshold
   policy, and whether a parameter may float.
7. If the new formula or compiler semantics change numerical amplitudes, bump
   `kGVVAmplitudeImplementationSignature` in `process/ModelCompiler.cu` in the
   same commit. This explicit contract is combined with the canonical model
   signature in every fitted state and checked by Post Calculation.

No propagator-specific edit should be needed in `WaveRegistry`,
`ParameterMapping`, `FitLikelihood`, `ProjectionWriter`, or Post Calculation.
Do not infer denominator orbital momentum from a Wave ID: the Resonance total
width hypothesis remains an explicit propagator contract.

## Term objects

A Term binds one registered Wave to one Resonance and assigns one coupling:

```json
{
  "id": "f0_1710_00",
  "label": "f_{0}(1710)",
  "wave": "gvv.scalar_00",
  "active": true,
  "coupling": {
    "mode": "positive_real",
    "reference": "phase",
    "initial": 0.1
  },
  "dynamics": {
    "type": "gvv_x_to_omega_omega",
    "resonance": "f0_1710"
  }
}
```

`active` defaults to `true`. The current process accepts exactly these two
dynamics fields on an active Term:

```json
"dynamics": {
  "type": "gvv_x_to_omega_omega",
  "resonance": "a_resonance_id"
}
```

The Wave ID must exist in the registry in `process/WaveRegistry.cu`, and the
Resonance ID must resolve after active dependency pruning.

## Couplings and phase conventions

The coupling contract is intentionally small:

| Mode | JSON `initial` | Minuit parameters | Required reference |
|---|---|---|---|
| `complex_cartesian` | `[real, imaginary]` | `Re_<term>`, `Im_<term>` | `none` or omitted |
| `positive_real` | positive number | `log_rho_<term>` | `phase` |
| `fixed_complex` | nonzero `[real, imaginary]` | none | `scale_and_phase` |

The current interface does not provide a fixed non-reference coupling or an
ordinary non-reference positive-real coupling. Extending those policies would
require a code change; do not emulate them with undocumented fields.

The amplitude has one global scale-and-phase redundancy and can have an
independent phase redundancy in each block of Waves that does not interfere
with other blocks. The rules are therefore:

1. The active model must contain exactly one nonzero `fixed_complex`
   `scale_and_phase` reference. `[1.0, 0.0]` is the usual convention.
2. Every active `coherence_class` must contain exactly one reference of either
   kind.
3. A class containing the global fixed reference must not also contain a
   `positive_real` phase reference.
4. Every other active class needs one `positive_real` phase reference.

The current Wave registry has two such classes:

| `coherence_class` | Registered Waves | Nominal reference |
|---|---|---|
| `positive_parity` | `gvv.scalar_00`, `gvv.scalar_22`, and all `gvv.tensor_*` Waves | `f0_1710_00`, `positive_real` phase reference |
| `negative_parity` | `gvv.pseudoscalar_11` | `eta_1760_11`, fixed global `scale_and_phase` reference |

`coherence_class` belongs to registered Wave metadata, not to `model.json`.
It does not turn interference on or off. The intensity engine always retains
all Wave cross terms. Separate classes are valid only when the corresponding
Wave Gram-matrix block is physically zero; this must be demonstrated by the
GPU numerical tests. The present names describe the verified GVV blocks, but
parity alone is not sufficient to classify a future Wave.

Choose a phase reference expected to remain significant. If a Term must be
tested against zero, first transfer the phase role to another active Term in
the same class and change the tested Term to `complex_cartesian`. Artificially
forcing the reference magnitude away from zero with a hard lower fit bound can
bias a physical zero-amplitude solution.

## How `active: false` works

Only Terms have an `active` flag. There is no Resonance-level active flag.
To disable a Resonance contribution, disable every Term that references that
Resonance.

Pruning occurs in this order:

1. The generic loader parses every Resonance and Term and validates JSON
   structure and coupling syntax.
2. The GVV compiler examines active Terms and resolves their dynamics.
3. Only Resonances referenced by active Terms are compiled.
4. Only Waves used by active Terms receive dense GPU slots.
5. The Minuit parameter layout is generated from the resulting compact model.

Consequently, an inactive Term contributes no coupling parameter, GPU entry,
fit-report component, projection component, or Post component. A Resonance
used only by inactive Terms is also absent, so none of its free mass, width, or
ratio parameters enters Minuit. If another active Term shares the Resonance,
the Resonance and its one shared set of free parameters correctly remain
active.

This behavior is independent of propagator type. It is not implemented by
setting a coupling to zero.

The compiler intentionally does not resolve the Wave, dynamics fields, or
propagator contract of an inactive-only dependency. This allows robust
pruning, but it is not a reason to leave a dormant scan entry broken: keep its
Wave and Resonance definitions valid so that re-enabling it is a safe one-line
operation.

Changing `active`, even from `true` to `false`, changes the canonical model
definition and its signature. Each schema-version-2 fit state embeds the exact
definition used by its Fit, so Post reconstructs the old or new model from the
selected state instead of reading the current `config/model.json`.

## Parameter mapping into Minuit

`process/ParameterMapping.cu` is the only translation between the compiled
model and the flat fit vector.

For active Terms, in JSON order:

- `fixed_complex` contributes no free parameter;
- `positive_real` contributes `log_rho_<term-id>`, starting from
  `log(initial)`, with step `0.10`;
- `complex_cartesian` contributes `Re_<term-id>` and `Im_<term-id>`, each with
  step `0.05`.

The current JSON coupling object does not expose coupling steps or bounds.
Later multistart seeds randomize free complex magnitudes/phases according to
the fit configuration; the fixed reference remains unchanged.

After all coupling parameters, active Resonances contribute:

- `mass_<resonance-id>` for a free identity-coordinate mass;
- `width_<resonance-id>` for a free identity-coordinate width;
- `log_rDS_<resonance-id>` for a free `sd_ratio`;
- `log_Romega_<resonance-id>` for a free `omegaomega_ratio`.

Their steps and bounds come from the corresponding parameter objects. Mass
and width coordinates are already physical GeV values; ratio coordinates are
in log space and are exponentiated back to positive physical values at every
likelihood evaluation. One binding is emitted per free parameter of each
active Resonance instance, independent of how many Terms reference it.

In the current nominal model, `eta_1760` and `f0_2020` expose their mass and
width as free identity-coordinate parameters. The `f0_2020` starts from the
PDG 2025 Breit-Wigner summary values and uses broad, user-adjustable physical
bounds. The retained `eta_2225` parameter objects are also configured as free,
but its only Term is inactive, so neither the Resonance nor its parameters
enter the compiled model or Minuit layout.

Because inactive objects are removed before this layout is built, no source
constant describes the number or order of Minuit parameters.

## Worked model-editing procedures

The following objects are array entries. Insert them into `resonances` and
`terms` without copying explanatory prose into JSON.

### Add one Resonance on an existing Wave

For a new scalar S-wave hypothesis, add the Resonance:

```json
{
  "id": "f0_2020",
  "label": "f0(2020)",
  "propagator": "two_body_running_bw",
  "parameters": {
    "mass": {"value": 1.992, "fixed": true},
    "width": {"value": 0.200, "fixed": true},
    "orbital_l": {"value": 0, "fixed": true}
  }
}
```

Then add its Term:

```json
{
  "id": "f0_2020_00",
  "label": "f_{0}(2020)",
  "wave": "gvv.scalar_00",
  "active": true,
  "coupling": {
    "mode": "complex_cartesian",
    "initial": [0.1, 0.0]
  },
  "dynamics": {
    "type": "gvv_x_to_omega_omega",
    "resonance": "f0_2020"
  }
}
```

No count, C++ array, likelihood code, Projection code, or Post code changes.
The existing `positive_parity` phase reference remains the only reference in
that class, so the new coupling must be ordinary complex Cartesian.

If this is the first Term in a newly registered coherence class, choose one
stable Term in that class as `positive_real` with `reference: "phase"`. Do not
add a second global `scale_and_phase` reference.

### Add S- and D-wave Terms sharing one scalar denominator

When the physics model specifically calls for a shared scalar S+D total
width, add one `scalar_sd_running_bw` Resonance and two Terms referencing it:

```json
{
  "id": "f0_2020_sd",
  "label": "f0(2020) S+D",
  "propagator": "scalar_sd_running_bw",
  "parameters": {
    "mass": {"value": 1.992, "fixed": true},
    "width": {"value": 0.200, "fixed": true},
    "sd_ratio": {
      "value": 0.10,
      "fixed": false,
      "transform": "log",
      "step": 0.10,
      "bounds": [-6.0, 3.0]
    }
  }
}
```

```json
{
  "id": "f0_2020_sd_00",
  "label": "f_{0}(2020)~(00)",
  "wave": "gvv.scalar_00",
  "coupling": {
    "mode": "complex_cartesian",
    "initial": [0.1, 0.0]
  },
  "dynamics": {
    "type": "gvv_x_to_omega_omega",
    "resonance": "f0_2020_sd"
  }
}
```

```json
{
  "id": "f0_2020_sd_22",
  "label": "f_{0}(2020)~(22)",
  "wave": "gvv.scalar_22",
  "coupling": {
    "mode": "complex_cartesian",
    "initial": [0.05, 0.0]
  },
  "dynamics": {
    "type": "gvv_x_to_omega_omega",
    "resonance": "f0_2020_sd"
  }
}
```

Both Terms remain in the `positive_parity` coherence class and interfere. Note
again that `gvv.scalar_22` is the scalar \(0^{++}\), \(L=S=2\) basis; it is
not a spin-two \(2^{++}\) Wave.

The nominal `f0(1500)` and `f0(1710)` models also activate both scalar
Wave Terms, but deliberately retain their existing Resonance-wide denominator
hypotheses. Selecting `gvv.scalar_22` adds the D-wave numerator and its
barrier factor; it does not implicitly change the Resonance propagator.

### Disable an ordinary Term

For a reversible scan, add or change only:

```json
"active": false
```

For example, disabling `f0_1500_00` removes only that coupling because
`f0_1500_22` still uses the same Resonance. The `f0_1500` propagator and its
free `log_Romega_f0_1500` parameter remain active. Disable both
`f0_1500_00` and `f0_1500_22` to remove the shared propagator and its free
parameter from the runtime model. The Resonance object may remain in the JSON
for later reuse.

Do not replace this operation with an initial coupling of `[0.0, 0.0]`; that
would leave two free parameters in Minuit and would not disable the Term.

### Disable the phase reference of one class

Suppose `f0_1710_00`, the nominal `positive_parity` phase reference, is to be
disabled while any other Term in that class remains active, including a
`2++` Term. Transfer the phase convention first. For example, change
`f0_1500_00` to:

```json
"coupling": {
  "mode": "positive_real",
  "reference": "phase",
  "initial": 0.1
}
```

and set:

```json
"id": "f0_1710_00",
"active": false
```

The inactive Term may retain its old coupling object because inactive
references do not count. If every active Term in `positive_parity` is
disabled, that class disappears and no replacement reference is needed. If
all scalar Terms are disabled while a tensor Term remains active, however,
one stable tensor Term must become the `positive_real` reference.

### Disable the global scale-and-phase reference

The nominal global reference is `eta_1760_11`. If it is disabled while other
Terms in `negative_parity` remain, transfer the fixed convention within that
class in the same edit. For example, make `eta_c_11`:

```json
"coupling": {
  "mode": "fixed_complex",
  "reference": "scale_and_phase",
  "initial": [1.0, 0.0]
}
```

and set `eta_1760_11` to `"active": false`. Moving the fixed reference within
the same coherence class is the least disruptive convention change. If the
fixed reference is moved into another active class, change that destination
class's former `positive_real` phase reference to `complex_cartesian` in the
same edit. If the original class remains active, give it one `positive_real`
reference; if it becomes empty, no replacement is required.

### Re-enable a Term

`"active": true` and an omitted `active` field are equivalent, but references
must be reconsidered.

If the `positive_parity` phase role was moved from `f0_1710_00` to
`f0_1500_00`, choose one of these two valid conventions when re-enabling:

- keep `f0_1500_00` as `positive_real` and change `f0_1710_00` to
  `complex_cartesian` with an array initial value; or
- restore `f0_1500_00` to `complex_cartesian` and restore `f0_1710_00` as the
  sole `positive_real` phase reference.

Likewise, a re-enabled former global reference must become ordinary
`complex_cartesian` if another active Term still owns `scale_and_phase`, or the
fixed role must be transferred back atomically. Merely flipping `active` can
otherwise create two references in one class or two global fixed references.

Before re-enabling, also confirm that its Wave is still registered, its
dynamics names an existing Resonance, and that Resonance still satisfies the
current propagator contract.

### Delete a Term and Resonance permanently

For example, to remove the nominal `X(2370)` hypothesis permanently:

1. Remove the complete Term object with ID `X_2370_11` from `terms`.
2. Search all remaining Term `dynamics.resonance` values for `X_2370`.
3. If there are no remaining references, remove the Resonance object with ID
   `X_2370` from `resonances`.
4. Recheck all phase references and the single global fixed reference.

If a Resonance has several Terms, removing one Term does not justify deleting
the shared Resonance. Conversely, deleting a Resonance while an active Term
still names it is rejected. An inactive Term can technically retain a dormant
reference to a deleted Resonance because active dependency resolution skips
it, but that produces a configuration that cannot be safely re-enabled. For a
permanent deletion, remove all such dormant Terms as well.

Use `active: false` for reversible model scans and physical-significance
tests. Delete objects only after the hypothesis is no longer meant to be
recoverable from that model document.

## Validation layers and common failures

Validation is intentionally split at ownership boundaries:

1. `model.schema.json` checks basic document shape in compatible editors.
2. `framework/model/Model.cpp` checks strict JSON structure, IDs, parameter
   syntax, coupling modes, nonzero global reference, and the global reference
   count.
3. `process/ModelCompiler.cu` checks the process ID, active dynamics, Wave
   registration, Resonance references, pruning, and one reference per active
   coherence class.
4. `process/PropagatorCompiler.cu` checks exact propagator fields, positive
   width, pole threshold, transforms, and supported \(L\).
5. `process/ParameterMapping.cu` creates the runtime-sized Minuit layout from
   coupling bindings and compiler-produced propagator bindings.
6. GPU tests check the numerical equivalence of the Term and Wave intensity
   paths and the registered Wave physics invariants.

Typical diagnostics have direct meanings:

| Diagnostic | Likely cause |
|---|---|
| `unknown field` | Misspelled or unsupported JSON member |
| `exactly one scale_and_phase reference` | The active global fixed reference is missing or duplicated |
| `coherence class ... requires exactly one phase reference` | No reference or multiple references in one active Wave block |
| `unregistered GVV wave` | Source-level Wave registration is absent or the ID is misspelled |
| `references unknown resonance` | Active Term dynamics names a missing Resonance |
| `does not accept parameter` | Propagator parameter is extra or misspelled |
| `requires positive log-transformed ...` | Ratio is non-positive or does not specify `transform: "log"` |
| `requires finite bounds with a positive lower limit` | A free mass or width lacks a finite positive physical interval or its initial value is outside it |
| `mass lower bound must remain above` | A free pole-normalized running-width mass range crosses the nominal omega-omega threshold |
| `fixed integer in the supported range` | `orbital_l` is not fixed identity integer 0, 1, or 2 |

## Safe scan checklist

Before editing:

- decide whether the change is reversible (`active: false`) or permanent
  deletion;
- identify every Term using the affected Resonance;
- identify the current global reference and the reference in every affected
  coherence class;
- derive the propagator hypothesis independently of the numerator Wave;
- remember that sharing is controlled by the Resonance ID, not by the
  propagator type or Wave ID.

While editing:

- use stable unique IDs and valid labels;
- keep the Resonance and Term edits coordinated;
- keep exactly one global `scale_and_phase` reference;
- keep exactly one reference in every active coherence class;
- keep log parameter values positive and log-coordinate initial values inside
  configured bounds;
- change the output tag in `fit.json` when scan products must coexist. Reusing
  a tag intentionally overwrites its previous products.

Before fitting:

```bash
python3 -m json.tool config/model.json >/dev/null
source config/gvv_env.sh
make -j2
make check
make gpu-tests
```

`make check` uses the repository nominal model as a regression fixture. For a
transient scan, it is often cleaner to keep `config/model.json` as the tested
baseline, copy the scan model to another JSON file, and point `fit.json` at
that file. A deliberate change to the repository nominal model must update its
development fixtures consistently.

Run `make check-gpu` only on an allocated CUDA device. It is never part of the
ordinary login-node check. The user controls all actual Fit/Post job
submission.

After a fit:

- inspect the active Term, Wave, Resonance, and parameter lists in
  `fit_result-<tag>.txt`;
- confirm that disabled couplings and inactive-only propagator parameters are
  absent;
- inspect convergence, covariance, boundaries, and closure diagnostics;
- retain the schema-version-2 fit-state JSON; it contains the exact formatted
  model definition, fitted coordinates, covariance, and compatibility
  signatures required by Post;
- do not manually edit the embedded model or its signatures.

For a new covariant basis, stop at the model layer and follow [Wave
development](WAVE_DEVELOPMENT.md). Do not invent an unregistered Wave ID or
put tensor formulae into JSON.

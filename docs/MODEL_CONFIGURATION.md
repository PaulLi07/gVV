# Model configuration

`config/model.json` is the single user-written description of an amplitude
model. It declares propagator instances, selects registered process Waves, and
combines them into active Terms. No generated source file and no hard-coded
model size is involved.

This document describes the contract implemented by
`framework/model/Model.cpp`, `process/WaveRegistry.cu`, and
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
that order, followed by free propagator parameters in active Resonance order.

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

The generic schema permits these fields on any parameter, but the current GVV
process compiler deliberately keeps all masses and widths fixed with the
identity transform. The only supported free propagator parameters are the
positive log ratios described below.

## Supported propagators

The parameter contract is exact. Missing, extra, or misspelled parameters are
rejected when the Resonance is needed by an active Term.

| `propagator` | Required parameters | Current fit policy |
|---|---|---|
| `nonresonant` | none | Unity, no propagator parameter |
| `fixed_width_bw` | `mass`, `width` | Both fixed, identity |
| `two_body_running_bw` | `mass`, `width`, `orbital_l` | All fixed; \(L=0,1,2\) |
| `scalar_sd_running_bw` | `mass`, `width`, `sd_ratio` | Mass/width fixed; positive log ratio fixed or free |
| `subtracted_effective_flatte` | `mass`, `width`, `omegaomega_ratio` | Mass/width fixed; positive log ratio fixed or free |

Mass must be positive and width must be non-negative.

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

`coherence_class` belongs to registered Wave metadata, not to `model.json`.
It does not turn interference on or off. The intensity engine always retains
all Wave cross terms. Separate classes are valid only when the corresponding
Wave Gram-matrix block is physically zero; this must be demonstrated by the
GPU numerical tests.

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
used only by inactive Terms is also absent, so even a free `sd_ratio` or
`omegaomega_ratio` on that Resonance does not enter Minuit. If another active
Term shares the Resonance, the Resonance and its free parameters correctly
remain active.

This behavior is independent of propagator type. It is not implemented by
setting a coupling to zero.

The compiler intentionally does not resolve the Wave, dynamics fields, or
propagator contract of an inactive-only dependency. This allows robust
pruning, but it is not a reason to leave a dormant scan entry broken: keep its
Wave and Resonance definitions valid so that re-enabling it is a safe one-line
operation.

Changing `active`, even from `true` to `false`, changes the canonical model
signature. A fit state from the old document cannot be supplied to Post with
the new document.

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

- `log_rDS_<resonance-id>` for a free `sd_ratio`;
- `log_Romega_<resonance-id>` for a free `omegaomega_ratio`.

Their steps and bounds come from the corresponding parameter objects and are
already in log space. At every likelihood evaluation the mapping exponentiates
these coordinates back to positive physical values.

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
The existing scalar phase reference remains the only reference in the scalar
coherence class, so the new coupling must be ordinary complex Cartesian.

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

Both Terms remain in the scalar coherence class and interfere. Note again
that `gvv.scalar_22` is the scalar \(0^{++}\), \(L=S=2\) basis; it is not a
spin-two \(2^{++}\) Wave.

### Disable an ordinary Term

For a reversible scan, add or change only:

```json
"active": false
```

For example, disabling `f0_1500_00` removes its coupling. Because no other
nominal Term uses `f0_1500`, the `f0_1500` propagator and its free
`log_Romega_f0_1500` parameter are also removed. The Resonance object may
remain in the JSON for later reuse.

Do not replace this operation with an initial coupling of `[0.0, 0.0]`; that
would leave two free parameters in Minuit and would not disable the Term.

### Disable the phase reference of one class

Suppose `f0_1710_00` is to be disabled while another scalar Term remains
active. Transfer the scalar phase convention first. For example, change
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
references do not count. If every scalar Term is disabled, the scalar class
disappears and no replacement scalar reference is needed.

### Disable the global scale-and-phase reference

The nominal global reference is `eta_1760_11`. If it is disabled while other
pseudoscalar Terms remain, transfer the fixed convention within the same class
in the same edit. For example, make `eta_c_11`:

```json
"coupling": {
  "mode": "fixed_complex",
  "reference": "scale_and_phase",
  "initial": [1.0, 0.0]
}
```

and set `eta_1760_11` to `"active": false`. Moving the fixed reference within
the same coherence class is the least disruptive convention change. If it is
moved to another class instead, the original active class still needs its own
`positive_real` phase reference.

### Re-enable a Term

`"active": true` and an omitted `active` field are equivalent, but references
must be reconsidered.

If the scalar phase role was moved from `f0_1710_00` to `f0_1500_00`, choose
one of these two valid conventions when re-enabling:

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
3. `process/WaveRegistry.cu` checks the process ID, active dynamics, Wave
   registration, Resonance references, exact propagator fields, supported
   \(L\), and one reference per active coherence class.
4. `process/ParameterMapping.cu` creates the runtime-sized Minuit layout.
5. GPU tests check the numerical equivalence of the Term and Wave intensity
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
| `fixed integer in the supported range` | `orbital_l` is not fixed identity integer 0, 1, or 2 |

## Safe scan checklist

Before editing:

- decide whether the change is reversible (`active: false`) or permanent
  deletion;
- identify every Term using the affected Resonance;
- identify the current global reference and the reference in every affected
  coherence class;
- derive the propagator hypothesis independently of the numerator Wave;
- preserve the exact model file needed by any pending Post calculation.

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
- retain the exact model JSON until all Post calculations for that fit are
  complete;
- do not combine a fit state with a model whose signature has changed.

For a new covariant basis, stop at the model layer and follow [Wave
development](WAVE_DEVELOPMENT.md). Do not invent an unregistered Wave ID or
put tensor formulae into JSON.

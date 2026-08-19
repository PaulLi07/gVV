# Model and fit configuration

The project has two user configuration files. `model.json` describes the
amplitude model; `fit.json` describes one run. No Python layer generates either
document.

## `model.json`

Top-level fields are `schema_version`, `process`, optional `metadata`,
`resonances`, and `terms`. Unknown fields and duplicate IDs are rejected before
GPU allocation.

### Resonance

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

Supported GVV propagator contracts are:

| Propagator | Parameters |
|---|---|
| `nonresonant` | empty object |
| `fixed_width_bw` | fixed identity `mass`, `width` |
| `two_body_running_bw` | fixed identity `mass`, `width`, integer `orbital_l` in 0--2 |
| `scalar_sd_running_bw` | fixed `mass`, `width`; positive log `sd_ratio` |
| `subtracted_effective_flatte` | fixed `mass`, `width`; positive log `omegaomega_ratio` |

A parameter may contain `value`, `fixed`, `transform`, `step`, and `bounds`.
For a log-transformed parameter, `value` is physical while `step` and `bounds`
are expressed in the transformed Minuit coordinate.

### Term

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

Coupling modes are:

- `complex_cartesian`: free real and imaginary parts;
- `positive_real`: free log magnitude and fixed phase;
- `fixed_complex`: fixed scale-and-phase reference, normally `1+0i`.

Every active coherence class must have exactly one phase reference. The class
is registered with the Wave and is not repeated in every Term.

The fixed `scale_and_phase` reference defines the overall amplitude scale and
phase and therefore must have a nonzero complex value. A `positive_real`
phase reference is fitted in its log magnitude. Choose a Term that is expected
to remain significant; if that Term is disabled or tested against zero, assign
the phase role to another active Term and make the tested coupling ordinary
`complex_cartesian`.

`two_body_running_bw` is independent of the registered Wave ID, but its
running width is not independent of the physical orbital angular momentum.
The explicit `orbital_l` controls the `q^(2L+1) B_L^2` behavior and may be 0,
1, or 2 with the current barrier-factor library. It is not inferred from the
Wave. A Resonance whose total width contains several partial waves requires a
corresponding multi-channel width model such as `scalar_sd_running_bw`.

Setting `active: false` removes the Term before model compilation. Its coupling
and a Resonance used only by inactive Terms do not enter Minuit, the report,
the GPU arrays, or downstream maps. This behavior is independent of propagator
type and whether that Resonance would otherwise have free parameters.

### Editing the model

For an existing Wave, adding or removing a Resonance requires only coordinated
edits to the `resonances` and `terms` arrays. No source-level counts or array
sizes change. A new covariant basis must first be implemented and registered as
described in `WAVE_DEVELOPMENT.md`.

## `fit.json`

```json
{
  "schema_version": 1,
  "model": "config/model.json",
  "inputs": {
    "data": "RootSet/data.root",
    "normalization_mc": "RootSet/normalization_mc.root",
    "backgrounds": [
      {"label": "SB1", "file": "RootSet/SB1.root", "coefficient": -0.5},
      {"label": "SB2", "file": "RootSet/SB2.root", "coefficient": 0.25}
    ]
  },
  "minimizer": {
    "n_starts": 10,
    "base_seed": 20260815,
    "maximum_edm": 0.001,
    "maximum_calls": 20000,
    "tolerance": 0.1,
    "error_definition": 0.5,
    "random_magnitude": [0.05, 5.0]
  },
  "output": {
    "directory": "results",
    "log_directory": "runlog",
    "tag": "initial"
  }
}
```

Backgrounds enter as
`ln L_eff += coefficient * sum(log(P(event)))`. Start zero uses model initial
values; later starts randomize free coupling magnitude/phase. The output tag
may contain letters, digits, dot, underscore, and hyphen. Reusing a tag
overwrites its report, state, projection, and log products.

Truth MC is intentionally not a fit input. It is supplied separately to Post
Calculation, together with the selected normalization MC.

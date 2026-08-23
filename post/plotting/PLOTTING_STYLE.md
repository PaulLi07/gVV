# GVV Plotting Style and Physics Conventions

This document defines the presentation and naming contract for figures made
from `results/projection-<tag>.root`. Plot-specific choices remain in the
configuration block at the top of each ROOT macro. Shared headers provide only
projection I/O, histogram construction, common ROOT defaults, label conversion,
vertical-range discovery, and other mechanics that have the same meaning in
every figure.

## Visual roles

The following styles have fixed meanings throughout the plotting system.

| Object | Required appearance | Notes |
| --- | --- | --- |
| Data | Black markers with statistical error bars | Draw the markers again after all histograms so they remain visible. |
| Background | Gray histogram with diagonal fill style `3004` | The histogram may be signed. Its negative bins must remain visible. |
| Total fit | Solid `kBlue + 1` line, width 2 | This style is reserved. No group or component may reuse this color. |
| Coherent JPC group | Line style 2 (dashed), common width, one distinct color per group | Legend entries use `coherent <JPC>`, for example `coherent 2++`; they never use `<JPC> coherence class`. |
| Term component | Unique and stable color/line-style combination | A component style is derived from model metadata, not its current list position. Related waves of one resonance may use related colors, but no two active Terms may be visually identical. |
| Zero reference | Thin gray dotted line | Draw for signed nonzero angular moments. It is a reference, not a fitted curve. |

The coherent-group palette must exclude black, gray, and the Total-fit blue.
When a model adds a group, extend the palette before colors repeat. Component
legends must be generated dynamically from the projection metadata.

## Layout and legends

- The main projection and omega-decay check figures use a `3 x 2` layout.
- The polarization figure uses three horizontal panels.
- The component figure uses six `3 x 2` physics panels plus a full-height
  legend column at the far right. A large component list must never force the
  physics panels to shrink into a narrow `4 x 2` grid.
- Even angular moments use `2 x 2`; the ordered-omega odd diagnostic uses
  `3 x 1`.
- Except for the component figure's dedicated right-hand column, every shared
  legend is drawn inside the first subplot. That first subplot alone receives
  additional vertical headroom; the remaining panels keep their normal scale.
- A legend must not obscure data, errors, peaks, or fitted curves.
- The number of legend columns is chosen from the number and length of entries.
  Labels must remain readable at the final PDF size.
- `projection_detailed-<tag>` is obsolete and is not part of the supported
  plotting output set.

## Axes and automatic ranges

Every vertical range is calculated from everything that is actually drawn:

1. data bin content plus and minus its error;
2. signed Background;
3. Total fit;
4. every coherent-group or Term-component curve in that panel.

After finding this envelope, the macro applies a small plot-specific headroom
factor. For every non-component figure with a shared legend, the first subplot
deliberately uses a larger upper factor. A style change must never clip a
negative Background bin, a component larger than the coherent Total, or the
end of a data error bar.

Mass-axis bin widths are derived from the configured limits and number of bins.
Do not hard-code text such as `50 MeV`. Use `GeV/c^2` on mass x axes and
`MeV/c^2` when quoting a mass-bin width on a y axis. Azimuthal bin widths carry
the unit `rad`; cosine variables are dimensionless.

## Coordinate and observable names

### Production frames

- `cos(theta_gamma)` is measured in the psi rest frame relative to the beam
  `+z` axis.
- In the `X = omega omega` rest frame, `+z_X` points opposite the radiative
  photon. The production plane fixes the other axes. The omega direction is
  therefore labelled with the `X helicity` frame.
- For omega candidate `i`, the omega helicity-frame `+z_i` follows `omega_i`
  in the X rest frame. Its `y_i` axis is normal to the
  `X -> omega omega` decay plane and `x_i` completes the right-handed basis.

Use frame qualifiers when a bare name could be ambiguous, for example
`cos(theta_omega) [X helicity]` and
`cos(theta_pi+) [omega helicity]`.

### Oriented omega decay plane

In the corresponding omega rest frame, the directed analyzer normal is

```text
n_i = unit( p(pi+_i) cross p(pi-_i) ).
```

The polarization panels are named

- `cos(theta_n_omega) [omega helicity]`;
- `phi_n_omega [omega helicity]`;
- `Delta phi(n_1, n_2)`.

The last quantity is the signed, wrapped local-azimuth difference
`phi_n1 - phi_n2`, not an unsigned geometric plane-opening angle.

### Identical-omega treatment

Candidate-combined histograms fill omega1 and omega2 with one-half weight each.
Exchange-symmetrized omega-direction observables additionally fill
`cos(theta_omega)` together with its negative and `phi_omega` together with
`phi_omega + pi` after wrapping. The decay-plane difference is filled together
with its negative. State this convention once in a figure note or macro
introduction; do not shorten it to an unexplained `sym.` in an axis title.

The omega-decay checks combine the two candidates but do not reflect the pion
cosines. Their axes are

- `cos(theta_pi+) [omega helicity]`;
- `cos(theta_pi-) [omega helicity]`;
- `cos(theta_pi0) [omega helicity]`;
- `M(pi+ pi-)`, `M(pi+ pi0)`, and `M(pi- pi0)`.

## Meaning of fitted curves

The Total-fit projection is the fitted signal MC plus the signed projection
Background. A curve labelled `coherent <JPC>` contains the coherent
contribution within that JPC group. It does not display interference with
other groups, so the visible group curves need not sum to the Total fit.

A Term-component curve is the diagonal contribution `|A_i|^2` only. It omits
all pairwise interference terms and must be drawn with `HIST SAME`, without
spline smoothing. Component figures must state that interference is omitted.

Angular-moment points are `Data - signed Background`; their comparison curve is
the fitted signal MC. The histogram content is an unnormalized binwise sum,

```text
sum_events P_l(cos(theta_omega)),
```

not an event-normalized average. Even moments use the exchange-symmetric omega
direction. Odd moments use the ordered `omega1` direction and are diagnostic
only; they are not exchange-invariant observables of the identical-omega final
state. Put that warning on the odd-moment canvas once, not in every panel.

## Running the macros

Run all supported plots with the convenience script:

```bash
bash post/plotting/draw.sh
```

Each macro also runs directly with its documented defaults, for example:

```bash
root post/plotting/macros/Draw_polarization.cxx
```

Specify another projection file and output prefix when needed:

```bash
root -l -b -q 'post/plotting/macros/Draw_polarization.cxx(
  "results/projection-myfit.root",
  "post/plotting/results/polarization-myfit")'
```

The plotting code creates the output directory when needed and writes both PDF
and EPS files. Plotting an existing projection ROOT file does not require a new
fit, a GPU, or a Slurm job.

## Review checklist

Before accepting a plotting change:

1. Run every supported macro against the same projection ROOT file.
2. Confirm histogram bin contents, integrals, and printed chi-square diagnostics
   are unchanged by a presentation-only edit.
3. Check that Background is gray, Total fit retains its reserved style, and all
   coherent JPC groups are dashed and labelled `coherent <JPC>`.
4. Check that no class or component uses the Total-fit blue and no two active
   Terms have an identical final appearance.
5. Inspect the rendered PDFs at normal reading size for clipped labels, clipped
   errors, legend overlap, weak contrast, and empty layout cells.
6. Check every axis for the correct frame, candidate convention, bin-width unit,
   and physical meaning.
7. Verify that signed quantities include zero where appropriate and that no
   negative content is hidden.
8. Confirm that the component and coherent-group annotations describe omitted
   interference correctly.

New plotting macros must follow this checklist and keep their user-editable
configuration in the macro introduction rather than moving figure-specific
choices into a shared header.

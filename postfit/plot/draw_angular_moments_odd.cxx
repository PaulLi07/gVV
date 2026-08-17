#include "postfit/GVVAngularMoments.h"

// Odd moments use the input-labelled omega1 without exchange symmetrization.
// They diagnose pairing/order bias and are not label-independent observables
// of the identical-omega final state.
void draw_angular_moments_odd(
    const char* input_file = "results/projection0.root",
    const char* output_prefix = "results/plot/angular_moments_odd_diagnostic")
{
    gvvplot::DrawAngularMoments(input_file, output_prefix, true);
}

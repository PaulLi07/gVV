#include "../GVVAngularMoments.h"

// Odd moments use the input-labelled omega1 without exchange symmetrization.
// They diagnose pairing/order bias and are not label-independent observables
// of the identical-omega final state.
void draw_angular_moments_odd(
    const char* input_file = "results/projection-initial.root",
    const char* output_prefix = "post/plotting/results/angular_moments_odd_diagnostic-initial")
{
    gvvplot::DrawAngularMoments(input_file, output_prefix, true);
}

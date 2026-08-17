#include "postfit/GVVAngularMoments.h"

// Physical, omega-exchange-symmetrized even moments P0/P2/P4/P6.
void draw_angular_moments(
    const char* input_file = "results/projection0.root",
    const char* output_prefix = "results/plot/angular_moments")
{
    gvvplot::DrawAngularMoments(input_file, output_prefix, false);
}

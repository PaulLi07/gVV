#include "../GVVAngularMoments.h"

// Physical, omega-exchange-symmetrized even moments P0/P2/P4/P6.
void draw_angular_moments(
    const char* input_file = "results/projection-initial.root",
    const char* output_prefix = "post/plotting/results/angular_moments-initial")
{
    gvvplot::DrawAngularMoments(input_file, output_prefix, false);
}

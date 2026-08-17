#include "postfit/GVVPlotUtils.h"

// Main publication-style 3x2 projection.  All observables involving a choice
// of omega1/omega2 use the explicit exchange-symmetric fill convention.
void Draw_projection_2_3(
    const char* input_file = "results/projection0.root",
    const char* output_prefix = "results/plot/projection")
{
    gvvplot::DrawProjection(input_file, output_prefix, false, false);
}

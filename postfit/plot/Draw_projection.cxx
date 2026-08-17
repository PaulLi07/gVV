#include "postfit/GVVPlotUtils.h"

// Detailed 4x2 diagnostic: the six symmetric main observables plus the
// combined omega -> 3pi mass, with a dedicated legend pad.
void Draw_projection(
    const char* input_file = "results/projection0.root",
    const char* output_prefix = "results/plot/projection_detailed")
{
    gvvplot::DrawProjection(input_file, output_prefix, true, false);
}

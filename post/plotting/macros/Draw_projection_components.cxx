#include "../GVVPlotUtils.h"

// Separate diagonal-component diagnostic.  Curves are |A_i|^2 terms read
// from weight_component[i][i]; they do not sum to the coherent total because
// the pairwise interference terms are intentionally omitted.
void Draw_projection_components(
    const char* input_file = "results/projection-initial.root",
    const char* output_prefix = "post/plotting/results/projection_components-initial")
{
    gvvplot::DrawProjection(input_file, output_prefix, false, true);
}

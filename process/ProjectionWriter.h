// ROOT serialization contract between the gVV fit and downstream projection
// tools. FitLikelihood supplies samples/intensities; this module owns the
// process-specific tree schema, derived observables, weights, and metadata.
#ifndef CTPWA_PROCESS_PROJECTION_WRITER_H
#define CTPWA_PROCESS_PROJECTION_WRITER_H

#include <string>

class FitLikelihood;

void write_gvv_projection(
    FitLikelihood& likelihood,
    const std::string& save_name,
    int best_start,
    long long best_seed,
    double minimum);

#endif // CTPWA_PROCESS_PROJECTION_WRITER_H

// Stable text and covariance output interface for a completed generic fit.
// Process-specific physical state can be appended through FitDetailWriter.
#ifndef CTPWA_FRAMEWORK_FIT_OUTPUT_H
#define CTPWA_FRAMEWORK_FIT_OUTPUT_H

#include "framework/fit/FitEngine.h"

#include <functional>
#include <iosfwd>
#include <string>
#include <vector>

namespace ctpwa {

struct FitResultContext {
    std::string fit_config_file;
    std::string model_config_file;
    std::string model_name;
    int data_entries = 0;
    int normalization_mc_entries = 0;
};

using FitDetailWriter = std::function<void(std::ostream&)>;

void write_fit_result(
    const std::string& file_name,
    const FitAttempt& best,
    const FitOptions& options,
    const std::vector<FitParameterSpec>& parameters,
    const FitResultContext& context,
    const FitDetailWriter& write_details);

void write_covariance_matrix(
    const std::string& file_name,
    const FitAttempt& best);

} // namespace ctpwa

#endif // CTPWA_FRAMEWORK_FIT_OUTPUT_H

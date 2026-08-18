// Human-readable fit diagnostics. Machine consumers use FitState instead.
#ifndef CTPWA_FRAMEWORK_FIT_OUTPUT_H
#define CTPWA_FRAMEWORK_FIT_OUTPUT_H

#include "framework/fit/FitEngine.h"

#include <functional>
#include <iosfwd>
#include <string>
#include <vector>

namespace ctpwa {

struct FitSampleSummary {
    std::string role;
    std::string label;
    std::string file;
    int entries = 0;
    double likelihood_coefficient = 0.0;
};

struct FitResultContext {
    std::string output_tag;
    std::string fit_config_file;
    std::string model_config_file;
    std::string model_name;
    std::string model_signature;
    std::vector<FitSampleSummary> samples;
};

using FitDetailWriter = std::function<void(std::ostream&)>;

void write_fit_result(
    const std::string& file_name,
    const FitSummary& summary,
    const FitOptions& options,
    const std::vector<FitParameterSpec>& parameters,
    const FitResultContext& context,
    const FitDetailWriter& write_details);

} // namespace ctpwa

#endif // CTPWA_FRAMEWORK_FIT_OUTPUT_H

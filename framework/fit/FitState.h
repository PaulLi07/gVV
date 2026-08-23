// Machine-readable handoff between Fit and downstream numerical tools.
// The complete canonical model document travels with the minimizer state so a
// downstream program never has to recover the model from an external path.
#ifndef CTPWA_FRAMEWORK_FIT_STATE_H
#define CTPWA_FRAMEWORK_FIT_STATE_H

#include "framework/fit/FitEngine.h"

#include <string>
#include <vector>

namespace ctpwa {

struct FitState {
    int schema_version = 2;
    std::string output_tag;
    std::string fit_config_file;
    // Provenance only. Consumers reconstruct the model from model_json.
    std::string model_config_file;
    std::string model_name;
    std::string model_json;
    std::string model_definition_signature;
    std::string model_implementation_signature;
    std::string model_signature;
    FitAttempt best;
    std::vector<FitParameterSpec> parameters;
};

void write_fit_state(
    const std::string& file_name,
    const FitState& state);

FitState read_fit_state(const std::string& file_name);

} // namespace ctpwa

#endif // CTPWA_FRAMEWORK_FIT_STATE_H

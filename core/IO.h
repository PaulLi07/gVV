#pragma once

#include "core/Minuit.h"
#include <string>
#include <vector>

// Typed representation of one fit run: samples, minimizer settings, model
// location, and the unified output tag.

namespace ctpwa {

struct WeightedSampleConfig {
    std::string label;
    std::string file;
    double likelihood_coefficient = 0.0;
};

struct FitInputConfig {
    std::string data_file;
    std::string normalization_mc_file;
    std::vector<WeightedSampleConfig> backgrounds;
};

struct FitOutputConfig {
    std::string directory = "results";
    std::string log_directory = "runlog";
    std::string tag = "initial";

    std::string result_file() const;
    std::string state_file() const;
    std::string projection_file() const;
    std::string log_file() const;
};

struct FitRunConfig {
    int schema_version = 1;
    std::string source_file;
    std::string model_file;
    FitInputConfig inputs;
    FitOptions minimizer;
    FitOutputConfig output;
};

FitRunConfig load_fit_run_config(const std::string& file_name);

} // namespace ctpwa

// Machine-readable handoff between Fit and downstream numerical tools.
// The complete canonical model document travels with the minimizer state so a
// downstream program never has to recover the model from an external path.

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

class GVVAmplitude;
class GVVSample;

// Non-owning projection input; background coefficients use the signed NLL convention.
struct GVVProjectionBackground {
    const GVVSample* sample;
    double likelihood_coefficient;
};
void write_gvv_projection(
    GVVAmplitude& amplitude, GVVSample& normalization_mc,
    const GVVSample& data, const std::vector<GVVProjectionBackground>& backgrounds,
    const std::string& save_name, const std::string& output_tag,
    const std::string& model_signature, int best_start,
    long long best_seed, double minimum);

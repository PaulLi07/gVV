// Process-neutral multi-start Minuit API. The caller supplies a flat parameter
// layout and an objective callback; no Resonance/Wave types enter this layer.
#ifndef CTPWA_FRAMEWORK_FIT_ENGINE_H
#define CTPWA_FRAMEWORK_FIT_ENGINE_H

#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace ctpwa {

// The fit engine knows how a parameter is randomized, but it deliberately
// knows nothing about resonances, waves, or a particular decay process.
enum class ParameterRandomization {
    None,
    ComplexReal,
    ComplexImaginary,
    LogMagnitude
};

struct FitParameterSpec {
    std::string name;
    double initial_value = 0.0;
    double step = 0.1;
    bool has_lower_bound = false;
    bool has_upper_bound = false;
    double lower_bound = 0.0;
    double upper_bound = 0.0;
    ParameterRandomization randomization = ParameterRandomization::None;
    int randomization_group = -1;
};

struct FitOptions {
    int number_starts = 1;
    long long base_seed = 20260815;
    double maximum_edm = 1.0e-3;
    int maximum_calls = 20000;
    double tolerance = 0.1;
    double error_definition = 0.5;
    double random_magnitude_min = 0.05;
    double random_magnitude_max = 5.0;
};

struct FitAttempt {
    int start_index = -1;
    long long seed = 0;
    std::vector<double> initial_values;
    std::vector<double> values;
    std::vector<double> errors;
    std::vector<double> covariance;
    int migrad_status = -1;
    int hesse_status = -1;
    int covariance_status = 0;
    double minimum = std::numeric_limits<double>::infinity();
    double edm = std::numeric_limits<double>::infinity();
    double error_definition = 0.5;
    double elapsed_seconds = 0.0;
    bool at_parameter_boundary = false;
    bool valid = false;
};

struct FitSummary {
    FitAttempt best;
    std::vector<FitAttempt> attempts;
};

using FitObjective = std::function<double(const std::vector<double>&)>;

FitSummary run_multistart_fit(
    const std::vector<FitParameterSpec>& parameters,
    const FitOptions& options,
    const FitObjective& objective);

} // namespace ctpwa

#endif // CTPWA_FRAMEWORK_FIT_ENGINE_H

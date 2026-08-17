#ifndef GVV_FIT_PARAMETERS_H
#define GVV_FIT_PARAMETERS_H

#include "process/WaveRegistry.cuh"

#include <string>
#include <vector>

enum class GVVFitParameterTarget {
    CouplingReal,
    CouplingImaginary,
    CouplingLogMagnitude,
    ResonanceLogSDRatio,
    ResonanceLogFlatteRatio
};

// Generated from the compiled model. This is the only description of
// Minuit/covariance ordering, initial values, steps, bounds, and state target.
struct GVVFitParameterSpec {
    std::string name;
    GVVFitParameterTarget target = GVVFitParameterTarget::CouplingReal;
    int target_index = -1;
    double initial_value = 0.0;
    double step = 0.1;
    bool has_lower_bound = false;
    bool has_upper_bound = false;
    double lower_bound = 0.0;
    double upper_bound = 0.0;
};

struct GVVParsedFitResult {
    GVVCompiledModel compiled_model;
    std::vector<std::string> parameter_names;
    std::vector<double> values;
    std::vector<double> errors;
    bool used_machine_readable_rows = false;
};

std::vector<GVVFitParameterSpec> gvv_fit_parameter_layout(
    const GVVCompiledModel& model);

std::vector<std::string> gvv_fit_parameter_names(
    const GVVCompiledModel& model);

std::vector<double> gvv_fit_parameters_from_model(
    const GVVCompiledModel& model);

void gvv_apply_fit_parameters_to_model(
    GVVCompiledModel& model,
    const std::vector<double>& parameters);

// Machine-readable parameter rows support any model. Historical nominal
// coupling/resonance rows are also accepted, but are resolved against ids in
// the supplied model rather than against a compiled-in default model.
GVVParsedFitResult gvv_read_fit_result(
    const std::string& file_name,
    GVVCompiledModel model);

std::vector<double> gvv_read_covariance_matrix(
    const std::string& file_name,
    int number_parameters);

void gvv_validate_covariance_matrix(
    const std::vector<double>& covariance,
    int number_parameters);

#endif // GVV_FIT_PARAMETERS_H

#ifndef GVV_FIT_PARAMETERS_H
#define GVV_FIT_PARAMETERS_H

#include "GVVModel.h"
#include "process/GVVProcessModel.h"

#include <array>
#include <string>
#include <vector>

enum class GVVFitParameterTarget {
    CouplingReal,
    CouplingImaginary,
    CouplingLogMagnitude,
    ResonanceLogSDRatio,
    ResonanceLogFlatteRatio
};

// Generated once from the compiled model.  This is the only description of
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

std::vector<GVVFitParameterSpec> gvv_fit_parameter_layout(
    const GVVCompiledModel& model);

std::vector<std::string> gvv_fit_parameter_names(
    const GVVCompiledModel& model);

std::vector<double> gvv_fit_parameters_from_model(
    const GVVCompiledModel& model);

void gvv_apply_fit_parameters_to_model(
    GVVCompiledModel& model,
    const std::vector<double>& parameters);

// Single source of truth for the Minuit/covariance parameter ordering.  Both
// Fit.exe and PostFit.exe use this structure; post-fit code must never infer a
// new ordering from a table or from the order in which couplings are printed.
struct GVVFitState {
    std::array<GVVResonanceParameters, GVV_NRESONANCES> resonances;
    std::array<GVVTermSpec, GVV_NTERMS> terms;
    std::array<DeviceComplex, GVV_NTERMS> couplings;
};

struct GVVParsedFitResult {
    GVVFitState state;
    std::vector<std::string> parameter_names;
    std::vector<double> values;
    std::vector<double> errors;
    bool used_machine_readable_rows = false;
};

GVVFitState gvv_default_fit_state();

std::vector<std::string> gvv_fit_parameter_names(
    const std::array<GVVResonanceParameters, GVV_NRESONANCES>& resonances);

std::vector<double> gvv_fit_parameters_from_state(const GVVFitState& state);

void gvv_apply_fit_parameters_to_state(
    GVVFitState& state,
    const std::vector<double>& parameters);

GVVParsedFitResult gvv_read_fit_result(const std::string& file_name);

std::vector<double> gvv_read_covariance_matrix(
    const std::string& file_name,
    int number_parameters);

// Checks finite entries, symmetry, positive diagonal and positive
// definiteness.  It throws with a diagnostic instead of allowing an invalid
// error matrix to enter the finite-difference propagation.
void gvv_validate_covariance_matrix(
    const std::vector<double>& covariance,
    int number_parameters);

bool gvv_fit_parameter_bounds(
    const std::string& name,
    double& lower,
    double& upper);

#endif // GVV_FIT_PARAMETERS_H

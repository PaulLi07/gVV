// Sole translation layer between the generic flat Minuit vector and mutable
// gVV couplings/propagator parameters.
#ifndef CTPWA_PROCESS_PARAMETER_MAPPING_H
#define CTPWA_PROCESS_PARAMETER_MAPPING_H

#include "framework/fit/FitEngine.h"
#include "process/PropagatorCompiler.h"

#include <iosfwd>
#include <vector>

enum class GVVFitParameterTarget {
    CouplingReal,
    CouplingImaginary,
    CouplingLogMagnitude,
    PropagatorParameter,
    OmegaResolutionSigma
};

struct GVVFitParameterBinding {
    ctpwa::FitParameterSpec fit;
    GVVFitParameterTarget target = GVVFitParameterTarget::CouplingReal;
    int target_index = -1;
    GVVPropagatorParameterTarget propagator_target =
        GVVPropagatorParameterTarget::Mass;
    GVVFitTransform transform = GVVFitTransform::Identity;
};

// This is the single translation layer between generic Minuit ordering and
// process-specific GVV model state. It is generated entirely from model.json
// and acts directly on GVVCompiledModel, without depending on FitLikelihood.
std::vector<GVVFitParameterBinding> gvv_fit_parameter_layout(
    const GVVCompiledModel& model);

std::vector<ctpwa::FitParameterSpec> gvv_fit_parameter_specs(
    const std::vector<GVVFitParameterBinding>& layout);

void gvv_apply_fit_parameters(
    GVVCompiledModel& model,
    const std::vector<GVVFitParameterBinding>& layout,
    const std::vector<double>& values);

void gvv_write_fit_details(
    std::ostream& output,
    const GVVCompiledModel& model,
    const std::vector<GVVFitParameterBinding>& layout,
    const ctpwa::FitAttempt& best);

#endif // CTPWA_PROCESS_PARAMETER_MAPPING_H

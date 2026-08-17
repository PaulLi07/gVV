#ifndef CTPWA_PROCESS_PARAMETER_MAPPING_H
#define CTPWA_PROCESS_PARAMETER_MAPPING_H

#include "framework/fit/FitEngine.h"
#include "process/WaveRegistry.cuh"

#include <iosfwd>
#include <vector>

class FitLikelihood;

enum class GVVFitParameterTarget {
    CouplingReal,
    CouplingImaginary,
    CouplingLogMagnitude,
    ResonanceLogSDRatio,
    ResonanceLogFlatteRatio
};

struct GVVFitParameterBinding {
    ctpwa::FitParameterSpec fit;
    GVVFitParameterTarget target = GVVFitParameterTarget::CouplingReal;
    int target_index = -1;
};

// This is the single translation layer between generic Minuit ordering and
// process-specific GVV model state. It is generated entirely from model.json.
std::vector<GVVFitParameterBinding> gvv_fit_parameter_layout(
    const GVVCompiledModel& model);

std::vector<ctpwa::FitParameterSpec> gvv_fit_parameter_specs(
    const std::vector<GVVFitParameterBinding>& layout);

void gvv_apply_fit_parameters(
    FitLikelihood& likelihood,
    const std::vector<GVVFitParameterBinding>& layout,
    const std::vector<double>& values);

void gvv_write_fit_details(
    std::ostream& output,
    const FitLikelihood& likelihood,
    const std::vector<GVVFitParameterBinding>& layout,
    const ctpwa::FitAttempt& best);

#endif // CTPWA_PROCESS_PARAMETER_MAPPING_H

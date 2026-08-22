// GVV Resonance compiler: validates process-owned JSON contracts and produces
// self-contained framework propagator descriptors plus generic fit bindings.
#ifndef CTPWA_PROCESS_PROPAGATOR_COMPILER_H
#define CTPWA_PROCESS_PROPAGATOR_COMPILER_H

#include "process/ProcessModel.h"

struct GVVCompiledResonance {
    ctpwa::PropagatorParameters propagator;
    std::vector<GVVPropagatorParameterMetadata> parameter_metadata;
    std::vector<GVVPropagatorFitBinding> fit_bindings;
};

GVVCompiledResonance gvv_compile_resonance(
    const ctpwa::ResonanceDefinition& input,
    int resonance_index);

double gvv_propagator_parameter_value(
    const ctpwa::PropagatorParameters& propagator,
    GVVPropagatorParameterTarget target);

void gvv_set_propagator_parameter(
    ctpwa::PropagatorParameters& propagator,
    GVVPropagatorParameterTarget target,
    double value);

#endif // CTPWA_PROCESS_PROPAGATOR_COMPILER_H

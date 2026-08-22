// Host/device model records for psi(2S) -> gamma omega omega. This file owns
// the compiled process representation; Wave registration and JSON compilation
// are separate boundaries.
#ifndef CTPWA_PROCESS_PROCESS_MODEL_H
#define CTPWA_PROCESS_PROCESS_MODEL_H

#include "framework/dynamics/PropagatorRegistry.cuh"
#include "framework/math/DeviceComplex.cuh"
#include "framework/model/Model.h"

#include <string>
#include <vector>

// Dense records used by CUDA kernels. Stable user-facing IDs live only in the
// host metadata below.
enum CouplingParameterization {
    COUPLING_COMPLEX = 0,
    COUPLING_FIXED_SCALE_AND_PHASE = 1,
    COUPLING_POSITIVE_REAL = 2
};

struct TermSpec {
    int resonance_index = 0;
    int wave_slot = 0;
    int registered_wave_type = 0;
};

enum class GVVPropagatorParameterTarget {
    Mass,
    PoleWidth,
    OrbitalL,
    SDRatio,
    FlatteRatio
};

enum class GVVFitTransform {
    Identity,
    LogPositive
};

struct GVVPropagatorParameterMetadata {
    std::string source_name;
    std::string display_name;
    std::string unit;
    GVVPropagatorParameterTarget target =
        GVVPropagatorParameterTarget::Mass;
};

// PropagatorCompiler creates these bindings once. ParameterMapping consumes
// them generically and therefore does not know concrete line-shape models.
struct GVVPropagatorFitBinding {
    std::string fit_name;
    double initial_coordinate = 0.0;
    double step = 0.1;
    bool has_lower_bound = false;
    bool has_upper_bound = false;
    double lower_bound = 0.0;
    double upper_bound = 0.0;
    int resonance_index = -1;
    GVVPropagatorParameterTarget target =
        GVVPropagatorParameterTarget::Mass;
    GVVFitTransform transform = GVVFitTransform::Identity;
    std::string source_name;
};

struct GVVResonanceMetadata {
    std::string id;
    std::string label;
    std::string propagator_id;
    std::vector<GVVPropagatorParameterMetadata> parameters;
};

struct GVVTermMetadata {
    std::string id;
    std::string label;
    std::string wave_id;
    std::string jpc;
    std::string wave_latex;
    std::string coherence_class;
    int registered_wave_type = -1;
    int coupling_parameterization = COUPLING_COMPLEX;
    ctpwa::CouplingReference reference = ctpwa::CouplingReference::None;
};

struct GVVCompiledModel {
    ctpwa::ModelDefinition definition;
    std::vector<ctpwa::PropagatorParameters> resonances;
    std::vector<TermSpec> terms;
    std::vector<DeviceComplex> initial_couplings;
    std::vector<int> active_wave_types;
    std::vector<GVVResonanceMetadata> resonance_metadata;
    std::vector<GVVTermMetadata> term_metadata;
    std::vector<GVVPropagatorFitBinding> propagator_fit_bindings;

    int find_resonance(const std::string& id) const;
    int find_term(const std::string& id) const;
};

#endif // CTPWA_PROCESS_PROCESS_MODEL_H

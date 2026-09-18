#pragma once

#include "core/Minuit.h"
#include "core/physics/AmplitudeTypes.h"
#include <string>
#include <unordered_map>
#include <vector>

#include "core/math/Complex.cuh"

// Channel model definitions, immutable compiled topology, and numeric fit state.

namespace ctpwa {

enum class CouplingMode {
    ComplexCartesian,
    FixedComplex,
    PositiveReal
};

enum class CouplingReference {
    None,
    Phase,
    ScaleAndPhase
};

struct ParameterDefinition {
    double value = 0.0;
    bool fixed = true;
    std::string transform = "identity";
    double step = 0.1;
    bool has_lower_bound = false;
    bool has_upper_bound = false;
    double lower_bound = 0.0;
    double upper_bound = 0.0;
};

struct ResonanceDefinition {
    std::string id;
    std::string label;
    std::string propagator;
    std::unordered_map<std::string, ParameterDefinition> parameters;
};

struct CouplingDefinition {
    CouplingMode mode = CouplingMode::ComplexCartesian;
    CouplingReference reference = CouplingReference::None;
    double initial_real = 0.1;
    double initial_imag = 0.0;
};

struct TermDefinition {
    std::string id;
    std::string label;
    std::string wave;
    bool active = true;
    CouplingDefinition coupling;

    // Parsed once; validation of active-only dependencies happens at compilation.
    struct Dynamics {
        std::string type;
        std::string resonance;
        bool has_string_fields = false;
        std::vector<std::string> unknown_fields;
    } dynamics;
};

struct ModelDefinition {
    int schema_version = 0;
    std::string process;
    std::string name;
    std::string description;
    // Shared process parameters reuse the ordinary parameter contract.
    // Their names and physical meaning are validated by the process compiler.
    std::unordered_map<std::string, ParameterDefinition> process_parameters;
    std::vector<ResonanceDefinition> resonances;
    std::vector<TermDefinition> terms;
    std::string canonical_json;

    const ResonanceDefinition& resonance(const std::string& id) const;
    const TermDefinition& term(const std::string& id) const;
};

ModelDefinition load_model_definition(const std::string& file_name);

// Exposed for unit tests and tools that already own a model document.  The
// same validation is used for file and in-memory input.
ModelDefinition parse_model_definition(
    const std::string& json_text,
    const std::string& source_name = "<memory>");

// Stable identifier used to ensure that a serialized fit state is applied to
// exactly the model document from which its parameter layout was generated.
std::string model_definition_signature(const ModelDefinition& definition);

const char* coupling_mode_name(CouplingMode mode);
const char* coupling_reference_name(CouplingReference reference);

} // namespace ctpwa

// Host/device model records for psi(2S) -> gamma omega omega. This file owns
// the compiled process representation; Wave registration and JSON compilation
// are separate boundaries.

// Dense records used by CUDA kernels. Stable user-facing IDs live only in the
// host metadata below.
enum CouplingParameterization {
    COUPLING_COMPLEX = 0,
    COUPLING_FIXED_SCALE_AND_PHASE = 1,
    COUPLING_POSITIVE_REAL = 2
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
    // Phase-reference block metadata; never used to mask Gram-matrix entries.
    std::string coherence_class;
    int registered_wave_type = -1;
    int coupling_parameterization = COUPLING_COMPLEX;
    ctpwa::CouplingReference reference = ctpwa::CouplingReference::None;
};

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

// The only state changed by Minuit or a Post covariance displacement.
// Copying this record never copies JSON, labels, Terms, or parameter bindings.
struct GVVParameterState {
    std::vector<ctpwa::PropagatorParameters> resonances;
    std::vector<DeviceComplex> couplings;
    double omega_resolution_sigma = 0.0;
};

struct GVVCompiledModel {
    ctpwa::ModelDefinition definition;
    GVVParameterState initial_parameters;
    std::vector<TermSpec> terms;
    std::vector<int> active_wave_types;
    std::vector<GVVResonanceMetadata> resonance_metadata;
    std::vector<GVVTermMetadata> term_metadata;
    std::vector<GVVFitParameterBinding> parameters;

    int find_resonance(const std::string& id) const;
    int find_term(const std::string& id) const;
};

// GVV Resonance compiler: validates process-owned JSON contracts and produces
// self-contained framework propagator descriptors plus generic fit bindings.

struct GVVCompiledResonance {
    ctpwa::PropagatorParameters propagator;
    std::vector<GVVPropagatorParameterMetadata> parameter_metadata;
    std::vector<GVVFitParameterBinding> fit_bindings;
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

// Complete GVV process compiler. It resolves active Term dependencies, calls
// the independent Wave and propagator registries, and builds dense GPU slots.

GVVCompiledModel gvv_compile_model(
    const ctpwa::ModelDefinition& definition);

GVVCompiledModel gvv_load_compiled_model(const std::string& file_name);

// Bump this contract identifier whenever a change alters the numerical meaning
// of a registered Wave, Resonance propagator, Term assembly, or fit binding.
const char* gvv_amplitude_implementation_signature();

// Preserve the legacy key for models without an explicit resolution setting.
std::string gvv_model_implementation_signature(
    const ctpwa::ModelDefinition& definition);

// Combined compatibility key for one declarative model and this GVV amplitude
// implementation. Fit and Post Calculation must agree on this value.
std::string gvv_model_signature(
    const ctpwa::ModelDefinition& definition);

// The compiled model owns the final layout; evaluation updates only numeric state.
const std::vector<GVVFitParameterBinding>& gvv_fit_parameter_layout(
    const GVVCompiledModel& model);

std::vector<ctpwa::FitParameterSpec> gvv_fit_parameter_specs(
    const std::vector<GVVFitParameterBinding>& layout);

void gvv_apply_fit_parameters(
    GVVParameterState& state,
    const std::vector<GVVFitParameterBinding>& layout,
    const std::vector<double>& values);

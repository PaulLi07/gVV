#ifndef CTPWA_FRAMEWORK_MODEL_DEFINITION_H
#define CTPWA_FRAMEWORK_MODEL_DEFINITION_H

#include <string>
#include <unordered_map>
#include <vector>

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

    // The common framework deliberately treats process dynamics as opaque.
    // The active process parses this canonical JSON object when compiling its
    // device term representation.
    std::string dynamics_json;
};

struct ModelDefinition {
    int schema_version = 0;
    std::string process;
    std::string name;
    std::string description;
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

const char* coupling_mode_name(CouplingMode mode);
const char* coupling_reference_name(CouplingReference reference);

} // namespace ctpwa

// Compact runtime objects shared by the compiled model and CUDA kernels.
// Stable user-facing ids remain in ModelDefinition metadata.
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

#endif // CTPWA_FRAMEWORK_MODEL_DEFINITION_H

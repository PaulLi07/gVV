#ifndef CTPWA_PROCESS_GVV_PROCESS_MODEL_H
#define CTPWA_PROCESS_GVV_PROCESS_MODEL_H

#include "../DeviceComplex.h"
#include "../GVVModel.h"
#include "../framework/ModelDefinition.h"

#include <string>
#include <vector>

struct GVVWaveMetadata {
    std::string id;
    std::string jpc;
    std::string latex;
    std::string coherence_class;
    int wave_type = -1;
};

struct GVVResonanceMetadata {
    std::string id;
    std::string label;
    std::string propagator_id;
};

struct GVVTermMetadata {
    std::string id;
    std::string label;
    std::string wave_id;
    std::string jpc;
    std::string latex;
    std::string coherence_class;
    int registered_wave_type = -1;
    int coupling_parameterization = GVV_COUPLING_COMPLEX;
    ctpwa::CouplingReference reference = ctpwa::CouplingReference::None;
};

// Host-side process model compiled from the generic JSON definition.  Numeric
// indices are dense, runtime-only device layout; stable ids live in metadata.
struct GVVCompiledModel {
    ctpwa::ModelDefinition definition;
    std::vector<GVVResonanceParameters> resonances;
    std::vector<GVVTermSpec> terms;
    std::vector<DeviceComplex> initial_couplings;
    std::vector<int> active_wave_types;
    std::vector<GVVResonanceMetadata> resonance_metadata;
    std::vector<GVVTermMetadata> term_metadata;

    int find_resonance(const std::string& id) const;
    int find_term(const std::string& id) const;
};

const std::vector<GVVWaveMetadata>& gvv_wave_registry();
const GVVWaveMetadata& gvv_registered_wave(const std::string& id);

GVVCompiledModel gvv_compile_model(
    const ctpwa::ModelDefinition& definition);

GVVCompiledModel gvv_load_compiled_model(const std::string& file_name);

#endif // CTPWA_PROCESS_GVV_PROCESS_MODEL_H

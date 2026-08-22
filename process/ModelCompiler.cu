// Assemble the active GVV model from independently registered Waves and
// independently compiled Resonances.
#include "process/ModelCompiler.h"

#include "process/PropagatorCompiler.h"
#include "process/WaveRegistry.cuh"

#include <nlohmann/json.hpp>

#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace {

using Json = nlohmann::json;

int coupling_parameterization(ctpwa::CouplingMode mode)
{
    if (mode == ctpwa::CouplingMode::FixedComplex) {
        return COUPLING_FIXED_SCALE_AND_PHASE;
    }
    if (mode == ctpwa::CouplingMode::PositiveReal) {
        return COUPLING_POSITIVE_REAL;
    }
    return COUPLING_COMPLEX;
}

std::string compile_term_dynamics(const ctpwa::TermDefinition& term)
{
    const Json dynamics = Json::parse(term.dynamics_json);
    for (const auto& item : dynamics.items()) {
        if (item.key() != "type" && item.key() != "resonance") {
            throw std::runtime_error(
                "term '" + term.id
                + "' has unknown dynamics field '" + item.key() + "'");
        }
    }
    if (!dynamics.contains("type") || !dynamics["type"].is_string()
        || !dynamics.contains("resonance")
        || !dynamics["resonance"].is_string()) {
        throw std::runtime_error(
            "term '" + term.id
            + "' dynamics requires string fields type and resonance");
    }
    const std::string type = dynamics["type"].get<std::string>();
    if (type != "gvv_x_to_omega_omega") {
        throw std::runtime_error(
            "term '" + term.id + "' has unsupported dynamics type '"
            + type + "'");
    }
    return dynamics["resonance"].get<std::string>();
}

} // namespace

int GVVCompiledModel::find_resonance(const std::string& id) const
{
    for (std::size_t index = 0; index < resonance_metadata.size(); ++index) {
        if (resonance_metadata[index].id == id) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

int GVVCompiledModel::find_term(const std::string& id) const
{
    for (std::size_t index = 0; index < term_metadata.size(); ++index) {
        if (term_metadata[index].id == id) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

GVVCompiledModel gvv_compile_model(
    const ctpwa::ModelDefinition& definition)
{
    if (definition.process != "psi2s_to_gamma_omega_omega") {
        throw std::runtime_error(
            "GVV process compiler cannot load process '"
            + definition.process + "'");
    }

    GVVCompiledModel result;
    result.definition = definition;

    // Only active Terms define runtime Resonance dependencies. An invalid or
    // unsupported Resonance used exclusively by inactive Terms is irrelevant
    // to both the GPU model and the Minuit layout.
    std::unordered_map<std::string, std::string> active_term_resonances;
    std::unordered_set<std::string> active_resonance_ids;
    for (const ctpwa::TermDefinition& term : definition.terms) {
        if (!term.active) {
            continue;
        }
        const std::string resonance_id = compile_term_dynamics(term);
        active_term_resonances.emplace(term.id, resonance_id);
        active_resonance_ids.insert(resonance_id);
    }

    for (const ctpwa::ResonanceDefinition& resonance : definition.resonances) {
        if (active_resonance_ids.find(resonance.id)
            == active_resonance_ids.end()) {
            continue;
        }
        const int resonance_index =
            static_cast<int>(result.resonances.size());
        GVVCompiledResonance compiled =
            gvv_compile_resonance(resonance, resonance_index);
        result.resonances.push_back(compiled.propagator);
        result.resonance_metadata.push_back({
            resonance.id,
            resonance.label,
            resonance.propagator,
            std::move(compiled.parameter_metadata)});
        result.propagator_fit_bindings.insert(
            result.propagator_fit_bindings.end(),
            compiled.fit_bindings.begin(),
            compiled.fit_bindings.end());
    }

    std::unordered_map<int, int> active_wave_slots;
    std::map<std::string, int> reference_counts;
    for (const ctpwa::TermDefinition& term : definition.terms) {
        if (!term.active) {
            continue;
        }
        const GVVWaveMetadata& wave = gvv_registered_wave(term.wave);
        if (active_wave_slots.find(wave.wave_type) == active_wave_slots.end()) {
            const int slot = static_cast<int>(result.active_wave_types.size());
            active_wave_slots.emplace(wave.wave_type, slot);
            result.active_wave_types.push_back(wave.wave_type);
        }

        const std::string resonance_id =
            active_term_resonances.at(term.id);
        const int resonance_index = result.find_resonance(resonance_id);
        if (resonance_index < 0) {
            throw std::runtime_error(
                "term '" + term.id + "' references unknown resonance '"
                + resonance_id + "'");
        }

        result.terms.push_back(TermSpec{
            resonance_index,
            active_wave_slots.at(wave.wave_type),
            wave.wave_type});
        result.initial_couplings.emplace_back(
            term.coupling.initial_real, term.coupling.initial_imag);
        result.term_metadata.push_back({
            term.id,
            term.label,
            wave.id,
            wave.jpc,
            wave.latex,
            wave.coherence_class,
            wave.wave_type,
            coupling_parameterization(term.coupling.mode),
            term.coupling.reference});

        if (term.coupling.reference != ctpwa::CouplingReference::None) {
            ++reference_counts[wave.coherence_class];
        }
    }

    if (result.terms.empty()) {
        throw std::runtime_error("GVV model has no active terms");
    }
    for (const GVVTermMetadata& term : result.term_metadata) {
        if (reference_counts[term.coherence_class] != 1) {
            throw std::runtime_error(
                "coherence class '" + term.coherence_class
                + "' requires exactly one phase reference");
        }
    }
    return result;
}

GVVCompiledModel gvv_load_compiled_model(const std::string& file_name)
{
    return gvv_compile_model(ctpwa::load_model_definition(file_name));
}

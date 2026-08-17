#include "process/WaveRegistry.cuh"

#include <nlohmann/json.hpp>

#include <cmath>
#include <map>
#include <stdexcept>
#include <unordered_map>

namespace {

using Json = nlohmann::json;

const ctpwa::ParameterDefinition& require_parameter(
    const ctpwa::ResonanceDefinition& resonance,
    const std::string& name)
{
    const auto found = resonance.parameters.find(name);
    if (found == resonance.parameters.end()) {
        throw std::runtime_error(
            "resonance '" + resonance.id + "' with propagator '"
            + resonance.propagator + "' requires parameter '" + name + "'");
    }
    return found->second;
}

void require_fixed_integer(
    const ctpwa::ResonanceDefinition& resonance,
    const std::string& name,
    int expected_minimum,
    int expected_maximum,
    int& output)
{
    const ctpwa::ParameterDefinition& parameter =
        require_parameter(resonance, name);
    const double rounded = std::round(parameter.value);
    if (!parameter.fixed || parameter.transform != "identity"
        || std::fabs(parameter.value - rounded) > 1.0e-12
        || rounded < expected_minimum || rounded > expected_maximum) {
        throw std::runtime_error(
            "resonance '" + resonance.id + "' parameter '" + name
            + "' must be a fixed integer in the supported range");
    }
    output = static_cast<int>(rounded);
}

ResonanceParameters compile_resonance(
    const ctpwa::ResonanceDefinition& input)
{
    if (input.propagator == "nonresonant") {
        if (!input.parameters.empty()) {
            throw std::runtime_error(
                "nonresonant component '" + input.id
                + "' must not define propagator parameters");
        }
        return ResonanceParameters(
            PROP_NONRESONANT, 0.0, 0.0, 0.0, 0);
    }

    const ctpwa::ParameterDefinition& mass =
        require_parameter(input, "mass");
    const ctpwa::ParameterDefinition& width =
        require_parameter(input, "width");
    if (!mass.fixed || !width.fixed
        || mass.transform != "identity" || width.transform != "identity") {
        throw std::runtime_error(
            "GVV v1 currently requires fixed identity mass/width for '"
            + input.id + "'");
    }
    if (!(mass.value > 0.0) || !(width.value >= 0.0)) {
        throw std::runtime_error(
            "resonance '" + input.id + "' has invalid mass/width");
    }

    if (input.propagator == "fixed_width_bw") {
        return ResonanceParameters(
            PROP_FIXED_BW, mass.value, width.value, 0.0, 0);
    }
    if (input.propagator == "two_body_running_bw") {
        int orbital_l = 0;
        require_fixed_integer(input, "orbital_l", 0, 1, orbital_l);
        const int model = orbital_l == 0
                              ? PROP_SCALAR_SWAVE_BWR
                              : PROP_PWAVE_BWR;
        return ResonanceParameters(
            model, mass.value, width.value, 0.0, 0);
    }
    if (input.propagator == "scalar_sd_running_bw") {
        const ctpwa::ParameterDefinition& ratio =
            require_parameter(input, "sd_ratio");
        if (ratio.transform != "log" || !(ratio.value > 0.0)) {
            throw std::runtime_error(
                "resonance '" + input.id
                + "' requires positive log-transformed sd_ratio");
        }
        return ResonanceParameters(
            PROP_SCALAR_SD_BWR,
            mass.value,
            width.value,
            ratio.value,
            ratio.fixed ? 0 : 1);
    }
    if (input.propagator == "subtracted_effective_flatte") {
        const ctpwa::ParameterDefinition& ratio =
            require_parameter(input, "omegaomega_ratio");
        if (ratio.transform != "log" || !(ratio.value > 0.0)) {
            throw std::runtime_error(
                "resonance '" + input.id
                + "' requires positive log-transformed omegaomega_ratio");
        }
        return ResonanceParameters(
            PROP_SUBTRACTED_FLATTE,
            mass.value,
            width.value,
            0.0,
            0,
            ratio.value,
            ratio.fixed ? 0 : 1);
    }
    throw std::runtime_error(
        "unsupported GVV propagator '" + input.propagator
        + "' for resonance '" + input.id + "'");
}

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

const std::vector<GVVWaveMetadata>& gvv_wave_registry()
{
    // This is the only host registration point for complete GVV waves. The
    // matching device dispatch is the single gvv_wave_tensor function in
    // WaveRegistry.cuh; individual formulae stay in process/waves/.
    static const std::vector<GVVWaveMetadata> registry = {
        {"gvv.scalar_00", "0++", "0^{++}(00)", "scalar", GVV_SCALAR_00},
        {"gvv.scalar_22", "0++", "0^{++}(22)", "scalar", GVV_SCALAR_22},
        {"gvv.pseudoscalar_11", "0-+", "0^{-+}(11)",
         "pseudoscalar", GVV_PSEUDOSCALAR_11}
    };
    return registry;
}

const GVVWaveMetadata& gvv_registered_wave(const std::string& id)
{
    for (const GVVWaveMetadata& wave : gvv_wave_registry()) {
        if (wave.id == id) {
            return wave;
        }
    }
    throw std::runtime_error(
        "model references unregistered GVV wave '" + id + "'");
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

    for (const ctpwa::ResonanceDefinition& resonance : definition.resonances) {
        result.resonances.push_back(compile_resonance(resonance));
        result.resonance_metadata.push_back(
            {resonance.id, resonance.label, resonance.propagator});
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

        const Json dynamics = Json::parse(term.dynamics_json);
        const std::string type = dynamics.value("type", "");
        if (type != "gvv_x_to_omega_omega") {
            throw std::runtime_error(
                "term '" + term.id + "' has unsupported dynamics type '"
                + type + "'");
        }
        const std::string resonance_id = dynamics.value("resonance", "");
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
            term.label,
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

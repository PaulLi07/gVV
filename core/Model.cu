#include "core/physics/Propagators.cuh"
#include "core/Model.h"
#include "core/physics/OmegaDecay.cuh"
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <unordered_set>
#include "core/physics/Waves.cuh"
#include "core/physics/OmegaResolution.cuh"
#include <algorithm>
#include <ostream>
#include <map>
#include <unordered_map>
#include <utility>

// Compile one process-level Resonance into a self-contained numerical
// descriptor. This is the only place that maps model.json propagator strings
// and named parameters to framework storage fields.

namespace {

void require_exact_parameters(
    const ctpwa::ResonanceDefinition& resonance,
    std::initializer_list<const char*> allowed)
{
    std::unordered_set<std::string> names;
    for (const char* name : allowed) {
        names.insert(name);
    }
    for (const auto& parameter : resonance.parameters) {
        if (names.find(parameter.first) == names.end()) {
            throw std::runtime_error(
                "resonance '" + resonance.id + "' propagator '"
                + resonance.propagator + "' does not accept parameter '"
                + parameter.first + "'");
        }
    }
}

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

void require_identity_parameter_policy(
    const ctpwa::ResonanceDefinition& resonance,
    const std::string& name,
    const ctpwa::ParameterDefinition& parameter)
{
    if (parameter.transform != "identity") {
        throw std::runtime_error(
            "resonance '" + resonance.id + "' parameter '" + name
            + "' must use the identity transform");
    }
    if (!parameter.fixed
        && (!parameter.has_lower_bound || !parameter.has_upper_bound
            || !std::isfinite(parameter.lower_bound)
            || !std::isfinite(parameter.upper_bound)
            || !(parameter.lower_bound > 0.0)
            || !(parameter.lower_bound < parameter.upper_bound)
            || parameter.value < parameter.lower_bound
            || parameter.value > parameter.upper_bound)) {
        throw std::runtime_error(
            "free parameter '" + name + "' for resonance '" + resonance.id
            + "' requires finite bounds with a positive lower limit "
              "containing its initial value");
    }
}

int require_fixed_integer(
    const ctpwa::ResonanceDefinition& resonance,
    const std::string& name,
    int minimum,
    int maximum)
{
    const ctpwa::ParameterDefinition& parameter =
        require_parameter(resonance, name);
    const double rounded = std::round(parameter.value);
    if (!parameter.fixed || parameter.transform != "identity"
        || std::fabs(parameter.value - rounded) > 1.0e-12
        || rounded < minimum || rounded > maximum) {
        throw std::runtime_error(
            "resonance '" + resonance.id + "' parameter '" + name
            + "' must be a fixed integer in the supported range");
    }
    return static_cast<int>(rounded);
}

GVVPropagatorParameterMetadata parameter_metadata(
    const char* source_name,
    const char* display_name,
    const char* unit,
    GVVPropagatorParameterTarget target)
{
    GVVPropagatorParameterMetadata result;
    result.source_name = source_name;
    result.display_name = display_name;
    result.unit = unit;
    result.target = target;
    return result;
}

GVVFitParameterBinding identity_binding(
    const ctpwa::ResonanceDefinition& resonance,
    const ctpwa::ParameterDefinition& parameter,
    const std::string& source_name,
    const std::string& fit_name,
    int resonance_index,
    GVVPropagatorParameterTarget target)
{
    GVVFitParameterBinding result;
    result.fit.name = fit_name + resonance.id;
    result.fit.initial_value = parameter.value;
    result.fit.step = parameter.step;
    result.fit.has_lower_bound = parameter.has_lower_bound;
    result.fit.has_upper_bound = parameter.has_upper_bound;
    result.fit.lower_bound = parameter.lower_bound;
    result.fit.upper_bound = parameter.upper_bound;
    result.target_index = resonance_index;
    result.target = GVVFitParameterTarget::PropagatorParameter;
    result.propagator_target = target;
    result.transform = GVVFitTransform::Identity;
    return result;
}

GVVFitParameterBinding log_binding(
    const ctpwa::ResonanceDefinition& resonance,
    const ctpwa::ParameterDefinition& parameter,
    const std::string& source_name,
    const std::string& fit_name,
    int resonance_index,
    GVVPropagatorParameterTarget target)
{
    GVVFitParameterBinding result;
    result.fit.name = fit_name + resonance.id;
    result.fit.initial_value = std::log(parameter.value);
    result.fit.step = parameter.step;
    result.fit.has_lower_bound = parameter.has_lower_bound;
    result.fit.has_upper_bound = parameter.has_upper_bound;
    result.fit.lower_bound = parameter.lower_bound;
    result.fit.upper_bound = parameter.upper_bound;
    result.target_index = resonance_index;
    result.target = GVVFitParameterTarget::PropagatorParameter;
    result.propagator_target = target;
    result.transform = GVVFitTransform::LogPositive;
    return result;
}

void require_positive_log_parameter(
    const ctpwa::ResonanceDefinition& resonance,
    const std::string& name,
    const ctpwa::ParameterDefinition& parameter)
{
    if (parameter.transform != "log" || !(parameter.value > 0.0)) {
        throw std::runtime_error(
            "resonance '" + resonance.id
            + "' requires positive log-transformed " + name);
    }
}

void require_open_omega_omega_pole(
    const ctpwa::ResonanceDefinition& resonance,
    const ctpwa::ParameterDefinition& mass)
{
    const double threshold = 2.0 * GVV_OMEGA_MASS;
    if (!(mass.value > threshold)) {
        throw std::runtime_error(
            "resonance '" + resonance.id
            + "' running-width pole must be above the nominal omega-omega "
              "threshold");
    }
    if (!mass.fixed && !(mass.lower_bound > threshold)) {
        throw std::runtime_error(
            "resonance '" + resonance.id
            + "' requires its entire free mass range to remain above the "
              "nominal omega-omega threshold");
    }
}

ctpwa::PropagatorParameters omega_omega_descriptor(
    int model,
    double mass,
    double width,
    int orbital_l = 0,
    double sd_ratio = 0.0,
    double flatte_ratio = 0.0)
{
    return ctpwa::PropagatorParameters(
        model,
        mass,
        width,
        orbital_l,
        sd_ratio,
        flatte_ratio,
        GVV_OMEGA_MASS,
        GVV_OMEGA_MASS,
        ctpwa::DEFAULT_BARRIER_RADIUS_FM);
}

} // namespace

GVVCompiledResonance gvv_compile_resonance(
    const ctpwa::ResonanceDefinition& input,
    int resonance_index)
{
    GVVCompiledResonance result;
    if (input.propagator == "nonresonant") {
        require_exact_parameters(input, {});
        result.propagator = ctpwa::PropagatorParameters();
        return result;
    }

    if (input.propagator == "fixed_width_bw") {
        require_exact_parameters(input, {"mass", "width"});
    } else if (input.propagator == "two_body_running_bw") {
        require_exact_parameters(input, {"mass", "width", "orbital_l"});
    } else if (input.propagator == "scalar_sd_running_bw") {
        require_exact_parameters(input, {"mass", "width", "sd_ratio"});
    } else if (input.propagator == "subtracted_effective_flatte") {
        require_exact_parameters(
            input, {"mass", "width", "omegaomega_ratio"});
    } else {
        throw std::runtime_error(
            "unsupported GVV propagator '" + input.propagator
            + "' for resonance '" + input.id + "'");
    }

    const ctpwa::ParameterDefinition& mass =
        require_parameter(input, "mass");
    const ctpwa::ParameterDefinition& width =
        require_parameter(input, "width");
    require_identity_parameter_policy(input, "mass", mass);
    require_identity_parameter_policy(input, "width", width);
    if (!(mass.value > 0.0) || !(width.value > 0.0)) {
        throw std::runtime_error(
            "resonance '" + input.id
            + "' requires positive mass and pole width");
    }
    if (input.propagator == "two_body_running_bw"
        || input.propagator == "scalar_sd_running_bw") {
        require_open_omega_omega_pole(input, mass);
    }

    result.parameter_metadata.push_back(parameter_metadata(
        "mass", "mass", "GeV", GVVPropagatorParameterTarget::Mass));
    result.parameter_metadata.push_back(parameter_metadata(
        "width",
        input.propagator == "subtracted_effective_flatte"
            ? "rest width" : "pole width",
        "GeV",
        GVVPropagatorParameterTarget::PoleWidth));

    if (!mass.fixed) {
        result.fit_bindings.push_back(identity_binding(
            input,
            mass,
            "mass",
            "mass_",
            resonance_index,
            GVVPropagatorParameterTarget::Mass));
    }
    if (!width.fixed) {
        result.fit_bindings.push_back(identity_binding(
            input,
            width,
            "width",
            "width_",
            resonance_index,
            GVVPropagatorParameterTarget::PoleWidth));
    }

    if (input.propagator == "fixed_width_bw") {
        result.propagator = omega_omega_descriptor(
            ctpwa::PROP_FIXED_BW, mass.value, width.value);
        return result;
    }

    if (input.propagator == "two_body_running_bw") {
        const int orbital_l = require_fixed_integer(
            input, "orbital_l", 0, 2);
        result.propagator = omega_omega_descriptor(
            ctpwa::PROP_TWO_BODY_RUNNING_BW,
            mass.value,
            width.value,
            orbital_l);
        result.parameter_metadata.push_back(parameter_metadata(
            "orbital_l",
            "width orbital L",
            "",
            GVVPropagatorParameterTarget::OrbitalL));
        return result;
    }

    if (input.propagator == "scalar_sd_running_bw") {
        const ctpwa::ParameterDefinition& ratio =
            require_parameter(input, "sd_ratio");
        require_positive_log_parameter(input, "sd_ratio", ratio);
        result.propagator = omega_omega_descriptor(
            ctpwa::PROP_SCALAR_SD_BWR,
            mass.value,
            width.value,
            0,
            ratio.value);
        result.parameter_metadata.push_back(parameter_metadata(
            "sd_ratio",
            "D/S pole-width ratio",
            "",
            GVVPropagatorParameterTarget::SDRatio));
        if (!ratio.fixed) {
            result.fit_bindings.push_back(log_binding(
                input,
                ratio,
                "sd_ratio",
                "log_rDS_",
                resonance_index,
                GVVPropagatorParameterTarget::SDRatio));
        }
        return result;
    }

    const ctpwa::ParameterDefinition& ratio =
        require_parameter(input, "omegaomega_ratio");
    require_positive_log_parameter(input, "omegaomega_ratio", ratio);
    result.propagator = omega_omega_descriptor(
        ctpwa::PROP_SUBTRACTED_FLATTE,
        mass.value,
        width.value,
        0,
        0.0,
        ratio.value);
    result.parameter_metadata.push_back(parameter_metadata(
        "omegaomega_ratio",
        "omega-omega effective coupling ratio",
        "",
        GVVPropagatorParameterTarget::FlatteRatio));
    if (!ratio.fixed) {
        result.fit_bindings.push_back(log_binding(
            input,
            ratio,
            "omegaomega_ratio",
            "log_Romega_",
            resonance_index,
            GVVPropagatorParameterTarget::FlatteRatio));
    }
    return result;
}

double gvv_propagator_parameter_value(
    const ctpwa::PropagatorParameters& propagator,
    GVVPropagatorParameterTarget target)
{
    if (target == GVVPropagatorParameterTarget::Mass) {
        return propagator.mass;
    }
    if (target == GVVPropagatorParameterTarget::PoleWidth) {
        return propagator.pole_width;
    }
    if (target == GVVPropagatorParameterTarget::OrbitalL) {
        return propagator.orbital_l;
    }
    if (target == GVVPropagatorParameterTarget::SDRatio) {
        return propagator.sd_ratio;
    }
    return propagator.flatte_ratio;
}

void gvv_set_propagator_parameter(
    ctpwa::PropagatorParameters& propagator,
    GVVPropagatorParameterTarget target,
    double value)
{
    if (target == GVVPropagatorParameterTarget::Mass) {
        propagator.mass = value;
    } else if (target == GVVPropagatorParameterTarget::PoleWidth) {
        propagator.pole_width = value;
    } else if (target == GVVPropagatorParameterTarget::OrbitalL) {
        propagator.orbital_l = static_cast<int>(std::lround(value));
    } else if (target == GVVPropagatorParameterTarget::SDRatio) {
        propagator.sd_ratio = value;
    } else {
        propagator.flatte_ratio = value;
    }
}

// Complete translation layer between compiled gVV model state and the flat
// Minuit vector: layout construction, state application, and TXT details.

namespace {

ctpwa::FitParameterSpec make_fit_spec(
    const std::string& name,
    double initial,
    double step)
{
    ctpwa::FitParameterSpec result;
    result.name = name;
    result.initial_value = initial;
    result.step = step;
    return result;
}

double physical_value(
    double coordinate,
    GVVFitTransform transform,
    const std::string& name)
{
    if (transform == GVVFitTransform::Identity) {
        return coordinate;
    }
    const double physical = std::exp(coordinate);
    if (!(physical > 0.0) || !std::isfinite(physical)) {
        throw std::invalid_argument(
            name + " is outside the numerical range");
    }
    return physical;
}

} // namespace

static std::vector<GVVFitParameterBinding> build_parameter_layout(
    const GVVCompiledModel& model,
    const std::vector<GVVFitParameterBinding>& resonance_bindings)
{
    std::vector<GVVFitParameterBinding> layout;
    for (std::size_t term = 0; term < model.terms.size(); ++term) {
        const GVVTermMetadata& metadata = model.term_metadata[term];
        const DeviceComplex coupling = model.initial_parameters.couplings[term];
        if (metadata.coupling_parameterization
            == COUPLING_FIXED_SCALE_AND_PHASE) {
            continue;
        }
        if (metadata.coupling_parameterization
            == COUPLING_POSITIVE_REAL) {
            GVVFitParameterBinding parameter;
            parameter.fit = make_fit_spec(
                "log_rho_" + metadata.id, std::log(coupling.real), 0.10);
            parameter.fit.randomization =
                ctpwa::ParameterRandomization::LogMagnitude;
            parameter.fit.randomization_group = static_cast<int>(term);
            parameter.target = GVVFitParameterTarget::CouplingLogMagnitude;
            parameter.target_index = static_cast<int>(term);
            layout.push_back(parameter);
            continue;
        }

        GVVFitParameterBinding real;
        real.fit = make_fit_spec(
            "Re_" + metadata.id, coupling.real, 0.05);
        real.fit.randomization =
            ctpwa::ParameterRandomization::ComplexReal;
        real.fit.randomization_group = static_cast<int>(term);
        real.target = GVVFitParameterTarget::CouplingReal;
        real.target_index = static_cast<int>(term);
        layout.push_back(real);

        GVVFitParameterBinding imaginary = real;
        imaginary.fit.name = "Im_" + metadata.id;
        imaginary.fit.initial_value = coupling.imag;
        imaginary.fit.randomization =
            ctpwa::ParameterRandomization::ComplexImaginary;
        imaginary.target = GVVFitParameterTarget::CouplingImaginary;
        layout.push_back(imaginary);
    }

    layout.insert(layout.end(), resonance_bindings.begin(), resonance_bindings.end());
    const auto omega_sigma = model.definition.process_parameters.find("omega_resolution_sigma");
    if (omega_sigma != model.definition.process_parameters.end() && !omega_sigma->second.fixed) {
        const auto& source = omega_sigma->second;
        GVVFitParameterBinding parameter;
        parameter.fit = make_fit_spec("log_sigma_omega", std::log(source.value), source.step);
        parameter.fit.has_lower_bound = source.has_lower_bound;
        parameter.fit.has_upper_bound = source.has_upper_bound;
        parameter.fit.lower_bound = source.lower_bound;
        parameter.fit.upper_bound = source.upper_bound;
        parameter.target = GVVFitParameterTarget::OmegaResolutionSigma;
        parameter.transform = GVVFitTransform::LogPositive;
        layout.push_back(parameter);
    }
    return layout;
}

std::vector<ctpwa::FitParameterSpec> gvv_fit_parameter_specs(
    const std::vector<GVVFitParameterBinding>& layout)
{
    std::vector<ctpwa::FitParameterSpec> result;
    result.reserve(layout.size());
    for (const GVVFitParameterBinding& parameter : layout) {
        result.push_back(parameter.fit);
    }
    return result;
}

void gvv_apply_fit_parameters(
    GVVParameterState& state,
    const std::vector<GVVFitParameterBinding>& layout,
    const std::vector<double>& values)
{
    if (layout.size() != values.size()) {
        throw std::invalid_argument("incorrect number of GVV fit parameters");
    }
    for (std::size_t cursor = 0; cursor < layout.size(); ++cursor) {
        const GVVFitParameterBinding& parameter = layout[cursor];
        if (parameter.target == GVVFitParameterTarget::CouplingReal) {
            state.couplings[parameter.target_index].real =
                values[cursor];
        } else if (
            parameter.target == GVVFitParameterTarget::CouplingImaginary) {
            state.couplings[parameter.target_index].imag =
                values[cursor];
        } else if (
            parameter.target == GVVFitParameterTarget::CouplingLogMagnitude) {
            state.couplings[parameter.target_index] = DeviceComplex(
                physical_value(
                    values[cursor],
                    GVVFitTransform::LogPositive,
                    "log coupling magnitude"),
                0.0);
        } else if (parameter.target == GVVFitParameterTarget::OmegaResolutionSigma) {
            const double sigma = physical_value(
                values[cursor], GVVFitTransform::LogPositive, parameter.fit.name);
            if (sigma > GVV_OMEGA_RESOLUTION_MAX_SIGMA * (1.0 + 1.e-12)) {
                throw std::invalid_argument("omega resolution sigma exceeds 0.05 GeV");
            }
            state.omega_resolution_sigma = sigma;
        } else {
            gvv_set_propagator_parameter(
                state.resonances[parameter.target_index],
                parameter.propagator_target,
                physical_value(
                    values[cursor],
                    parameter.transform,
                    parameter.fit.name));
        }
    }
}

const std::vector<GVVFitParameterBinding>& gvv_fit_parameter_layout(const GVVCompiledModel& model)
{
    return model.parameters;
}

// Assemble the active GVV model from independently registered Waves and
// independently compiled Resonances.

namespace {

constexpr const char* kGVVAmplitudeImplementationSignature =
    "gvv-amplitude-contract-v1";

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
    const auto& dynamics = term.dynamics;
    if (!dynamics.unknown_fields.empty()) {
        throw std::runtime_error("term '" + term.id
            + "' has unknown dynamics field '" + dynamics.unknown_fields.front() + "'");
    }
    if (!dynamics.has_string_fields) {
        throw std::runtime_error("term '" + term.id
            + "' dynamics requires string fields type and resonance");
    }
    if (dynamics.type != "gvv_x_to_omega_omega") {
        throw std::runtime_error("term '" + term.id
            + "' has unsupported dynamics type '" + dynamics.type + "'");
    }
    return dynamics.resonance;
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
    std::vector<GVVFitParameterBinding> resonance_bindings;
    for (const auto& entry : definition.process_parameters) {
        if (entry.first != "omega_resolution_sigma") {
            throw std::runtime_error("unknown GVV process parameter '" + entry.first + "'");
        }
        const auto& sigma = entry.second;
        if (!std::isfinite(sigma.value) || sigma.value < 0.0
            || (!sigma.fixed && sigma.transform != "log")) {
            throw std::runtime_error(
                "omega_resolution_sigma must be nonnegative; a free sigma requires transform=log");
        }
        // Explicit finite log bounds keep every Minuit trial within the
        // supported convolution range; fixed zero is the exact legacy mode.
        if (!sigma.fixed && (!sigma.has_lower_bound || !sigma.has_upper_bound
            || !(std::exp(sigma.lower_bound) > 0.0)
            || !std::isfinite(std::exp(sigma.upper_bound)))) {
            throw std::runtime_error("free omega_resolution_sigma requires finite physical log bounds");
        }
        const double maximum_sigma = sigma.fixed ? sigma.value : std::exp(sigma.upper_bound);
        if (maximum_sigma > GVV_OMEGA_RESOLUTION_MAX_SIGMA * (1.0 + 1.e-12)) {
            throw std::runtime_error("omega_resolution_sigma exceeds the supported 0.05 GeV range");
        }
        result.initial_parameters.omega_resolution_sigma = sigma.value;
    }

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
            static_cast<int>(result.initial_parameters.resonances.size());
        GVVCompiledResonance compiled =
            gvv_compile_resonance(resonance, resonance_index);
        result.initial_parameters.resonances.push_back(compiled.propagator);
        result.resonance_metadata.push_back({
            resonance.id,
            resonance.label,
            resonance.propagator,
            std::move(compiled.parameter_metadata)});
        resonance_bindings.insert(
            resonance_bindings.end(),
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
        result.initial_parameters.couplings.emplace_back(
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
    result.parameters = build_parameter_layout(result, resonance_bindings);
    return result;
}

GVVCompiledModel gvv_load_compiled_model(const std::string& file_name)
{
    return gvv_compile_model(ctpwa::load_model_definition(file_name));
}

const char* gvv_amplitude_implementation_signature()
{
    return kGVVAmplitudeImplementationSignature;
}

std::string gvv_model_implementation_signature(
    const ctpwa::ModelDefinition& definition)
{
    return definition.process_parameters.count("omega_resolution_sigma")
        ? GVV_OMEGA_RESOLUTION_IMPLEMENTATION
        : gvv_amplitude_implementation_signature();
}

std::string gvv_model_signature(
    const ctpwa::ModelDefinition& definition)
{
    return gvv_model_implementation_signature(definition) + ':'
           + ctpwa::model_definition_signature(definition);
}

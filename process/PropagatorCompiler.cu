// Compile one process-level Resonance into a self-contained numerical
// descriptor. This is the only place that maps model.json propagator strings
// and named parameters to framework storage fields.
#include "process/PropagatorCompiler.h"

#include "process/OmegaDecayModel.cuh"

#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <unordered_set>

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

GVVPropagatorFitBinding identity_binding(
    const ctpwa::ResonanceDefinition& resonance,
    const ctpwa::ParameterDefinition& parameter,
    const std::string& source_name,
    const std::string& fit_name,
    int resonance_index,
    GVVPropagatorParameterTarget target)
{
    GVVPropagatorFitBinding result;
    result.fit_name = fit_name + resonance.id;
    result.initial_coordinate = parameter.value;
    result.step = parameter.step;
    result.has_lower_bound = parameter.has_lower_bound;
    result.has_upper_bound = parameter.has_upper_bound;
    result.lower_bound = parameter.lower_bound;
    result.upper_bound = parameter.upper_bound;
    result.resonance_index = resonance_index;
    result.target = target;
    result.transform = GVVFitTransform::Identity;
    result.source_name = source_name;
    return result;
}

GVVPropagatorFitBinding log_binding(
    const ctpwa::ResonanceDefinition& resonance,
    const ctpwa::ParameterDefinition& parameter,
    const std::string& source_name,
    const std::string& fit_name,
    int resonance_index,
    GVVPropagatorParameterTarget target)
{
    GVVPropagatorFitBinding result;
    result.fit_name = fit_name + resonance.id;
    result.initial_coordinate = std::log(parameter.value);
    result.step = parameter.step;
    result.has_lower_bound = parameter.has_lower_bound;
    result.has_upper_bound = parameter.has_upper_bound;
    result.lower_bound = parameter.lower_bound;
    result.upper_bound = parameter.upper_bound;
    result.resonance_index = resonance_index;
    result.target = target;
    result.transform = GVVFitTransform::LogPositive;
    result.source_name = source_name;
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

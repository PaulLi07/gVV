// Host regression for the process-owned propagator compiler. Every supported
// model is exercised without loading samples or launching a CUDA kernel.
#include "core/Model.h"
#include "core/physics/OmegaDecay.cuh"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

ctpwa::ParameterDefinition fixed(double value)
{
    ctpwa::ParameterDefinition result;
    result.value = value;
    return result;
}

ctpwa::ParameterDefinition free_identity(
    double value,
    double lower,
    double upper,
    double step)
{
    ctpwa::ParameterDefinition result;
    result.value = value;
    result.fixed = false;
    result.transform = "identity";
    result.step = step;
    result.has_lower_bound = true;
    result.has_upper_bound = true;
    result.lower_bound = lower;
    result.upper_bound = upper;
    return result;
}

ctpwa::ParameterDefinition free_log(double value)
{
    ctpwa::ParameterDefinition result;
    result.value = value;
    result.fixed = false;
    result.transform = "log";
    result.step = 0.2;
    result.has_lower_bound = true;
    result.has_upper_bound = true;
    result.lower_bound = -5.0;
    result.upper_bound = 4.0;
    return result;
}

ctpwa::ResonanceDefinition resonance(
    const std::string& id,
    const std::string& propagator,
    double mass = 1.80,
    double width = 0.20)
{
    ctpwa::ResonanceDefinition result;
    result.id = id;
    result.label = id;
    result.propagator = propagator;
    if (propagator != "nonresonant") {
        result.parameters.emplace("mass", fixed(mass));
        result.parameters.emplace("width", fixed(width));
    }
    return result;
}

void require_invalid(
    const ctpwa::ResonanceDefinition& definition,
    const std::string& expected)
{
    try {
        (void)gvv_compile_resonance(definition, 0);
    } catch (const std::exception& error) {
        require(
            std::string(error.what()).find(expected) != std::string::npos,
            "propagator compiler diagnostic did not contain expected text");
        return;
    }
    throw std::runtime_error("invalid propagator definition was accepted");
}

void require_identity_binding(
    const GVVFitParameterBinding& binding,
    const std::string& fit_name,
    GVVPropagatorParameterTarget target,
    double initial,
    double step,
    double lower,
    double upper,
    const char* message)
{
    require(
        binding.fit.name == fit_name
            && binding.propagator_target == target
            && binding.transform == GVVFitTransform::Identity
            && std::fabs(binding.fit.initial_value - initial) < 1.0e-14
            && std::fabs(binding.fit.step - step) < 1.0e-14
            && binding.fit.has_lower_bound
            && binding.fit.has_upper_bound
            && std::fabs(binding.fit.lower_bound - lower) < 1.0e-14
            && std::fabs(binding.fit.upper_bound - upper) < 1.0e-14,
        message);
}

} // namespace

int main()
{
    try {
        const GVVCompiledResonance nonresonant = gvv_compile_resonance(
            resonance("nr", "nonresonant"), 0);
        require(
            nonresonant.propagator.propagator_model
                == ctpwa::PROP_NONRESONANT,
            "nonresonant model compilation failed");

        const GVVCompiledResonance fixed_width = gvv_compile_resonance(
            resonance("fixed", "fixed_width_bw", 1.50, 0.10), 1);
        require(
            fixed_width.propagator.propagator_model == ctpwa::PROP_FIXED_BW,
            "fixed-width model compilation failed");

        ctpwa::ResonanceDefinition floating_fixed = resonance(
            "floating_fixed", "fixed_width_bw", 1.50, 0.10);
        floating_fixed.parameters.at("mass") =
            free_identity(1.50, 1.40, 1.60, 0.001);
        floating_fixed.parameters.at("width") =
            free_identity(0.10, 0.05, 0.20, 0.002);
        const GVVCompiledResonance compiled_floating_fixed =
            gvv_compile_resonance(floating_fixed, 2);
        require(
            compiled_floating_fixed.fit_bindings.size() == 2,
            "fixed-width floating binding count is wrong");
        require_identity_binding(
            compiled_floating_fixed.fit_bindings[0],
            "mass_floating_fixed",
            GVVPropagatorParameterTarget::Mass,
            1.50, 0.001, 1.40, 1.60,
            "fixed-width floating mass binding is wrong");
        require_identity_binding(
            compiled_floating_fixed.fit_bindings[1],
            "width_floating_fixed",
            GVVPropagatorParameterTarget::PoleWidth,
            0.10, 0.002, 0.05, 0.20,
            "fixed-width floating width binding is wrong");

        ctpwa::ResonanceDefinition running = resonance(
            "running", "two_body_running_bw");
        running.parameters.emplace("orbital_l", fixed(2.0));
        const GVVCompiledResonance compiled_running =
            gvv_compile_resonance(running, 2);
        require(
            compiled_running.propagator.propagator_model
                    == ctpwa::PROP_TWO_BODY_RUNNING_BW
                && compiled_running.propagator.orbital_l == 2
                && compiled_running.propagator.daughter_mass1
                    == GVV_OMEGA_MASS
                && compiled_running.propagator.daughter_mass2
                    == GVV_OMEGA_MASS,
            "running-width channel context was not compiled");

        ctpwa::ResonanceDefinition floating_running = running;
        floating_running.id = "floating_running";
        floating_running.label = "floating_running";
        floating_running.parameters.at("mass") =
            free_identity(1.80, 1.70, 1.95, 0.001);
        floating_running.parameters.at("width") =
            free_identity(0.20, 0.05, 0.40, 0.002);
        const GVVCompiledResonance compiled_floating_running =
            gvv_compile_resonance(floating_running, 3);
        require(
            compiled_floating_running.fit_bindings.size() == 2,
            "running-width floating binding count is wrong");
        require_identity_binding(
            compiled_floating_running.fit_bindings[0],
            "mass_floating_running",
            GVVPropagatorParameterTarget::Mass,
            1.80, 0.001, 1.70, 1.95,
            "running-width floating mass binding is wrong");
        require_identity_binding(
            compiled_floating_running.fit_bindings[1],
            "width_floating_running",
            GVVPropagatorParameterTarget::PoleWidth,
            0.20, 0.002, 0.05, 0.40,
            "running-width floating width binding is wrong");

        ctpwa::ResonanceDefinition scalar_sd = resonance(
            "scalar", "scalar_sd_running_bw");
        scalar_sd.parameters.emplace("sd_ratio", free_log(0.5));
        const GVVCompiledResonance compiled_sd =
            gvv_compile_resonance(scalar_sd, 3);
        require(
            compiled_sd.propagator.propagator_model
                    == ctpwa::PROP_SCALAR_SD_BWR
                && compiled_sd.fit_bindings.size() == 1
                && compiled_sd.fit_bindings[0].propagator_target
                    == GVVPropagatorParameterTarget::SDRatio,
            "S/D running-width fit binding was not compiled");

        ctpwa::ResonanceDefinition flatte = resonance(
            "flatte", "subtracted_effective_flatte", 1.522, 0.108);
        flatte.parameters.at("mass") =
            free_identity(1.522, 1.45, 1.60, 0.001);
        flatte.parameters.at("width") =
            free_identity(0.108, 0.05, 0.20, 0.002);
        flatte.parameters.emplace("omegaomega_ratio", free_log(1.0));
        const GVVCompiledResonance compiled_flatte =
            gvv_compile_resonance(flatte, 4);
        require(
            compiled_flatte.propagator.propagator_model
                    == ctpwa::PROP_SUBTRACTED_FLATTE
                && compiled_flatte.fit_bindings.size() == 3
                && compiled_flatte.fit_bindings[0].fit.name
                    == "mass_flatte"
                && compiled_flatte.fit_bindings[1].fit.name
                    == "width_flatte"
                && compiled_flatte.fit_bindings[2].fit.name
                    == "log_Romega_flatte",
            "Flatte fit binding was not compiled");

        ctpwa::PropagatorParameters mutable_state =
            compiled_flatte.propagator;
        gvv_set_propagator_parameter(
            mutable_state,
            GVVPropagatorParameterTarget::FlatteRatio,
            0.75);
        require(
            std::fabs(gvv_propagator_parameter_value(
                          mutable_state,
                          GVVPropagatorParameterTarget::FlatteRatio)
                      - 0.75) < 1.0e-14,
            "generic propagator parameter access failed");

        ctpwa::ResonanceDefinition zero_width = running;
        zero_width.parameters.at("width").value = 0.0;
        require_invalid(zero_width, "positive mass and pole width");

        ctpwa::ResonanceDefinition below_threshold = running;
        below_threshold.parameters.at("mass").value = 1.50;
        require_invalid(below_threshold, "above the nominal omega-omega");

        ctpwa::ResonanceDefinition transformed_mass = floating_fixed;
        transformed_mass.parameters.at("mass").transform = "log";
        require_invalid(transformed_mass, "must use the identity transform");

        ctpwa::ResonanceDefinition missing_bounds = floating_fixed;
        missing_bounds.id = "missing_bounds";
        missing_bounds.parameters.at("mass").has_upper_bound = false;
        require_invalid(missing_bounds, "requires finite bounds");

        ctpwa::ResonanceDefinition nonpositive_bound = floating_fixed;
        nonpositive_bound.id = "nonpositive_bound";
        nonpositive_bound.parameters.at("width").lower_bound = 0.0;
        require_invalid(nonpositive_bound, "positive lower limit");

        ctpwa::ResonanceDefinition threshold_crossing = floating_running;
        threshold_crossing.id = "threshold_crossing";
        threshold_crossing.parameters.at("mass").lower_bound = 1.50;
        require_invalid(threshold_crossing, "entire free mass range");

        ctpwa::ResonanceDefinition floating_orbital_l = running;
        floating_orbital_l.parameters.at("orbital_l").fixed = false;
        require_invalid(floating_orbital_l, "must be a fixed integer");

        ctpwa::ResonanceDefinition identity_ratio = flatte;
        identity_ratio.parameters.at("omegaomega_ratio").transform =
            "identity";
        require_invalid(identity_ratio, "positive log-transformed");

        ctpwa::ResonanceDefinition unknown = running;
        unknown.parameters.emplace("widht", fixed(0.2));
        require_invalid(unknown, "does not accept parameter");

        std::cout << "GVV propagator compiler tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << "GVV propagator compiler test failed: "
                  << error.what() << '\n';
        return 1;
    }
    return 0;
}

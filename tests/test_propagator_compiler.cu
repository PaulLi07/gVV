// Host regression for the process-owned propagator compiler. Every supported
// model is exercised without loading samples or launching a CUDA kernel.
#include "process/PropagatorCompiler.h"
#include "process/OmegaDecayModel.cuh"

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

        ctpwa::ResonanceDefinition scalar_sd = resonance(
            "scalar", "scalar_sd_running_bw");
        scalar_sd.parameters.emplace("sd_ratio", free_log(0.5));
        const GVVCompiledResonance compiled_sd =
            gvv_compile_resonance(scalar_sd, 3);
        require(
            compiled_sd.propagator.propagator_model
                    == ctpwa::PROP_SCALAR_SD_BWR
                && compiled_sd.fit_bindings.size() == 1
                && compiled_sd.fit_bindings[0].target
                    == GVVPropagatorParameterTarget::SDRatio,
            "S/D running-width fit binding was not compiled");

        ctpwa::ResonanceDefinition flatte = resonance(
            "flatte", "subtracted_effective_flatte", 1.522, 0.108);
        flatte.parameters.emplace("omegaomega_ratio", free_log(1.0));
        const GVVCompiledResonance compiled_flatte =
            gvv_compile_resonance(flatte, 4);
        require(
            compiled_flatte.propagator.propagator_model
                    == ctpwa::PROP_SUBTRACTED_FLATTE
                && compiled_flatte.fit_bindings.size() == 1
                && compiled_flatte.fit_bindings[0].fit_name
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

// Verifies that the model-generated Minuit layout has stable ordering for the
// nominal fixture while remaining sized from runtime vectors.
#include "process/ParameterMapping.h"
#include "process/TermEvaluator.cuh"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

int main()
{
    GVVCompiledModel compiled =
        gvv_load_compiled_model("config/model.json");
    const std::vector<GVVFitParameterBinding> layout =
        gvv_fit_parameter_layout(compiled);
    const std::vector<ctpwa::FitParameterSpec> parameters =
        gvv_fit_parameter_specs(layout);
    if (layout.size() != 12 || parameters.size() != layout.size()
        || parameters[0].name != "Re_f0_1500_00"
        || parameters[2].name != "log_rho_f0_1710_00"
        || parameters[11].name != "log_Romega_f0_1500"
        || !parameters[11].has_lower_bound
        || parameters[11].lower_bound != -6.0
        || parameters[11].upper_bound != 3.0
        || parameters[0].randomization
               != ctpwa::ParameterRandomization::ComplexReal
        || parameters[1].randomization
               != ctpwa::ParameterRandomization::ComplexImaginary
        || parameters[2].randomization
               != ctpwa::ParameterRandomization::LogMagnitude
        || !compiled.resonance_metadata[0].fit_flatte_ratio
        || layout[11].target
               != GVVFitParameterTarget::ResonanceLogFlatteRatio) {
        std::cerr << "runtime GVV fit parameter layout is wrong\n";
        return 1;
    }

    std::vector<double> values;
    values.reserve(parameters.size());
    for (const ctpwa::FitParameterSpec& parameter : parameters) {
        values.push_back(parameter.initial_value);
    }
    values[0] = 0.25;
    values[1] = -0.50;
    values[2] = std::log(2.0);
    values[11] = std::log(0.75);
    gvv_apply_fit_parameters(compiled, layout, values);
    const int cartesian_term = layout[0].target_index;
    const int phase_reference_term = layout[2].target_index;
    const int flatte_resonance = layout[11].target_index;
    if (compiled.initial_couplings[cartesian_term].real != 0.25
        || compiled.initial_couplings[cartesian_term].imag != -0.50
        || std::fabs(
               compiled.initial_couplings[phase_reference_term].real - 2.0)
               > 1.0e-12
        || compiled.initial_couplings[phase_reference_term].imag != 0.0
        || std::fabs(
               compiled.resonances[flatte_resonance].flatte_ratio - 0.75)
               > 1.0e-12) {
        std::cerr << "GVV fit-parameter application is wrong\n";
        return 2;
    }

    ctpwa::ModelDefinition reduced_definition = compiled.definition;
    reduced_definition.terms[0].active = false;
    const GVVCompiledModel reduced =
        gvv_compile_model(reduced_definition);
    const std::vector<GVVFitParameterBinding> reduced_layout =
        gvv_fit_parameter_layout(reduced);
    bool has_inactive_flatte_parameter = false;
    for (const GVVFitParameterBinding& parameter : reduced_layout) {
        if (parameter.fit.name == "log_Romega_f0_1500") {
            has_inactive_flatte_parameter = true;
        }
    }
    if (reduced.resonances.size() != 6
        || reduced.find_resonance("f0_1500") != -1
        || reduced_layout.size() != 9
        || has_inactive_flatte_parameter) {
        std::cerr << "inactive Term leaked Resonance fit parameters\n";
        return 3;
    }

    const int number_terms = static_cast<int>(compiled.terms.size());
    const int number_pairs = ctpwa::component_pair_count(number_terms);
    std::vector<bool> seen(number_pairs, false);
    for (int first = 0; first < number_terms; ++first) {
        for (int second = first; second < number_terms; ++second) {
            const int index = ctpwa::component_pair_index(
                first, second, number_terms);
            if (index < 0 || index >= number_pairs || seen[index]) {
                std::cerr << "component-pair compact index is wrong\n";
                return 4;
            }
            seen[index] = true;
        }
    }

    std::cout << "GVV fit-parameter tests passed\n";
    return 0;
}

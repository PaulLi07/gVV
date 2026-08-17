// Verifies that the model-generated Minuit layout has stable ordering for the
// nominal fixture while remaining sized from runtime vectors.
#include "process/ParameterMapping.h"
#include "process/TermEvaluator.cuh"

#include <iostream>
#include <vector>

int main()
{
    const GVVCompiledModel compiled =
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

    const int number_terms = static_cast<int>(compiled.terms.size());
    const int number_pairs = ctpwa::component_pair_count(number_terms);
    std::vector<bool> seen(number_pairs, false);
    for (int first = 0; first < number_terms; ++first) {
        for (int second = first; second < number_terms; ++second) {
            const int index = ctpwa::component_pair_index(
                first, second, number_terms);
            if (index < 0 || index >= number_pairs || seen[index]) {
                std::cerr << "component-pair compact index is wrong\n";
                return 2;
            }
            seen[index] = true;
        }
    }

    std::cout << "GVV fit-parameter tests passed\n";
    return 0;
}

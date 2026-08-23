// Verifies that the model-generated Minuit layout follows active Terms and
// shares each Resonance parameter set across every Wave that uses it.
#include "process/ParameterMapping.h"
#include "process/ModelCompiler.h"
#include "process/TermEvaluator.cuh"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int find_parameter(
    const std::vector<ctpwa::FitParameterSpec>& parameters,
    const std::string& name)
{
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        if (parameters[index].name == name) {
            return static_cast<int>(index);
        }
    }
    throw std::runtime_error("missing fit parameter '" + name + "'");
}

bool has_parameter(
    const std::vector<GVVFitParameterBinding>& layout,
    const std::string& name)
{
    for (const GVVFitParameterBinding& parameter : layout) {
        if (parameter.fit.name == name) {
            return true;
        }
    }
    return false;
}

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

} // namespace

int main()
{
    try {
        GVVCompiledModel compiled =
            gvv_load_compiled_model("config/model.json");
        const std::vector<GVVFitParameterBinding> layout =
            gvv_fit_parameter_layout(compiled);
        const std::vector<ctpwa::FitParameterSpec> parameters =
            gvv_fit_parameter_specs(layout);

        // Six active legacy Terms contribute nine coupling coordinates. The six
        // tensor Terms add twelve more, and the two threshold line shapes add
        // one shared log-coupling coordinate each.
        require(layout.size() == 23 && parameters.size() == layout.size(),
                "runtime GVV fit parameter count is wrong");
        require(compiled.propagator_fit_bindings.size() == 2,
                "free propagator parameter count is wrong");

        const int re_f0 = find_parameter(parameters, "Re_f0_1500_00");
        const int im_f0 = find_parameter(parameters, "Im_f0_1500_00");
        const int phase_reference =
            find_parameter(parameters, "log_rho_f0_1710_00");
        const int f0_ratio =
            find_parameter(parameters, "log_Romega_f0_1500");
        const int f2_ratio =
            find_parameter(parameters, "log_Romega_f2_1565");
        const int re_f2_u1 =
            find_parameter(parameters, "Re_f2_1565_02_u1");
        const int im_f2_u1 =
            find_parameter(parameters, "Im_f2_1565_02_u1");

        require(parameters[re_f0].randomization
                    == ctpwa::ParameterRandomization::ComplexReal,
                "Cartesian real randomization policy is wrong");
        require(parameters[im_f0].randomization
                    == ctpwa::ParameterRandomization::ComplexImaginary,
                "Cartesian imaginary randomization policy is wrong");
        require(parameters[phase_reference].randomization
                    == ctpwa::ParameterRandomization::LogMagnitude,
                "phase-reference randomization policy is wrong");
        for (const int ratio : {f0_ratio, f2_ratio}) {
            require(parameters[ratio].has_lower_bound
                        && parameters[ratio].has_upper_bound
                        && parameters[ratio].lower_bound == -6.0
                        && parameters[ratio].upper_bound == 3.0,
                    "threshold-line-shape bounds are wrong");
            require(layout[ratio].target
                        == GVVFitParameterTarget::PropagatorParameter
                        && layout[ratio].propagator_target
                            == GVVPropagatorParameterTarget::FlatteRatio,
                    "threshold-line-shape parameter target is wrong");
        }

        const int f2_1565 = compiled.find_resonance("f2_1565");
        const int f2_1810 = compiled.find_resonance("f2_1810");
        require(f2_1565 >= 0 && f2_1810 >= 0 && f2_1565 != f2_1810,
                "tensor Resonance indices are wrong");
        require(layout[f2_ratio].target_index == f2_1565,
                "f2(1565) line-shape parameter points to the wrong Resonance");
        for (const std::string& id : {
                 "f2_1565_02_u1", "f2_1565_02_u2", "f2_1565_02_u3"}) {
            require(compiled.terms[compiled.find_term(id)].resonance_index
                        == f2_1565,
                    "f2(1565) Terms do not share one propagator");
        }
        for (const std::string& id : {
                 "f2_1810_02_u1", "f2_1810_02_u2", "f2_1810_02_u3"}) {
            require(compiled.terms[compiled.find_term(id)].resonance_index
                        == f2_1810,
                    "f2(1810) Terms do not share one propagator");
        }

        std::vector<double> values;
        values.reserve(parameters.size());
        for (const ctpwa::FitParameterSpec& parameter : parameters) {
            values.push_back(parameter.initial_value);
        }
        values[re_f0] = 0.25;
        values[im_f0] = -0.50;
        values[phase_reference] = std::log(2.0);
        values[f0_ratio] = std::log(0.75);
        values[f2_ratio] = std::log(1.25);
        values[re_f2_u1] = 0.33;
        values[im_f2_u1] = -0.22;
        gvv_apply_fit_parameters(compiled, layout, values);

        const int f0_term = layout[re_f0].target_index;
        const int reference_term = layout[phase_reference].target_index;
        const int f2_u1_term = layout[re_f2_u1].target_index;
        require(compiled.initial_couplings[f0_term].real == 0.25
                    && compiled.initial_couplings[f0_term].imag == -0.50,
                "Cartesian coupling application is wrong");
        require(std::fabs(
                    compiled.initial_couplings[reference_term].real - 2.0)
                    < 1.0e-12
                    && compiled.initial_couplings[reference_term].imag == 0.0,
                "positive-real coupling application is wrong");
        require(std::fabs(compiled.resonances[
                              layout[f0_ratio].target_index].flatte_ratio
                          - 0.75)
                    < 1.0e-12
                    && std::fabs(compiled.resonances[f2_1565].flatte_ratio
                                 - 1.25)
                        < 1.0e-12,
                "shared propagator parameter application is wrong");
        require(compiled.initial_couplings[f2_u1_term].real == 0.33
                    && compiled.initial_couplings[f2_u1_term].imag == -0.22,
                "tensor coupling application is wrong");

        ctpwa::ModelDefinition reduced_definition = compiled.definition;
        reduced_definition.terms[0].active = false;
        const GVVCompiledModel reduced =
            gvv_compile_model(reduced_definition);
        const std::vector<GVVFitParameterBinding> reduced_layout =
            gvv_fit_parameter_layout(reduced);
        require(reduced.resonances.size() == 7
                    && reduced.find_resonance("f0_1500") == -1
                    && reduced_layout.size() == layout.size() - 3
                    && !has_parameter(
                        reduced_layout, "log_Romega_f0_1500"),
                "inactive Term leaked Resonance fit parameters");

        // Disabling all three f2(1565) Terms must remove its one shared
        // propagator coordinate while leaving the f2(1810) LS=02 basis active.
        ctpwa::ModelDefinition no_f2_1565 = compiled.definition;
        for (ctpwa::TermDefinition& term : no_f2_1565.terms) {
            if (term.id.rfind("f2_1565_02_", 0) == 0) {
                term.active = false;
            }
        }
        for (ctpwa::ResonanceDefinition& resonance : no_f2_1565.resonances) {
            if (resonance.id == "f2_1565") {
                resonance.propagator = "ignored_for_inactive_terms";
            }
        }
        const GVVCompiledModel without_f2_1565 =
            gvv_compile_model(no_f2_1565);
        const std::vector<GVVFitParameterBinding> without_f2_layout =
            gvv_fit_parameter_layout(without_f2_1565);
        require(without_f2_1565.find_resonance("f2_1565") == -1
                    && without_f2_1565.find_resonance("f2_1810") >= 0
                    && without_f2_1565.terms.size() == 9
                    && without_f2_1565.active_wave_types.size() == 5
                    && without_f2_layout.size() == layout.size() - 7
                    && !has_parameter(
                        without_f2_layout, "log_Romega_f2_1565"),
                "inactive tensor Resonance leaked into the fit layout");

        const int number_terms = static_cast<int>(compiled.terms.size());
        const int number_pairs = ctpwa::component_pair_count(number_terms);
        std::vector<bool> seen(number_pairs, false);
        for (int first = 0; first < number_terms; ++first) {
            for (int second = first; second < number_terms; ++second) {
                const int index = ctpwa::component_pair_index(
                    first, second, number_terms);
                require(index >= 0 && index < number_pairs && !seen[index],
                        "component-pair compact index is wrong");
                seen[index] = true;
            }
        }

        std::cout << "GVV fit-parameter tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << "GVV fit-parameter test failed: "
                  << error.what() << '\n';
        return 1;
    }
    return 0;
}

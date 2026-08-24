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

ctpwa::ResonanceDefinition& mutable_resonance(
    ctpwa::ModelDefinition& model,
    const std::string& id)
{
    for (ctpwa::ResonanceDefinition& resonance : model.resonances) {
        if (resonance.id == id) {
            return resonance;
        }
    }
    throw std::runtime_error("missing Resonance definition '" + id + "'");
}

void configure_free_identity(
    ctpwa::ParameterDefinition& parameter,
    double lower,
    double upper,
    double step)
{
    parameter.fixed = false;
    parameter.transform = "identity";
    parameter.step = step;
    parameter.has_lower_bound = true;
    parameter.has_upper_bound = true;
    parameter.lower_bound = lower;
    parameter.upper_bound = upper;
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

        // Nine active non-tensor Terms contribute fifteen coupling
        // coordinates. The six tensor Terms add twelve more, the two
        // threshold line shapes add one log-ratio each, and the floating
        // eta(1760)/eta(2225) pole parameters add four coordinates.
        require(layout.size() == 33 && parameters.size() == layout.size(),
                "runtime GVV fit parameter count is wrong");
        require(compiled.propagator_fit_bindings.size() == 6,
                "free propagator parameter count is wrong");

        const int re_f0 = find_parameter(parameters, "Re_f0_1500_00");
        const int im_f0 = find_parameter(parameters, "Im_f0_1500_00");
        const int re_f0_1500_22 =
            find_parameter(parameters, "Re_f0_1500_22");
        const int im_f0_1500_22 =
            find_parameter(parameters, "Im_f0_1500_22");
        const int re_f0_1710_22 =
            find_parameter(parameters, "Re_f0_1710_22");
        const int im_f0_1710_22 =
            find_parameter(parameters, "Im_f0_1710_22");
        const int phase_reference =
            find_parameter(parameters, "log_rho_f0_1710_00");
        const int f0_ratio =
            find_parameter(parameters, "log_Romega_f0_1500");
        const int f2_ratio =
            find_parameter(parameters, "log_Romega_f2_1565");
        const int eta_1760_mass =
            find_parameter(parameters, "mass_eta_1760");
        const int eta_1760_width =
            find_parameter(parameters, "width_eta_1760");
        const int eta_2225_mass =
            find_parameter(parameters, "mass_eta_2225");
        const int eta_2225_width =
            find_parameter(parameters, "width_eta_2225");
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
        require(parameters[re_f0_1500_22].randomization
                    == ctpwa::ParameterRandomization::ComplexReal
                    && parameters[im_f0_1500_22].randomization
                        == ctpwa::ParameterRandomization::ComplexImaginary
                    && parameters[re_f0_1710_22].randomization
                        == ctpwa::ParameterRandomization::ComplexReal
                    && parameters[im_f0_1710_22].randomization
                        == ctpwa::ParameterRandomization::ComplexImaginary,
                "scalar LS=22 randomization policy is wrong");
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
        require(
            parameters[eta_2225_mass].lower_bound == 2.211
                && parameters[eta_2225_mass].upper_bound == 2.234
                && parameters[eta_2225_width].lower_bound == 0.165
                && parameters[eta_2225_width].upper_bound == 0.225,
            "eta(2225) PDG parameter bounds are wrong");
        for (const int pole_parameter : {
                 eta_1760_mass, eta_1760_width,
                 eta_2225_mass, eta_2225_width}) {
            require(
                parameters[pole_parameter].has_lower_bound
                    && parameters[pole_parameter].has_upper_bound
                    && layout[pole_parameter].target
                        == GVVFitParameterTarget::PropagatorParameter,
                "floating eta pole-parameter binding is wrong");
        }

        const int f0_1500 = compiled.find_resonance("f0_1500");
        const int f0_1710 = compiled.find_resonance("f0_1710");
        require(f0_1500 >= 0 && f0_1710 >= 0 && f0_1500 != f0_1710,
                "scalar Resonance indices are wrong");
        for (const std::string& id : {"f0_1500_00", "f0_1500_22"}) {
            require(compiled.terms[compiled.find_term(id)].resonance_index
                        == f0_1500,
                    "f0(1500) Terms do not share one propagator");
        }
        for (const std::string& id : {"f0_1710_00", "f0_1710_22"}) {
            require(compiled.terms[compiled.find_term(id)].resonance_index
                        == f0_1710,
                    "f0(1710) Terms do not share one propagator");
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
        values[re_f0_1500_22] = -0.15;
        values[im_f0_1500_22] = 0.20;
        values[re_f0_1710_22] = 0.35;
        values[im_f0_1710_22] = -0.10;
        values[phase_reference] = std::log(2.0);
        values[f0_ratio] = std::log(0.75);
        values[f2_ratio] = std::log(1.25);
        values[eta_1760_mass] = 1.80;
        values[eta_1760_width] = 0.25;
        values[eta_2225_mass] = 2.222;
        values[eta_2225_width] = 0.190;
        values[re_f2_u1] = 0.33;
        values[im_f2_u1] = -0.22;
        gvv_apply_fit_parameters(compiled, layout, values);

        const int f0_term = layout[re_f0].target_index;
        const int f0_1500_22_term = layout[re_f0_1500_22].target_index;
        const int f0_1710_22_term = layout[re_f0_1710_22].target_index;
        const int reference_term = layout[phase_reference].target_index;
        const int f2_u1_term = layout[re_f2_u1].target_index;
        require(compiled.initial_couplings[f0_term].real == 0.25
                    && compiled.initial_couplings[f0_term].imag == -0.50,
                "Cartesian coupling application is wrong");
        require(
            compiled.initial_couplings[f0_1500_22_term].real == -0.15
                && compiled.initial_couplings[f0_1500_22_term].imag == 0.20
                && compiled.initial_couplings[f0_1710_22_term].real == 0.35
                && compiled.initial_couplings[f0_1710_22_term].imag == -0.10,
            "scalar LS=22 coupling application is wrong");
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
        require(
            std::fabs(compiled.resonances[
                          compiled.find_resonance("eta_1760")].mass
                      - 1.80) < 1.0e-12
                && std::fabs(compiled.resonances[
                                  compiled.find_resonance("eta_1760")]
                                  .pole_width
                              - 0.25)
                    < 1.0e-12
                && std::fabs(compiled.resonances[
                                  compiled.find_resonance("eta_2225")].mass
                              - 2.222)
                    < 1.0e-12
                && std::fabs(compiled.resonances[
                                  compiled.find_resonance("eta_2225")]
                                  .pole_width
                              - 0.190)
                    < 1.0e-12,
            "floating eta pole-parameter application is wrong");

        // Disabling one of two Terms sharing f0(1500) removes only that
        // coupling. The Resonance and its propagator coordinate stay active.
        ctpwa::ModelDefinition one_f0_1500_wave = compiled.definition;
        for (ctpwa::TermDefinition& term : one_f0_1500_wave.terms) {
            if (term.id == "f0_1500_00") {
                term.active = false;
            }
        }
        const GVVCompiledModel with_f0_1500_22_only =
            gvv_compile_model(one_f0_1500_wave);
        const std::vector<GVVFitParameterBinding> one_f0_1500_layout =
            gvv_fit_parameter_layout(with_f0_1500_22_only);
        require(with_f0_1500_22_only.resonances.size() == 9
                    && with_f0_1500_22_only.find_resonance("f0_1500") >= 0
                    && with_f0_1500_22_only.terms.size() == 14
                    && one_f0_1500_layout.size() == layout.size() - 2
                    && has_parameter(
                        one_f0_1500_layout, "log_Romega_f0_1500"),
                "active shared Term did not retain its Resonance parameters");

        // Disabling both f0(1500) Terms removes the shared Resonance and its
        // one propagator coordinate from the runtime fit layout.
        ctpwa::ModelDefinition no_f0_1500 = compiled.definition;
        for (ctpwa::TermDefinition& term : no_f0_1500.terms) {
            if (term.id.rfind("f0_1500_", 0) == 0) {
                term.active = false;
            }
        }
        mutable_resonance(no_f0_1500, "f0_1500").propagator =
            "ignored_for_inactive_terms";
        const GVVCompiledModel without_f0_1500 =
            gvv_compile_model(no_f0_1500);
        const std::vector<GVVFitParameterBinding> without_f0_1500_layout =
            gvv_fit_parameter_layout(without_f0_1500);
        require(without_f0_1500.resonances.size() == 8
                    && without_f0_1500.find_resonance("f0_1500") == -1
                    && without_f0_1500.terms.size() == 13
                    && without_f0_1500_layout.size() == layout.size() - 5
                    && !has_parameter(
                        without_f0_1500_layout, "log_Romega_f0_1500"),
                "inactive shared Resonance leaked into the fit layout");

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
                    && without_f2_1565.terms.size() == 12
                    && without_f2_1565.active_wave_types.size() == 6
                    && without_f2_layout.size() == layout.size() - 7
                    && !has_parameter(
                        without_f2_layout, "log_Romega_f2_1565"),
                "inactive tensor Resonance leaked into the fit layout");

        ctpwa::ModelDefinition shared_definition = compiled.definition;
        ctpwa::ResonanceDefinition& shared_input =
            mutable_resonance(shared_definition, "f2_1810");
        configure_free_identity(
            shared_input.parameters.at("mass"), 1.70, 1.95, 0.001);
        configure_free_identity(
            shared_input.parameters.at("width"), 0.05, 0.40, 0.002);
        GVVCompiledModel shared = gvv_compile_model(shared_definition);
        const std::vector<GVVFitParameterBinding> shared_layout =
            gvv_fit_parameter_layout(shared);
        const std::vector<ctpwa::FitParameterSpec> shared_parameters =
            gvv_fit_parameter_specs(shared_layout);
        require(
            shared_layout.size() == layout.size() + 2
                && shared.propagator_fit_bindings.size() == 8,
            "one shared Resonance produced duplicate fit coordinates");
        const int shared_mass =
            find_parameter(shared_parameters, "mass_f2_1810");
        const int shared_width =
            find_parameter(shared_parameters, "width_f2_1810");
        const int shared_resonance_index =
            shared.find_resonance("f2_1810");
        require(
            shared_layout[shared_mass].target_index
                    == shared_resonance_index
                && shared_layout[shared_width].target_index
                    == shared_resonance_index
                && shared_layout[shared_mass].propagator_target
                    == GVVPropagatorParameterTarget::Mass
                && shared_layout[shared_width].propagator_target
                    == GVVPropagatorParameterTarget::PoleWidth,
            "shared Resonance mass/width targets are wrong");
        for (const std::string& id : {
                 "f2_1810_02_u1", "f2_1810_02_u2", "f2_1810_02_u3"}) {
            require(
                shared.terms[shared.find_term(id)].resonance_index
                    == shared_resonance_index,
                "Terms using one Resonance ID did not share one parameter set");
        }
        std::vector<double> shared_values;
        for (const ctpwa::FitParameterSpec& parameter : shared_parameters) {
            shared_values.push_back(parameter.initial_value);
        }
        shared_values[shared_mass] = 1.84;
        shared_values[shared_width] = 0.21;
        gvv_apply_fit_parameters(shared, shared_layout, shared_values);
        require(
            std::fabs(gvv_propagator_parameter_value(
                          shared.resonances[shared_resonance_index],
                          GVVPropagatorParameterTarget::Mass)
                      - 1.84) < 1.0e-12
                && std::fabs(gvv_propagator_parameter_value(
                                 shared.resonances[shared_resonance_index],
                                 GVVPropagatorParameterTarget::PoleWidth)
                             - 0.21)
                    < 1.0e-12,
            "shared Resonance mass/width application is wrong");

        ctpwa::ModelDefinition independent_definition =
            compiled.definition;
        configure_free_identity(
            mutable_resonance(independent_definition, "f0_1710")
                .parameters.at("mass"),
            1.68, 1.78, 0.001);
        configure_free_identity(
            mutable_resonance(independent_definition, "f2_1810")
                .parameters.at("mass"),
            1.70, 1.95, 0.001);
        GVVCompiledModel independent =
            gvv_compile_model(independent_definition);
        const std::vector<GVVFitParameterBinding> independent_layout =
            gvv_fit_parameter_layout(independent);
        const std::vector<ctpwa::FitParameterSpec> independent_parameters =
            gvv_fit_parameter_specs(independent_layout);
        const int f0_mass =
            find_parameter(independent_parameters, "mass_f0_1710");
        const int f2_mass =
            find_parameter(independent_parameters, "mass_f2_1810");
        require(
            independent_layout[f0_mass].target_index
                    == independent.find_resonance("f0_1710")
                && independent_layout[f2_mass].target_index
                    == independent.find_resonance("f2_1810")
                && independent_layout[f0_mass].target_index
                    != independent_layout[f2_mass].target_index,
            "distinct Resonance IDs did not produce independent bindings");
        std::vector<double> independent_values;
        for (const ctpwa::FitParameterSpec& parameter :
             independent_parameters) {
            independent_values.push_back(parameter.initial_value);
        }
        independent_values[f0_mass] = 1.71;
        independent_values[f2_mass] = 1.84;
        gvv_apply_fit_parameters(
            independent, independent_layout, independent_values);
        require(
            std::fabs(gvv_propagator_parameter_value(
                          independent.resonances[
                              independent.find_resonance("f0_1710")],
                          GVVPropagatorParameterTarget::Mass)
                      - 1.71) < 1.0e-12
                && std::fabs(gvv_propagator_parameter_value(
                                 independent.resonances[
                                     independent.find_resonance("f2_1810")],
                                 GVVPropagatorParameterTarget::Mass)
                             - 1.84)
                    < 1.0e-12,
            "distinct Resonance mass bindings were not independent");

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

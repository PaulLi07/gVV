// Process-compiler regression: validates nominal migration, dynamic model
// sizing, and strict rejection of misspelled process fields.
#include "process/WaveRegistry.cuh"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

bool close(double first, double second)
{
    return std::fabs(first - second) < 1.0e-14;
}

void require_compile_invalid(
    const ctpwa::ModelDefinition& definition,
    const std::string& expected)
{
    try {
        (void)gvv_compile_model(definition);
    } catch (const std::exception& error) {
        require(
            std::string(error.what()).find(expected) != std::string::npos,
            "process-compiler diagnostic did not contain expected text");
        return;
    }
    throw std::runtime_error("invalid process model was accepted");
}

} // namespace

int main(int argc, char* argv[])
{
    try {
        const std::string file_name =
            argc > 1 ? argv[1] : "config/model.json";
        const GVVCompiledModel model = gvv_load_compiled_model(file_name);

        const std::vector<std::string> resonance_ids = {
            "f0_1500", "f0_1710", "eta_1760", "eta_c_1S",
            "X_1835", "X_2370", "NR_0mp"};
        const std::vector<int> propagators = {
            ctpwa::PROP_SUBTRACTED_FLATTE,
            ctpwa::PROP_SCALAR_SWAVE_BWR,
            ctpwa::PROP_PWAVE_BWR,
            ctpwa::PROP_PWAVE_BWR,
            ctpwa::PROP_PWAVE_BWR,
            ctpwa::PROP_PWAVE_BWR,
            ctpwa::PROP_NONRESONANT};
        const std::vector<double> masses = {
            1.522, 1.723, 1.751, 2.98409, 1.8340, 2.377, 0.0};
        const std::vector<double> widths = {
            0.108, 0.149, 0.240, 0.0300, 0.130, 0.148, 0.0};
        const std::vector<std::string> term_ids = {
            "f0_1500_00", "f0_1710_00", "eta_1760_11", "eta_c_11",
            "X_1835_11", "X_2370_11", "NR_0mp_11"};
        const std::vector<int> term_resonances = {0, 1, 2, 3, 4, 5, 6};
        const std::vector<int> term_waves = {
            GVV_SCALAR_00, GVV_SCALAR_00,
            GVV_PSEUDOSCALAR_11, GVV_PSEUDOSCALAR_11,
            GVV_PSEUDOSCALAR_11, GVV_PSEUDOSCALAR_11,
            GVV_PSEUDOSCALAR_11};
        const std::vector<int> coupling_policies = {
            COUPLING_COMPLEX,
            COUPLING_POSITIVE_REAL,
            COUPLING_FIXED_SCALE_AND_PHASE,
            COUPLING_COMPLEX,
            COUPLING_COMPLEX,
            COUPLING_COMPLEX,
            COUPLING_COMPLEX};

        require(model.resonances.size() == resonance_ids.size(),
                "compiled resonance count mismatch");
        require(model.terms.size() == term_ids.size(),
                "compiled Term count mismatch");
        require(model.initial_couplings.size() == term_ids.size(),
                "coupling count mismatch");
        require(model.active_wave_types.size() == 2,
                "active Wave count mismatch");
        require(model.find_resonance("f0_1710") == 1,
                "resonance lookup mismatch");
        require(model.find_term("eta_1760_11") == 2,
                "Term lookup mismatch");

        for (std::size_t index = 0; index < resonance_ids.size(); ++index) {
            require(model.resonance_metadata[index].id == resonance_ids[index],
                    "resonance id migration mismatch");
            require(model.resonances[index].propagator_model
                        == propagators[index],
                    "propagator migration mismatch");
            require(close(model.resonances[index].mass, masses[index]),
                    "mass migration mismatch");
            require(close(model.resonances[index].pole_width, widths[index]),
                    "width migration mismatch");
        }
        for (std::size_t index = 0; index < term_ids.size(); ++index) {
            require(model.term_metadata[index].id == term_ids[index],
                    "Term id migration mismatch");
            require(model.terms[index].resonance_index
                        == term_resonances[index],
                    "Term resonance migration mismatch");
            require(model.terms[index].registered_wave_type
                        == term_waves[index],
                    "Term Wave migration mismatch");
            require(model.term_metadata[index].coupling_parameterization
                        == coupling_policies[index],
                    "coupling policy migration mismatch");
        }
        require(close(
                    model.initial_couplings[model.find_term("eta_1760_11")].real,
                    1.0),
                "global reference initial value mismatch");

        // Runtime dimensions must follow the model rather than the nominal
        // seven-Term fixture. Disabling a non-reference Term changes only the
        // compiled dense layout.
        ctpwa::ModelDefinition reduced_definition = model.definition;
        reduced_definition.terms[0].active = false;
        const GVVCompiledModel reduced =
            gvv_compile_model(reduced_definition);
        require(reduced.terms.size() == 6,
                "disabled Term was not removed from runtime layout");
        require(reduced.active_wave_types.size() == 2,
                "reduced active Wave layout mismatch");

        // Likewise, a model larger than the nominal fixture compiles without
        // source-level count changes when it uses registered physics pieces.
        ctpwa::ModelDefinition expanded_definition = model.definition;
        ctpwa::ResonanceDefinition extra_resonance =
            expanded_definition.resonances.back();
        extra_resonance.id = "NR_extra";
        extra_resonance.label = "extra nonresonant test component";
        expanded_definition.resonances.push_back(extra_resonance);
        ctpwa::TermDefinition extra_term = expanded_definition.terms.back();
        extra_term.id = "NR_extra_11";
        extra_term.label = "extra test Term";
        extra_term.dynamics_json =
            R"({"type":"gvv_x_to_omega_omega","resonance":"NR_extra"})";
        expanded_definition.terms.push_back(extra_term);
        const GVVCompiledModel expanded =
            gvv_compile_model(expanded_definition);
        require(expanded.resonances.size() == 8,
                "expanded resonance layout mismatch");
        require(expanded.terms.size() == 8,
                "expanded Term layout mismatch");
        require(expanded.find_term("NR_extra_11") == 7,
                "expanded stable Term id mismatch");

        ctpwa::ModelDefinition bad_parameter = model.definition;
        bad_parameter.resonances[1].parameters.emplace(
            "widht", ctpwa::ParameterDefinition());
        require_compile_invalid(bad_parameter, "does not accept parameter");

        ctpwa::ModelDefinition bad_dynamics = model.definition;
        bad_dynamics.terms[0].dynamics_json =
            R"({"type":"gvv_x_to_omega_omega","resonance":"f0_1500","extra":1})";
        require_compile_invalid(bad_dynamics, "unknown dynamics field");

        std::cout << "GVV process-model compilation tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << "GVV process-model test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

// Process-compiler regression: validates nominal migration, dynamic model
// sizing, and strict rejection of misspelled process fields.
#include "core/Model.h"
#include "core/physics/OmegaDecay.cuh"
#include "core/physics/Waves.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
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

        const std::vector<GVVWaveMetadata>& registry = gvv_wave_registry();
        require(registry.size() == GVV_NBASIS,
                "Wave registry size does not match GVV_NBASIS");
        std::set<std::string> registered_ids;
        std::vector<bool> registered_types(GVV_NBASIS, false);
        for (const GVVWaveMetadata& wave : registry) {
            require(registered_ids.insert(wave.id).second,
                    "duplicate Wave id in registry");
            require(wave.wave_type >= 0 && wave.wave_type < GVV_NBASIS,
                    "registered Wave type is outside the enum range");
            require(!registered_types[wave.wave_type],
                    "duplicate Wave enum in registry");
            registered_types[wave.wave_type] = true;
            require(
                wave.coherence_class
                    == (wave.wave_type == GVV_PSEUDOSCALAR_11
                            ? "negative_parity"
                            : "positive_parity"),
                "Wave coherence-class registration mismatch");
        }
        require(
            gvv_registered_wave("gvv.tensor_02_u1").wave_type
                    == GVV_TENSOR_02_U1
                && gvv_registered_wave("gvv.tensor_02_u2").wave_type
                    == GVV_TENSOR_02_U2
                && gvv_registered_wave("gvv.tensor_02_u3").wave_type
                    == GVV_TENSOR_02_U3,
            "LS=02 tensor Wave registration mismatch");
        const std::vector<std::string> resonance_ids = {
            "f0_1500", "f0_1710", "f0_2020", "eta_1760", "eta_c_1S",
            "X_1835", "NR_0mp", "f2_1565", "f2_1810"};
        const std::vector<int> propagators = {
            ctpwa::PROP_SUBTRACTED_FLATTE,
            ctpwa::PROP_TWO_BODY_RUNNING_BW,
            ctpwa::PROP_TWO_BODY_RUNNING_BW,
            ctpwa::PROP_TWO_BODY_RUNNING_BW,
            ctpwa::PROP_TWO_BODY_RUNNING_BW,
            ctpwa::PROP_TWO_BODY_RUNNING_BW,
            ctpwa::PROP_NONRESONANT,
            ctpwa::PROP_SUBTRACTED_FLATTE,
            ctpwa::PROP_TWO_BODY_RUNNING_BW};
        const std::vector<int> orbital_l = {0, 0, 0, 1, 1, 1, 0, 0, 0};
        const std::vector<double> masses = {
            1.522, 1.723, 1.982, 1.751, 2.98409,
            1.8340, 0.0, 1.571, 1.815};
        const std::vector<double> widths = {
            0.108, 0.149, 0.440, 0.240, 0.0300,
            0.130, 0.0, 0.132, 0.197};
        const std::vector<std::string> term_ids = {
            "f0_1500_00", "f0_1500_22",
            "f0_1710_00", "f0_1710_22",
            "f0_2020_00", "f0_2020_22",
            "eta_1760_11", "eta_c_11", "X_1835_11", "NR_0mp_11",
            "f2_1565_02_u1", "f2_1565_02_u2", "f2_1565_02_u3",
            "f2_1810_02_u1", "f2_1810_02_u2", "f2_1810_02_u3"};
        const std::vector<int> term_resonances = {
            0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 7, 7, 8, 8, 8};
        const std::vector<int> term_waves = {
            GVV_SCALAR_00, GVV_SCALAR_22,
            GVV_SCALAR_00, GVV_SCALAR_22,
            GVV_SCALAR_00, GVV_SCALAR_22,
            GVV_PSEUDOSCALAR_11, GVV_PSEUDOSCALAR_11,
            GVV_PSEUDOSCALAR_11,
            GVV_PSEUDOSCALAR_11,
            GVV_TENSOR_02_U1, GVV_TENSOR_02_U2, GVV_TENSOR_02_U3,
            GVV_TENSOR_02_U1, GVV_TENSOR_02_U2, GVV_TENSOR_02_U3};
        const std::vector<int> coupling_policies = {
            COUPLING_COMPLEX,
            COUPLING_COMPLEX,
            COUPLING_POSITIVE_REAL,
            COUPLING_COMPLEX,
            COUPLING_COMPLEX,
            COUPLING_COMPLEX,
            COUPLING_FIXED_SCALE_AND_PHASE,
            COUPLING_COMPLEX,
            COUPLING_COMPLEX,
            COUPLING_COMPLEX,
            COUPLING_COMPLEX,
            COUPLING_COMPLEX,
            COUPLING_COMPLEX,
            COUPLING_COMPLEX,
            COUPLING_COMPLEX,
            COUPLING_COMPLEX};

        require(model.initial_parameters.resonances.size() == resonance_ids.size(),
                "compiled resonance count mismatch");
        require(model.terms.size() == term_ids.size(),
                "compiled Term count mismatch");
        require(model.initial_parameters.couplings.size() == term_ids.size(),
                "coupling count mismatch");
        require(model.active_wave_types.size() == 6,
                "active Wave count mismatch");
        require(model.find_resonance("X_2370") == -1,
                "inactive X(2370) Resonance entered the compiled model");
        require(model.find_resonance("eta_2225") == -1,
                "inactive eta(2225) Resonance entered the compiled model");
        require(model.find_resonance("f0_1710") == 1,
                "resonance lookup mismatch");
        require(model.find_term("eta_1760_11") == 6,
                "Term lookup mismatch");
        require(model.find_term("f2_1810_02_u3") == 15,
                "tensor Term lookup mismatch");

        for (std::size_t index = 0; index < resonance_ids.size(); ++index) {
            require(model.resonance_metadata[index].id == resonance_ids[index],
                    "resonance id migration mismatch");
            require(model.initial_parameters.resonances[index].propagator_model
                        == propagators[index],
                    "propagator migration mismatch");
            require(model.initial_parameters.resonances[index].orbital_l == orbital_l[index],
                    "propagator orbital-L migration mismatch");
            require(close(model.initial_parameters.resonances[index].mass, masses[index]),
                    "mass migration mismatch");
            require(close(model.initial_parameters.resonances[index].pole_width, widths[index]),
                    "width migration mismatch");
            if (propagators[index] != ctpwa::PROP_NONRESONANT) {
                require(close(
                            model.initial_parameters.resonances[index].daughter_mass1,
                            GVV_OMEGA_MASS)
                            && close(
                                model.initial_parameters.resonances[index].daughter_mass2,
                                GVV_OMEGA_MASS),
                        "compiled omega-omega channel masses mismatch");
            }
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
            require(
                model.term_metadata[index].wave_latex
                    == gvv_registered_wave(
                           model.term_metadata[index].wave_id).latex,
                "Term Wave latex metadata mismatch");
        }
        require(close(
                    model.initial_parameters.couplings[model.find_term("eta_1760_11")].real,
                    1.0),
                "global reference initial value mismatch");

        // Runtime dimensions must follow active Terms rather than the nominal
        // fixture. A Resonance is omitted only after all Terms using it are
        // inactive.
        ctpwa::ModelDefinition reduced_definition = model.definition;
        for (ctpwa::TermDefinition& term : reduced_definition.terms) {
            if (term.id.rfind("f0_1500_", 0) == 0) {
                term.active = false;
            }
        }
        reduced_definition.resonances[0].propagator =
            "ignored_for_inactive_terms";
        const GVVCompiledModel reduced =
            gvv_compile_model(reduced_definition);
        require(reduced.initial_parameters.resonances.size() == 8,
                "inactive-only Resonance was not removed");
        require(reduced.terms.size() == 14,
                "disabled Terms were not removed from runtime layout");
        require(reduced.active_wave_types.size() == 6,
                "reduced active Wave layout mismatch");
        require(reduced.find_resonance("f0_1500") == -1,
                "inactive-only Resonance remained addressable");
        require(reduced.find_resonance("f0_1710") == 0,
                "active Resonance indices were not compacted");
        require(reduced.terms[0].resonance_index == 0,
                "active Term did not use the compacted Resonance index");

        // Likewise, a model larger than the nominal fixture compiles without
        // source-level count changes when it uses registered physics pieces.
        ctpwa::ModelDefinition expanded_definition = model.definition;
        const auto nr_resonance = std::find_if(
            expanded_definition.resonances.begin(),
            expanded_definition.resonances.end(),
            [](const ctpwa::ResonanceDefinition& resonance) {
                return resonance.id == "NR_0mp";
            });
        require(nr_resonance != expanded_definition.resonances.end(),
                "nonresonant fixture is missing");
        ctpwa::ResonanceDefinition extra_resonance = *nr_resonance;
        extra_resonance.id = "NR_extra";
        extra_resonance.label = "extra nonresonant test component";
        expanded_definition.resonances.push_back(extra_resonance);
        const auto nr_term = std::find_if(
            expanded_definition.terms.begin(),
            expanded_definition.terms.end(),
            [](const ctpwa::TermDefinition& term) {
                return term.id == "NR_0mp_11";
            });
        require(nr_term != expanded_definition.terms.end(),
                "nonresonant Term fixture is missing");
        ctpwa::TermDefinition extra_term = *nr_term;
        extra_term.id = "NR_extra_11";
        extra_term.label = "extra test Term";
        extra_term.dynamics.resonance = "NR_extra";
        expanded_definition.terms.push_back(extra_term);
        const GVVCompiledModel expanded =
            gvv_compile_model(expanded_definition);
        require(expanded.initial_parameters.resonances.size() == 10,
                "expanded resonance layout mismatch");
        require(expanded.terms.size() == 17,
                "expanded Term layout mismatch");
        require(expanded.find_term("NR_extra_11") == 16,
                "expanded stable Term id mismatch");

        ctpwa::ModelDefinition bad_parameter = model.definition;
        bad_parameter.resonances[1].parameters.emplace(
            "widht", ctpwa::ParameterDefinition());
        require_compile_invalid(bad_parameter, "does not accept parameter");

        ctpwa::ModelDefinition d_wave_propagator = model.definition;
        d_wave_propagator.resonances[1].parameters.at("orbital_l").value = 2.0;
        const GVVCompiledModel d_wave_compiled =
            gvv_compile_model(d_wave_propagator);
        require(d_wave_compiled.initial_parameters.resonances[1].orbital_l == 2,
                "two-body running BW did not preserve orbital L=2");

        ctpwa::ModelDefinition unsupported_l = model.definition;
        unsupported_l.resonances[1].parameters.at("orbital_l").value = 3.0;
        require_compile_invalid(unsupported_l, "supported range");

        ctpwa::ModelDefinition zero_width = model.definition;
        zero_width.resonances[1].parameters.at("width").value = 0.0;
        require_compile_invalid(zero_width, "positive mass and pole width");

        ctpwa::ModelDefinition closed_pole = model.definition;
        closed_pole.resonances[1].parameters.at("mass").value = 1.50;
        require_compile_invalid(closed_pole, "above the nominal omega-omega");

        ctpwa::ModelDefinition bad_dynamics = model.definition;
        bad_dynamics.terms[0].dynamics.unknown_fields.push_back("extra");
        require_compile_invalid(bad_dynamics, "unknown dynamics field");

        std::cout << "GVV process-model compilation tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << "GVV process-model test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

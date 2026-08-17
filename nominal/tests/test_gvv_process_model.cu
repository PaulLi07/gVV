#include "../include/process/GVVProcessModel.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool close(double first, double second)
{
    return std::fabs(first - second) < 1.0e-14;
}

} // namespace

int main(int argc, char* argv[])
{
    try {
        const std::string file_name =
            argc > 1 ? argv[1] : "config/model.json";
        const GVVCompiledModel model = gvv_load_compiled_model(file_name);

        require(model.resonances.size() == 7, "compiled resonance count mismatch");
        require(model.terms.size() == 7, "compiled term count mismatch");
        require(model.initial_couplings.size() == 7, "coupling count mismatch");
        require(model.active_wave_types.size() == 2, "active wave count mismatch");
        require(model.find_resonance("f0_1710") == 1, "resonance lookup mismatch");
        require(model.find_term("eta_1760_11") == 2, "term lookup mismatch");

        // Migration bridge: the JSON-compiled nominal model must reproduce
        // every legacy default before the production path switches over.
        for (int index = 0; index < GVV_NRESONANCES; ++index) {
            const GVVResonanceParameters expected = gvv_default_resonance(index);
            const GVVResonanceParameters actual = model.resonances[index];
            require(
                actual.propagator_model == expected.propagator_model,
                "propagator migration mismatch");
            require(close(actual.mass, expected.mass), "mass migration mismatch");
            require(
                close(actual.pole_width, expected.pole_width),
                "width migration mismatch");
            require(
                close(actual.sd_ratio, expected.sd_ratio),
                "S/D migration mismatch");
            require(
                actual.fit_sd_ratio == expected.fit_sd_ratio,
                "S/D fit-policy migration mismatch");
            require(
                close(actual.flatte_ratio, expected.flatte_ratio),
                "Flatte migration mismatch");
            require(
                actual.fit_flatte_ratio == expected.fit_flatte_ratio,
                "Flatte fit-policy migration mismatch");
            require(
                model.resonance_metadata[index].id == gvv_resonance_name(index),
                "resonance id migration mismatch");
        }
        for (int index = 0; index < GVV_NTERMS; ++index) {
            const GVVTermSpec expected = gvv_default_term(index);
            require(
                model.terms[index].resonance_index == expected.resonance_index,
                "term resonance migration mismatch");
            require(
                model.terms[index].registered_wave_type
                    == expected.registered_wave_type,
                "term wave migration mismatch");
            require(
                model.term_metadata[index].id == gvv_term_name(index),
                "term id migration mismatch");
            require(
                model.term_metadata[index].coupling_parameterization
                    == gvv_coupling_parameterization(index),
                "coupling policy migration mismatch");
        }
        require(
            close(model.initial_couplings[GVV_SCALE_AND_PHASE_REFERENCE_TERM].real, 1.0),
            "global reference initial value mismatch");

        std::cout << "GVV process-model compilation tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << "GVV process-model test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

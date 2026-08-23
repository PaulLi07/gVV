// Generic model-parser test: nominal document plus strict invalid-input cases.
#include "framework/model/Model.h"

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

void require_invalid(const std::string& json, const char* expected)
{
    try {
        (void)ctpwa::parse_model_definition(json, "invalid-test");
    } catch (const std::exception& error) {
        require(
            std::string(error.what()).find(expected) != std::string::npos,
            "invalid-model diagnostic did not contain expected text");
        return;
    }
    throw std::runtime_error("invalid model was accepted");
}

} // namespace

int main(int argc, char* argv[])
{
    try {
        const std::string model_file =
            argc > 1 ? argv[1] : "config/model.json";
        const ctpwa::ModelDefinition model =
            ctpwa::load_model_definition(model_file);

        require(model.schema_version == 1, "schema version mismatch");
        require(
            model.process == "psi2s_to_gamma_omega_omega",
            "process id mismatch");
        require(model.resonances.size() == 9, "nominal resonance count mismatch");
        require(model.terms.size() == 13, "nominal term count mismatch");
        require(
            model.resonance("f0_1710").propagator
                == "two_body_running_bw",
            "f0(1710) propagator mismatch");
        require(
            model.term("f0_1710_00").coupling.mode
                == ctpwa::CouplingMode::PositiveReal,
            "positive-parity phase-reference coupling mismatch");
        require(
            model.term("eta_1760_11").coupling.reference
                == ctpwa::CouplingReference::ScaleAndPhase,
            "global reference coupling mismatch");
        require(
            model.resonance("f2_1565").propagator
                == "subtracted_effective_flatte",
            "f2(1565) propagator mismatch");
        require(
            model.resonance("f2_1810").propagator
                == "two_body_running_bw",
            "f2(1810) propagator mismatch");
        require(
            model.term("f2_1565_02_u1").wave == "gvv.tensor_02_u1"
                && model.term("f2_1565_02_u2").wave == "gvv.tensor_02_u2"
                && model.term("f2_1565_02_u3").wave == "gvv.tensor_02_u3",
            "f2(1565) LS=02 wave assignment mismatch");
        require(
            model.term("f2_1810_02_u1").wave == "gvv.tensor_02_u1"
                && model.term("f2_1810_02_u2").wave == "gvv.tensor_02_u2"
                && model.term("f2_1810_02_u3").wave == "gvv.tensor_02_u3",
            "f2(1810) LS=02 wave assignment mismatch");
        require(!model.term("X_2370_11").active, "inactive term mismatch");
        require(!model.canonical_json.empty(), "canonical JSON is empty");

        require_invalid(
            R"({
              "schema_version": 1,
              "process": "test_process",
              "resonances": [
                {"id":"r","propagator":"nonresonant","parameters":{}},
                {"id":"r","propagator":"nonresonant","parameters":{}}
              ],
              "terms": [{
                "id":"t","wave":"test.wave",
                "coupling":{"mode":"fixed_complex","reference":"scale_and_phase","initial":[1,0]},
                "dynamics":{}
              }]
            })",
            "duplicate resonance id");

        require_invalid(
            R"({
              "schema_version": 1,
              "process": "test_process",
              "resonances": [
                {"id":"r","propagator":"nonresonant","parameters":{}}
              ],
              "terms": [{
                "id":"t","wave":"test.wave",
                "coupling":{"mode":"complex_cartesian","initial":[1,0]},
                "dynamics":{}
              }]
            })",
            "exactly one scale_and_phase reference");

        require_invalid(
            R"({
              "schema_version": 1,
              "process": "test_process",
              "resonances": [
                {"id":"r","propagator":"nonresonant","parameters":{},"widht":1}
              ],
              "terms": [{
                "id":"t","wave":"test.wave",
                "coupling":{"mode":"fixed_complex","reference":"scale_and_phase","initial":[1,0]},
                "dynamics":{}
              }]
            })",
            "unknown field");

        require_invalid(
            R"({
              "schema_version": 1,
              "process": "test_process",
              "resonances": [
                {"id":"r","propagator":"nonresonant","parameters":{}}
              ],
              "terms": [{
                "id":"t","wave":"test.wave",
                "coupling":{"mode":"fixed_complex","reference":"scale_and_phase","initial":[0,0]},
                "dynamics":{}
              }]
            })",
            "reference amplitude must be nonzero");

        std::cout << "Runtime model-definition tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << "Model-definition test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

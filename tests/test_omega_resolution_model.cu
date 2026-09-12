// Process parameter validation, legacy compatibility and saved-fit recovery.
#include "process/ModelCompiler.h"
#include "process/ParameterMapping.h"
#include "framework/fit/FitState.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
using Json = nlohmann::json;
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
GVVCompiledModel compile(const Json& document)
{
    return gvv_compile_model(ctpwa::parse_model_definition(document.dump(), "resolution-test"));
}
void invalid(const Json& document)
{
    try { (void)compile(document); }
    catch (const std::exception&) { return; }
    throw std::runtime_error("invalid omega resolution configuration was accepted");
}
int main()
{
    const std::string path = "/tmp/gvv-omega-state-" + std::to_string(getpid()) + ".json";
    try {
        const auto definition = ctpwa::load_model_definition("config/model.json");
        const Json nominal = Json::parse(definition.canonical_json);
        auto model = compile(nominal);
        const auto layout = gvv_fit_parameter_layout(model);
        const auto specs = gvv_fit_parameter_specs(layout);
        require(specs.size() == 36 && specs.back().name == "log_sigma_omega",
                "nominal layout must append exactly one shared sigma");
        require(std::fabs(std::exp(specs.back().initial_value) - 0.005) < 1.e-15,
                "sigma initial value has incorrect units or transform");
        Json legacy = nominal;
        legacy.erase("process_parameters");
        const auto old = compile(legacy);
        require(old.omega_resolution_sigma == 0.0 && gvv_fit_parameter_layout(old).size() == 35,
                "missing process parameter must preserve legacy layout");
        require(gvv_model_implementation_signature(old.definition) == gvv_amplitude_implementation_signature()
                    && gvv_model_signature(old.definition) != gvv_model_signature(model.definition),
                "new and legacy implementation signatures were not distinguished");
        const auto old_specs = gvv_fit_parameter_specs(gvv_fit_parameter_layout(old));
        for (std::size_t i=0; i<old_specs.size(); ++i) {
            require(old_specs[i].name == specs[i].name
                        && old_specs[i].initial_value == specs[i].initial_value,
                    "existing parameter order or values changed");
        }
        for (double sigma : {0.0, 0.007}) {
            Json fixed = nominal;
            fixed["process_parameters"]["omega_resolution_sigma"] = {
                {"value", sigma}, {"fixed", true}, {"transform", "identity"}};
            const auto state = compile(fixed);
            require(state.omega_resolution_sigma == sigma && gvv_fit_parameter_layout(state).size() == 35,
                    "fixed sigma added a free parameter");
        }
        Json bad = nominal;
        bad["process_parameters"]["unknown"] = bad["process_parameters"]["omega_resolution_sigma"];
        invalid(bad);
        for (double value : {-0.001, 0.0, 0.051}) {
            bad = nominal; bad["process_parameters"]["omega_resolution_sigma"]["value"] = value; invalid(bad);
        }
        bad = nominal; bad["process_parameters"]["omega_resolution_sigma"]["transform"] = "identity"; invalid(bad);
        bad = nominal; bad["process_parameters"]["omega_resolution_sigma"].erase("bounds"); invalid(bad);
        bad = nominal; bad["process_parameters"]["omega_resolution_sigma"]["bounds"] = {-10.0, -2.0}; invalid(bad);

        Json maximum = nominal;
        maximum["process_parameters"]["omega_resolution_sigma"]["bounds"] = {-10.0, std::log(0.05)};
        (void)compile(maximum); // exp(log(0.05)) may round just above 0.05.

        ctpwa::FitState saved;
        saved.output_tag = "resolution-test";
        saved.model_json = definition.canonical_json;
        saved.model_name = definition.name;
        saved.model_definition_signature = ctpwa::model_definition_signature(definition);
        saved.model_implementation_signature = gvv_model_implementation_signature(definition);
        saved.model_signature = gvv_model_signature(definition);
        saved.parameters = specs;
        for (const auto& spec : specs) saved.best.values.push_back(spec.initial_value);
        saved.best.initial_values = saved.best.values;
        saved.best.values.back() = std::log(0.007);
        saved.best.errors.assign(specs.size(), 0.02);
        saved.best.covariance.assign(specs.size()*specs.size(), 0.0);
        for (std::size_t i=0; i<specs.size(); ++i) saved.best.covariance[i*specs.size()+i] = 0.0004;
        saved.best.minimum = 1.0; saved.best.edm = 1.e-6;
        saved.best.valid = true;
        ctpwa::write_fit_state(path, saved);
        const auto read = ctpwa::read_fit_state(path);
        auto restored = gvv_compile_model(ctpwa::parse_model_definition(read.model_json, "restored"));
        const auto restored_layout = gvv_fit_parameter_layout(restored);
        gvv_apply_fit_parameters(restored, restored_layout, read.best.values);
        require(std::fabs(restored.omega_resolution_sigma-0.007)<1.e-15
                    && read.best.covariance == saved.best.covariance
                    && read.model_signature == gvv_model_signature(restored.definition),
                "saved fit did not recover sigma, covariance or signature");
        // Sigma has no binding into any X propagator or complex coupling.
        for (std::size_t i=0; i<model.resonances.size(); ++i) {
            for (const auto& parameter : model.resonance_metadata[i].parameters) {
                require(gvv_propagator_parameter_value(model.resonances[i], parameter.target)
                            == gvv_propagator_parameter_value(restored.resonances[i], parameter.target),
                        "varying sigma changed an X propagator parameter");
            }
        }
        for (std::size_t i=0; i<model.initial_couplings.size(); ++i) {
            require(std::fabs(model.initial_couplings[i].real-restored.initial_couplings[i].real)<1.e-15
                        && std::fabs(model.initial_couplings[i].imag-restored.initial_couplings[i].imag)<1.e-15,
                    "varying sigma changed a coupling");
        }
        auto invalid_values = read.best.values;
        invalid_values.back() = std::log(0.06);
        bool rejected = false;
        try { gvv_apply_fit_parameters(restored, restored_layout, invalid_values); }
        catch (const std::exception&) { rejected = true; }
        require(rejected, "out-of-range restored sigma was accepted");
        gvv_apply_fit_parameters(restored, restored_layout, read.best.values);
        std::ostringstream details;
        gvv_write_fit_details(details, restored, restored_layout, read.best);
        require(details.str().find("log_sigma_omega") != std::string::npos,
                "human-readable result omitted sigma");
        std::remove(path.c_str());
        std::cout << "Omega resolution model/state tests passed\n";
    } catch (const std::exception& error) {
        std::remove(path.c_str());
        std::cerr << error.what() << '\n';
        return 1;
    }
}

#include "framework/fit/FitState.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

nlohmann::json read_json(const char* file_name)
{
    std::ifstream input(file_name);
    nlohmann::json document;
    input >> document;
    return document;
}

void write_json(const char* file_name, const nlohmann::json& document)
{
    std::ofstream output(file_name, std::ios::trunc);
    output << document.dump(2) << '\n';
}

void require_invalid(
    const char* file_name,
    const nlohmann::json& document,
    const std::string& expected)
{
    write_json(file_name, document);
    try {
        (void)ctpwa::read_fit_state(file_name);
    } catch (const std::exception& error) {
        require(
            std::string(error.what()).find(expected) != std::string::npos,
            "fit-state diagnostic did not contain expected text");
        return;
    }
    throw std::runtime_error("invalid fit state was accepted");
}

} // namespace

int main()
{
    const char* file_name = "/tmp/gvv-test-fit-state.json";
    try {
        ctpwa::FitState source;
        source.output_tag = "unit";
        source.fit_config_file = "config/fit.json";
        source.model_config_file = "config/model.json";
        source.model_name = "unit model";
        source.model_signature = "fnv1a64:0123456789abcdef";
        ctpwa::FitParameterSpec parameter;
        parameter.name = "x";
        parameter.initial_value = 1.0;
        parameter.step = 0.1;
        parameter.has_lower_bound = true;
        parameter.lower_bound = -2.0;
        source.parameters.push_back(parameter);
        ctpwa::FitParameterSpec second = parameter;
        second.name = "y";
        second.initial_value = -0.5;
        second.has_lower_bound = false;
        second.has_upper_bound = true;
        second.upper_bound = 2.0;
        source.parameters.push_back(second);
        source.best.start_index = 2;
        source.best.seed = 11;
        source.best.initial_values = {1.0, -0.5};
        source.best.values = {1.25, -0.45};
        source.best.errors = {0.2, 0.3};
        source.best.covariance = {0.04, 0.01, 0.01, 0.09};
        source.best.minimum = 12.5;
        source.best.edm = 1.0e-5;
        source.best.error_definition = 0.5;
        source.best.migrad_status = 0;
        source.best.hesse_status = 0;
        source.best.covariance_status = 3;
        source.best.elapsed_seconds = 0.25;
        source.best.valid = true;

        ctpwa::write_fit_state(file_name, source);
        const ctpwa::FitState restored = ctpwa::read_fit_state(file_name);
        require(restored.output_tag == "unit", "output tag was not restored");
        require(restored.parameters.size() == 2, "parameter count changed");
        require(restored.parameters[0].name == "x", "parameter name changed");
        require(std::fabs(restored.best.values[0] - 1.25) < 1.0e-12,
                "parameter value changed");
        require(std::fabs(restored.best.covariance[0] - 0.04) < 1.0e-12,
                "covariance changed");

        const nlohmann::json valid = read_json(file_name);
        nlohmann::json invalid = valid;
        invalid["output_tag"] = "../unit";
        require_invalid(file_name, invalid, "unsafe characters");

        invalid = valid;
        invalid["parameters"][1]["name"] = "x";
        require_invalid(file_name, invalid, "duplicate fit-state parameter");

        invalid = valid;
        invalid["covariance"][0][1] = 0.02;
        require_invalid(file_name, invalid, "not symmetric");

        invalid = valid;
        invalid["covariance"][1][1] = -0.01;
        require_invalid(file_name, invalid, "negative diagonal");

        std::remove(file_name);
        std::cout << "FitState round-trip test passed\n";
    } catch (const std::exception& error) {
        std::remove(file_name);
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}

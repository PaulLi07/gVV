#include "framework/fit/FitState.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
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
        source.best.start_index = 2;
        source.best.seed = 11;
        source.best.initial_values = {1.0};
        source.best.values = {1.25};
        source.best.errors = {0.2};
        source.best.covariance = {0.04};
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
        require(restored.parameters.size() == 1, "parameter count changed");
        require(restored.parameters[0].name == "x", "parameter name changed");
        require(std::fabs(restored.best.values[0] - 1.25) < 1.0e-12,
                "parameter value changed");
        require(std::fabs(restored.best.covariance[0] - 0.04) < 1.0e-12,
                "covariance changed");
        std::remove(file_name);
        std::cout << "FitState round-trip test passed\n";
    } catch (const std::exception& error) {
        std::remove(file_name);
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}

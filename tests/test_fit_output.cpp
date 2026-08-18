#include "framework/fit/FitOutput.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

int main()
{
    const char* file_name = "/tmp/gvv-test-fit-report.txt";
    try {
        ctpwa::FitParameterSpec parameter;
        parameter.name = "x";
        parameter.initial_value = 0.5;
        parameter.step = 0.1;
        ctpwa::FitAttempt attempt;
        attempt.start_index = 0;
        attempt.seed = 7;
        attempt.initial_values = {0.5};
        attempt.values = {1.0};
        attempt.errors = {0.2};
        attempt.covariance = {0.04};
        attempt.minimum = -12.0;
        attempt.edm = 1.0e-6;
        attempt.migrad_status = 0;
        attempt.hesse_status = 0;
        attempt.covariance_status = 3;
        attempt.elapsed_seconds = 0.1;
        attempt.valid = true;
        ctpwa::FitSummary summary;
        summary.best = attempt;
        summary.attempts.push_back(attempt);
        ctpwa::FitResultContext context;
        context.output_tag = "unit";
        context.fit_config_file = "config/fit.json";
        context.model_config_file = "config/model.json";
        context.model_name = "unit model";
        context.model_signature = "fnv1a64:0123456789abcdef";
        context.samples.push_back({"data", "data", "data.root", 10, 1.0});

        ctpwa::write_fit_result(
            file_name, summary, ctpwa::FitOptions(), {parameter}, context,
            [](std::ostream& output) { output << "term active_unit\n"; });
        std::ifstream input(file_name);
        std::ostringstream text;
        text << input.rdbuf();
        const std::string report = text.str();
        const char* required[] = {
            "[multistart_attempts]", "[best_fit]", "[free_parameters]",
            "[active_physical_model]", "term active_unit",
            "[covariance_matrix]", "[correlation_matrix]"};
        for (const char* marker : required) {
            if (report.find(marker) == std::string::npos) {
                throw std::runtime_error(
                    std::string("fit report is missing ") + marker);
            }
        }
        std::remove(file_name);
        std::cout << "FitOutput report test passed\n";
    } catch (const std::exception& error) {
        std::remove(file_name);
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}

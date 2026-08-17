#include "framework/fit/FitParameters.h"
#include "process/TermEvaluator.cuh"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

bool close(double first, double second)
{
    return std::fabs(first - second) < 1.0e-12;
}

} // namespace

int main(int argc, char* argv[])
{
    const GVVCompiledModel compiled =
        gvv_load_compiled_model("config/model.json");
    const std::vector<GVVFitParameterSpec> layout =
        gvv_fit_parameter_layout(compiled);
    const std::vector<std::string> names =
        gvv_fit_parameter_names(compiled);
    const std::vector<double> nominal =
        gvv_fit_parameters_from_model(compiled);
    if (layout.size() != 12 || names.size() != layout.size()
        || nominal.size() != layout.size()
        || names[0] != "Re_f0_1500_00"
        || names[2] != "log_rho_f0_1710_00"
        || names[11] != "log_Romega_f0_1500"
        || !layout[11].has_lower_bound
        || layout[11].lower_bound != -6.0
        || layout[11].upper_bound != 3.0) {
        std::cerr << "runtime GVV fit parameter layout is wrong\n";
        return 1;
    }

    GVVCompiledModel round_trip = compiled;
    std::vector<double> changed = nominal;
    changed[0] = 0.37;
    changed[1] = -0.29;
    changed[2] = std::log(0.42);
    changed[11] = std::log(1.7);
    gvv_apply_fit_parameters_to_model(round_trip, changed);
    const std::vector<double> reconstructed =
        gvv_fit_parameters_from_model(round_trip);
    for (std::size_t index = 0; index < changed.size(); ++index) {
        if (!close(reconstructed[index], changed[index])) {
            std::cerr << "runtime parameter/model round trip failed\n";
            return 2;
        }
    }

    const GVVParsedFitResult legacy = gvv_read_fit_result(
        "tests/fit_result_legacy_fixture.txt", compiled);
    if (legacy.used_machine_readable_rows
        || legacy.values.size() != 12
        || !close(legacy.values[0], 2.0)
        || !close(legacy.values[2], std::log(3.0))
        || !close(legacy.values[11], std::log(0.4))
        || !close(legacy.errors[11], 0.2)) {
        std::cerr << "legacy fit-result parser is wrong\n";
        return 3;
    }
    const GVVParsedFitResult machine = gvv_read_fit_result(
        "tests/fit_result_machine_fixture.txt", compiled);
    if (!machine.used_machine_readable_rows
        || machine.parameter_names != names
        || machine.values.size() != legacy.values.size()) {
        std::cerr << "machine-readable fit-result parser is wrong\n";
        return 4;
    }
    for (std::size_t index = 0; index < machine.values.size(); ++index) {
        if (!close(machine.values[index], legacy.values[index])) {
            std::cerr << "machine/legacy fit-result parsers disagree\n";
            return 5;
        }
    }

    std::vector<double> identity(layout.size() * layout.size(), 0.0);
    for (std::size_t index = 0; index < layout.size(); ++index) {
        identity[index * layout.size() + index] = 1.0;
    }
    gvv_validate_covariance_matrix(identity, static_cast<int>(layout.size()));
    identity[1] = 0.5;
    bool rejected = false;
    try {
        gvv_validate_covariance_matrix(
            identity, static_cast<int>(layout.size()));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    if (!rejected) {
        std::cerr << "asymmetric covariance was not rejected\n";
        return 6;
    }

    if (argc == 2) {
        const GVVParsedFitResult external =
            gvv_read_fit_result(argv[1], compiled);
        if (external.values.size() != names.size()
            || external.parameter_names != names) {
            std::cerr << "external fit result does not match model layout\n";
            return 7;
        }
    } else if (argc > 2) {
        std::cerr << "Usage: " << argv[0] << " [fit_result.txt]\n";
        return 8;
    }

    const int number_terms = static_cast<int>(compiled.terms.size());
    const int number_pairs = ctpwa::component_pair_count(number_terms);
    std::vector<bool> seen(number_pairs, false);
    for (int first = 0; first < number_terms; ++first) {
        for (int second = first; second < number_terms; ++second) {
            const int index = ctpwa::component_pair_index(
                first, second, number_terms);
            if (index < 0 || index >= number_pairs || seen[index]) {
                std::cerr << "component-pair compact index is wrong\n";
                return 9;
            }
            seen[index] = true;
        }
    }

    std::cout << "GVV fit-parameter tests passed\n";
    return 0;
}

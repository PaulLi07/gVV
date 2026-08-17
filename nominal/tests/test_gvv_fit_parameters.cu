#include "../include/GVVFitParameters.h"
#include "../include/kernel.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char* argv[])
{
    const GVVCompiledModel compiled =
        gvv_load_compiled_model("config/model.json");
    const std::vector<GVVFitParameterSpec> runtime_layout =
        gvv_fit_parameter_layout(compiled);
    const std::vector<std::string> runtime_names =
        gvv_fit_parameter_names(compiled);
    const std::vector<double> runtime_values =
        gvv_fit_parameters_from_model(compiled);
    if (runtime_layout.size() != 12
        || runtime_names.size() != runtime_layout.size()
        || runtime_values.size() != runtime_layout.size()
        || runtime_names[0] != "Re_f0_1500_00"
        || runtime_names[2] != "log_rho_f0_1710_00"
        || runtime_names[11] != "log_Romega_f0_1500"
        || !runtime_layout[11].has_lower_bound
        || runtime_layout[11].lower_bound != -6.0
        || runtime_layout[11].upper_bound != 3.0) {
        std::cerr << "runtime GVV fit parameter layout is wrong\n";
        return 10;
    }
    GVVCompiledModel runtime_round_trip = compiled;
    std::vector<double> changed = runtime_values;
    changed[0] = 0.37;
    changed[1] = -0.29;
    changed[2] = std::log(0.42);
    changed[11] = std::log(1.7);
    gvv_apply_fit_parameters_to_model(runtime_round_trip, changed);
    const std::vector<double> reconstructed_runtime =
        gvv_fit_parameters_from_model(runtime_round_trip);
    for (std::size_t index = 0; index < changed.size(); ++index) {
        if (std::fabs(reconstructed_runtime[index] - changed[index])
            > 1.0e-12) {
            std::cerr << "runtime parameter/model round trip failed\n";
            return 11;
        }
    }

    const GVVFitState defaults = gvv_default_fit_state();
    const std::vector<std::string> names =
        gvv_fit_parameter_names(defaults.resonances);
    if (names.size() != 12
        || names[0] != "Re_f0_1500_00"
        || names[2] != "log_rho_f0_1710_00"
        || names[11] != "log_Romega_f0_1500") {
        std::cerr << "GVV fit parameter layout is wrong\n";
        return 1;
    }

    const GVVParsedFitResult legacy =
        gvv_read_fit_result("tests/fit_result_legacy_fixture.txt");
    if (legacy.used_machine_readable_rows
        || legacy.values.size() != 12
        || std::fabs(legacy.values[0] - 2.0) > 1.0e-12
        || std::fabs(legacy.values[2] - std::log(3.0)) > 1.0e-12
        || std::fabs(legacy.values[11] - std::log(0.4)) > 1.0e-12
        || std::fabs(legacy.errors[11] - 0.2) > 1.0e-12) {
        std::cerr << "legacy fit-result parser is wrong\n";
        return 2;
    }
    const GVVParsedFitResult machine =
        gvv_read_fit_result("tests/fit_result_machine_fixture.txt");
    if (!machine.used_machine_readable_rows
        || machine.parameter_names != names
        || machine.values.size() != legacy.values.size()) {
        std::cerr << "machine-readable fit-result parser is wrong\n";
        return 3;
    }
    for (std::size_t index = 0; index < machine.values.size(); ++index) {
        if (std::fabs(machine.values[index] - legacy.values[index]) > 1.0e-12) {
            std::cerr << "machine/legacy fit-result parsers disagree\n";
            return 4;
        }
    }

    GVVFitState round_trip = gvv_default_fit_state();
    gvv_apply_fit_parameters_to_state(round_trip, legacy.values);
    const std::vector<double> reconstructed =
        gvv_fit_parameters_from_state(round_trip);
    for (std::size_t index = 0; index < reconstructed.size(); ++index) {
        if (std::fabs(reconstructed[index] - legacy.values[index]) > 1.0e-12) {
            std::cerr << "fit parameter state round trip failed\n";
            return 5;
        }
    }

    std::vector<double> identity(12 * 12, 0.0);
    for (int index = 0; index < 12; ++index) {
        identity[index * 12 + index] = 1.0;
    }
    gvv_validate_covariance_matrix(identity, 12);
    identity[1] = 0.5;
    bool rejected = false;
    try {
        gvv_validate_covariance_matrix(identity, 12);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    if (!rejected) {
        std::cerr << "asymmetric covariance was not rejected\n";
        return 6;
    }

    if (argc == 2) {
        const GVVParsedFitResult external = gvv_read_fit_result(argv[1]);
        if (external.values.size() != names.size()
            || external.parameter_names != names) {
            std::cerr << "external fit result does not match the GVV layout\n";
            return 7;
        }
    } else if (argc > 2) {
        std::cerr << "Usage: " << argv[0] << " [fit_result.txt]\n";
        return 8;
    }

    bool seen[GVV_NCOMPONENT_PAIRS] = {false};
    for (int first = 0; first < GVV_NTERMS; ++first) {
        for (int second = first; second < GVV_NTERMS; ++second) {
            const int index = gvv_component_pair_index(first, second);
            if (index < 0 || index >= GVV_NCOMPONENT_PAIRS || seen[index]) {
                std::cerr << "component-pair compact index is wrong\n";
            return 12;
            }
            seen[index] = true;
        }
    }

    std::cout << "GVV fit-parameter tests passed\n";
    return 0;
}

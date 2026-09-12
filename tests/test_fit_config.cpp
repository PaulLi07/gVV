// Contract test with controlled minimizer/output settings, independent of user tuning.
#include "framework/fit/FitConfig.h"

#include <iostream>
#include <fstream>
#include <cstdio>
#include <unistd.h>
#include <nlohmann/json.hpp>

int main()
{
    std::ifstream input("config/fit.json");
    nlohmann::json fixture;
    input >> fixture;
    fixture["minimizer"]["n_starts"] = 7;
    fixture["output"]["tag"] = "omega_config_test";
    const std::string path = "/tmp/gvv-fit-config-" + std::to_string(getpid()) + ".json";
    { std::ofstream output(path); output << fixture.dump(); }
    ctpwa::FitRunConfig config;
    try { config = ctpwa::load_fit_run_config(path); }
    catch (...) { std::remove(path.c_str()); throw; }
    std::remove(path.c_str());
    if (config.schema_version != 1
        || config.model_file != "config/model.json"
        || config.inputs.backgrounds.size() != 2
        || config.inputs.backgrounds[0].likelihood_coefficient != -0.5
        || config.inputs.backgrounds[1].likelihood_coefficient != 0.25
        || config.minimizer.number_starts != 7
        || config.output.result_file()
               != "results/fit_result-omega_config_test.txt"
        || config.output.state_file()
               != "results/fit_state-omega_config_test.json"
        || config.output.projection_file()
               != "results/projection-omega_config_test.root"
        || config.output.log_file() != "runlog/fit-omega_config_test.log") {
        std::cerr << "fit configuration contract is wrong\n";
        return 1;
    }
    std::cout << "Fit configuration tests passed\n";
    return 0;
}

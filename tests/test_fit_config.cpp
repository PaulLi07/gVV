// Contract test for the nominal fit.json and unified output-name derivation.
#include "framework/fit/FitConfig.h"

#include <iostream>

int main()
{
    const ctpwa::FitRunConfig config =
        ctpwa::load_fit_run_config("config/fit.json");
    if (config.schema_version != 1
        || config.model_file != "config/model.json"
        || config.inputs.backgrounds.size() != 2
        || config.inputs.backgrounds[0].likelihood_coefficient != -0.5
        || config.inputs.backgrounds[1].likelihood_coefficient != 0.25
        || config.minimizer.number_starts != 10
        || config.output.result_file()
               != "results/fit_result-initial.txt"
        || config.output.state_file()
               != "results/fit_state-initial.json"
        || config.output.projection_file()
               != "results/projection-initial.root"
        || config.output.log_file() != "runlog/fit-initial.log") {
        std::cerr << "fit configuration contract is wrong\n";
        return 1;
    }
    std::cout << "Fit configuration tests passed\n";
    return 0;
}

#ifndef CTPWA_FRAMEWORK_FIT_CONFIG_H
#define CTPWA_FRAMEWORK_FIT_CONFIG_H

#include "framework/fit/FitEngine.h"

#include <string>
#include <vector>

namespace ctpwa {

struct WeightedSampleConfig {
    std::string label;
    std::string file;
    double likelihood_coefficient = 0.0;
};

struct FitInputConfig {
    std::string data_file;
    std::string normalization_mc_file;
    std::vector<WeightedSampleConfig> backgrounds;
};

struct FitOutputConfig {
    std::string directory = "results";
    std::string log_directory = "runlog";
    std::string tag = "initial";

    std::string result_file() const;
    std::string covariance_file() const;
    std::string projection_file() const;
    std::string log_file() const;
};

struct FitRunConfig {
    int schema_version = 1;
    std::string source_file;
    std::string model_file;
    FitInputConfig inputs;
    FitOptions minimizer;
    FitOutputConfig output;
};

FitRunConfig load_fit_run_config(const std::string& file_name);

} // namespace ctpwa

#endif // CTPWA_FRAMEWORK_FIT_CONFIG_H

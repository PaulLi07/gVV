#include "framework/fit/FitOutput.h"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace ctpwa {

void write_fit_result(
    const std::string& file_name,
    const FitAttempt& best,
    const FitOptions& options,
    const std::vector<FitParameterSpec>& parameters,
    const FitResultContext& context,
    const FitDetailWriter& write_details)
{
    if (!best.valid || best.values.size() != parameters.size()
        || best.errors.size() != parameters.size()) {
        throw std::invalid_argument(
            "cannot write an invalid or inconsistent fit result");
    }
    std::ofstream output(file_name.c_str(), std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot write fit result: " + file_name);
    }
    output << std::setprecision(12);
    output << "# ctpwa_fit_result format_version 1\n";
    output << "# fit_config " << context.fit_config_file << '\n';
    output << "# model_config " << context.model_config_file << '\n';
    output << "# model_name " << context.model_name << '\n';
    output << "# multistart n_starts " << options.number_starts
           << " base_seed " << options.base_seed
           << " best_start " << best.start_index
           << " best_seed " << best.seed
           << " elapsed_seconds " << best.elapsed_seconds
           << " at_parameter_boundary "
           << (best.at_parameter_boundary ? 1 : 0) << '\n';
    output << "# samples data " << context.data_entries
           << " normalization_mc " << context.normalization_mc_entries
           << '\n';
    output << "# minimization migrad_status " << best.migrad_status
           << " hesse_status " << best.hesse_status
           << " covariance_status " << best.covariance_status
           << " minimum " << best.minimum
           << " edm " << best.edm
           << " error_definition " << best.error_definition << '\n';
    output << "# parameter index name value error\n";
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        output << "parameter " << index << ' ' << parameters[index].name
               << ' ' << best.values[index]
               << ' ' << best.errors[index] << '\n';
    }
    if (write_details) write_details(output);
    output.close();
    if (!output) {
        throw std::runtime_error("failed to write fit result: " + file_name);
    }
}

void write_covariance_matrix(
    const std::string& file_name,
    const FitAttempt& best)
{
    const int number_parameters = static_cast<int>(best.values.size());
    if (!best.valid || number_parameters <= 0
        || static_cast<int>(best.covariance.size())
               != number_parameters * number_parameters) {
        throw std::invalid_argument("invalid covariance matrix dimensions");
    }
    std::ofstream output(file_name.c_str(), std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "cannot write covariance matrix: " + file_name);
    }
    output << std::setprecision(12);
    for (int row = 0; row < number_parameters; ++row) {
        for (int column = 0; column < number_parameters; ++column) {
            if (column != 0) output << ' ';
            output << best.covariance[
                static_cast<std::size_t>(row) * number_parameters + column];
        }
        output << '\n';
    }
    output.close();
    if (!output) {
        throw std::runtime_error(
            "failed to write covariance matrix: " + file_name);
    }
}

} // namespace ctpwa

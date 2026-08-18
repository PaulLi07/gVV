// Complete user-facing diagnostics for one fit. This format is intentionally
// optimized for reading, while FitState JSON is the stable software contract.
#include "framework/fit/FitOutput.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace ctpwa {
namespace {

const char* yes_no(bool value)
{
    return value ? "yes" : "no";
}

void write_matrix(
    std::ostream& output,
    const char* title,
    const std::vector<double>& matrix,
    const std::vector<FitParameterSpec>& parameters,
    bool correlation)
{
    const std::size_t size = parameters.size();
    output << "\n[" << title << "]\n";
    output << "# rows and columns follow the free-parameter order above\n";
    output << "# index";
    for (std::size_t column = 0; column < size; ++column) {
        output << ' ' << column;
    }
    output << '\n';
    for (std::size_t row = 0; row < size; ++row) {
        output << row;
        for (std::size_t column = 0; column < size; ++column) {
            double value = matrix[row * size + column];
            if (correlation) {
                const double denominator = std::sqrt(std::max(
                    0.0,
                    matrix[row * size + row]
                    * matrix[column * size + column]));
                value = denominator > 0.0 ? value / denominator : 0.0;
            }
            output << ' ' << value;
        }
        output << '\n';
    }
}

} // namespace

void write_fit_result(
    const std::string& file_name,
    const FitSummary& summary,
    const FitOptions& options,
    const std::vector<FitParameterSpec>& parameters,
    const FitResultContext& context,
    const FitDetailWriter& write_details)
{
    const FitAttempt& best = summary.best;
    const std::size_t size = parameters.size();
    if (!best.valid || best.values.size() != size
        || best.errors.size() != size
        || best.initial_values.size() != size
        || best.covariance.size() != size * size) {
        throw std::invalid_argument(
            "cannot write an invalid or inconsistent fit result");
    }
    std::ofstream output(file_name.c_str(), std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot write fit result: " + file_name);
    }
    output << std::setprecision(12);
    output << "GVV COVARIANT-TENSOR PARTIAL-WAVE FIT REPORT\n"
           << "===============================================\n"
           << "This file is for inspection. Post Calculation reads the "
              "matching fit_state JSON.\n\n"
           << "[provenance]\n"
           << "output_tag: " << context.output_tag << '\n'
           << "fit_config: " << context.fit_config_file << '\n'
           << "model_config: " << context.model_config_file << '\n'
           << "model_name: " << context.model_name << '\n'
           << "model_signature: " << context.model_signature << '\n';

    output << "\n[samples]\n"
           << "# role label entries likelihood_coefficient file\n";
    for (const FitSampleSummary& sample : context.samples) {
        output << sample.role << ' ' << sample.label << ' '
               << sample.entries << ' ' << sample.likelihood_coefficient
               << ' ' << sample.file << '\n';
    }

    output << "\n[minimizer_configuration]\n"
           << "n_starts: " << options.number_starts << '\n'
           << "base_seed: " << options.base_seed << '\n'
           << "maximum_edm: " << options.maximum_edm << '\n'
           << "maximum_calls: " << options.maximum_calls << '\n'
           << "tolerance: " << options.tolerance << '\n'
           << "error_definition: " << options.error_definition << '\n'
           << "random_magnitude_range: ["
           << options.random_magnitude_min << ", "
           << options.random_magnitude_max << "]\n";

    output << "\n[multistart_attempts]\n"
           << "# start seed final_nll edm migrad hesse covariance boundary "
              "seconds accepted\n";
    for (const FitAttempt& attempt : summary.attempts) {
        output << attempt.start_index << ' ' << attempt.seed << ' '
               << attempt.minimum << ' ' << attempt.edm << ' '
               << attempt.migrad_status << ' ' << attempt.hesse_status << ' '
               << attempt.covariance_status << ' '
               << yes_no(attempt.at_parameter_boundary) << ' '
               << attempt.elapsed_seconds << ' ' << yes_no(attempt.valid)
               << '\n';
    }

    output << "\n[best_fit]\n"
           << "best_start: " << best.start_index << '\n'
           << "best_seed: " << best.seed << '\n'
           << "minimum_nll: " << best.minimum << '\n'
           << "edm: " << best.edm << '\n'
           << "migrad_status: " << best.migrad_status << '\n'
           << "hesse_status: " << best.hesse_status << '\n'
           << "covariance_status: " << best.covariance_status << '\n'
           << "error_definition: " << best.error_definition << '\n'
           << "at_parameter_boundary: "
           << yes_no(best.at_parameter_boundary) << '\n'
           << "elapsed_seconds: " << best.elapsed_seconds << '\n';

    output << "\n[free_parameters]\n"
           << "# index name initial value error step lower upper\n";
    for (std::size_t index = 0; index < size; ++index) {
        const FitParameterSpec& parameter = parameters[index];
        output << index << ' ' << parameter.name << ' '
               << best.initial_values[index] << ' ' << best.values[index]
               << ' ' << best.errors[index] << ' ' << parameter.step << ' ';
        if (parameter.has_lower_bound) output << parameter.lower_bound;
        else output << "none";
        output << ' ';
        if (parameter.has_upper_bound) output << parameter.upper_bound;
        else output << "none";
        output << '\n';
    }

    if (write_details) {
        output << "\n[active_physical_model]\n";
        write_details(output);
    }
    write_matrix(output, "covariance_matrix", best.covariance, parameters, false);
    write_matrix(output, "correlation_matrix", best.covariance, parameters, true);
    output << "\n[end]\n";
    if (!output) {
        throw std::runtime_error("failed to write fit result: " + file_name);
    }
}

} // namespace ctpwa

#include "framework/fit/FitEngine.h"

#include "TMinuit.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>

namespace ctpwa {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kBoundaryTolerance = 1.0e-6;
constexpr double kNllTieTolerance = 1.0e-8;
constexpr double kFailedObjective = 1.0e100;

const FitObjective* active_objective = nullptr;
std::size_t active_parameter_count = 0;

void minuit_objective(
    Int_t& number_parameters,
    Double_t* gradient,
    Double_t& result,
    Double_t* parameters,
    Int_t flag)
{
    (void)gradient;
    (void)flag;
    if (active_objective == nullptr
        || number_parameters != static_cast<Int_t>(active_parameter_count)) {
        result = kFailedObjective;
        return;
    }
    try {
        const std::vector<double> values(
            parameters, parameters + active_parameter_count);
        result = (*active_objective)(values);
        if (!std::isfinite(result)) {
            result = kFailedObjective;
        }
    } catch (const std::exception& error) {
        std::cerr << "fit objective error: " << error.what() << '\n';
        result = kFailedObjective;
    }
}

void validate_inputs(
    const std::vector<FitParameterSpec>& parameters,
    const FitOptions& options,
    const FitObjective& objective)
{
    if (parameters.empty()) {
        throw std::invalid_argument("fit requires at least one parameter");
    }
    if (!objective) {
        throw std::invalid_argument("fit objective is empty");
    }
    if (options.number_starts <= 0 || options.base_seed < 0
        || !(options.maximum_edm > 0.0) || options.maximum_calls <= 0
        || !(options.tolerance > 0.0)
        || !(options.error_definition > 0.0)
        || !(options.random_magnitude_min > 0.0)
        || !(options.random_magnitude_max
             > options.random_magnitude_min)) {
        throw std::invalid_argument("invalid generic fit options");
    }
    for (const FitParameterSpec& parameter : parameters) {
        if (parameter.name.empty() || !std::isfinite(parameter.initial_value)
            || !(parameter.step > 0.0)
            || parameter.has_lower_bound != parameter.has_upper_bound
            || (parameter.has_lower_bound
                && !(parameter.lower_bound < parameter.upper_bound))) {
            throw std::invalid_argument(
                "invalid fit parameter specification for '"
                + parameter.name + "'");
        }
    }
}

std::vector<double> randomized_start(
    const std::vector<FitParameterSpec>& parameters,
    long long seed,
    double magnitude_min,
    double magnitude_max)
{
    std::vector<double> values;
    values.reserve(parameters.size());
    for (const FitParameterSpec& parameter : parameters) {
        values.push_back(parameter.initial_value);
    }

    std::mt19937_64 generator(
        static_cast<std::mt19937_64::result_type>(seed));
    std::uniform_real_distribution<double> log_magnitude(
        std::log(magnitude_min), std::log(magnitude_max));
    std::uniform_real_distribution<double> phase(-kPi, kPi);

    for (std::size_t cursor = 0; cursor < parameters.size(); ++cursor) {
        const FitParameterSpec& parameter = parameters[cursor];
        if (parameter.randomization
            == ParameterRandomization::LogMagnitude) {
            values[cursor] = log_magnitude(generator);
            continue;
        }
        if (parameter.randomization
            != ParameterRandomization::ComplexReal) {
            continue;
        }
        if (cursor + 1 >= parameters.size()
            || parameters[cursor + 1].randomization
                   != ParameterRandomization::ComplexImaginary
            || parameters[cursor + 1].randomization_group
                   != parameter.randomization_group) {
            throw std::runtime_error(
                "complex randomized parameters must be adjacent pairs");
        }
        const double magnitude = std::exp(log_magnitude(generator));
        const double angle = phase(generator);
        values[cursor] = magnitude * std::cos(angle);
        values[cursor + 1] = magnitude * std::sin(angle);
        ++cursor;
    }
    return values;
}

bool at_boundary(
    const std::vector<FitParameterSpec>& parameters,
    const std::vector<double>& values)
{
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        if (parameters[index].has_lower_bound
            && std::fabs(values[index] - parameters[index].lower_bound)
                   < kBoundaryTolerance) {
            return true;
        }
        if (parameters[index].has_upper_bound
            && std::fabs(values[index] - parameters[index].upper_bound)
                   < kBoundaryTolerance) {
            return true;
        }
    }
    return false;
}

void print_parameters(
    const char* heading,
    const std::vector<FitParameterSpec>& parameters,
    const std::vector<double>& values,
    const std::vector<double>* errors = nullptr)
{
    std::cout << heading << '\n' << std::setprecision(12);
    for (std::size_t index = 0; index < values.size(); ++index) {
        std::cout << "  " << std::setw(26) << std::left
                  << parameters[index].name << std::right << "  "
                  << std::setw(18) << values[index];
        if (errors != nullptr) {
            std::cout << " +/- " << (*errors)[index];
        }
        std::cout << '\n';
    }
}

FitAttempt run_attempt(
    const std::vector<FitParameterSpec>& parameters,
    const FitOptions& options,
    const FitObjective& objective,
    int start_index,
    const std::vector<double>& initial_values)
{
    FitAttempt attempt;
    attempt.start_index = start_index;
    attempt.seed = options.base_seed + start_index;
    attempt.initial_values = initial_values;
    attempt.values.assign(parameters.size(), 0.0);
    attempt.errors.assign(parameters.size(), 0.0);
    attempt.covariance.assign(
        parameters.size() * parameters.size(), 0.0);

    std::cout << "\n============================================================\n"
              << "MULTISTART FIT " << start_index
              << "  seed=" << attempt.seed << '\n'
              << "============================================================\n";
    print_parameters("Initial free parameters:", parameters, initial_values);

    const auto start_time = std::chrono::steady_clock::now();
    TMinuit minuit(static_cast<Int_t>(parameters.size()));
    minuit.SetFCN(minuit_objective);
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        const FitParameterSpec& parameter = parameters[index];
        minuit.DefineParameter(
            static_cast<Int_t>(index),
            parameter.name.c_str(),
            initial_values[index],
            parameter.step,
            parameter.has_lower_bound ? parameter.lower_bound : 0.0,
            parameter.has_upper_bound ? parameter.upper_bound : 0.0);
    }

    active_objective = &objective;
    active_parameter_count = parameters.size();
    Int_t status = 0;
    Double_t arguments[2] = {options.error_definition, 0.0};
    minuit.mnexcm("SET ERR", arguments, 1, status);
    arguments[0] = options.maximum_calls;
    arguments[1] = options.tolerance;
    minuit.mnexcm("MIGRAD", arguments, 2, status);
    attempt.migrad_status = status;
    minuit.mnexcm("HESSE", arguments, 0, status);
    attempt.hesse_status = status;
    active_objective = nullptr;
    active_parameter_count = 0;

    Int_t number_variable_parameters = 0;
    Int_t number_parameters = 0;
    minuit.mnstat(
        attempt.minimum,
        attempt.edm,
        attempt.error_definition,
        number_variable_parameters,
        number_parameters,
        attempt.covariance_status);
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        minuit.GetParameter(
            static_cast<Int_t>(index),
            attempt.values[index],
            attempt.errors[index]);
    }
    minuit.mnemat(
        attempt.covariance.data(), static_cast<Int_t>(parameters.size()));
    attempt.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time).count();
    attempt.at_parameter_boundary = at_boundary(parameters, attempt.values);
    attempt.valid = attempt.migrad_status == 0
                    && attempt.hesse_status == 0
                    && attempt.covariance_status >= 2
                    && std::isfinite(attempt.minimum)
                    && std::isfinite(attempt.edm)
                    && attempt.edm <= options.maximum_edm;

    minuit.mnprin(3, attempt.minimum);
    print_parameters(
        "Final free parameters:", parameters, attempt.values, &attempt.errors);
    std::cout << std::setprecision(12)
              << "Start summary: index=" << attempt.start_index
              << " seed=" << attempt.seed
              << " NLL=" << attempt.minimum
              << " MIGRAD=" << attempt.migrad_status
              << " HESSE=" << attempt.hesse_status
              << " covariance_status=" << attempt.covariance_status
              << " EDM=" << attempt.edm
              << " at_boundary="
              << (attempt.at_parameter_boundary ? "yes" : "no")
              << " elapsed_seconds=" << attempt.elapsed_seconds
              << " accepted=" << (attempt.valid ? "yes" : "no") << '\n';
    return attempt;
}

bool is_better(const FitAttempt& candidate, const FitAttempt& best)
{
    if (!candidate.valid) return false;
    if (!best.valid) return true;
    if (candidate.minimum < best.minimum - kNllTieTolerance) return true;
    if (std::fabs(candidate.minimum - best.minimum) <= kNllTieTolerance) {
        if (candidate.covariance_status != best.covariance_status) {
            return candidate.covariance_status > best.covariance_status;
        }
        if (candidate.edm != best.edm) return candidate.edm < best.edm;
        return candidate.start_index < best.start_index;
    }
    return false;
}

} // namespace

FitSummary run_multistart_fit(
    const std::vector<FitParameterSpec>& parameters,
    const FitOptions& options,
    const FitObjective& objective)
{
    validate_inputs(parameters, options, objective);
    FitSummary summary;
    summary.attempts.reserve(options.number_starts);
    for (int start = 0; start < options.number_starts; ++start) {
        std::vector<double> initial_values;
        if (start == 0) {
            for (const FitParameterSpec& parameter : parameters) {
                initial_values.push_back(parameter.initial_value);
            }
        } else {
            initial_values = randomized_start(
                parameters,
                options.base_seed + start,
                options.random_magnitude_min,
                options.random_magnitude_max);
        }
        FitAttempt attempt = run_attempt(
            parameters, options, objective, start, initial_values);
        if (is_better(attempt, summary.best)) {
            summary.best = attempt;
            std::cout << "Start " << start
                      << " is the current best converged solution\n";
        }
        summary.attempts.push_back(std::move(attempt));
    }
    return summary;
}

} // namespace ctpwa

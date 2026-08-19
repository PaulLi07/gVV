// JSON serialization for the stable Fit -> Post Calculation boundary.
#include "framework/fit/FitState.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <unordered_set>

namespace ctpwa {
namespace {

using Json = nlohmann::json;

Json optional_bound(bool present, double value)
{
    return present ? Json(value) : Json(nullptr);
}

double required_number(const Json& value, const std::string& path)
{
    if (!value.is_number()) {
        throw std::runtime_error("fit state " + path + " must be a number");
    }
    const double result = value.get<double>();
    if (!std::isfinite(result)) {
        throw std::runtime_error("fit state " + path + " must be finite");
    }
    return result;
}

void require_safe_output_tag(const std::string& value)
{
    static const std::regex safe_tag("^[A-Za-z0-9][A-Za-z0-9._-]*$");
    if (!std::regex_match(value, safe_tag)) {
        throw std::runtime_error(
            "fit state output_tag contains unsafe characters");
    }
}

bool covariance_entries_match(double first, double second)
{
    const double scale = std::max(
        1.0, std::max(std::fabs(first), std::fabs(second)));
    return std::fabs(first - second) <= 1.0e-12 * scale;
}

} // namespace

void write_fit_state(const std::string& file_name, const FitState& state)
{
    const std::size_t size = state.parameters.size();
    if (!state.best.valid || state.best.values.size() != size
        || state.best.errors.size() != size
        || state.best.initial_values.size() != size
        || state.best.covariance.size() != size * size) {
        throw std::invalid_argument("inconsistent fit state dimensions");
    }

    Json parameters = Json::array();
    for (std::size_t index = 0; index < size; ++index) {
        const FitParameterSpec& specification = state.parameters[index];
        parameters.push_back({
            {"index", index},
            {"name", specification.name},
            {"value", state.best.values[index]},
            {"error", state.best.errors[index]},
            {"nominal_value", specification.initial_value},
            {"start_value", state.best.initial_values[index]},
            {"step", specification.step},
            {"lower_bound", optional_bound(
                specification.has_lower_bound,
                specification.lower_bound)},
            {"upper_bound", optional_bound(
                specification.has_upper_bound,
                specification.upper_bound)}});
    }

    Json covariance = Json::array();
    for (std::size_t row = 0; row < size; ++row) {
        Json values = Json::array();
        for (std::size_t column = 0; column < size; ++column) {
            values.push_back(state.best.covariance[row * size + column]);
        }
        covariance.push_back(std::move(values));
    }

    Json document = {
        {"schema_version", state.schema_version},
        {"output_tag", state.output_tag},
        {"fit_config", state.fit_config_file},
        {"model", {
            {"file", state.model_config_file},
            {"name", state.model_name},
            {"signature", state.model_signature}}},
        {"best_fit", {
            {"start_index", state.best.start_index},
            {"seed", state.best.seed},
            {"minimum_nll", state.best.minimum},
            {"edm", state.best.edm},
            {"error_definition", state.best.error_definition},
            {"migrad_status", state.best.migrad_status},
            {"hesse_status", state.best.hesse_status},
            {"covariance_status", state.best.covariance_status},
            {"at_parameter_boundary", state.best.at_parameter_boundary},
            {"elapsed_seconds", state.best.elapsed_seconds}}},
        {"parameters", std::move(parameters)},
        {"covariance", std::move(covariance)}};

    std::ofstream output(file_name.c_str(), std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot write fit state: " + file_name);
    }
    output << document.dump(2) << '\n';
    if (!output) {
        throw std::runtime_error("failed to write fit state: " + file_name);
    }
}

FitState read_fit_state(const std::string& file_name)
{
    std::ifstream input(file_name.c_str());
    if (!input) {
        throw std::runtime_error("cannot read fit state: " + file_name);
    }
    Json document;
    try {
        input >> document;
    } catch (const Json::exception& error) {
        throw std::runtime_error(
            "cannot parse fit state '" + file_name + "': " + error.what());
    }

    FitState result;
    result.schema_version = document.at("schema_version").get<int>();
    if (result.schema_version != 1) {
        throw std::runtime_error("unsupported fit-state schema version");
    }
    result.output_tag = document.at("output_tag").get<std::string>();
    require_safe_output_tag(result.output_tag);
    result.fit_config_file = document.at("fit_config").get<std::string>();
    const Json& model = document.at("model");
    result.model_config_file = model.at("file").get<std::string>();
    result.model_name = model.at("name").get<std::string>();
    result.model_signature = model.at("signature").get<std::string>();

    const Json& best = document.at("best_fit");
    result.best.start_index = best.at("start_index").get<int>();
    result.best.seed = best.at("seed").get<long long>();
    result.best.minimum = required_number(best.at("minimum_nll"), "minimum_nll");
    result.best.edm = required_number(best.at("edm"), "edm");
    result.best.error_definition = required_number(
        best.at("error_definition"), "error_definition");
    result.best.migrad_status = best.at("migrad_status").get<int>();
    result.best.hesse_status = best.at("hesse_status").get<int>();
    result.best.covariance_status = best.at("covariance_status").get<int>();
    result.best.at_parameter_boundary =
        best.at("at_parameter_boundary").get<bool>();
    result.best.elapsed_seconds = required_number(
        best.at("elapsed_seconds"), "elapsed_seconds");
    result.best.valid = true;

    const Json& parameters = document.at("parameters");
    if (!parameters.is_array() || parameters.empty()) {
        throw std::runtime_error("fit state has no parameters");
    }
    std::unordered_set<std::string> parameter_names;
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        const Json& item = parameters[index];
        if (item.at("index").get<std::size_t>() != index) {
            throw std::runtime_error("fit-state parameter indices are not ordered");
        }
        FitParameterSpec specification;
        specification.name = item.at("name").get<std::string>();
        if (specification.name.empty()) {
            throw std::runtime_error(
                "fit-state parameter name must not be empty");
        }
        if (!parameter_names.insert(specification.name).second) {
            throw std::runtime_error(
                "duplicate fit-state parameter name '"
                + specification.name + "'");
        }
        specification.initial_value = required_number(
            item.at("nominal_value"), "parameter.nominal_value");
        specification.step = required_number(item.at("step"), "parameter.step");
        if (!(specification.step > 0.0)) {
            throw std::runtime_error(
                "fit-state parameter step must be positive");
        }
        const Json& lower = item.at("lower_bound");
        const Json& upper = item.at("upper_bound");
        specification.has_lower_bound = !lower.is_null();
        specification.has_upper_bound = !upper.is_null();
        if (specification.has_lower_bound) {
            specification.lower_bound = required_number(lower, "lower_bound");
        }
        if (specification.has_upper_bound) {
            specification.upper_bound = required_number(upper, "upper_bound");
        }
        if (specification.has_lower_bound
            && specification.has_upper_bound
            && !(specification.lower_bound < specification.upper_bound)) {
            throw std::runtime_error(
                "fit-state parameter bounds must be increasing");
        }
        const double value = required_number(item.at("value"), "value");
        const double error = required_number(item.at("error"), "error");
        if (error < 0.0) {
            throw std::runtime_error(
                "fit-state parameter error must be nonnegative");
        }
        const double start_value = required_number(
            item.at("start_value"), "parameter.start_value");
        const auto outside_bounds = [&](double candidate) {
            return (specification.has_lower_bound
                    && candidate < specification.lower_bound)
                   || (specification.has_upper_bound
                       && candidate > specification.upper_bound);
        };
        if (outside_bounds(specification.initial_value)
            || outside_bounds(start_value) || outside_bounds(value)) {
            throw std::runtime_error(
                "fit-state parameter value is outside its bounds");
        }
        result.parameters.push_back(specification);
        result.best.values.push_back(value);
        result.best.errors.push_back(error);
        result.best.initial_values.push_back(start_value);
    }

    const Json& covariance = document.at("covariance");
    const std::size_t size = result.parameters.size();
    if (!covariance.is_array() || covariance.size() != size) {
        throw std::runtime_error("fit-state covariance has the wrong size");
    }
    for (std::size_t row = 0; row < size; ++row) {
        if (!covariance[row].is_array() || covariance[row].size() != size) {
            throw std::runtime_error("fit-state covariance row has the wrong size");
        }
        for (std::size_t column = 0; column < size; ++column) {
            result.best.covariance.push_back(required_number(
                covariance[row][column], "covariance"));
        }
    }
    for (std::size_t row = 0; row < size; ++row) {
        if (result.best.covariance[row * size + row] < 0.0) {
            throw std::runtime_error(
                "fit-state covariance has a negative diagonal entry");
        }
        for (std::size_t column = row + 1; column < size; ++column) {
            if (!covariance_entries_match(
                    result.best.covariance[row * size + column],
                    result.best.covariance[column * size + row])) {
                throw std::runtime_error(
                    "fit-state covariance is not symmetric");
            }
        }
    }
    return result;
}

} // namespace ctpwa

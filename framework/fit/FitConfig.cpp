#include "framework/fit/FitConfig.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <initializer_list>
#include <regex>
#include <stdexcept>
#include <unordered_set>

namespace ctpwa {
namespace {

using Json = nlohmann::json;

[[noreturn]] void fail(
    const std::string& source,
    const std::string& path,
    const std::string& message)
{
    throw std::runtime_error(
        "invalid fit configuration '" + source + "' at " + path
        + ": " + message);
}

void reject_unknown(
    const Json& object,
    std::initializer_list<const char*> allowed,
    const std::string& source,
    const std::string& path)
{
    std::unordered_set<std::string> names;
    for (const char* name : allowed) names.insert(name);
    for (const auto& item : object.items()) {
        if (names.find(item.key()) == names.end()) {
            fail(source, path + "." + item.key(), "unknown field");
        }
    }
}

const Json& require_object(
    const Json& parent,
    const char* key,
    const std::string& source,
    const std::string& path)
{
    const auto found = parent.find(key);
    if (found == parent.end() || !found->is_object()) {
        fail(source, path + "." + key, "expected an object");
    }
    return *found;
}

std::string require_string(
    const Json& parent,
    const char* key,
    const std::string& source,
    const std::string& path)
{
    const auto found = parent.find(key);
    if (found == parent.end() || !found->is_string()) {
        fail(source, path + "." + key, "expected a string");
    }
    const std::string value = found->get<std::string>();
    if (value.empty()) fail(source, path + "." + key, "must not be empty");
    return value;
}

template <typename T>
T value_or(
    const Json& parent,
    const char* key,
    T default_value,
    const std::string& source,
    const std::string& path)
{
    const auto found = parent.find(key);
    if (found == parent.end()) return default_value;
    try {
        return found->get<T>();
    } catch (const nlohmann::json::exception&) {
        fail(source, path + "." + key, "has the wrong type");
    }
}

void validate_output_token(
    const std::string& value,
    const std::string& source,
    const std::string& path)
{
    static const std::regex safe_tag("^[A-Za-z0-9][A-Za-z0-9._-]*$");
    if (!std::regex_match(value, safe_tag)) {
        fail(
            source,
            path,
            "must contain only letters, digits, dot, underscore, or hyphen");
    }
}

std::string joined(const std::string& directory, const std::string& file)
{
    if (directory.empty() || directory == ".") return file;
    return directory.back() == '/' ? directory + file
                                   : directory + "/" + file;
}

} // namespace

std::string FitOutputConfig::result_file() const
{
    return joined(directory, "fit_result-" + tag + ".txt");
}

std::string FitOutputConfig::covariance_file() const
{
    return joined(directory, "Cova_matrix-" + tag + ".dat");
}

std::string FitOutputConfig::projection_file() const
{
    return joined(directory, "projection-" + tag + ".root");
}

std::string FitOutputConfig::log_file() const
{
    return joined(log_directory, "fit-" + tag + ".log");
}

FitRunConfig load_fit_run_config(const std::string& file_name)
{
    std::ifstream input(file_name.c_str());
    if (!input) {
        throw std::runtime_error(
            "cannot read fit configuration: " + file_name);
    }
    Json document;
    try {
        input >> document;
    } catch (const Json::exception& error) {
        throw std::runtime_error(
            "cannot parse fit configuration '" + file_name
            + "': " + error.what());
    }
    if (!document.is_object()) fail(file_name, "$", "expected an object");
    reject_unknown(
        document,
        {"schema_version", "model", "inputs", "minimizer", "output"},
        file_name,
        "$");

    FitRunConfig result;
    result.source_file = file_name;
    result.schema_version = value_or<int>(
        document, "schema_version", 0, file_name, "$");
    if (result.schema_version != 1) {
        fail(file_name, "$.schema_version", "only version 1 is supported");
    }
    result.model_file = require_string(document, "model", file_name, "$");

    const Json& inputs = require_object(document, "inputs", file_name, "$");
    reject_unknown(
        inputs,
        {"data", "normalization_mc", "backgrounds"},
        file_name,
        "$.inputs");
    result.inputs.data_file = require_string(
        inputs, "data", file_name, "$.inputs");
    result.inputs.normalization_mc_file = require_string(
        inputs, "normalization_mc", file_name, "$.inputs");
    const auto backgrounds = inputs.find("backgrounds");
    if (backgrounds != inputs.end()) {
        if (!backgrounds->is_array()) {
            fail(file_name, "$.inputs.backgrounds", "expected an array");
        }
        for (std::size_t index = 0; index < backgrounds->size(); ++index) {
            const Json& item = (*backgrounds)[index];
            const std::string path =
                "$.inputs.backgrounds[" + std::to_string(index) + "]";
            if (!item.is_object()) fail(file_name, path, "expected an object");
            reject_unknown(
                item, {"label", "file", "coefficient"}, file_name, path);
            WeightedSampleConfig sample;
            sample.label = require_string(item, "label", file_name, path);
            sample.file = require_string(item, "file", file_name, path);
            sample.likelihood_coefficient = value_or<double>(
                item, "coefficient", 0.0, file_name, path);
            if (!std::isfinite(sample.likelihood_coefficient)) {
                fail(file_name, path + ".coefficient", "must be finite");
            }
            result.inputs.backgrounds.push_back(std::move(sample));
        }
    }

    const Json& minimizer = require_object(
        document, "minimizer", file_name, "$");
    reject_unknown(
        minimizer,
        {"n_starts", "base_seed", "maximum_edm", "maximum_calls",
         "tolerance", "error_definition", "random_magnitude"},
        file_name,
        "$.minimizer");
    result.minimizer.number_starts = value_or<int>(
        minimizer, "n_starts", 1, file_name, "$.minimizer");
    result.minimizer.base_seed = value_or<long long>(
        minimizer, "base_seed", 20260815LL, file_name, "$.minimizer");
    result.minimizer.maximum_edm = value_or<double>(
        minimizer, "maximum_edm", 1.0e-3, file_name, "$.minimizer");
    result.minimizer.maximum_calls = value_or<int>(
        minimizer, "maximum_calls", 20000, file_name, "$.minimizer");
    result.minimizer.tolerance = value_or<double>(
        minimizer, "tolerance", 0.1, file_name, "$.minimizer");
    result.minimizer.error_definition = value_or<double>(
        minimizer, "error_definition", 0.5, file_name, "$.minimizer");
    const auto magnitude = minimizer.find("random_magnitude");
    if (magnitude != minimizer.end()) {
        if (!magnitude->is_array() || magnitude->size() != 2
            || !(*magnitude)[0].is_number()
            || !(*magnitude)[1].is_number()) {
            fail(
                file_name,
                "$.minimizer.random_magnitude",
                "expected [minimum, maximum]");
        }
        result.minimizer.random_magnitude_min = (*magnitude)[0].get<double>();
        result.minimizer.random_magnitude_max = (*magnitude)[1].get<double>();
    }
    if (result.minimizer.number_starts <= 0
        || result.minimizer.base_seed < 0
        || !(result.minimizer.maximum_edm > 0.0)
        || result.minimizer.maximum_calls <= 0
        || !(result.minimizer.tolerance > 0.0)
        || !(result.minimizer.error_definition > 0.0)
        || !(result.minimizer.random_magnitude_min > 0.0)
        || !(result.minimizer.random_magnitude_max
             > result.minimizer.random_magnitude_min)) {
        fail(file_name, "$.minimizer", "contains an invalid fit option");
    }

    const Json& output = require_object(document, "output", file_name, "$");
    reject_unknown(
        output,
        {"directory", "log_directory", "tag"},
        file_name,
        "$.output");
    result.output.directory = require_string(
        output, "directory", file_name, "$.output");
    result.output.log_directory = require_string(
        output, "log_directory", file_name, "$.output");
    result.output.tag = require_string(output, "tag", file_name, "$.output");
    validate_output_token(result.output.tag, file_name, "$.output.tag");
    return result;
}

} // namespace ctpwa

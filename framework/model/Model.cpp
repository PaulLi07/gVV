#include "framework/model/Model.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <initializer_list>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace ctpwa {
namespace {

using Json = nlohmann::json;

[[noreturn]] void fail(
    const std::string& source,
    const std::string& path,
    const std::string& message);

void reject_unknown_members(
    const Json& object,
    std::initializer_list<const char*> allowed,
    const std::string& source,
    const std::string& path)
{
    std::unordered_set<std::string> names;
    for (const char* name : allowed) {
        names.insert(name);
    }
    for (const auto& item : object.items()) {
        if (names.find(item.key()) == names.end()) {
            fail(source, path + "." + item.key(), "unknown field");
        }
    }
}

[[noreturn]] void fail(
    const std::string& source,
    const std::string& path,
    const std::string& message)
{
    throw std::runtime_error(
        "invalid model '" + source + "' at " + path + ": " + message);
}

const Json& require_object_member(
    const Json& object,
    const char* key,
    const std::string& source,
    const std::string& path)
{
    const auto found = object.find(key);
    if (found == object.end() || !found->is_object()) {
        fail(source, path + "." + key, "expected an object");
    }
    return *found;
}

const Json& require_array_member(
    const Json& object,
    const char* key,
    const std::string& source,
    const std::string& path)
{
    const auto found = object.find(key);
    if (found == object.end() || !found->is_array()) {
        fail(source, path + "." + key, "expected an array");
    }
    return *found;
}

std::string require_string(
    const Json& object,
    const char* key,
    const std::string& source,
    const std::string& path)
{
    const auto found = object.find(key);
    if (found == object.end() || !found->is_string()) {
        fail(source, path + "." + key, "expected a string");
    }
    const std::string value = found->get<std::string>();
    if (value.empty()) {
        fail(source, path + "." + key, "must not be empty");
    }
    return value;
}

double require_finite_number(
    const Json& object,
    const char* key,
    const std::string& source,
    const std::string& path)
{
    const auto found = object.find(key);
    if (found == object.end() || !found->is_number()) {
        fail(source, path + "." + key, "expected a number");
    }
    const double value = found->get<double>();
    if (!std::isfinite(value)) {
        fail(source, path + "." + key, "must be finite");
    }
    return value;
}

void validate_id(
    const std::string& id,
    const std::string& source,
    const std::string& path)
{
    static const std::regex pattern("^[A-Za-z][A-Za-z0-9_.-]*$");
    if (!std::regex_match(id, pattern)) {
        fail(
            source,
            path,
            "must start with a letter and contain only letters, digits, "
            "underscore, dot, or hyphen");
    }
}

ParameterDefinition parse_parameter(
    const Json& document,
    const std::string& source,
    const std::string& path)
{
    if (!document.is_object()) {
        fail(source, path, "expected a parameter object");
    }
    reject_unknown_members(
        document,
        {"value", "fixed", "transform", "step", "bounds"},
        source,
        path);
    ParameterDefinition result;
    result.value = require_finite_number(document, "value", source, path);

    const auto fixed = document.find("fixed");
    if (fixed != document.end()) {
        if (!fixed->is_boolean()) {
            fail(source, path + ".fixed", "expected a boolean");
        }
        result.fixed = fixed->get<bool>();
    }

    const auto transform = document.find("transform");
    if (transform != document.end()) {
        if (!transform->is_string()) {
            fail(source, path + ".transform", "expected a string");
        }
        result.transform = transform->get<std::string>();
    }
    if (result.transform != "identity" && result.transform != "log") {
        fail(source, path + ".transform", "supported values are identity/log");
    }
    if (result.transform == "log" && !(result.value > 0.0)) {
        fail(source, path + ".value", "log-transformed value must be positive");
    }

    const auto step = document.find("step");
    if (step != document.end()) {
        if (!step->is_number()) {
            fail(source, path + ".step", "expected a number");
        }
        result.step = step->get<double>();
    }
    if (!(result.step > 0.0) || !std::isfinite(result.step)) {
        fail(source, path + ".step", "must be finite and positive");
    }

    const auto bounds = document.find("bounds");
    if (bounds != document.end()) {
        if (!bounds->is_array() || bounds->size() != 2
            || !(*bounds)[0].is_number() || !(*bounds)[1].is_number()) {
            fail(source, path + ".bounds", "expected [lower, upper]");
        }
        result.lower_bound = (*bounds)[0].get<double>();
        result.upper_bound = (*bounds)[1].get<double>();
        result.has_lower_bound = true;
        result.has_upper_bound = true;
        if (!std::isfinite(result.lower_bound)
            || !std::isfinite(result.upper_bound)
            || !(result.lower_bound < result.upper_bound)) {
            fail(source, path + ".bounds", "must be finite and increasing");
        }
    }
    return result;
}

CouplingMode parse_coupling_mode(
    const std::string& value,
    const std::string& source,
    const std::string& path)
{
    if (value == "complex_cartesian") {
        return CouplingMode::ComplexCartesian;
    }
    if (value == "fixed_complex") {
        return CouplingMode::FixedComplex;
    }
    if (value == "positive_real") {
        return CouplingMode::PositiveReal;
    }
    fail(source, path, "unknown coupling mode '" + value + "'");
}

CouplingReference parse_coupling_reference(
    const std::string& value,
    const std::string& source,
    const std::string& path)
{
    if (value == "none") {
        return CouplingReference::None;
    }
    if (value == "phase") {
        return CouplingReference::Phase;
    }
    if (value == "scale_and_phase") {
        return CouplingReference::ScaleAndPhase;
    }
    fail(source, path, "unknown coupling reference '" + value + "'");
}

CouplingDefinition parse_coupling(
    const Json& document,
    const std::string& source,
    const std::string& path)
{
    reject_unknown_members(
        document,
        {"mode", "reference", "initial"},
        source,
        path);
    CouplingDefinition result;
    result.mode = parse_coupling_mode(
        require_string(document, "mode", source, path),
        source,
        path + ".mode");

    const auto reference = document.find("reference");
    if (reference != document.end()) {
        if (!reference->is_string()) {
            fail(source, path + ".reference", "expected a string");
        }
        result.reference = parse_coupling_reference(
            reference->get<std::string>(), source, path + ".reference");
    }

    const auto initial = document.find("initial");
    if (initial == document.end()) {
        fail(source, path + ".initial", "is required");
    }
    if (result.mode == CouplingMode::PositiveReal) {
        if (!initial->is_number()) {
            fail(source, path + ".initial", "expected a positive number");
        }
        result.initial_real = initial->get<double>();
        result.initial_imag = 0.0;
        if (!(result.initial_real > 0.0)
            || !std::isfinite(result.initial_real)) {
            fail(source, path + ".initial", "must be finite and positive");
        }
        if (result.reference != CouplingReference::Phase) {
            fail(
                source,
                path + ".reference",
                "positive_real requires the phase reference role");
        }
    } else {
        if (!initial->is_array() || initial->size() != 2
            || !(*initial)[0].is_number() || !(*initial)[1].is_number()) {
            fail(source, path + ".initial", "expected [real, imaginary]");
        }
        result.initial_real = (*initial)[0].get<double>();
        result.initial_imag = (*initial)[1].get<double>();
        if (!std::isfinite(result.initial_real)
            || !std::isfinite(result.initial_imag)) {
            fail(source, path + ".initial", "values must be finite");
        }
    }

    if (result.mode == CouplingMode::FixedComplex
        && result.reference != CouplingReference::ScaleAndPhase) {
        fail(
            source,
            path + ".reference",
            "fixed_complex requires the scale_and_phase reference role");
    }
    if (result.mode == CouplingMode::ComplexCartesian
        && result.reference != CouplingReference::None) {
        fail(
            source,
            path + ".reference",
            "complex_cartesian cannot be a reference coupling");
    }
    return result;
}

} // namespace

const ResonanceDefinition& ModelDefinition::resonance(
    const std::string& id) const
{
    for (const ResonanceDefinition& value : resonances) {
        if (value.id == id) {
            return value;
        }
    }
    throw std::out_of_range("unknown resonance id '" + id + "'");
}

const TermDefinition& ModelDefinition::term(const std::string& id) const
{
    for (const TermDefinition& value : terms) {
        if (value.id == id) {
            return value;
        }
    }
    throw std::out_of_range("unknown term id '" + id + "'");
}

ModelDefinition load_model_definition(const std::string& file_name)
{
    std::ifstream input(file_name.c_str());
    if (!input) {
        throw std::runtime_error("cannot open model JSON: " + file_name);
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    return parse_model_definition(contents.str(), file_name);
}

ModelDefinition parse_model_definition(
    const std::string& json_text,
    const std::string& source_name)
{
    Json document;
    try {
        document = Json::parse(json_text);
    } catch (const Json::parse_error& error) {
        throw std::runtime_error(
            "cannot parse model JSON '" + source_name + "': " + error.what());
    }
    if (!document.is_object()) {
        fail(source_name, "$", "expected a JSON object");
    }
    reject_unknown_members(
        document,
        {"schema_version", "process", "metadata", "resonances", "terms"},
        source_name,
        "$");

    ModelDefinition result;
    const auto schema_version = document.find("schema_version");
    if (schema_version == document.end() || !schema_version->is_number_integer()) {
        fail(source_name, "$.schema_version", "expected an integer");
    }
    result.schema_version = schema_version->get<int>();
    if (result.schema_version != 1) {
        fail(source_name, "$.schema_version", "only schema version 1 is supported");
    }
    result.process = require_string(document, "process", source_name, "$");
    validate_id(result.process, source_name, "$.process");

    const auto metadata = document.find("metadata");
    if (metadata != document.end()) {
        if (!metadata->is_object()) {
            fail(source_name, "$.metadata", "expected an object");
        }
        reject_unknown_members(
            *metadata,
            {"name", "description"},
            source_name,
            "$.metadata");
        const auto name = metadata->find("name");
        if (name != metadata->end()) {
            if (!name->is_string()) {
                fail(source_name, "$.metadata.name", "expected a string");
            }
            result.name = name->get<std::string>();
        }
        const auto description = metadata->find("description");
        if (description != metadata->end()) {
            if (!description->is_string()) {
                fail(source_name, "$.metadata.description", "expected a string");
            }
            result.description = description->get<std::string>();
        }
    }

    std::unordered_set<std::string> resonance_ids;
    const Json& resonances = require_array_member(
        document, "resonances", source_name, "$");
    if (resonances.empty()) {
        fail(source_name, "$.resonances", "must contain at least one resonance");
    }
    for (std::size_t index = 0; index < resonances.size(); ++index) {
        const std::string path = "$.resonances[" + std::to_string(index) + "]";
        const Json& entry = resonances[index];
        if (!entry.is_object()) {
            fail(source_name, path, "expected an object");
        }
        reject_unknown_members(
            entry,
            {"id", "label", "propagator", "parameters"},
            source_name,
            path);
        ResonanceDefinition resonance;
        resonance.id = require_string(entry, "id", source_name, path);
        validate_id(resonance.id, source_name, path + ".id");
        if (!resonance_ids.insert(resonance.id).second) {
            fail(source_name, path + ".id", "duplicate resonance id");
        }
        resonance.label = entry.value("label", resonance.id);
        resonance.propagator = require_string(
            entry, "propagator", source_name, path);
        validate_id(resonance.propagator, source_name, path + ".propagator");

        const Json& parameters = require_object_member(
            entry, "parameters", source_name, path);
        for (const auto& item : parameters.items()) {
            validate_id(item.key(), source_name, path + ".parameters");
            resonance.parameters.emplace(
                item.key(),
                parse_parameter(
                    item.value(),
                    source_name,
                    path + ".parameters." + item.key()));
        }
        result.resonances.push_back(std::move(resonance));
    }

    std::unordered_set<std::string> term_ids;
    int scale_references = 0;
    const Json& terms = require_array_member(document, "terms", source_name, "$");
    if (terms.empty()) {
        fail(source_name, "$.terms", "must contain at least one term");
    }
    for (std::size_t index = 0; index < terms.size(); ++index) {
        const std::string path = "$.terms[" + std::to_string(index) + "]";
        const Json& entry = terms[index];
        if (!entry.is_object()) {
            fail(source_name, path, "expected an object");
        }
        reject_unknown_members(
            entry,
            {"id", "label", "wave", "active", "coupling", "dynamics"},
            source_name,
            path);
        TermDefinition term;
        term.id = require_string(entry, "id", source_name, path);
        validate_id(term.id, source_name, path + ".id");
        if (!term_ids.insert(term.id).second) {
            fail(source_name, path + ".id", "duplicate term id");
        }
        term.label = entry.value("label", term.id);
        term.wave = require_string(entry, "wave", source_name, path);
        validate_id(term.wave, source_name, path + ".wave");
        const auto active = entry.find("active");
        if (active != entry.end()) {
            if (!active->is_boolean()) {
                fail(source_name, path + ".active", "expected a boolean");
            }
            term.active = active->get<bool>();
        }
        term.coupling = parse_coupling(
            require_object_member(entry, "coupling", source_name, path),
            source_name,
            path + ".coupling");
        const Json& dynamics = require_object_member(
            entry, "dynamics", source_name, path);
        term.dynamics_json = dynamics.dump();
        if (term.active
            && term.coupling.reference == CouplingReference::ScaleAndPhase) {
            ++scale_references;
        }
        result.terms.push_back(std::move(term));
    }
    if (scale_references != 1) {
        fail(
            source_name,
            "$.terms",
            "active model requires exactly one scale_and_phase reference");
    }

    result.canonical_json = document.dump(2) + "\n";
    return result;
}

const char* coupling_mode_name(CouplingMode mode)
{
    switch (mode) {
    case CouplingMode::ComplexCartesian:
        return "complex_cartesian";
    case CouplingMode::FixedComplex:
        return "fixed_complex";
    case CouplingMode::PositiveReal:
        return "positive_real";
    }
    return "unknown";
}

const char* coupling_reference_name(CouplingReference reference)
{
    switch (reference) {
    case CouplingReference::None:
        return "none";
    case CouplingReference::Phase:
        return "phase";
    case CouplingReference::ScaleAndPhase:
        return "scale_and_phase";
    }
    return "unknown";
}

} // namespace ctpwa

#include "../include/GVVFitParameters.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace {

int find_term(const std::string& name)
{
    for (int term = 0; term < GVV_NTERMS; ++term) {
        if (name == gvv_term_name(term)) {
            return term;
        }
    }
    return -1;
}

int find_resonance(const std::string& name)
{
    for (int resonance = 0; resonance < GVV_NRESONANCES; ++resonance) {
        if (name == gvv_resonance_name(resonance)) {
            return resonance;
        }
    }
    return -1;
}

std::vector<std::string> split(const std::string& line)
{
    std::istringstream input(line);
    std::vector<std::string> tokens;
    std::string token;
    while (input >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

double parse_number(const std::string& text, const std::string& context)
{
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != text.size() || !std::isfinite(value)) {
        throw std::runtime_error("invalid finite number for " + context);
    }
    return value;
}

int token_after(
    const std::vector<std::string>& tokens,
    const std::string& key)
{
    for (std::size_t index = 0; index + 1 < tokens.size(); ++index) {
        if (tokens[index] == key) {
            return static_cast<int>(index + 1);
        }
    }
    return -1;
}

} // namespace

std::vector<GVVFitParameterSpec> gvv_fit_parameter_layout(
    const GVVCompiledModel& model)
{
    if (model.terms.size() != model.initial_couplings.size()
        || model.terms.size() != model.term_metadata.size()
        || model.resonances.size() != model.resonance_metadata.size()) {
        throw std::invalid_argument("incomplete compiled GVV model layout");
    }

    std::vector<GVVFitParameterSpec> layout;
    for (std::size_t term = 0; term < model.terms.size(); ++term) {
        const GVVTermMetadata& metadata = model.term_metadata[term];
        const DeviceComplex coupling = model.initial_couplings[term];
        if (metadata.coupling_parameterization
            == GVV_COUPLING_FIXED_SCALE_AND_PHASE) {
            continue;
        }
        if (metadata.coupling_parameterization
            == GVV_COUPLING_POSITIVE_REAL) {
            if (!(coupling.real > 0.0) || coupling.imag != 0.0) {
                throw std::runtime_error(
                    "positive-real coupling '" + metadata.id
                    + "' is invalid");
            }
            GVVFitParameterSpec parameter;
            parameter.name = "log_rho_" + metadata.id;
            parameter.target = GVVFitParameterTarget::CouplingLogMagnitude;
            parameter.target_index = static_cast<int>(term);
            parameter.initial_value = std::log(coupling.real);
            parameter.step = 0.10;
            layout.push_back(parameter);
            continue;
        }

        GVVFitParameterSpec real;
        real.name = "Re_" + metadata.id;
        real.target = GVVFitParameterTarget::CouplingReal;
        real.target_index = static_cast<int>(term);
        real.initial_value = coupling.real;
        real.step = 0.05;
        layout.push_back(real);

        GVVFitParameterSpec imaginary = real;
        imaginary.name = "Im_" + metadata.id;
        imaginary.target = GVVFitParameterTarget::CouplingImaginary;
        imaginary.initial_value = coupling.imag;
        layout.push_back(imaginary);
    }

    for (std::size_t resonance = 0;
         resonance < model.resonances.size();
         ++resonance) {
        const GVVResonanceParameters& values = model.resonances[resonance];
        const GVVResonanceMetadata& metadata =
            model.resonance_metadata[resonance];
        const ctpwa::ResonanceDefinition& definition =
            model.definition.resonance(metadata.id);

        auto append_log_parameter = [&](const std::string& source_name,
                                        const std::string& fit_name,
                                        GVVFitParameterTarget target,
                                        double physical_value) {
            const auto found = definition.parameters.find(source_name);
            if (found == definition.parameters.end() || found->second.fixed
                || found->second.transform != "log") {
                throw std::runtime_error(
                    "compiled fitted parameter '" + source_name
                    + "' is inconsistent for resonance '" + metadata.id
                    + "'");
            }
            if (!(physical_value > 0.0)) {
                throw std::runtime_error(
                    "non-positive fitted parameter for resonance '"
                    + metadata.id + "'");
            }
            GVVFitParameterSpec parameter;
            parameter.name = fit_name + metadata.id;
            parameter.target = target;
            parameter.target_index = static_cast<int>(resonance);
            parameter.initial_value = std::log(physical_value);
            parameter.step = found->second.step;
            parameter.has_lower_bound = found->second.has_lower_bound;
            parameter.has_upper_bound = found->second.has_upper_bound;
            parameter.lower_bound = found->second.lower_bound;
            parameter.upper_bound = found->second.upper_bound;
            layout.push_back(parameter);
        };

        if (values.fit_sd_ratio) {
            append_log_parameter(
                "sd_ratio",
                "log_rDS_",
                GVVFitParameterTarget::ResonanceLogSDRatio,
                values.sd_ratio);
        }
        if (values.fit_flatte_ratio) {
            append_log_parameter(
                "omegaomega_ratio",
                "log_Romega_",
                GVVFitParameterTarget::ResonanceLogFlatteRatio,
                values.flatte_ratio);
        }
    }
    return layout;
}

std::vector<std::string> gvv_fit_parameter_names(
    const GVVCompiledModel& model)
{
    const std::vector<GVVFitParameterSpec> layout =
        gvv_fit_parameter_layout(model);
    std::vector<std::string> names;
    names.reserve(layout.size());
    for (const GVVFitParameterSpec& parameter : layout) {
        names.push_back(parameter.name);
    }
    return names;
}

std::vector<double> gvv_fit_parameters_from_model(
    const GVVCompiledModel& model)
{
    const std::vector<GVVFitParameterSpec> layout =
        gvv_fit_parameter_layout(model);
    std::vector<double> values;
    values.reserve(layout.size());
    for (const GVVFitParameterSpec& parameter : layout) {
        values.push_back(parameter.initial_value);
    }
    return values;
}

void gvv_apply_fit_parameters_to_model(
    GVVCompiledModel& model,
    const std::vector<double>& parameters)
{
    const std::vector<GVVFitParameterSpec> layout =
        gvv_fit_parameter_layout(model);
    if (parameters.size() != layout.size()) {
        throw std::invalid_argument("incorrect number of GVV fit parameters");
    }
    for (std::size_t index = 0; index < layout.size(); ++index) {
        const double value = parameters[index];
        if (!std::isfinite(value)) {
            throw std::invalid_argument("non-finite GVV fit parameter");
        }
        const GVVFitParameterSpec& parameter = layout[index];
        if (parameter.target == GVVFitParameterTarget::CouplingReal) {
            model.initial_couplings[parameter.target_index].real = value;
        } else if (
            parameter.target == GVVFitParameterTarget::CouplingImaginary) {
            model.initial_couplings[parameter.target_index].imag = value;
        } else if (
            parameter.target == GVVFitParameterTarget::CouplingLogMagnitude) {
            model.initial_couplings[parameter.target_index] =
                DeviceComplex(std::exp(value), 0.0);
        } else if (
            parameter.target == GVVFitParameterTarget::ResonanceLogSDRatio) {
            model.resonances[parameter.target_index].sd_ratio =
                std::exp(value);
        } else if (
            parameter.target
            == GVVFitParameterTarget::ResonanceLogFlatteRatio) {
            model.resonances[parameter.target_index].flatte_ratio =
                std::exp(value);
        }
    }
}

GVVFitState gvv_default_fit_state()
{
    GVVFitState state;
    for (int resonance = 0; resonance < GVV_NRESONANCES; ++resonance) {
        state.resonances[resonance] = gvv_default_resonance(resonance);
    }
    for (int term = 0; term < GVV_NTERMS; ++term) {
        state.terms[term] = gvv_default_term(term);
        state.couplings[term] = DeviceComplex(0.10, 0.0);
    }
    state.couplings[GVV_SCALE_AND_PHASE_REFERENCE_TERM] =
        DeviceComplex(1.0, 0.0);
    state.couplings[GVV_SCALAR_PHASE_REFERENCE_TERM] =
        DeviceComplex(0.10, 0.0);
    return state;
}

std::vector<std::string> gvv_fit_parameter_names(
    const std::array<GVVResonanceParameters, GVV_NRESONANCES>& resonances)
{
    std::vector<std::string> names;
    for (int term = 0; term < GVV_NTERMS; ++term) {
        const int parameterization = gvv_coupling_parameterization(term);
        if (parameterization == GVV_COUPLING_FIXED_SCALE_AND_PHASE) {
            continue;
        }
        if (parameterization == GVV_COUPLING_POSITIVE_REAL) {
            names.push_back(std::string("log_rho_") + gvv_term_name(term));
        } else {
            names.push_back(std::string("Re_") + gvv_term_name(term));
            names.push_back(std::string("Im_") + gvv_term_name(term));
        }
    }
    for (int resonance = 0; resonance < GVV_NRESONANCES; ++resonance) {
        if (resonances[resonance].fit_sd_ratio) {
            names.push_back(
                std::string("log_rDS_") + gvv_resonance_name(resonance));
        }
        if (resonances[resonance].fit_flatte_ratio) {
            names.push_back(
                std::string("log_Romega_")
                + gvv_resonance_name(resonance));
        }
    }
    return names;
}

std::vector<double> gvv_fit_parameters_from_state(const GVVFitState& state)
{
    std::vector<double> values;
    for (int term = 0; term < GVV_NTERMS; ++term) {
        const int parameterization = gvv_coupling_parameterization(term);
        if (parameterization == GVV_COUPLING_FIXED_SCALE_AND_PHASE) {
            continue;
        }
        const DeviceComplex coupling = state.couplings[term];
        if (parameterization == GVV_COUPLING_POSITIVE_REAL) {
            if (!(coupling.real > 0.0) || coupling.imag != 0.0) {
                throw std::runtime_error(
                    "positive-real phase reference is invalid");
            }
            values.push_back(std::log(coupling.real));
        } else {
            values.push_back(coupling.real);
            values.push_back(coupling.imag);
        }
    }
    for (int resonance = 0; resonance < GVV_NRESONANCES; ++resonance) {
        const GVVResonanceParameters& parameters = state.resonances[resonance];
        if (parameters.fit_sd_ratio) {
            if (!(parameters.sd_ratio > 0.0)) {
                throw std::runtime_error("non-positive fitted S/D ratio");
            }
            values.push_back(std::log(parameters.sd_ratio));
        }
        if (parameters.fit_flatte_ratio) {
            if (!(parameters.flatte_ratio > 0.0)) {
                throw std::runtime_error("non-positive fitted Flatte ratio");
            }
            values.push_back(std::log(parameters.flatte_ratio));
        }
    }
    return values;
}

void gvv_apply_fit_parameters_to_state(
    GVVFitState& state,
    const std::vector<double>& parameters)
{
    const std::vector<std::string> names =
        gvv_fit_parameter_names(state.resonances);
    if (parameters.size() != names.size()) {
        throw std::invalid_argument("incorrect number of GVV fit parameters");
    }

    std::size_t cursor = 0;
    for (int term = 0; term < GVV_NTERMS; ++term) {
        const int parameterization = gvv_coupling_parameterization(term);
        if (parameterization == GVV_COUPLING_FIXED_SCALE_AND_PHASE) {
            state.couplings[term] = DeviceComplex(1.0, 0.0);
            continue;
        }
        if (parameterization == GVV_COUPLING_POSITIVE_REAL) {
            const double magnitude = std::exp(parameters[cursor++]);
            if (!(magnitude > 0.0) || !std::isfinite(magnitude)) {
                throw std::invalid_argument(
                    "positive-real coupling exponent is invalid");
            }
            state.couplings[term] = DeviceComplex(magnitude, 0.0);
        } else {
            const double real = parameters[cursor++];
            const double imag = parameters[cursor++];
            if (!std::isfinite(real) || !std::isfinite(imag)) {
                throw std::invalid_argument("non-finite complex coupling");
            }
            state.couplings[term] = DeviceComplex(real, imag);
        }
    }
    for (int resonance = 0; resonance < GVV_NRESONANCES; ++resonance) {
        GVVResonanceParameters& values = state.resonances[resonance];
        if (values.fit_sd_ratio) {
            values.sd_ratio = std::exp(parameters[cursor++]);
        }
        if (values.fit_flatte_ratio) {
            values.flatte_ratio = std::exp(parameters[cursor++]);
        }
    }
}

GVVParsedFitResult gvv_read_fit_result(const std::string& file_name)
{
    std::ifstream input(file_name.c_str());
    if (!input) {
        throw std::runtime_error("cannot read fit result: " + file_name);
    }

    GVVParsedFitResult result;
    result.state = gvv_default_fit_state();
    const std::vector<std::string> expected =
        gvv_fit_parameter_names(result.state.resonances);
    result.parameter_names = expected;
    result.errors.assign(expected.size(), 0.0);

    std::unordered_map<std::string, std::pair<double, double>> machine_rows;
    std::array<double, GVV_NTERMS> coupling_errors_real{};
    std::array<double, GVV_NTERMS> coupling_errors_imag{};
    std::array<double, GVV_NTERMS> coupling_errors_log{};
    std::array<double, GVV_NRESONANCES> sd_errors_log{};
    std::array<double, GVV_NRESONANCES> flatte_errors_log{};

    std::string line;
    while (std::getline(input, line)) {
        const std::vector<std::string> tokens = split(line);
        if (tokens.empty() || tokens[0][0] == '#') {
            continue;
        }
        if (tokens[0] == "parameter") {
            if (tokens.size() != 5) {
                throw std::runtime_error(
                    "malformed machine-readable parameter row");
            }
            const std::string& name = tokens[2];
            machine_rows[name] = std::make_pair(
                parse_number(tokens[3], name),
                parse_number(tokens[4], name + " error"));
            continue;
        }
        if (tokens[0] == "coupling" && tokens.size() >= 4) {
            const int term = find_term(tokens[1]);
            if (term < 0) {
                throw std::runtime_error("unknown coupling " + tokens[1]);
            }
            const double real = parse_number(tokens[2], tokens[1]);
            const double imag = parse_number(tokens[3], tokens[1]);
            result.state.couplings[term] = DeviceComplex(real, imag);
            if (tokens.size() >= 6 && tokens[4] != "fixed") {
                coupling_errors_real[term] =
                    parse_number(tokens[4], tokens[1] + " Re error");
                coupling_errors_imag[term] =
                    parse_number(tokens[5], tokens[1] + " Im error");
            }
            const int log_error = token_after(tokens, "log_error");
            if (log_error >= 0) {
                coupling_errors_log[term] = parse_number(
                    tokens[log_error], tokens[1] + " log error");
            }
            continue;
        }
        if (tokens[0] == "resonance" && tokens.size() >= 2) {
            const int resonance = find_resonance(tokens[1]);
            if (resonance < 0) {
                throw std::runtime_error("unknown resonance " + tokens[1]);
            }
            GVVResonanceParameters& state = result.state.resonances[resonance];
            const int sd = token_after(tokens, "r_D_over_S");
            if (sd >= 0) {
                state.sd_ratio = parse_number(tokens[sd], tokens[1] + " rDS");
            }
            const int flatte = token_after(tokens, "R_omegaomega");
            if (flatte >= 0) {
                state.flatte_ratio = parse_number(
                    tokens[flatte], tokens[1] + " Romega");
            }
            const int log_error = token_after(tokens, "log_error");
            if (log_error >= 0) {
                const double error = parse_number(
                    tokens[log_error], tokens[1] + " log error");
                if (state.fit_sd_ratio) {
                    sd_errors_log[resonance] = error;
                } else if (state.fit_flatte_ratio) {
                    flatte_errors_log[resonance] = error;
                }
            }
        }
    }

    if (!machine_rows.empty()) {
        if (machine_rows.size() != expected.size()) {
            throw std::runtime_error(
                "machine-readable parameter count does not match model");
        }
        result.values.resize(expected.size());
        for (std::size_t index = 0; index < expected.size(); ++index) {
            const auto found = machine_rows.find(expected[index]);
            if (found == machine_rows.end()) {
                throw std::runtime_error(
                    "missing machine-readable parameter " + expected[index]);
            }
            result.values[index] = found->second.first;
            result.errors[index] = found->second.second;
        }
        gvv_apply_fit_parameters_to_state(result.state, result.values);
        result.used_machine_readable_rows = true;
        return result;
    }

    result.values = gvv_fit_parameters_from_state(result.state);
    std::size_t cursor = 0;
    for (int term = 0; term < GVV_NTERMS; ++term) {
        const int parameterization = gvv_coupling_parameterization(term);
        if (parameterization == GVV_COUPLING_FIXED_SCALE_AND_PHASE) {
            continue;
        }
        if (parameterization == GVV_COUPLING_POSITIVE_REAL) {
            result.errors[cursor++] = coupling_errors_log[term];
        } else {
            result.errors[cursor++] = coupling_errors_real[term];
            result.errors[cursor++] = coupling_errors_imag[term];
        }
    }
    for (int resonance = 0; resonance < GVV_NRESONANCES; ++resonance) {
        if (result.state.resonances[resonance].fit_sd_ratio) {
            result.errors[cursor++] = sd_errors_log[resonance];
        }
        if (result.state.resonances[resonance].fit_flatte_ratio) {
            result.errors[cursor++] = flatte_errors_log[resonance];
        }
    }
    return result;
}

std::vector<double> gvv_read_covariance_matrix(
    const std::string& file_name,
    int number_parameters)
{
    std::ifstream input(file_name.c_str());
    if (!input) {
        throw std::runtime_error("cannot read covariance matrix: " + file_name);
    }
    std::vector<double> covariance;
    double value = 0.0;
    while (input >> value) {
        covariance.push_back(value);
    }
    if (static_cast<int>(covariance.size())
        != number_parameters * number_parameters) {
        throw std::runtime_error("covariance matrix has incorrect dimensions");
    }
    gvv_validate_covariance_matrix(covariance, number_parameters);
    return covariance;
}

void gvv_validate_covariance_matrix(
    const std::vector<double>& covariance,
    int number_parameters)
{
    if (number_parameters <= 0
        || static_cast<int>(covariance.size())
               != number_parameters * number_parameters) {
        throw std::invalid_argument("invalid covariance dimensions");
    }
    for (int row = 0; row < number_parameters; ++row) {
        for (int column = 0; column < number_parameters; ++column) {
            const double value = covariance[
                static_cast<std::size_t>(row) * number_parameters + column];
            if (!std::isfinite(value)) {
                throw std::runtime_error("covariance contains non-finite value");
            }
            const double transpose = covariance[
                static_cast<std::size_t>(column) * number_parameters + row];
            const double scale = std::max(
                1.0, std::max(std::fabs(value), std::fabs(transpose)));
            if (std::fabs(value - transpose) > 1.0e-9 * scale) {
                throw std::runtime_error("covariance matrix is not symmetric");
            }
        }
    }

    // Cholesky factorization is a compact positive-definiteness check and is
    // independent of ROOT, so it is also available to the unit tests.
    std::vector<double> lower(covariance.size(), 0.0);
    for (int row = 0; row < number_parameters; ++row) {
        for (int column = 0; column <= row; ++column) {
            double sum = covariance[
                static_cast<std::size_t>(row) * number_parameters + column];
            for (int k = 0; k < column; ++k) {
                sum -= lower[
                           static_cast<std::size_t>(row) * number_parameters + k]
                       * lower[
                           static_cast<std::size_t>(column) * number_parameters
                           + k];
            }
            if (row == column) {
                if (!(sum > 0.0) || !std::isfinite(sum)) {
                    throw std::runtime_error(
                        "covariance matrix is not positive definite");
                }
                lower[static_cast<std::size_t>(row) * number_parameters + column]
                    = std::sqrt(sum);
            } else {
                lower[static_cast<std::size_t>(row) * number_parameters + column]
                    = sum / lower[
                        static_cast<std::size_t>(column) * number_parameters
                        + column];
            }
        }
    }
}

bool gvv_fit_parameter_bounds(
    const std::string& name,
    double& lower,
    double& upper)
{
    if (name.rfind("log_Romega_", 0) == 0) {
        lower = GVV_FLATTE_LOG_RATIO_MIN;
        upper = GVV_FLATTE_LOG_RATIO_MAX;
        return true;
    }
    if (name.rfind("log_rDS_", 0) == 0) {
        lower = -10.0;
        upper = 10.0;
        return true;
    }
    lower = -std::numeric_limits<double>::infinity();
    upper = std::numeric_limits<double>::infinity();
    return false;
}

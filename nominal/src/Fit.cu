#include "../include/Minuit.h"
#include "../include/GVVFitParameters.h"

#include "TMinuit.h"

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kDefaultNumberStarts = 1;
constexpr long long kDefaultBaseSeed = 20260815;
constexpr double kMaximumAcceptedEDM = 1.0e-3;
constexpr double kRandomCouplingMagnitudeMin = 0.05;
constexpr double kRandomCouplingMagnitudeMax = 5.0;
constexpr double kPi = 3.14159265358979323846;
constexpr double kSB1LikelihoodCoefficient = -0.5;
constexpr double kSB2LikelihoodCoefficient = +0.25;
const char* const kProjectionFile = "results/projection0.root";
const char* const kCovarianceFile = "results/Cova_matrix.dat";

struct FitAttempt {
    int start_index = -1;
    long long seed = 0;
    std::vector<Double_t> initial_values;
    std::vector<Double_t> values;
    std::vector<Double_t> errors;
    std::vector<Double_t> covariance;
    Int_t migrad_status = -1;
    Int_t hesse_status = -1;
    Int_t covariance_status = 0;
    Double_t minimum = std::numeric_limits<Double_t>::infinity();
    Double_t edm = std::numeric_limits<Double_t>::infinity();
    Double_t error_definition = 0.5;
    double elapsed_seconds = 0.0;
    bool at_parameter_boundary = false;
    bool valid = false;
};

void print_usage(const char* executable)
{
    std::cerr
        << "Usage: " << executable
        << " data.root normalization_mc.root"
        << " SB1.root SB2.root"
        << " [fit_result.txt [n_starts [base_seed [model.json]]]]\n\n"
        << "Required tree/branch contract (Double_t[4], px,py,pz,E):\n"
        << "  tree: Pwa\n"
        << "  p4_pip1 p4_pim1 p4_pi01 p4_pip2 p4_pim2 p4_pi02 p4_gam\n"
        << "Sideband likelihood coefficients are fixed to SB1=-0.5,"
        << " SB2=+0.25.\n"
        << "n_starts defaults to 1. Only the best converged start writes "
        << "the fit result, results/Cova_matrix.dat, and "
        << "results/projection0.root. model.json defaults to "
        << "config/model.json.\n";
}

int parse_number_starts(const char* text)
{
    std::size_t consumed = 0;
    const std::string argument(text);
    const long long value = std::stoll(argument, &consumed, 10);
    if (consumed != argument.size() || value <= 0
        || value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "n_starts must be a positive integer");
    }
    return static_cast<int>(value);
}

long long parse_base_seed(const char* text)
{
    std::size_t consumed = 0;
    const std::string argument(text);
    const long long value = std::stoll(argument, &consumed, 10);
    if (consumed != argument.size() || value < 0) {
        throw std::invalid_argument(
            "base_seed must be a non-negative integer");
    }
    return value;
}

void randomize_free_couplings(
    std::vector<Double_t>& values,
    const std::vector<GVVFitParameterSpec>& layout,
    long long seed)
{
    std::mt19937_64 generator(static_cast<std::mt19937_64::result_type>(seed));
    std::uniform_real_distribution<double> log_magnitude(
        std::log(kRandomCouplingMagnitudeMin),
        std::log(kRandomCouplingMagnitudeMax));
    std::uniform_real_distribution<double> phase(-kPi, kPi);

    for (std::size_t cursor = 0; cursor < layout.size(); ++cursor) {
        const GVVFitParameterSpec& parameter = layout[cursor];
        const double random_log_magnitude = log_magnitude(generator);
        if (parameter.target
            == GVVFitParameterTarget::CouplingLogMagnitude) {
            values[cursor] = random_log_magnitude;
            continue;
        }
        if (parameter.target != GVVFitParameterTarget::CouplingReal) {
            continue;
        }
        if (cursor + 1 >= layout.size()
            || layout[cursor + 1].target
                   != GVVFitParameterTarget::CouplingImaginary
            || layout[cursor + 1].target_index != parameter.target_index) {
            throw std::runtime_error(
                "complex coupling parameter layout is not adjacent");
        }
        const double magnitude = std::exp(random_log_magnitude);
        const double angle = phase(generator);
        values[cursor] = magnitude * std::cos(angle);
        values[cursor + 1] = magnitude * std::sin(angle);
        ++cursor;
    }
    // Non-coupling physics parameters keep their nominal start values.  In
    // particular, fitted log_Romega starts from its documented nominal value.
}

void define_fit_parameters(
    TMinuit& minuit,
    const std::vector<GVVFitParameterSpec>& layout,
    const std::vector<Double_t>& initial_values)
{
    if (initial_values.size() != layout.size()) {
        throw std::invalid_argument("incorrect number of initial parameters");
    }

    for (std::size_t parameter = 0;
         parameter < layout.size();
         ++parameter) {
        const GVVFitParameterSpec& specification = layout[parameter];
        if (specification.has_lower_bound != specification.has_upper_bound) {
            throw std::runtime_error(
                "TMinuit requires either two bounds or no bounds");
        }
        minuit.DefineParameter(
            static_cast<int>(parameter),
            specification.name.c_str(),
            initial_values[parameter],
            specification.step,
            specification.has_lower_bound ? specification.lower_bound : 0.0,
            specification.has_upper_bound ? specification.upper_bound : 0.0);
    }
}

void print_parameter_vector(
    const char* heading,
    const std::vector<std::string>& names,
    const std::vector<Double_t>& values,
    const std::vector<Double_t>* errors = nullptr)
{
    std::cout << heading << '\n' << std::setprecision(12);
    for (std::size_t index = 0; index < values.size(); ++index) {
        std::cout << "  " << std::setw(26) << std::left << names[index]
                  << std::right << "  " << std::setw(18) << values[index];
        if (errors != nullptr) {
            std::cout << " +/- " << (*errors)[index];
        }
        std::cout << '\n';
    }
}

bool has_parameter_at_boundary(
    const std::vector<GVVFitParameterSpec>& layout,
    const std::vector<Double_t>& values)
{
    constexpr double boundary_tolerance = 1.0e-6;
    for (std::size_t cursor = 0; cursor < layout.size(); ++cursor) {
        if (layout[cursor].has_lower_bound
            && std::fabs(values[cursor] - layout[cursor].lower_bound)
                   < boundary_tolerance) {
            return true;
        }
        if (layout[cursor].has_upper_bound
            && std::fabs(values[cursor] - layout[cursor].upper_bound)
                   < boundary_tolerance) {
            return true;
        }
    }
    return false;
}

FitAttempt run_fit_attempt(
    NLL_estimator& estimator,
    const std::vector<std::string>& parameter_names,
    const std::vector<GVVFitParameterSpec>& parameter_layout,
    const std::vector<Double_t>& initial_values,
    int start_index,
    long long seed)
{
    FitAttempt attempt;
    attempt.start_index = start_index;
    attempt.seed = seed;
    attempt.initial_values = initial_values;
    attempt.values.assign(estimator.NumberFitParameters(), 0.0);
    attempt.errors.assign(estimator.NumberFitParameters(), 0.0);
    attempt.covariance.assign(
        static_cast<std::size_t>(estimator.NumberFitParameters())
            * estimator.NumberFitParameters(),
        0.0);

    std::cout
        << "\n============================================================\n"
        << "GVV MULTISTART FIT " << start_index << "  seed=" << seed << '\n'
        << "============================================================\n";
    print_parameter_vector(
        "Initial free parameters:", parameter_names, initial_values);

    const auto start_time = std::chrono::steady_clock::now();
    apply_gvv_fit_parameters(estimator, initial_values.data());
    TMinuit minuit(estimator.NumberFitParameters());
    minuit.SetObjectFit(&estimator);
    minuit.SetFCN(objective_function);
    define_fit_parameters(minuit, parameter_layout, initial_values);

    Int_t status = 0;
    Double_t arguments[2] = {0.0, 0.0};
    arguments[0] = 0.5;
    minuit.mnexcm("SET ERR", arguments, 1, status);
    arguments[0] = 20000;
    arguments[1] = 0.1;
    minuit.mnexcm("MIGRAD", arguments, 2, status);
    attempt.migrad_status = status;
    minuit.mnexcm("HESSE", arguments, 0, status);
    attempt.hesse_status = status;

    Int_t number_variable_parameters = 0;
    Int_t number_parameters = 0;
    minuit.mnstat(
        attempt.minimum,
        attempt.edm,
        attempt.error_definition,
        number_variable_parameters,
        number_parameters,
        attempt.covariance_status);
    for (int parameter = 0;
         parameter < estimator.NumberFitParameters();
         ++parameter) {
        minuit.GetParameter(
            parameter, attempt.values[parameter], attempt.errors[parameter]);
    }
    minuit.mnemat(
        attempt.covariance.data(), estimator.NumberFitParameters());
    apply_gvv_fit_parameters(estimator, attempt.values.data());
    attempt.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time).count();
    attempt.at_parameter_boundary = has_parameter_at_boundary(
        parameter_layout, attempt.values);

    attempt.valid =
        attempt.migrad_status == 0
        && attempt.hesse_status == 0
        && attempt.covariance_status >= 2
        && std::isfinite(attempt.minimum)
        && std::isfinite(attempt.edm)
        && attempt.edm <= kMaximumAcceptedEDM;

    minuit.mnprin(3, attempt.minimum);
    print_parameter_vector(
        "Final free parameters:",
        parameter_names,
        attempt.values,
        &attempt.errors);
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

void save_fit_result(
    const std::string& file_name,
    NLL_estimator& fitter,
    const FitAttempt& best,
    int number_starts,
    long long base_seed)
{
    apply_gvv_fit_parameters(fitter, best.values.data());

    std::ofstream output(file_name.c_str());
    if (!output) {
        throw std::runtime_error("cannot write fit result: " + file_name);
    }
    output << std::setprecision(12);
    output << "# GVV fit result: best converged multistart solution\n";
    output << "# Model ids and reference conventions come from model.json.\n";
    output << "# SB1 coefficient -0.5; SB2 coefficient +0.25.\n";
    output << "# multistart n_starts " << number_starts
           << " base_seed " << base_seed
           << " best_start " << best.start_index
           << " best_seed " << best.seed
           << " elapsed_seconds " << best.elapsed_seconds
           << " at_parameter_boundary "
           << (best.at_parameter_boundary ? 1 : 0) << "\n";
    output << "# samples data " << fitter.DataEntries()
           << " normalization_mc " << fitter.NormalizationMCEntries() << "\n";
    output << "# minimization migrad_status " << best.migrad_status
           << " hesse_status " << best.hesse_status
           << " covariance_status " << best.covariance_status
           << " minimum " << best.minimum
           << " edm " << best.edm
           << " error_definition " << best.error_definition << "\n";

    const GVVCompiledModel& model = fitter.Model();
    const std::vector<std::string> parameter_names =
        gvv_fit_parameter_names(model);
    if (parameter_names.size() != best.values.size()
        || best.errors.size() != best.values.size()) {
        throw std::runtime_error(
            "fit-result parameter layout does not match best fit");
    }
    output << "# Machine-readable covariance ordering follows.\n";
    for (std::size_t index = 0; index < parameter_names.size(); ++index) {
        output << "parameter " << index << ' ' << parameter_names[index]
               << ' ' << best.values[index]
               << ' ' << best.errors[index] << '\n';
    }

    int parameter = 0;
    for (int term = 0; term < fitter.NumberTerms(); ++term) {
        const GVVTermMetadata& metadata = model.term_metadata[term];
        const int parameterization = metadata.coupling_parameterization;
        if (parameterization == GVV_COUPLING_FIXED_SCALE_AND_PHASE) {
            const DeviceComplex value = fitter.Coupling(term);
            output << "coupling " << metadata.id << ' '
                   << value.real << ' ' << value.imag << " fixed\n";
            continue;
        }
        if (parameterization == GVV_COUPLING_POSITIVE_REAL) {
            const double log_magnitude = best.values[parameter];
            const double magnitude = std::exp(log_magnitude);
            const double magnitude_error =
                magnitude * best.errors[parameter];
            output << "coupling " << metadata.id << ' '
                   << magnitude << " 0 "
                   << magnitude_error << " 0 phase_fixed"
                   << " log_rho " << log_magnitude
                   << " log_error " << best.errors[parameter] << '\n';
            ++parameter;
            continue;
        }
        output << "coupling " << metadata.id << ' '
               << best.values[parameter] << ' '
               << best.values[parameter + 1] << ' '
               << best.errors[parameter] << ' '
               << best.errors[parameter + 1] << '\n';
        parameter += 2;
    }
    for (int resonance = 0;
         resonance < fitter.NumberResonances();
         ++resonance) {
        const GVVResonanceParameters& state = fitter.Resonance(resonance);
        output << "resonance " << model.resonance_metadata[resonance].id
               << " model " << gvv_propagator_name(state.propagator_model)
               << " mass " << state.mass << " fixed";
        if (state.propagator_model == GVV_PROP_SUBTRACTED_FLATTE) {
            output << " Gamma_rest " << state.pole_width << " fixed";
        } else {
            output << " width " << state.pole_width << " fixed";
        }
        if (state.fit_sd_ratio) {
            output << " r_D_over_S " << state.sd_ratio
                   << " log_error " << best.errors[parameter++];
        }
        if (state.propagator_model == GVV_PROP_SUBTRACTED_FLATTE) {
            output << " R_omegaomega " << state.flatte_ratio;
            if (state.fit_flatte_ratio) {
                output << " log_error " << best.errors[parameter++];
            } else {
                output << " fixed";
            }
        }
        output << '\n';
    }
    output.close();
    if (!output) {
        throw std::runtime_error("failed to write fit result: " + file_name);
    }

    const std::string snapshot_name = file_name + ".model.json";
    std::ofstream snapshot(snapshot_name.c_str());
    if (!snapshot) {
        throw std::runtime_error(
            "cannot write fitted model snapshot: " + snapshot_name);
    }
    snapshot << model.definition.canonical_json;
    snapshot.close();
    if (!snapshot) {
        throw std::runtime_error(
            "failed to write fitted model snapshot: " + snapshot_name);
    }
}

void save_covariance_matrix(
    const std::string& file_name,
    const FitAttempt& best)
{
    const int number_parameters = static_cast<int>(best.values.size());
    if (static_cast<int>(best.covariance.size())
        != number_parameters * number_parameters) {
        throw std::runtime_error("invalid best-fit covariance dimensions");
    }
    std::ofstream output(file_name.c_str());
    if (!output) {
        throw std::runtime_error(
            "cannot write covariance matrix: " + file_name);
    }
    output << std::setprecision(12);
    for (int row = 0; row < number_parameters; ++row) {
        for (int column = 0; column < number_parameters; ++column) {
            if (column != 0) {
                output << ' ';
            }
            output << best.covariance[
                static_cast<std::size_t>(row) * number_parameters + column];
        }
        output << '\n';
    }
}

bool is_better_attempt(const FitAttempt& candidate, const FitAttempt& best)
{
    if (!candidate.valid) {
        return false;
    }
    if (!best.valid) {
        return true;
    }
    constexpr double nll_tie_tolerance = 1.0e-8;
    if (candidate.minimum < best.minimum - nll_tie_tolerance) {
        return true;
    }
    if (std::fabs(candidate.minimum - best.minimum) <= nll_tie_tolerance) {
        if (candidate.covariance_status != best.covariance_status) {
            return candidate.covariance_status > best.covariance_status;
        }
        if (candidate.edm != best.edm) {
            return candidate.edm < best.edm;
        }
        return candidate.start_index < best.start_index;
    }
    return false;
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 5 || argc > 9) {
        print_usage(argv[0]);
        return 2;
    }

    const std::string data_file = argv[1];
    const std::string normalization_mc_file = argv[2];
    const std::string sb1_file = argv[3];
    const std::string sb2_file = argv[4];
    const std::string result_file =
        argc >= 6 ? argv[5] : "results/fit_result.txt";
    const std::string model_file =
        argc >= 9 ? argv[8] : "config/model.json";

    try {
        const int number_starts = argc >= 7
                                      ? parse_number_starts(argv[6])
                                      : kDefaultNumberStarts;
        const long long base_seed = argc >= 8
                                        ? parse_base_seed(argv[7])
                                        : kDefaultBaseSeed;

        // ------------------------ User input contract -----------------
        // This block is the only place that maps an input ROOT production
        // onto the internal [px,py,pz,E] GVV event representation.
        GVVBranchConfig branches;
        branches.tree_name = "Pwa";
        branches.branches = {{
            "p4_pip1", "p4_pim1", "p4_pi01",
            "p4_pip2", "p4_pim2", "p4_pi02", "p4_gam"}};
        branches.input_order = GVV_PX_PY_PZ_E;
        // -------------------------------------------------------------

        NLL_estimator estimator(
            gvv_load_compiled_model(model_file), branches);
        estimator.PrintModelSummary();
        estimator.LoadData(data_file);
        estimator.LoadNormalizationMC(normalization_mc_file);
        estimator.AddBackground(
            sb1_file, kSB1LikelihoodCoefficient, "SB1");
        estimator.AddBackground(
            sb2_file, kSB2LikelihoodCoefficient, "SB2");
        estimator.Prepare();

        const std::vector<GVVFitParameterSpec> parameter_layout =
            gvv_fit_parameter_layout(estimator.Model());
        const std::vector<std::string> parameter_names =
            gvv_fit_parameter_names(estimator.Model());
        const std::vector<Double_t> nominal_values =
            gvv_fit_parameters_from_model(estimator.Model());
        if (parameter_names.size() != nominal_values.size()) {
            throw std::runtime_error(
                "fit parameter name/value ordering is inconsistent");
        }

        std::cout << "GVV multistart configuration: n_starts="
                  << number_starts << " base_seed=" << base_seed
                  << "; start 0 is nominal; later starts randomize only "
                     "free coupling magnitudes/phases while preserving "
                     "the reference conventions\n";

        FitAttempt best;
        std::vector<FitAttempt> summaries;
        summaries.reserve(number_starts);
        for (int start = 0; start < number_starts; ++start) {
            const long long seed = base_seed + start;
            std::vector<Double_t> initial_values = nominal_values;
            if (start > 0) {
                randomize_free_couplings(
                    initial_values, parameter_layout, seed);
            }
            FitAttempt attempt = run_fit_attempt(
                estimator,
                parameter_names,
                parameter_layout,
                initial_values,
                start,
                seed);
            if (is_better_attempt(attempt, best)) {
                best = attempt;
                std::cout << "Start " << start
                          << " is the current best converged solution\n";
            }
            summaries.push_back(std::move(attempt));
        }

        std::cout
            << "\n================ GVV MULTISTART SUMMARY ================\n"
            << "start seed NLL MIGRAD HESSE covariance EDM boundary "
               "seconds accepted\n";
        for (const FitAttempt& attempt : summaries) {
            std::cout << attempt.start_index << ' '
                      << attempt.seed << ' '
                      << std::setprecision(12) << attempt.minimum << ' '
                      << attempt.migrad_status << ' '
                      << attempt.hesse_status << ' '
                      << attempt.covariance_status << ' '
                      << attempt.edm << ' '
                      << (attempt.at_parameter_boundary ? "yes" : "no")
                      << ' ' << attempt.elapsed_seconds << ' '
                      << (attempt.valid ? "yes" : "no") << '\n';
        }
        if (!best.valid) {
            throw std::runtime_error(
                "no multistart attempt passed the convergence criteria; "
                "no final fit output was written");
        }

        apply_gvv_fit_parameters(estimator, best.values.data());
        std::cout << "Selected best start " << best.start_index
                  << " (seed " << best.seed << ") with NLL "
                  << std::setprecision(12) << best.minimum << '\n';

        save_fit_result(
            result_file, estimator, best, number_starts, base_seed);
        save_covariance_matrix(kCovarianceFile, best);
        estimator.Project_fit_result(
            kProjectionFile,
            best.start_index,
            best.seed,
            best.minimum);

        std::cout << "Best fit result written to " << result_file << '\n'
                  << "Best covariance matrix written to "
                  << kCovarianceFile << '\n'
                  << "Best-fit projection written to "
                  << kProjectionFile << '\n';
    } catch (const std::exception& error) {
        std::cerr << "GVV fit aborted: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

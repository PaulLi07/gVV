// Nominal psi(3686) -> gamma omega omega fit.
// Run from the project root: bin/Fit.exe config/fit.json (on an allocated GPU).
//
// USER SETTINGS: edit config/fit.json for input samples, signed background
// coefficients, multistart/minimizer settings, and output paths/tag. Edit
// config/model.json for active Terms, couplings, propagators, and shared sigma.
// The main function below is the complete run workflow; helpers in this file
// own its likelihood and text report. Only reusable mechanics live in core/.

#include "core/Model.h"
#include "core/Sample.h"
#include "core/Amplitude.h"
#include "core/Minuit.h"
#include "core/IO.h"
#include "core/physics/Waves.cuh"
#include "core/physics/Propagators.cuh"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <utility>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <functional>
#include <iosfwd>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>

// 1. Signed likelihood arithmetic (normalization is an MC mean).
// Pure arithmetic helpers; GPU evaluation and sample roles are defined below.

namespace ctpwa {

// Monte-Carlo normalization shared by decay processes. The process supplies
// only event intensities; this layer owns probability/likelihood semantics.
inline double monte_carlo_normalization(
    const double* intensity,
    std::size_t number_events)
{
    if (intensity == nullptr || number_events == 0) {
        throw std::invalid_argument("normalization sample is empty");
    }
    double sum = 0.0;
    for (std::size_t event = 0; event < number_events; ++event) {
        if (!(intensity[event] >= 0.0) || !std::isfinite(intensity[event])) {
            throw std::domain_error(
                "normalization contains an invalid intensity");
        }
        sum += intensity[event];
    }
    const double result = sum / static_cast<double>(number_events);
    if (!(result > 0.0) || !std::isfinite(result)) {
        throw std::domain_error("normalization is not finite and positive");
    }
    return result;
}

inline double log_likelihood_contribution(
    const double* intensity,
    std::size_t number_events,
    double normalization,
    double coefficient)
{
    if (intensity == nullptr || !std::isfinite(coefficient)
        || !(normalization > 0.0) || !std::isfinite(normalization)) {
        throw std::invalid_argument("invalid likelihood input");
    }
    const double log_normalization = std::log(normalization);
    double logarithm_sum = 0.0;
    for (std::size_t event = 0; event < number_events; ++event) {
        if (!(intensity[event] > 0.0)
            || !std::isfinite(intensity[event])) {
            throw std::domain_error(
                "likelihood contains an invalid intensity");
        }
        logarithm_sum += std::log(intensity[event]) - log_normalization;
    }
    return coefficient * logarithm_sum;
}

} // namespace ctpwa

// 2. Human-readable fit report.
// Human-readable fit diagnostics. Machine consumers use FitState instead.

namespace ctpwa {

struct FitSampleSummary {
    std::string role;
    std::string label;
    std::string file;
    int entries = 0;
    double likelihood_coefficient = 0.0;
};

struct FitResultContext {
    std::string output_tag;
    std::string fit_config_file;
    std::string model_config_file;
    std::string model_name;
    std::string model_signature;
    std::vector<FitSampleSummary> samples;
};

using FitDetailWriter = std::function<void(std::ostream&)>;

void write_fit_result(
    const std::string& file_name,
    const FitSummary& summary,
    const FitOptions& options,
    const std::vector<FitParameterSpec>& parameters,
    const FitResultContext& context,
    const FitDetailWriter& write_details);

} // namespace ctpwa

// Complete user-facing diagnostics for one fit. This format is intentionally
// optimized for reading, while FitState JSON is the stable software contract.

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

namespace {
int find_propagator_layout_index(
    const std::vector<GVVFitParameterBinding>& layout,
    int resonance_index,
    GVVPropagatorParameterTarget target)
{
    for (std::size_t index = 0; index < layout.size(); ++index) {
        if (layout[index].target == GVVFitParameterTarget::PropagatorParameter
            && layout[index].target_index == resonance_index
            && layout[index].propagator_target == target) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

void gvv_write_fit_details(
    std::ostream& output,
    const GVVCompiledModel& model,
    const GVVParameterState& parameters,
    const std::vector<GVVFitParameterBinding>& layout,
    const ctpwa::FitAttempt& best)
{
    if (layout.size() != best.values.size()
        || layout.size() != best.errors.size()) {
        throw std::invalid_argument("GVV fit detail layout mismatch");
    }
    output << "model: " << model.definition.name << '\n'
           << "active_resonances: " << parameters.resonances.size() << '\n'
           << "active_terms: " << model.terms.size() << '\n'
           << "active_waves: " << model.active_wave_types.size() << "\n\n"
           << "# WAVE REGISTRY USED BY THE ACTIVE MODEL\n"
           << "# wave_id JPC coherence_class latex\n";
    for (const GVVWaveMetadata& wave : gvv_wave_registry()) {
        if (std::find(
                model.active_wave_types.begin(),
                model.active_wave_types.end(),
                wave.wave_type) != model.active_wave_types.end()) {
            output << "wave " << wave.id << ' ' << wave.jpc << ' '
                   << wave.coherence_class << ' ' << wave.latex << '\n';
        }
    }

    output << "\n# ACTIVE TERMS AND COUPLINGS\n"
           << "# term id label wave resonance JPC coherence coupling policy\n";
    std::size_t parameter = 0;
    for (std::size_t term = 0; term < model.terms.size(); ++term) {
        const GVVTermMetadata& metadata = model.term_metadata[term];
        const GVVResonanceMetadata& resonance = model.resonance_metadata[
            model.terms[term].resonance_index];
        const int parameterization = metadata.coupling_parameterization;
        output << "term " << metadata.id << " label=\"" << metadata.label
               << "\" wave=" << metadata.wave_id
               << " resonance=" << resonance.id
               << " JPC=" << metadata.jpc
               << " coherence=" << metadata.coherence_class
               << " coupling_mode=";
        if (parameterization == COUPLING_FIXED_SCALE_AND_PHASE) {
            const DeviceComplex value = parameters.couplings[term];
            output << "fixed_complex reference="
                   << ctpwa::coupling_reference_name(metadata.reference)
                   << " value=(" << value.real << ',' << value.imag
                   << ") status=fixed_reference\n";
            continue;
        }
        if (parameterization == COUPLING_POSITIVE_REAL) {
            const double log_magnitude = best.values.at(parameter);
            const double magnitude = std::exp(log_magnitude);
            output << "positive_real reference="
                   << ctpwa::coupling_reference_name(metadata.reference)
                   << " value=(" << magnitude << ",0)"
                   << " magnitude_error="
                   << magnitude * best.errors.at(parameter)
                   << " fitted_as=" << layout.at(parameter).fit.name
                   << " fitted_value=" << log_magnitude
                   << " fitted_error=" << best.errors.at(parameter) << '\n';
            ++parameter;
            continue;
        }
        output << "complex_cartesian reference=none value=("
               << best.values.at(parameter) << ','
               << best.values.at(parameter + 1) << ')'
               << " error=(" << best.errors.at(parameter) << ','
               << best.errors.at(parameter + 1) << ')'
               << " fitted_as=(" << layout.at(parameter).fit.name << ','
               << layout.at(parameter + 1).fit.name << ")\n";
        parameter += 2;
    }

    output << "\n# ACTIVE RESONANCES\n"
           << "# Values are the final physical values. The source status is "
              "taken from model.json.\n";
    for (std::size_t resonance = 0;
         resonance < parameters.resonances.size();
         ++resonance) {
        const ctpwa::PropagatorParameters& state = parameters.resonances[resonance];
        const GVVResonanceMetadata& metadata =
            model.resonance_metadata[resonance];
        output << "resonance " << metadata.id << " label=\""
               << metadata.label << "\" propagator=" << metadata.propagator_id
               << " compiled_model=\""
               << ctpwa::propagator_name(state.propagator_model) << "\"\n";
        const ctpwa::ResonanceDefinition& definition =
            model.definition.resonance(metadata.id);
        for (const GVVPropagatorParameterMetadata& parameter_metadata :
             metadata.parameters) {
            const std::string& name = parameter_metadata.source_name;
            const ctpwa::ParameterDefinition& source =
                definition.parameters.at(name);
            const double value = gvv_propagator_parameter_value(
                state, parameter_metadata.target);
            output << "  parameter " << name << " value=" << value
                   << " status=" << (source.fixed ? "fixed" : "free")
                   << " transform=" << source.transform;
            if (!parameter_metadata.unit.empty()) {
                output << " unit=" << parameter_metadata.unit;
            }
            if (source.has_lower_bound || source.has_upper_bound) {
                output << " bounds=[";
                output << (source.has_lower_bound
                    ? std::to_string(source.lower_bound) : "-inf");
                output << ',';
                output << (source.has_upper_bound
                    ? std::to_string(source.upper_bound) : "+inf");
                output << ']';
            }
            output << '\n';
            const int fitted_index = find_propagator_layout_index(
                layout,
                static_cast<int>(resonance),
                parameter_metadata.target);
            if (fitted_index >= 0) {
                output << "  fitted_parameter "
                       << layout[fitted_index].fit.name
                       << " value=" << best.values[fitted_index]
                       << " error=" << best.errors[fitted_index] << '\n';
            }
        }
    }
    output << "\n# SHARED OMEGA EFFECTIVE RESOLUTION\n"
           << "omega_resolution_sigma=" << parameters.omega_resolution_sigma
           << " GeV; gaussian_mean=0; convolution=complex_amplitude_in_mass\n";
    for (std::size_t index = 0; index < layout.size(); ++index) {
        if (layout[index].target == GVVFitParameterTarget::OmegaResolutionSigma) {
            output << "  fitted_parameter " << layout[index].fit.name
                   << " value=" << best.values[index] << " error=" << best.errors[index]
                   << " physical_sigma_error=" << parameters.omega_resolution_sigma * best.errors[index]
                   << " GeV (linear error propagation)\n";
        }
    }

}

// 3. Nominal sample roles and shared amplitude evaluation.

// Sample roles and signed likelihood policy are specific to this nominal fit.
// A future FitBins.cu can own a different objective while reusing GVVAmplitude.
class FitLikelihood {
public:
    explicit FitLikelihood(GVVCompiledModel model)
        : model_(std::move(model)), parameters_(model_.initial_parameters), amplitude_(model_) {}
    void LoadData(const std::string& file)
    {
        data_ = LoadSample(file, "data");
    }
    void LoadNormalizationMC(const std::string& file)
    {
        normalization_mc_ = LoadSample(file, "normalization MC");
    }
    void AddBackground(const std::string& file, double coefficient, const std::string& label)
    {
        backgrounds_.push_back({LoadSample(file, label), coefficient});
    }
    void Prepare()
    {
        amplitude_.Prepare(*normalization_mc_);
        amplitude_.Prepare(*data_);
        for (auto& background : backgrounds_) {
            amplitude_.Prepare(*background.sample);
        }
        const auto& width = amplitude_.WidthConfig();
        std::cout << "GVV samples, F matrices, and omega width table prepared ("
                  << width.table_size << " mass points, " << width.dalitz_bins << "x"
                  << width.dalitz_bins << " Dalitz midpoint grid, " << width.minimum_mass
                  << "-" << width.maximum_mass << " GeV, clamped outside)\n";
    }
    double LogLikelihood()
    {
        // One parameter update serves data, accepted MC, and every sideband.
        amplitude_.SetParameters(parameters_);
        const double normalization = ctpwa::monte_carlo_normalization(
            amplitude_.EvaluateIntensity(*normalization_mc_), normalization_mc_->Entries());
        double result = ctpwa::log_likelihood_contribution(
            amplitude_.EvaluateIntensity(*data_), data_->Entries(), normalization, +1.0);
        if (!std::isfinite(result)) return -1.0e100;
        for (auto& background : backgrounds_) {
            const double contribution = ctpwa::log_likelihood_contribution(
                amplitude_.EvaluateIntensity(*background.sample), background.sample->Entries(),
                normalization, background.likelihood_coefficient);
            if (!std::isfinite(contribution)) return -1.0e100;
            result += contribution;
        }
        return result;
    }
    GVVParameterState& Parameters() { return parameters_; }
    const GVVCompiledModel& Model() const { return model_; }
    int NumberTerms() const { return static_cast<int>(model_.terms.size()); }
    int NumberResonances() const { return static_cast<int>(parameters_.resonances.size()); }
    int DataEntries() const { return data_->Entries(); }
    int NormalizationMCEntries() const { return normalization_mc_->Entries(); }
    const GVVSample& BackgroundSampleAt(std::size_t i) const { return *backgrounds_[i].sample; }
    void PrintModelSummary() const;
    void WriteProjection(const std::string& file, const std::string& tag,
                         const std::string& signature, const ctpwa::FitAttempt& best)
    {
        amplitude_.SetParameters(parameters_);
        std::vector<GVVProjectionBackground> backgrounds;
        for (const auto& item : backgrounds_)
            backgrounds.push_back({item.sample.get(), item.likelihood_coefficient});
        write_gvv_projection(amplitude_, *normalization_mc_, *data_, backgrounds,
            file, tag, signature, best.start_index, best.seed, best.minimum);
    }
private:
    struct Background {
        std::unique_ptr<GVVSample> sample;
        double likelihood_coefficient;
    };
    std::unique_ptr<GVVSample> LoadSample(const std::string& file, const std::string& label)
    {
        auto sample = std::make_unique<GVVSample>(label);
        sample->Load(file, GVVBranchConfig{});
        return sample;
    }
    const GVVCompiledModel model_;
    GVVParameterState parameters_;
    GVVAmplitude amplitude_;
    std::unique_ptr<GVVSample> normalization_mc_, data_;
    std::vector<Background> backgrounds_;
};
void FitLikelihood::PrintModelSummary() const
{
    std::cout << "GVV model '" << model_.definition.name << "': "
              << NumberResonances() << " active Resonances, "
              << NumberTerms() << " active coherent Terms, "
              << model_.active_wave_types.size() << " active Waves\n";
    for (int index = 0; index < NumberResonances(); ++index) {
        const ctpwa::PropagatorParameters& resonance = parameters_.resonances[index];
        const GVVResonanceMetadata& metadata =
            model_.resonance_metadata[index];
        std::cout << "  " << std::setw(10)
                  << metadata.id
                  << "  model=" << ctpwa::propagator_name(
                         resonance.propagator_model);
        for (const GVVPropagatorParameterMetadata& parameter :
             metadata.parameters) {
            const double value = gvv_propagator_parameter_value(
                resonance, parameter.target);
            std::cout << "  " << parameter.display_name << '=' << value;
            if (!parameter.unit.empty()) {
                std::cout << ' ' << parameter.unit;
            }
            for (const GVVFitParameterBinding& fit :
                 model_.parameters) {
                if (fit.target_index == index
                    && fit.target == GVVFitParameterTarget::PropagatorParameter
                    && fit.propagator_target == parameter.target) {
                    std::cout << " (fitted as " << fit.fit.name << ')';
                }
            }
        }
        std::cout << '\n';
    }
    for (int term = 0; term < NumberTerms(); ++term) {
        const GVVTermMetadata& metadata = model_.term_metadata[term];
        if (metadata.reference == ctpwa::CouplingReference::ScaleAndPhase) {
            const DeviceComplex coupling = parameters_.couplings[term];
            std::cout << "  scale-and-phase reference amplitude: "
                      << metadata.id << " = " << coupling.real;
            if (coupling.imag >= 0.0) {
                std::cout << " + " << coupling.imag << "i\n";
            } else {
                std::cout << " - " << -coupling.imag << "i\n";
            }
        } else if (metadata.reference == ctpwa::CouplingReference::Phase) {
            std::cout << "  " << metadata.coherence_class
                      << " phase reference amplitude: " << metadata.id
                      << " = rho + 0i, rho > 0 and fitted as log(rho)\n";
        }
    }
}

} // namespace

// 4. Run configuration -> samples -> multistart -> selected fit -> outputs.
namespace {

void print_usage(const char* executable)
{
    std::cerr << "Usage: " << executable << " [config/fit.json]\n"
              << "All model, sample, minimizer, and output settings are read "
                 "from that one fit configuration.\n";
}

void print_multistart_summary(const ctpwa::FitSummary& summary)
{
    std::cout
        << "\n================ MULTISTART SUMMARY ================\n"
        << "start seed NLL MIGRAD HESSE covariance EDM boundary seconds "
           "accepted\n";
    for (const ctpwa::FitAttempt& attempt : summary.attempts) {
        std::cout << attempt.start_index << ' '
                  << attempt.seed << ' '
                  << std::setprecision(12) << attempt.minimum << ' '
                  << attempt.migrad_status << ' '
                  << attempt.hesse_status << ' '
                  << attempt.covariance_status << ' '
                  << attempt.edm << ' '
                  << (attempt.at_parameter_boundary ? "yes" : "no") << ' '
                  << attempt.elapsed_seconds << ' '
                  << (attempt.valid ? "yes" : "no") << '\n';
    }
}

} // namespace

#ifndef GVV_FIT_NO_MAIN
int main(int argc, char* argv[])
{
    if (argc > 2) {
        print_usage(argv[0]);
        return 2;
    }
    const std::string fit_config_file =
        argc == 2 ? argv[1] : "config/fit.json";

    try {
        const ctpwa::FitRunConfig config =
            ctpwa::load_fit_run_config(fit_config_file);
        std::filesystem::create_directories(config.output.directory);
        std::filesystem::create_directories(config.output.log_directory);

        // A. Load the active model and assign this analysis's sample roles.
        FitLikelihood likelihood(
            gvv_load_compiled_model(config.model_file));
        likelihood.PrintModelSummary();
        likelihood.LoadData(config.inputs.data_file);
        likelihood.LoadNormalizationMC(
            config.inputs.normalization_mc_file);
        for (const ctpwa::WeightedSampleConfig& background :
             config.inputs.backgrounds) {
            likelihood.AddBackground(
                background.file,
                background.likelihood_coefficient,
                background.label);
        }
        likelihood.Prepare();

        const std::vector<GVVFitParameterBinding> mapping =
            gvv_fit_parameter_layout(likelihood.Model());
        const std::vector<ctpwa::FitParameterSpec> parameters =
            gvv_fit_parameter_specs(mapping);

        std::cout << "Fit configuration: " << fit_config_file << '\n'
                  << "Model configuration: " << config.model_file << '\n'
                  << "Multistart: n_starts="
                  << config.minimizer.number_starts
                  << " base_seed=" << config.minimizer.base_seed
                  << "; start 0 is nominal and later starts randomize only "
                     "free coupling magnitudes/phases\n"
                  << "Output tag: " << config.output.tag
                  << " (existing files with this tag are overwritten)\n";

        // B. Minuit sees a flat vector; Model updates the small numeric state.
        const ctpwa::FitObjective objective =
            [&](const std::vector<double>& values) {
                gvv_apply_fit_parameters(
                    likelihood.Parameters(), mapping, values);
                return -likelihood.LogLikelihood();
            };
        const ctpwa::FitSummary summary = ctpwa::run_multistart_fit(
            parameters, config.minimizer, objective);
        print_multistart_summary(summary);
        if (!summary.best.valid) {
            throw std::runtime_error(
                "no multistart attempt passed the convergence criteria; "
                "no final fit output was written");
        }

        // C. Restore the selected attempt before writing any result or projection.
        gvv_apply_fit_parameters(
            likelihood.Parameters(), mapping, summary.best.values);
        std::cout << "Selected best start " << summary.best.start_index
                  << " (seed " << summary.best.seed << ") with NLL "
                  << std::setprecision(12) << summary.best.minimum << '\n';

        // D. Record sample provenance, readable diagnostics, and exact fitted state.
        ctpwa::FitResultContext context;
        context.output_tag = config.output.tag;
        context.fit_config_file = fit_config_file;
        context.model_config_file = config.model_file;
        context.model_name = likelihood.Model().definition.name;
        context.model_signature = gvv_model_signature(
            likelihood.Model().definition);
        context.samples.push_back({
            "data", "data", config.inputs.data_file,
            likelihood.DataEntries(), +1.0});
        context.samples.push_back({
            "normalization_mc", "normalization MC",
            config.inputs.normalization_mc_file,
            likelihood.NormalizationMCEntries(), 0.0});
        for (std::size_t index = 0;
             index < config.inputs.backgrounds.size();
             ++index) {
            const ctpwa::WeightedSampleConfig& background =
                config.inputs.backgrounds[index];
            context.samples.push_back({
                "background", background.label, background.file,
                likelihood.BackgroundSampleAt(index).Entries(),
                background.likelihood_coefficient});
        }
        ctpwa::write_fit_result(
            config.output.result_file(),
            summary,
            config.minimizer,
            parameters,
            context,
            [&](std::ostream& output) {
                gvv_write_fit_details(
                    output, likelihood.Model(), likelihood.Parameters(), mapping, summary.best);
            });

        ctpwa::FitState state;
        state.output_tag = config.output.tag;
        state.fit_config_file = fit_config_file;
        state.model_config_file = config.model_file;
        state.model_name = context.model_name;
        state.model_json = likelihood.Model().definition.canonical_json;
        state.model_definition_signature =
            ctpwa::model_definition_signature(likelihood.Model().definition);
        state.model_implementation_signature =
            gvv_model_implementation_signature(likelihood.Model().definition);
        state.model_signature = context.model_signature;
        state.best = summary.best;
        state.parameters = parameters;
        ctpwa::write_fit_state(config.output.state_file(), state);
        likelihood.WriteProjection(config.output.projection_file(),
            config.output.tag, context.model_signature, summary.best);

        std::cout << "Fit result written to "
                  << config.output.result_file() << '\n'
                  << "Machine-readable fit state written to "
                  << config.output.state_file() << '\n'
                  << "Projection written to "
                  << config.output.projection_file() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "GVV fit aborted: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

#endif // GVV_FIT_NO_MAIN

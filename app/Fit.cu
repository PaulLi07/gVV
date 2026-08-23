// Fit executable glue: load one run configuration, assemble the gVV process
// likelihood, invoke the generic fit engine, and write the agreed outputs.
// Physics formulae and Minuit implementation details deliberately live below
// this layer.
#include "framework/fit/FitConfig.h"
#include "framework/fit/FitEngine.h"
#include "framework/fit/FitOutput.h"
#include "framework/fit/FitState.h"
#include "process/FitLikelihood.h"
#include "process/ModelCompiler.h"
#include "process/ParameterMapping.h"
#include "process/ProjectionWriter.h"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

GVVBranchConfig gvv_input_contract()
{
    // The process layer owns this ROOT-to-event boundary. A future decay
    // process replaces this function and its ProcessEvent/SampleLoader code;
    // the framework fit engine and likelihood arithmetic remain unchanged.
    GVVBranchConfig branches;
    branches.tree_name = "Pwa";
    branches.branches = {{
        "p4_pip1", "p4_pim1", "p4_pi01",
        "p4_pip2", "p4_pim2", "p4_pi02", "p4_gam"}};
    branches.input_order = GVV_PX_PY_PZ_E;
    return branches;
}

} // namespace

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

        FitLikelihood likelihood(
            gvv_load_compiled_model(config.model_file),
            gvv_input_contract());
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

        const ctpwa::FitObjective objective =
            [&](const std::vector<double>& values) {
                gvv_apply_fit_parameters(
                    likelihood.MutableModel(), mapping, values);
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

        gvv_apply_fit_parameters(
            likelihood.MutableModel(), mapping, summary.best.values);
        std::cout << "Selected best start " << summary.best.start_index
                  << " (seed " << summary.best.seed << ") with NLL "
                  << std::setprecision(12) << summary.best.minimum << '\n';

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
                    output, likelihood.Model(), mapping, summary.best);
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
            gvv_amplitude_implementation_signature();
        state.model_signature = context.model_signature;
        state.best = summary.best;
        state.parameters = parameters;
        ctpwa::write_fit_state(config.output.state_file(), state);
        write_gvv_projection(
            likelihood,
            config.output.projection_file(),
            config.output.tag,
            context.model_signature,
            summary.best.start_index,
            summary.best.seed,
            summary.best.minimum);

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

// Post-fit fractions, interference, efficiencies, and covariance propagation.
// Run on an allocated GPU: bin/Post.exe FIT_STATE TRUTH_MC SELECTED_MC
// USER SETTINGS are the command-line inputs. Truth and selected MC must
// represent the same generated exposure; efficiencies use weighted sums.
// Observable definitions and finite differences stay together in this script.
// Post Calculation executable: restore a fitted model, integrate pairwise
// components over truth/selected MC, and propagate the fit covariance.

#include "core/Model.h"
#include "core/Sample.h"
#include "core/Amplitude.h"
#include "core/Minuit.h"
#include "core/IO.h"
#include "core/AmplitudeKernels.cuh"
#include "TFile.h"
#include "TMatrixDSym.h"
#include "TTree.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Observable {
    std::string category;
    std::string name;
    int first = -1;
    int second = -1;
    double value = 0.0;
    double error = 0.0;
    double truth_integral = 0.0;
    double selected_integral = 0.0;
};

struct ObservableSet {
    std::vector<Observable> values;
    double truth_total = 0.0;
    double selected_total = 0.0;
    double fraction_closure = 0.0;
    double group_closure = 0.0;
};

double sum(const std::vector<double>& values)
{
    return std::accumulate(values.begin(), values.end(), 0.0);
}
// GPU-backed evaluator shared by all Post Calculation observables. It owns
// truth/selected samples and reduces pairwise Term components in bounded
// event batches; no event-by-pair matrix is retained.

struct GVVIntegratedComponents {
    std::vector<double> truth;
    std::vector<double> selected;
};

class GVVComponentEvaluator {
public:
    GVVComponentEvaluator(
        const std::string& truth_file,
        const std::string& selected_file,
        const GVVBranchConfig& branches,
        GVVCompiledModel model,
        std::vector<GVVFitParameterBinding> layout);

    GVVComponentEvaluator(const GVVComponentEvaluator&) = delete;
    GVVComponentEvaluator& operator=(const GVVComponentEvaluator&) = delete;

    GVVIntegratedComponents Evaluate(const std::vector<double>& values);
    void ValidateTotal(
        const std::vector<double>& values,
        const GVVIntegratedComponents& components);

    int TruthEntries() const;
    int SelectedEntries() const;
    int NumberTerms() const;
    const GVVCompiledModel& Model() const;
    const std::vector<GVVFitParameterBinding>& Layout() const;

private:
    void SetValues(const std::vector<double>& values);
    std::vector<double> EvaluateSample(GVVSample& sample);
    void ValidateSample(GVVSample& sample, double component_sum);
    const GVVCompiledModel model_;
    const std::vector<GVVFitParameterBinding> layout_;
    GVVParameterState state_;
    GVVAmplitude amplitude_;
    GVVSample truth_;
    GVVSample selected_;
    int component_batch_capacity_;
};

constexpr int kPostComponentBatchSize = 4096;
GVVComponentEvaluator::GVVComponentEvaluator(
    const std::string& truth_file, const std::string& selected_file,
    const GVVBranchConfig& branches, GVVCompiledModel model,
    std::vector<GVVFitParameterBinding> layout)
    : model_(std::move(model)), layout_(std::move(layout)),
      state_(model_.initial_parameters), amplitude_(model_),
      truth_("generated truth MC"), selected_("selected normalization MC")
{
    truth_.Load(truth_file, branches);
    selected_.Load(selected_file, branches);
    amplitude_.Prepare(truth_);
    amplitude_.Prepare(selected_);
    component_batch_capacity_ = std::min(kPostComponentBatchSize,
        std::max(truth_.Entries(), selected_.Entries()));
}
void GVVComponentEvaluator::SetValues(const std::vector<double>& values)
{
    gvv_apply_fit_parameters(state_, layout_, values);
    amplitude_.SetParameters(state_);
}
std::vector<double> GVVComponentEvaluator::EvaluateSample(GVVSample& sample)
{
    return amplitude_.IntegrateComponents(sample, component_batch_capacity_);
}
GVVIntegratedComponents GVVComponentEvaluator::Evaluate(
    const std::vector<double>& values)
{
    SetValues(values);
    return {EvaluateSample(truth_), EvaluateSample(selected_)};
}

void GVVComponentEvaluator::ValidateSample(
    GVVSample& sample,
    double component_sum)
{
    amplitude_.EvaluateIntensity(sample);
    double direct_sum = 0.0;
    for (int event = 0; event < sample.Entries(); ++event) {
        direct_sum += sample.IntensityBuffer()[event];
    }
    const double relative = std::fabs(direct_sum - component_sum)
                            / std::max(1.0, std::fabs(direct_sum));
    if (relative > 1.0e-9) {
        throw std::runtime_error(
            "component/PDF closure failed for " + sample.Label());
    }
}

void GVVComponentEvaluator::ValidateTotal(
    const std::vector<double>& values,
    const GVVIntegratedComponents& components)
{
    SetValues(values);
    ValidateSample(truth_, sum(components.truth));
    ValidateSample(selected_, sum(components.selected));
}

int GVVComponentEvaluator::TruthEntries() const { return truth_.Entries(); }
int GVVComponentEvaluator::SelectedEntries() const { return selected_.Entries(); }
int GVVComponentEvaluator::NumberTerms() const
{
    return static_cast<int>(model_.terms.size());
}
const GVVCompiledModel& GVVComponentEvaluator::Model() const { return model_; }
const std::vector<GVVFitParameterBinding>& GVVComponentEvaluator::Layout() const
{
    return layout_;
}

double group_sum(
    const std::vector<double>& values,
    const GVVCompiledModel& model,
    const std::string& group)
{
    double result = 0.0;
    const int count = static_cast<int>(model.terms.size());
    for (int first = 0; first < count; ++first) {
        if (model.term_metadata[first].jpc != group) continue;
        for (int second = first; second < count; ++second) {
            if (model.term_metadata[second].jpc == group) {
                result += values[ctpwa::component_pair_index(
                    first, second, count)];
            }
        }
    }
    return result;
}

double cross_group_sum(
    const std::vector<double>& values,
    const GVVCompiledModel& model,
    const std::string& first_group,
    const std::string& second_group)
{
    double result = 0.0;
    const int count = static_cast<int>(model.terms.size());
    for (int first = 0; first < count; ++first) {
        for (int second = first + 1; second < count; ++second) {
            const std::string& left = model.term_metadata[first].jpc;
            const std::string& right = model.term_metadata[second].jpc;
            if ((left == first_group && right == second_group)
                || (left == second_group && right == first_group)) {
                result += values[ctpwa::component_pair_index(
                    first, second, count)];
            }
        }
    }
    return result;
}

std::vector<std::string> jpc_groups(const GVVCompiledModel& model)
{
    std::vector<std::string> result;
    for (const GVVTermMetadata& term : model.term_metadata) {
        if (std::find(result.begin(), result.end(), term.jpc) == result.end()) {
            result.push_back(term.jpc);
        }
    }
    return result;
}

ObservableSet build_observables(
    const GVVIntegratedComponents& components,
    const GVVCompiledModel& model)
{
    ObservableSet result;
    result.truth_total = sum(components.truth);
    result.selected_total = sum(components.selected);
    const int count = static_cast<int>(model.terms.size());
    double reconstructed = 0.0;

    for (int term = 0; term < count; ++term) {
        const int pair = ctpwa::component_pair_index(term, term, count);
        const double truth = components.truth[pair];
        const double selected = components.selected[pair];
        if (!(truth > 0.0)) {
            throw std::runtime_error(
                "non-positive truth integral for active Term "
                + model.term_metadata[term].id);
        }
        result.values.push_back({
            "fit_fraction", model.term_metadata[term].id, term, term,
            truth / result.truth_total, 0.0, truth, selected});
        result.values.push_back({
            "efficiency_component", model.term_metadata[term].id, term, term,
            selected / truth, 0.0, truth, selected});
        reconstructed += truth;
    }
    for (int first = 0; first < count; ++first) {
        for (int second = first + 1; second < count; ++second) {
            const int pair = ctpwa::component_pair_index(first, second, count);
            const double truth = components.truth[pair];
            result.values.push_back({
                "interference",
                model.term_metadata[first].id + "__"
                    + model.term_metadata[second].id,
                first, second, truth / result.truth_total, 0.0, truth,
                components.selected[pair]});
            reconstructed += truth;
        }
    }
    result.fraction_closure = reconstructed / result.truth_total - 1.0;
    result.values.push_back({
        "efficiency_total", "total_coherent_model", -1, -1,
        result.selected_total / result.truth_total, 0.0,
        result.truth_total, result.selected_total});

    const std::vector<std::string> groups = jpc_groups(model);
    double group_reconstruction = 0.0;
    for (const std::string& group : groups) {
        const double truth = group_sum(components.truth, model, group);
        const double selected = group_sum(components.selected, model, group);
        if (!(truth > 0.0)) {
            throw std::runtime_error(
                "non-positive truth integral for JPC group " + group);
        }
        result.values.push_back({
            "fit_fraction_group", group, -1, -1,
            truth / result.truth_total, 0.0, truth, selected});
        result.values.push_back({
            "efficiency_group", group, -1, -1,
            selected / truth, 0.0, truth, selected});
        group_reconstruction += truth;
    }
    for (std::size_t first = 0; first < groups.size(); ++first) {
        for (std::size_t second = first + 1; second < groups.size(); ++second) {
            const double truth = cross_group_sum(
                components.truth, model, groups[first], groups[second]);
            const double selected = cross_group_sum(
                components.selected, model, groups[first], groups[second]);
            result.values.push_back({
                "interference_group", groups[first] + "__" + groups[second],
                -1, -1, truth / result.truth_total, 0.0, truth, selected});
            group_reconstruction += truth;
        }
    }
    result.group_closure = group_reconstruction / result.truth_total - 1.0;
    return result;
}

void propagate_errors(
    GVVComponentEvaluator& evaluator,
    const ctpwa::FitState& fit,
    ObservableSet& central,
    std::vector<double>& observable_covariance)
{
    const int n_parameters = static_cast<int>(fit.parameters.size());
    const int n_observables = static_cast<int>(central.values.size());
    std::vector<double> gradients(
        static_cast<std::size_t>(n_parameters) * n_observables, 0.0);

    for (int parameter = 0; parameter < n_parameters; ++parameter) {
        const GVVFitParameterBinding& binding = evaluator.Layout()[parameter];
        const double variance = fit.best.covariance[
            static_cast<std::size_t>(parameter) * n_parameters + parameter];
        const double sigma = std::sqrt(std::max(0.0, variance));
        const double requested_step = std::max(
            1.0e-5 * std::max(1.0, std::fabs(fit.best.values[parameter])),
            0.05 * sigma);
        const double minus_room = binding.fit.has_lower_bound
            ? fit.best.values[parameter] - binding.fit.lower_bound
            : requested_step;
        const double plus_room = binding.fit.has_upper_bound
            ? binding.fit.upper_bound - fit.best.values[parameter]
            : requested_step;
        const bool full_minus = !binding.fit.has_lower_bound
            || minus_room > requested_step;
        const bool full_plus = !binding.fit.has_upper_bound
            || plus_room > requested_step;
        bool use_minus = false;
        bool use_plus = false;
        double minus_step = 0.0;
        double plus_step = 0.0;
        if (full_minus && full_plus) {
            // Use a central difference only when the requested step fits on
            // both sides. A nearly active bound must not collapse this step
            // to a noise-dominated value.
            use_minus = true;
            use_plus = true;
            minus_step = requested_step;
            plus_step = requested_step;
        } else if (full_plus) {
            use_plus = true;
            plus_step = requested_step;
        } else if (full_minus) {
            use_minus = true;
            minus_step = requested_step;
        } else if (plus_room >= minus_room && plus_room > 0.0) {
            use_plus = true;
            plus_step = 0.5 * plus_room;
        } else if (minus_room > 0.0) {
            use_minus = true;
            minus_step = 0.5 * minus_room;
        }
        const double coordinate_scale = std::max(
            1.0, std::fabs(fit.best.values[parameter]));
        const double minimum_step = std::sqrt(
            std::numeric_limits<double>::epsilon()) * coordinate_scale;
        if ((!use_minus && !use_plus)
            || (use_minus && minus_step <= minimum_step)
            || (use_plus && plus_step <= minimum_step)) {
            throw std::runtime_error(
                "fit-parameter bounds leave no resolvable finite-difference step");
        }

        ObservableSet plus;
        ObservableSet minus;
        if (use_plus) {
            std::vector<double> values = fit.best.values;
            values[parameter] += plus_step;
            plus = build_observables(evaluator.Evaluate(values), evaluator.Model());
        }
        if (use_minus) {
            std::vector<double> values = fit.best.values;
            values[parameter] -= minus_step;
            minus = build_observables(evaluator.Evaluate(values), evaluator.Model());
        }
        for (int observable = 0; observable < n_observables; ++observable) {
            double derivative = 0.0;
            if (use_plus && use_minus) {
                derivative = (plus.values[observable].value
                              - minus.values[observable].value)
                             / (plus_step + minus_step);
            } else if (use_plus) {
                derivative = (plus.values[observable].value
                              - central.values[observable].value) / plus_step;
            } else {
                derivative = (central.values[observable].value
                              - minus.values[observable].value) / minus_step;
            }
            gradients[static_cast<std::size_t>(observable) * n_parameters
                      + parameter] = derivative;
        }
        std::cout << "Post derivative " << parameter + 1 << '/'
                  << n_parameters << ": " << fit.parameters[parameter].name
                  << " step(-,+)=(" << minus_step << ',' << plus_step
                  << ")\n";
    }

    observable_covariance.assign(
        static_cast<std::size_t>(n_observables) * n_observables, 0.0);
    for (int first = 0; first < n_observables; ++first) {
        for (int second = 0; second < n_observables; ++second) {
            double value = 0.0;
            for (int row = 0; row < n_parameters; ++row) {
                for (int column = 0; column < n_parameters; ++column) {
                    value += gradients[
                                 static_cast<std::size_t>(first) * n_parameters
                                 + row]
                             * fit.best.covariance[
                                 static_cast<std::size_t>(row) * n_parameters
                                 + column]
                             * gradients[
                                 static_cast<std::size_t>(second) * n_parameters
                                 + column];
                }
            }
            observable_covariance[
                static_cast<std::size_t>(first) * n_observables + second] = value;
        }
        central.values[first].error = std::sqrt(std::max(
            0.0, observable_covariance[
                static_cast<std::size_t>(first) * n_observables + first]));
    }
}

void write_text(
    const std::string& file_name,
    const ctpwa::FitState& fit,
    const ObservableSet& result)
{
    std::ofstream output(file_name.c_str(), std::ios::trunc);
    if (!output) throw std::runtime_error("cannot write " + file_name);
    output << std::setprecision(12)
           << "# GVV Post Calculation observables\n"
           << "# output_tag " << fit.output_tag << '\n'
           << "# model_signature " << fit.model_signature << '\n'
           << "# truth_total " << result.truth_total
           << " selected_total " << result.selected_total << '\n'
           << "# fraction_closure " << result.fraction_closure
           << " group_closure " << result.group_closure << '\n'
           << "# category name value error truth_integral selected_integral\n";
    for (const Observable& value : result.values) {
        output << value.category << ' ' << value.name << ' '
               << value.value << ' ' << value.error << ' '
               << value.truth_integral << ' ' << value.selected_integral << '\n';
    }
}

void copy_text(char* destination, std::size_t size, const std::string& source)
{
    std::snprintf(destination, size, "%s", source.c_str());
}

void write_root(
    const std::string& file_name,
    const ctpwa::FitState& fit,
    const ObservableSet& result,
    const std::vector<double>& covariance_values,
    const GVVComponentEvaluator& evaluator)
{
    TFile output(file_name.c_str(), "RECREATE");
    if (output.IsZombie()) throw std::runtime_error("cannot write " + file_name);
    int index = 0;
    int first = -1;
    int second = -1;
    char category[32] = {0};
    char name[128] = {0};
    double value = 0.0;
    double error = 0.0;
    double truth_integral = 0.0;
    double selected_integral = 0.0;
    TTree observables("observables", "Post Calculation observables");
    observables.Branch("index", &index, "index/I");
    observables.Branch("category", category, "category/C");
    observables.Branch("name", name, "name/C");
    observables.Branch("first", &first, "first/I");
    observables.Branch("second", &second, "second/I");
    observables.Branch("value", &value, "value/D");
    observables.Branch("error", &error, "error/D");
    observables.Branch("truth_integral", &truth_integral, "truth_integral/D");
    observables.Branch(
        "selected_integral", &selected_integral, "selected_integral/D");
    for (std::size_t row = 0; row < result.values.size(); ++row) {
        const Observable& source = result.values[row];
        index = static_cast<int>(row);
        first = source.first;
        second = source.second;
        copy_text(category, sizeof(category), source.category);
        copy_text(name, sizeof(name), source.name);
        value = source.value;
        error = source.error;
        truth_integral = source.truth_integral;
        selected_integral = source.selected_integral;
        observables.Fill();
    }

    int n_truth = evaluator.TruthEntries();
    int n_selected = evaluator.SelectedEntries();
    int n_observables = static_cast<int>(result.values.size());
    int schema_version = 1;
    double fraction_closure = result.fraction_closure;
    double group_closure = result.group_closure;
    char output_tag[64] = {0};
    char model_signature[64] = {0};
    copy_text(output_tag, sizeof(output_tag), fit.output_tag);
    copy_text(model_signature, sizeof(model_signature), fit.model_signature);
    TTree metadata("metadata", "Post Calculation provenance");
    metadata.Branch("schema_version", &schema_version, "schema_version/I");
    metadata.Branch("output_tag", output_tag, "output_tag/C");
    metadata.Branch("model_signature", model_signature, "model_signature/C");
    metadata.Branch("n_truth", &n_truth, "n_truth/I");
    metadata.Branch("n_selected", &n_selected, "n_selected/I");
    metadata.Branch("n_observables", &n_observables, "n_observables/I");
    metadata.Branch("fraction_closure", &fraction_closure, "fraction_closure/D");
    metadata.Branch("group_closure", &group_closure, "group_closure/D");
    metadata.Fill();

    TMatrixDSym covariance(n_observables);
    TMatrixDSym correlation(n_observables);
    for (int row = 0; row < n_observables; ++row) {
        for (int column = 0; column < n_observables; ++column) {
            covariance(row, column) = covariance_values[
                static_cast<std::size_t>(row) * n_observables + column];
        }
    }
    for (int row = 0; row < n_observables; ++row) {
        for (int column = 0; column < n_observables; ++column) {
            const double denominator = std::sqrt(std::max(
                0.0, covariance(row, row) * covariance(column, column)));
            correlation(row, column) = denominator > 0.0
                ? covariance(row, column) / denominator : 0.0;
        }
    }
    covariance.Write("observable_covariance");
    correlation.Write("observable_correlation");
    observables.Write();
    metadata.Write();
    output.Close();
}

void write_latex(
    const std::string& file_name,
    const ObservableSet& result,
    const GVVCompiledModel& model)
{
    std::ofstream output(file_name.c_str(), std::ios::trunc);
    if (!output) throw std::runtime_error("cannot write " + file_name);
    const char* row_end = R"(\\)";
    output << "% Auto-generated by Post.exe\n"
           << "\\begin{tabular}{lc}\n\\hline\n"
           << "Component & Fit fraction (\\%) " << row_end
           << "\n\\hline\n";
    for (const Observable& value : result.values) {
        if (value.category != "fit_fraction") continue;
        output << '$' << model.term_metadata[value.first].label << "$ & "
               << std::fixed << std::setprecision(2)
               << 100.0 * value.value << " $\\pm$ "
               << 100.0 * value.error << ' ' << row_end << '\n';
    }
    output << "\\hline\n\\end{tabular}\n";
}

GVVCompiledModel compile_embedded_model(
    const ctpwa::FitState& fit,
    const std::string& fit_state_file)
{
    const ctpwa::ModelDefinition definition = ctpwa::parse_model_definition(
        fit.model_json,
        "embedded model in fit state '" + fit_state_file + "'");
    if (definition.name != fit.model_name) {
        throw std::runtime_error(
            "embedded model name '" + definition.name
            + "' does not match fit-state model name '"
            + fit.model_name + "'");
    }
    const std::string definition_signature =
        ctpwa::model_definition_signature(definition);
    if (definition_signature != fit.model_definition_signature) {
        throw std::runtime_error(
            "embedded model definition does not match its stored signature");
    }

    const std::string implementation_signature =
        gvv_model_implementation_signature(definition);
    if (implementation_signature != fit.model_implementation_signature) {
        throw std::runtime_error(
            "fit state requires GVV amplitude implementation '"
            + fit.model_implementation_signature
            + "' but this Post executable provides '"
            + implementation_signature + "'");
    }
    if (gvv_model_signature(definition) != fit.model_signature) {
        throw std::runtime_error(
            "fit-state model signature is inconsistent with its embedded "
            "definition and GVV implementation signature");
    }
    return gvv_compile_model(definition);
}

void validate_parameter_contract(
    const ctpwa::FitState& fit,
    const std::vector<GVVFitParameterBinding>& layout)
{
    if (fit.parameters.size() != layout.size()) {
        throw std::runtime_error("fit-state parameter count does not match model");
    }
    for (std::size_t index = 0; index < layout.size(); ++index) {
        if (fit.parameters[index].name != layout[index].fit.name) {
            throw std::runtime_error(
                "fit-state parameter order does not match model at index "
                + std::to_string(index));
        }
    }
}

void usage(const char* executable)
{
    std::cerr << "Usage: " << executable
              << " fit_state.json truth_mc.root normalization_mc.root\n";
}

} // namespace

#ifndef GVV_POST_NO_MAIN
int main(int argc, char* argv[])
{
    if (argc != 4) {
        usage(argv[0]);
        return 2;
    }
    try {
        const ctpwa::FitState fit = ctpwa::read_fit_state(argv[1]);
        GVVCompiledModel model = compile_embedded_model(fit, argv[1]);
        std::vector<GVVFitParameterBinding> layout =
            gvv_fit_parameter_layout(model);
        validate_parameter_contract(fit, layout);

        GVVBranchConfig branches;
        GVVComponentEvaluator evaluator(
            argv[2], argv[3], branches, std::move(model), std::move(layout));
        const GVVIntegratedComponents components =
            evaluator.Evaluate(fit.best.values);
        evaluator.ValidateTotal(fit.best.values, components);
        ObservableSet result = build_observables(components, evaluator.Model());
        if (std::fabs(result.fraction_closure) > 1.0e-9
            || std::fabs(result.group_closure) > 1.0e-9) {
            throw std::runtime_error("Post Calculation closure check failed");
        }

        std::vector<double> observable_covariance;
        propagate_errors(evaluator, fit, result, observable_covariance);

        const std::filesystem::path output_directory =
            "post/calculation/results";
        std::filesystem::create_directories(output_directory);
        const std::string base = "post_result-" + fit.output_tag;
        const std::string text_file = (output_directory / (base + ".txt")).string();
        const std::string root_file = (output_directory / (base + ".root")).string();
        const std::string latex_file =
            (output_directory / ("fit_fractions-" + fit.output_tag + ".tex")).string();
        write_text(text_file, fit, result);
        write_root(root_file, fit, result, observable_covariance, evaluator);
        write_latex(latex_file, result, evaluator.Model());

        std::cout << std::setprecision(12)
                  << "Post Calculation complete: efficiency="
                  << result.selected_total / result.truth_total
                  << ", fraction closure=" << result.fraction_closure
                  << ", group closure=" << result.group_closure << '\n'
                  << "Text: " << text_file << '\n'
                  << "ROOT: " << root_file << '\n'
                  << "LaTeX: " << latex_file << '\n';
    } catch (const std::exception& error) {
        std::cerr << "GVV Post Calculation aborted: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

#endif // GVV_POST_NO_MAIN

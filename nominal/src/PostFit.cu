#include "../include/GVVFitParameters.h"
#include "../include/GVVSample.h"
#include "../include/OmegaPropagator.h"
#include "../include/kernel.h"

#include "TFile.h"
#include "TMatrixDSym.h"
#include "TTree.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct IntegratedComponents {
    std::vector<double> truth;
    std::vector<double> selected;
};

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

void check_cuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

double sum_all(
    const std::vector<double>& values)
{
    double result = 0.0;
    for (double value : values) {
        result += value;
    }
    return result;
}

double sum_group(
    const std::vector<double>& values,
    const GVVCompiledModel& model,
    const std::string& jpc)
{
    double result = 0.0;
    const int number_terms = static_cast<int>(model.terms.size());
    for (int first = 0; first < number_terms; ++first) {
        if (model.term_metadata[first].jpc != jpc) {
            continue;
        }
        for (int second = first; second < number_terms; ++second) {
            if (model.term_metadata[second].jpc == jpc) {
                result += values[gvv_component_pair_index(
                    first, second, number_terms)];
            }
        }
    }
    return result;
}

double sum_cross_groups(
    const std::vector<double>& values,
    const GVVCompiledModel& model,
    const std::string& first_jpc,
    const std::string& second_jpc)
{
    double result = 0.0;
    const int number_terms = static_cast<int>(model.terms.size());
    for (int first = 0; first < number_terms; ++first) {
        for (int second = first + 1; second < number_terms; ++second) {
            const std::string& first_value = model.term_metadata[first].jpc;
            const std::string& second_value = model.term_metadata[second].jpc;
            if ((first_value == first_jpc && second_value == second_jpc)
                || (first_value == second_jpc
                    && second_value == first_jpc)) {
                result += values[gvv_component_pair_index(
                    first, second, number_terms)];
            }
        }
    }
    return result;
}

const char* component_jpc(const GVVCompiledModel& model, int term)
{
    return model.term_metadata.at(term).jpc.c_str();
}

const char* component_latex(const GVVCompiledModel& model, int term)
{
    return model.term_metadata.at(term).latex.c_str();
}

class GVVPostFitEvaluator {
public:
    GVVPostFitEvaluator(
        const std::string& truth_file,
        const std::string& selected_file,
        const GVVBranchConfig& branches,
        GVVCompiledModel model)
        : model_(std::move(model)),
          truth_("generated truth MC"),
          selected_("selected normalization MC"),
          device_resonances_(nullptr),
          device_terms_(nullptr),
          device_couplings_(nullptr),
          component_buffer_(nullptr)
    {
        if (model_.terms.empty() || model_.active_wave_types.empty()) {
            throw std::invalid_argument("PostFit received an empty model");
        }
        truth_.Load(truth_file, branches);
        selected_.Load(selected_file, branches);
        truth_.UploadAndBuildF(
            model_.active_wave_types, NumberTerms());
        selected_.UploadAndBuildF(
            model_.active_wave_types, NumberTerms());
        omega_width_table_.Build();
        omega_width_table_.Upload();

        check_cuda(
            cudaMallocManaged(
                &device_resonances_,
                model_.resonances.size()
                    * sizeof(GVVResonanceParameters)),
            "cudaMallocManaged PostFit resonances");
        check_cuda(
            cudaMallocManaged(
                &device_terms_,
                model_.terms.size() * sizeof(GVVTermSpec)),
            "cudaMallocManaged PostFit terms");
        check_cuda(
            cudaMallocManaged(
                &device_couplings_,
                model_.initial_couplings.size() * sizeof(DeviceComplex)),
            "cudaMallocManaged PostFit couplings");
        const int maximum_entries = std::max(
            truth_.Entries(), selected_.Entries());
        check_cuda(
            cudaMallocManaged(
                &component_buffer_,
                static_cast<std::size_t>(maximum_entries)
                    * NumberPairs() * sizeof(double)),
            "cudaMallocManaged PostFit component buffer");
    }

    ~GVVPostFitEvaluator()
    {
        if (device_resonances_ != nullptr) {
            cudaFree(device_resonances_);
        }
        if (device_terms_ != nullptr) {
            cudaFree(device_terms_);
        }
        if (device_couplings_ != nullptr) {
            cudaFree(device_couplings_);
        }
        if (component_buffer_ != nullptr) {
            cudaFree(component_buffer_);
        }
    }

    IntegratedComponents Evaluate(const std::vector<double>& parameters)
    {
        GVVCompiledModel state = model_;
        gvv_apply_fit_parameters_to_model(state, parameters);
        Upload(state);

        IntegratedComponents result;
        result.truth = EvaluateSample(truth_);
        result.selected = EvaluateSample(selected_);
        return result;
    }

    void ValidateAgainstTotalPDF(
        const std::vector<double>& parameters,
        const IntegratedComponents& components)
    {
        GVVCompiledModel state = model_;
        gvv_apply_fit_parameters_to_model(state, parameters);
        Upload(state);
        ValidateSample(truth_, sum_all(components.truth));
        ValidateSample(selected_, sum_all(components.selected));
    }

    int TruthEntries() const { return truth_.Entries(); }
    int SelectedEntries() const { return selected_.Entries(); }
    int NumberTerms() const { return static_cast<int>(model_.terms.size()); }
    int NumberPairs() const {
        return gvv_number_component_pairs(NumberTerms());
    }
    const GVVCompiledModel& Model() const { return model_; }

private:
    void Upload(const GVVCompiledModel& state)
    {
        check_cuda(
            cudaMemcpy(
                device_resonances_,
                state.resonances.data(),
                state.resonances.size() * sizeof(GVVResonanceParameters),
                cudaMemcpyHostToDevice),
            "cudaMemcpy PostFit resonances");
        check_cuda(
            cudaMemcpy(
                device_terms_,
                state.terms.data(),
                state.terms.size() * sizeof(GVVTermSpec),
                cudaMemcpyHostToDevice),
            "cudaMemcpy PostFit terms");
        check_cuda(
            cudaMemcpy(
                device_couplings_,
                state.initial_couplings.data(),
                state.initial_couplings.size() * sizeof(DeviceComplex),
                cudaMemcpyHostToDevice),
            "cudaMemcpy PostFit couplings");
    }

    std::vector<double> EvaluateSample(
        GVVSample& sample)
    {
        CalGVVComponentMatrix(
            sample.Momenta(),
            device_resonances_,
            device_terms_,
            device_couplings_,
            omega_width_table_.DeviceView(),
            sample.FMatrix(),
            sample.TermCoefficientBuffer(),
            component_buffer_,
            NumberTerms(),
            static_cast<int>(model_.active_wave_types.size()),
            sample.Entries());

        std::vector<double> sums(NumberPairs(), 0.0);
        for (int event = 0; event < sample.Entries(); ++event) {
            const std::size_t offset =
                static_cast<std::size_t>(event) * NumberPairs();
            for (int pair = 0; pair < NumberPairs(); ++pair) {
                const double value = component_buffer_[offset + pair];
                if (!std::isfinite(value)) {
                    throw std::runtime_error(
                        "non-finite PostFit component in " + sample.Label());
                }
                sums[pair] += value;
            }
        }
        if (!(sum_all(sums) > 0.0) || !std::isfinite(sum_all(sums))) {
            throw std::runtime_error(
                "non-positive integrated intensity in " + sample.Label());
        }
        return sums;
    }

    void ValidateSample(GVVSample& sample, double component_sum)
    {
        CalGVVPDF(
            sample.Momenta(),
            device_resonances_,
            device_terms_,
            device_couplings_,
            omega_width_table_.DeviceView(),
            sample.FMatrix(),
            sample.TermCoefficientBuffer(),
            sample.IntensityBuffer(),
            NumberTerms(),
            static_cast<int>(model_.active_wave_types.size()),
            sample.Entries());
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
        std::cout << "PostFit component closure " << sample.Label()
                  << ": relative residual=" << relative << '\n';
    }

    GVVCompiledModel model_;
    GVVSample truth_;
    GVVSample selected_;
    OmegaWidthTable omega_width_table_;
    GVVResonanceParameters* device_resonances_;
    GVVTermSpec* device_terms_;
    DeviceComplex* device_couplings_;
    double* component_buffer_;
};

ObservableSet build_observables(
    const IntegratedComponents& components,
    const GVVCompiledModel& model)
{
    ObservableSet result;
    result.truth_total = sum_all(components.truth);
    result.selected_total = sum_all(components.selected);
    if (!(result.truth_total > 0.0) || !(result.selected_total > 0.0)) {
        throw std::runtime_error("invalid total intensity in PostFit");
    }

    double fraction_sum = 0.0;
    const int number_terms = static_cast<int>(model.terms.size());
    for (int term = 0; term < number_terms; ++term) {
        const int pair = gvv_component_pair_index(
            term, term, number_terms);
        Observable value;
        value.category = "fit_fraction";
        value.name = model.term_metadata[term].id;
        value.first = term;
        value.second = term;
        value.truth_integral = components.truth[pair];
        value.selected_integral = components.selected[pair];
        value.value = value.truth_integral / result.truth_total;
        fraction_sum += value.value;
        result.values.push_back(value);
    }
    for (int first = 0; first < number_terms; ++first) {
        for (int second = first + 1; second < number_terms; ++second) {
            const int pair = gvv_component_pair_index(
                first, second, number_terms);
            Observable value;
            value.category = "interference";
            value.name = model.term_metadata[first].id
                         + "__" + model.term_metadata[second].id;
            value.first = first;
            value.second = second;
            value.truth_integral = components.truth[pair];
            value.selected_integral = components.selected[pair];
            value.value = value.truth_integral / result.truth_total;
            fraction_sum += value.value;
            result.values.push_back(value);
        }
    }
    result.fraction_closure = fraction_sum - 1.0;

    Observable total_efficiency;
    total_efficiency.category = "efficiency_total";
    total_efficiency.name = "total_coherent_model";
    total_efficiency.truth_integral = result.truth_total;
    total_efficiency.selected_integral = result.selected_total;
    total_efficiency.value = result.selected_total / result.truth_total;
    result.values.push_back(total_efficiency);

    for (int term = 0; term < number_terms; ++term) {
        const int pair = gvv_component_pair_index(
            term, term, number_terms);
        Observable efficiency;
        efficiency.category = "efficiency_component";
        efficiency.name = model.term_metadata[term].id;
        efficiency.first = term;
        efficiency.second = term;
        efficiency.truth_integral = components.truth[pair];
        efficiency.selected_integral = components.selected[pair];
        if (!(efficiency.truth_integral > 0.0)) {
            throw std::runtime_error(
                "non-positive diagonal truth integral for " + efficiency.name);
        }
        efficiency.value =
            efficiency.selected_integral / efficiency.truth_integral;
        result.values.push_back(efficiency);
    }

    std::vector<std::string> groups;
    for (const GVVTermMetadata& term : model.term_metadata) {
        if (std::find(groups.begin(), groups.end(), term.jpc) == groups.end()) {
            groups.push_back(term.jpc);
        }
    }
    double group_reconstruction = 0.0;
    for (const std::string& group : groups) {
        const double truth = sum_group(components.truth, model, group);
        const double selected = sum_group(components.selected, model, group);

        Observable fraction;
        fraction.category = "fit_fraction_group";
        fraction.name = group;
        fraction.truth_integral = truth;
        fraction.selected_integral = selected;
        fraction.value = truth / result.truth_total;
        result.values.push_back(fraction);

        Observable efficiency;
        efficiency.category = "efficiency_group";
        efficiency.name = group;
        efficiency.truth_integral = truth;
        efficiency.selected_integral = selected;
        efficiency.value = selected / truth;
        result.values.push_back(efficiency);
        group_reconstruction += truth;
    }

    for (std::size_t first = 0; first < groups.size(); ++first) {
        for (std::size_t second = first + 1;
             second < groups.size();
             ++second) {
            const double truth_cross = sum_cross_groups(
                components.truth, model, groups[first], groups[second]);
            Observable cross;
            cross.category = "interference_group";
            cross.name = groups[first] + "__" + groups[second];
            cross.truth_integral = truth_cross;
            cross.value = truth_cross / result.truth_total;
            result.values.push_back(cross);
            group_reconstruction += truth_cross;
        }
    }
    result.group_closure =
        group_reconstruction / result.truth_total - 1.0;
    return result;
}

void propagate_errors(
    GVVPostFitEvaluator& evaluator,
    const GVVParsedFitResult& fit,
    const std::vector<double>& covariance,
    ObservableSet& central,
    std::vector<double>& observable_covariance)
{
    const int number_parameters = static_cast<int>(fit.values.size());
    const int number_observables = static_cast<int>(central.values.size());
    const std::vector<GVVFitParameterSpec> parameter_layout =
        gvv_fit_parameter_layout(evaluator.Model());
    if (static_cast<int>(parameter_layout.size()) != number_parameters) {
        throw std::runtime_error(
            "PostFit parameter layout does not match fit result");
    }
    std::vector<double> gradients(
        static_cast<std::size_t>(number_observables) * number_parameters,
        0.0);

    for (int parameter = 0; parameter < number_parameters; ++parameter) {
        const double sigma = std::sqrt(covariance[
            static_cast<std::size_t>(parameter) * number_parameters + parameter]);
        const double scale_step =
            1.0e-5 * std::max(1.0, std::fabs(fit.values[parameter]));
        const double step = std::max(scale_step, 0.05 * sigma);
        if (!(step > 0.0) || !std::isfinite(step)) {
            throw std::runtime_error("invalid finite-difference step");
        }

        const GVVFitParameterSpec& specification =
            parameter_layout[parameter];
        const bool can_minus = !specification.has_lower_bound
                               || fit.values[parameter] - step
                                      > specification.lower_bound;
        const bool can_plus = !specification.has_upper_bound
                              || fit.values[parameter] + step
                                     < specification.upper_bound;
        if (!can_minus && !can_plus) {
            throw std::runtime_error(
                "fit parameter cannot be varied inside its bounds");
        }

        std::vector<double> plus_values = fit.values;
        std::vector<double> minus_values = fit.values;
        ObservableSet plus;
        ObservableSet minus;
        if (can_plus) {
            plus_values[parameter] += step;
            plus = build_observables(
                evaluator.Evaluate(plus_values), evaluator.Model());
        }
        if (can_minus) {
            minus_values[parameter] -= step;
            minus = build_observables(
                evaluator.Evaluate(minus_values), evaluator.Model());
        }
        for (int observable = 0;
             observable < number_observables;
             ++observable) {
            double derivative = 0.0;
            if (can_plus && can_minus) {
                derivative =
                    (plus.values[observable].value
                     - minus.values[observable].value)
                    / (2.0 * step);
            } else if (can_plus) {
                derivative =
                    (plus.values[observable].value
                     - central.values[observable].value)
                    / step;
            } else {
                derivative =
                    (central.values[observable].value
                     - minus.values[observable].value)
                    / step;
            }
            gradients[
                static_cast<std::size_t>(observable) * number_parameters
                + parameter] = derivative;
        }
        std::cout << "PostFit uncertainty derivative "
                  << parameter + 1 << '/' << number_parameters
                  << "  " << fit.parameter_names[parameter]
                  << "  step=" << step << '\n';
    }

    observable_covariance.assign(
        static_cast<std::size_t>(number_observables) * number_observables,
        0.0);
    for (int first = 0; first < number_observables; ++first) {
        for (int second = 0; second < number_observables; ++second) {
            double value = 0.0;
            for (int row = 0; row < number_parameters; ++row) {
                const double gradient_first = gradients[
                    static_cast<std::size_t>(first) * number_parameters + row];
                for (int column = 0; column < number_parameters; ++column) {
                    value += gradient_first
                             * covariance[
                                 static_cast<std::size_t>(row)
                                     * number_parameters
                                 + column]
                             * gradients[
                                 static_cast<std::size_t>(second)
                                     * number_parameters
                                 + column];
                }
            }
            observable_covariance[
                static_cast<std::size_t>(first) * number_observables + second]
                = value;
        }
        central.values[first].error = std::sqrt(std::max(
            0.0,
            observable_covariance[
                static_cast<std::size_t>(first) * number_observables + first]));
    }
}

void write_text(const std::string& file_name, const ObservableSet& result)
{
    std::ofstream output(file_name.c_str());
    if (!output) {
        throw std::runtime_error("cannot write PostFit text output");
    }
    output << std::setprecision(12)
           << "# GVV post-fit observables; fractions are dimensionless\n"
           << "# truth_total " << result.truth_total
           << " selected_total " << result.selected_total << '\n'
           << "# fraction_closure " << result.fraction_closure
           << " group_closure " << result.group_closure << '\n';
    for (const Observable& value : result.values) {
        output << value.category << ' ' << value.name << ' '
               << value.value << ' ' << value.error
               << " truth_integral " << value.truth_integral
               << " selected_integral " << value.selected_integral << '\n';
    }
}

void write_latex(
    const std::string& file_name,
    const ObservableSet& result,
    const GVVCompiledModel& model)
{
    std::ofstream output(file_name.c_str());
    if (!output) {
        throw std::runtime_error("cannot write PostFit LaTeX output");
    }
    const char* row_end = R"(\\)";
    output << "% Auto-generated by PostFit.exe\n"
           << "\\begin{table}[htbp]\n\\centering\n"
           << "\\begin{tabular}{lc}\n\\hline\n"
           << "Component & Fit fraction (\\%) " << row_end << '\n'
           << "\\hline\n";
    for (const Observable& value : result.values) {
        if (value.category == "fit_fraction") {
            output << '$' << component_latex(model, value.first) << "$ & "
                   << std::fixed << std::setprecision(2)
                   << 100.0 * value.value << " $\\pm$ "
                   << 100.0 * value.error << ' ' << row_end << '\n';
        }
    }
    output << "\\hline\n\\end{tabular}\n"
           << "\\caption{GVV truth-phase-space fit fractions.}\n"
           << "\\end{table}\n\n"
           << "\\begin{table}[htbp]\n\\centering\n"
           << "\\begin{tabular}{lc}\n\\hline\n"
           << "Pair & Interference fraction (\\%) " << row_end << '\n'
           << "\\hline\n";
    for (const Observable& value : result.values) {
        if (value.category == "interference") {
            output << '$' << component_latex(model, value.first) << "--"
                   << component_latex(model, value.second) << "$ & "
                   << std::fixed << std::setprecision(2)
                   << 100.0 * value.value << " $\\pm$ "
                   << 100.0 * value.error << ' ' << row_end << '\n';
        }
    }
    output << "\\hline\n\\end{tabular}\n"
           << "\\caption{GVV pairwise interference fractions.}\n"
           << "\\end{table}\n\n"
           << "\\begin{table}[htbp]\n\\centering\n"
           << "\\begin{tabular}{lc}\n\\hline\n"
           << "Component & Efficiency (\\%) " << row_end << '\n'
           << "\\hline\n";
    for (const Observable& value : result.values) {
        if (value.category == "efficiency_total") {
            output << "Total coherent model & ";
        } else if (value.category == "efficiency_component") {
            output << '$' << component_latex(model, value.first) << "$ & ";
        } else {
            continue;
        }
        output << std::fixed << std::setprecision(2)
               << 100.0 * value.value << " $\\pm$ "
               << 100.0 * value.error << ' ' << row_end << '\n';
    }
    output << "\\hline\n\\end{tabular}\n"
           << "\\caption{GVV model-weighted selection efficiencies.}\n"
           << "\\end{table}\n";
}

void copy_text(char* destination, std::size_t size, const std::string& source)
{
    std::snprintf(destination, size, "%s", source.c_str());
}

void write_root(
    const std::string& file_name,
    const ObservableSet& result,
    const std::vector<double>& observable_covariance,
    int truth_entries,
    int selected_entries,
    const GVVCompiledModel& model)
{
    TFile output(file_name.c_str(), "RECREATE");
    if (output.IsZombie()) {
        throw std::runtime_error("cannot create PostFit ROOT output");
    }

    int index = 0;
    int first = -1;
    int second = -1;
    char category[32] = {0};
    char name[128] = {0};
    char jpc[16] = {0};
    double value = 0.0;
    double error = 0.0;
    double truth_integral = 0.0;
    double selected_integral = 0.0;
    TTree observables("observables", "GVV fit fractions, interference, efficiency");
    observables.Branch("index", &index, "index/I");
    observables.Branch("category", category, "category/C");
    observables.Branch("name", name, "name/C");
    observables.Branch("jpc", jpc, "jpc/C");
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
        copy_text(
            jpc,
            sizeof(jpc),
            source.first >= 0
                ? component_jpc(model, source.first)
                : "combined");
        value = source.value;
        error = source.error;
        truth_integral = source.truth_integral;
        selected_integral = source.selected_integral;
        observables.Fill();
    }

    TTree metadata("metadata", "PostFit provenance and closure");
    double truth_total = result.truth_total;
    double selected_total = result.selected_total;
    double fraction_closure = result.fraction_closure;
    double group_closure = result.group_closure;
    int number_observables = static_cast<int>(result.values.size());
    metadata.Branch("n_truth", &truth_entries, "n_truth/I");
    metadata.Branch("n_selected", &selected_entries, "n_selected/I");
    metadata.Branch("n_observables", &number_observables, "n_observables/I");
    metadata.Branch("truth_total", &truth_total, "truth_total/D");
    metadata.Branch("selected_total", &selected_total, "selected_total/D");
    metadata.Branch(
        "fraction_closure", &fraction_closure, "fraction_closure/D");
    metadata.Branch("group_closure", &group_closure, "group_closure/D");
    metadata.Fill();

    TMatrixDSym covariance(number_observables);
    TMatrixDSym correlation(number_observables);
    for (int row = 0; row < number_observables; ++row) {
        for (int column = 0; column < number_observables; ++column) {
            covariance(row, column) = observable_covariance[
                static_cast<std::size_t>(row) * number_observables + column];
        }
    }
    for (int row = 0; row < number_observables; ++row) {
        for (int column = 0; column < number_observables; ++column) {
            const double denominator = std::sqrt(std::max(
                0.0, covariance(row, row) * covariance(column, column)));
            correlation(row, column) = denominator > 0.0
                                           ? covariance(row, column) / denominator
                                           : 0.0;
        }
    }
    covariance.Write("observable_covariance");
    correlation.Write("observable_correlation");
    observables.Write();
    metadata.Write();
    output.Close();
}

void usage(const char* executable)
{
    std::cerr
        << "Usage: " << executable
        << " fit_result.txt Cova_matrix.dat truth_mc.root"
        << " normalization_mc.root [output_prefix [model.json]]\n"
        << "       " << executable << " --self-test\n"
        << "truth_mc.root must contain every generated event from the same"
        << " PHSP production whose selected subset is normalization_mc.root.\n";
}

void run_postfit_math_self_test(const GVVCompiledModel& model)
{
    const int number_terms = static_cast<int>(model.terms.size());
    const int number_pairs = gvv_number_component_pairs(number_terms);
    IntegratedComponents components;
    components.truth.assign(number_pairs, 0.0);
    components.selected.assign(number_pairs, 0.0);
    for (int term = 0; term < number_terms; ++term) {
        const int diagonal = gvv_component_pair_index(
            term, term, number_terms);
        components.truth[diagonal] = 1.0 + term;
        components.selected[diagonal] = components.truth[diagonal];
    }
    if (number_terms >= 2) {
        const int pair = gvv_component_pair_index(0, 1, number_terms);
        components.truth[pair] = -0.25;
        components.selected[pair] = -0.25;
    }
    if (number_terms >= 4) {
        const int pair = gvv_component_pair_index(2, 3, number_terms);
        components.truth[pair] = 0.40;
        components.selected[pair] = 0.40;
    }
    const ObservableSet result = build_observables(components, model);
    if (std::fabs(result.fraction_closure) > 1.0e-12
        || std::fabs(result.group_closure) > 1.0e-12) {
        throw std::runtime_error("self-test fraction closure failed");
    }
    for (const Observable& value : result.values) {
        if (value.category.rfind("efficiency", 0) == 0
            && std::fabs(value.value - 1.0) > 1.0e-12) {
            throw std::runtime_error("self-test truth=selected efficiency failed");
        }
    }
    std::cout << "GVV PostFit algebra self-test passed\n";
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc == 2 && std::string(argv[1]) == "--self-test") {
        try {
            run_postfit_math_self_test(
                gvv_load_compiled_model("config/model.json"));
        } catch (const std::exception& error) {
            std::cerr << "GVV PostFit self-test failed: " << error.what() << '\n';
            return 1;
        }
        return 0;
    }
    if (argc < 5 || argc > 7) {
        usage(argv[0]);
        return 2;
    }
    try {
        const std::string output_prefix =
            argc >= 6 ? argv[5] : "results/postfit_result";
        const std::filesystem::path output_path(output_prefix);
        const std::filesystem::path output_directory =
            output_path.has_parent_path()
                ? output_path.parent_path()
                : std::filesystem::path(".");
        std::filesystem::create_directories(output_directory);

        const std::string snapshot_file = std::string(argv[1]) + ".model.json";
        const std::string model_file =
            argc >= 7
                ? argv[6]
                : (std::filesystem::exists(snapshot_file)
                       ? snapshot_file
                       : "config/model.json");
        const GVVParsedFitResult fit = gvv_read_fit_result(
            argv[1], gvv_load_compiled_model(model_file));
        const int number_parameters = static_cast<int>(fit.values.size());
        const std::vector<double> covariance =
            gvv_read_covariance_matrix(argv[2], number_parameters);
        std::cout << "PostFit loaded " << number_parameters
                  << " parameters using "
                  << (fit.used_machine_readable_rows
                          ? "machine-readable rows"
                          : "backward-compatible coupling/resonance rows")
                  << '\n';

        GVVBranchConfig branches;
        GVVPostFitEvaluator evaluator(
            argv[3], argv[4], branches, fit.compiled_model);
        const IntegratedComponents central_components =
            evaluator.Evaluate(fit.values);
        evaluator.ValidateAgainstTotalPDF(fit.values, central_components);
        ObservableSet central = build_observables(
            central_components, evaluator.Model());
        if (std::fabs(central.fraction_closure) > 1.0e-9
            || std::fabs(central.group_closure) > 1.0e-9) {
            throw std::runtime_error("PostFit fraction closure failed");
        }

        std::vector<double> observable_covariance;
        propagate_errors(
            evaluator,
            fit,
            covariance,
            central,
            observable_covariance);

        const std::string text_file = output_prefix + ".txt";
        const std::string root_file = output_prefix + ".root";
        const std::string latex_file =
            (output_directory / "fit_fractions.tex").string();
        write_text(text_file, central);
        write_root(
            root_file,
            central,
            observable_covariance,
            evaluator.TruthEntries(),
            evaluator.SelectedEntries(),
            evaluator.Model());
        write_latex(latex_file, central, evaluator.Model());

        std::cout << std::setprecision(12)
                  << "PostFit complete: total efficiency="
                  << central.selected_total / central.truth_total
                  << ", fraction closure=" << central.fraction_closure
                  << ", group closure=" << central.group_closure << '\n'
                  << "Text: " << text_file << '\n'
                  << "ROOT: " << root_file << '\n'
                  << "LaTeX: " << latex_file << '\n';
    } catch (const std::exception& error) {
        std::cerr << "GVV PostFit aborted: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

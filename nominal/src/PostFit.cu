#include "../include/GVVFitParameters.h"
#include "../include/GVVSample.h"
#include "../include/OmegaPropagator.h"
#include "../include/kernel.h"

#include "TFile.h"
#include "TMatrixDSym.h"
#include "TTree.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
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
#include <vector>

namespace {

struct IntegratedComponents {
    std::array<double, GVV_NCOMPONENT_PAIRS> truth{};
    std::array<double, GVV_NCOMPONENT_PAIRS> selected{};
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
    const std::array<double, GVV_NCOMPONENT_PAIRS>& values)
{
    double result = 0.0;
    for (double value : values) {
        result += value;
    }
    return result;
}

double sum_group(
    const std::array<double, GVV_NCOMPONENT_PAIRS>& values,
    int first_term,
    int last_term)
{
    double result = 0.0;
    for (int first = first_term; first <= last_term; ++first) {
        for (int second = first; second <= last_term; ++second) {
            result += values[gvv_component_pair_index(first, second)];
        }
    }
    return result;
}

double sum_cross_groups(
    const std::array<double, GVV_NCOMPONENT_PAIRS>& values,
    int first_begin,
    int first_end,
    int second_begin,
    int second_end)
{
    double result = 0.0;
    for (int first = first_begin; first <= first_end; ++first) {
        for (int second = second_begin; second <= second_end; ++second) {
            result += values[gvv_component_pair_index(first, second)];
        }
    }
    return result;
}

const char* component_jpc(int term)
{
    return gvv_default_term(term).wave_type == GVV_PSEUDOSCALAR_11
               ? "0-+"
               : "0++";
}

const char* component_latex(int term)
{
    static const char* names[GVV_NTERMS] = {
        "f_{0}(1500)",
        "f_{0}(1710)",
        "\\eta(1760)",
        "\\eta_{c}(1S)",
        "X(1835)",
        "X(2370)",
        "0^{-+}~\\mathrm{NR}"};
    return names[term];
}

class GVVPostFitEvaluator {
public:
    GVVPostFitEvaluator(
        const std::string& truth_file,
        const std::string& selected_file,
        const GVVBranchConfig& branches)
        : truth_("generated truth MC"),
          selected_("selected normalization MC"),
          device_resonances_(nullptr),
          device_terms_(nullptr),
          device_couplings_(nullptr),
          component_buffer_(nullptr)
    {
        truth_.Load(truth_file, branches);
        selected_.Load(selected_file, branches);
        truth_.UploadAndBuildF();
        selected_.UploadAndBuildF();
        omega_width_table_.Build();
        omega_width_table_.Upload();

        check_cuda(
            cudaMallocManaged(
                &device_resonances_,
                GVV_NRESONANCES * sizeof(GVVResonanceParameters)),
            "cudaMallocManaged PostFit resonances");
        check_cuda(
            cudaMallocManaged(
                &device_terms_, GVV_NTERMS * sizeof(GVVTermSpec)),
            "cudaMallocManaged PostFit terms");
        check_cuda(
            cudaMallocManaged(
                &device_couplings_, GVV_NTERMS * sizeof(DeviceComplex)),
            "cudaMallocManaged PostFit couplings");
        const int maximum_entries = std::max(
            truth_.Entries(), selected_.Entries());
        check_cuda(
            cudaMallocManaged(
                &component_buffer_,
                static_cast<std::size_t>(maximum_entries)
                    * GVV_NCOMPONENT_PAIRS * sizeof(double)),
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
        GVVFitState state = gvv_default_fit_state();
        gvv_apply_fit_parameters_to_state(state, parameters);
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
        GVVFitState state = gvv_default_fit_state();
        gvv_apply_fit_parameters_to_state(state, parameters);
        Upload(state);
        ValidateSample(truth_, sum_all(components.truth));
        ValidateSample(selected_, sum_all(components.selected));
    }

    int TruthEntries() const { return truth_.Entries(); }
    int SelectedEntries() const { return selected_.Entries(); }

private:
    void Upload(const GVVFitState& state)
    {
        check_cuda(
            cudaMemcpy(
                device_resonances_,
                state.resonances.data(),
                GVV_NRESONANCES * sizeof(GVVResonanceParameters),
                cudaMemcpyHostToDevice),
            "cudaMemcpy PostFit resonances");
        check_cuda(
            cudaMemcpy(
                device_terms_,
                state.terms.data(),
                GVV_NTERMS * sizeof(GVVTermSpec),
                cudaMemcpyHostToDevice),
            "cudaMemcpy PostFit terms");
        check_cuda(
            cudaMemcpy(
                device_couplings_,
                state.couplings.data(),
                GVV_NTERMS * sizeof(DeviceComplex),
                cudaMemcpyHostToDevice),
            "cudaMemcpy PostFit couplings");
    }

    std::array<double, GVV_NCOMPONENT_PAIRS> EvaluateSample(
        GVVSample& sample)
    {
        CalGVVComponentMatrix(
            sample.Momenta(),
            device_resonances_,
            device_terms_,
            device_couplings_,
            omega_width_table_.DeviceView(),
            sample.FMatrix(),
            component_buffer_,
            sample.Entries());

        std::array<double, GVV_NCOMPONENT_PAIRS> sums{};
        for (int event = 0; event < sample.Entries(); ++event) {
            const std::size_t offset =
                static_cast<std::size_t>(event) * GVV_NCOMPONENT_PAIRS;
            for (int pair = 0; pair < GVV_NCOMPONENT_PAIRS; ++pair) {
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
            sample.IntensityBuffer(),
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

    GVVSample truth_;
    GVVSample selected_;
    OmegaWidthTable omega_width_table_;
    GVVResonanceParameters* device_resonances_;
    GVVTermSpec* device_terms_;
    DeviceComplex* device_couplings_;
    double* component_buffer_;
};

ObservableSet build_observables(const IntegratedComponents& components)
{
    ObservableSet result;
    result.truth_total = sum_all(components.truth);
    result.selected_total = sum_all(components.selected);
    if (!(result.truth_total > 0.0) || !(result.selected_total > 0.0)) {
        throw std::runtime_error("invalid total intensity in PostFit");
    }

    double fraction_sum = 0.0;
    for (int term = 0; term < GVV_NTERMS; ++term) {
        const int pair = gvv_component_pair_index(term, term);
        Observable value;
        value.category = "fit_fraction";
        value.name = gvv_term_name(term);
        value.first = term;
        value.second = term;
        value.truth_integral = components.truth[pair];
        value.selected_integral = components.selected[pair];
        value.value = value.truth_integral / result.truth_total;
        fraction_sum += value.value;
        result.values.push_back(value);
    }
    for (int first = 0; first < GVV_NTERMS; ++first) {
        for (int second = first + 1; second < GVV_NTERMS; ++second) {
            const int pair = gvv_component_pair_index(first, second);
            Observable value;
            value.category = "interference";
            value.name = std::string(gvv_term_name(first))
                         + "__" + gvv_term_name(second);
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

    for (int term = 0; term < GVV_NTERMS; ++term) {
        const int pair = gvv_component_pair_index(term, term);
        Observable efficiency;
        efficiency.category = "efficiency_component";
        efficiency.name = gvv_term_name(term);
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

    const double truth_scalar = sum_group(components.truth, 0, 1);
    const double selected_scalar = sum_group(components.selected, 0, 1);
    const double truth_pseudoscalar = sum_group(components.truth, 2, 6);
    const double selected_pseudoscalar = sum_group(components.selected, 2, 6);
    const double truth_cross = sum_cross_groups(
        components.truth, 0, 1, 2, 6);

    for (int group = 0; group < 2; ++group) {
        const bool scalar = group == 0;
        const double truth = scalar ? truth_scalar : truth_pseudoscalar;
        const double selected = scalar ? selected_scalar : selected_pseudoscalar;
        const std::string name = scalar ? "0pp" : "0mp";

        Observable fraction;
        fraction.category = "fit_fraction_group";
        fraction.name = name;
        fraction.truth_integral = truth;
        fraction.selected_integral = selected;
        fraction.value = truth / result.truth_total;
        result.values.push_back(fraction);

        Observable efficiency;
        efficiency.category = "efficiency_group";
        efficiency.name = name;
        efficiency.truth_integral = truth;
        efficiency.selected_integral = selected;
        efficiency.value = selected / truth;
        result.values.push_back(efficiency);
    }

    Observable cross;
    cross.category = "interference_group";
    cross.name = "0pp__0mp";
    cross.truth_integral = truth_cross;
    cross.value = truth_cross / result.truth_total;
    result.values.push_back(cross);
    result.group_closure =
        (truth_scalar + truth_pseudoscalar + truth_cross)
            / result.truth_total
        - 1.0;
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

        double lower = 0.0;
        double upper = 0.0;
        const bool bounded = gvv_fit_parameter_bounds(
            fit.parameter_names[parameter], lower, upper);
        const bool can_minus = !bounded
                               || fit.values[parameter] - step > lower;
        const bool can_plus = !bounded
                              || fit.values[parameter] + step < upper;
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
            plus = build_observables(evaluator.Evaluate(plus_values));
        }
        if (can_minus) {
            minus_values[parameter] -= step;
            minus = build_observables(evaluator.Evaluate(minus_values));
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

#if 0
void write_latex_legacy_format(
    const std::string& file_name,
    const ObservableSet& result)
{
    std::ofstream output(file_name.c_str());
    if (!output) {
        throw std::runtime_error("cannot write PostFit LaTeX output");
    }
    output << "% Auto-generated by PostFit.exe\n"
           << "\\begin{table}[htbp]\n\\centering\n"
           << "\\begin{tabular}{lc}\n\\hline\n"
           << "Component & Fit fraction (\\%) \\\\\n+\\hline\n";
    for (const Observable& value : result.values) {
        if (value.category == "fit_fraction") {
            output << '$' << component_latex(value.first) << "$ & "
                   << std::fixed << std::setprecision(2)
                   << 100.0 * value.value << " $\\pm$ "
                   << 100.0 * value.error << " \\\\\n+";
        }
    }
    output << "\\hline\n\\end{tabular}\n"
           << "\\caption{GVV truth-phase-space fit fractions.}\n"
           << "\\end{table}\n\n"
           << "\\begin{table}[htbp]\n\\centering\n"
           << "\\begin{tabular}{lc}\n\\hline\n"
           << "Pair & Interference fraction (\\%) \\\\\n+\\hline\n";
    for (const Observable& value : result.values) {
        if (value.category == "interference") {
            output << '$' << component_latex(value.first) << "--"
                   << component_latex(value.second) << "$ & "
                   << std::fixed << std::setprecision(2)
                   << 100.0 * value.value << " $\\pm$ "
                   << 100.0 * value.error << " \\\\\n+";
        }
    }
    output << "\\hline\n\\end{tabular}\n"
           << "\\caption{GVV pairwise interference fractions.}\n"
           << "\\end{table}\n\n"
           << "\\begin{table}[htbp]\n\\centering\n"
           << "\\begin{tabular}{lc}\n\\hline\n"
           << "Component & Efficiency (\\%) \\\\\n+\\hline\n";
    for (const Observable& value : result.values) {
        if (value.category == "efficiency_total") {
            output << "Total coherent model & ";
        } else if (value.category == "efficiency_component") {
            output << '$' << component_latex(value.first) << "$ & ";
        } else {
            continue;
        }
        output << std::fixed << std::setprecision(2)
               << 100.0 * value.value << " $\\pm$ "
               << 100.0 * value.error << " \\\\\n+";
    }
    output << "\\hline\n\\end{tabular}\n"
           << "\\caption{GVV model-weighted selection efficiencies.}\n"
           << "\\end{table}\n";
}

#endif

void write_latex(const std::string& file_name, const ObservableSet& result)
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
            output << '$' << component_latex(value.first) << "$ & "
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
            output << '$' << component_latex(value.first) << "--"
                   << component_latex(value.second) << "$ & "
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
            output << '$' << component_latex(value.first) << "$ & ";
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
    int selected_entries)
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
            source.first >= 0 ? component_jpc(source.first) : "combined");
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
        << " normalization_mc.root [output_prefix]\n"
        << "       " << executable << " --self-test\n"
        << "truth_mc.root must contain every generated event from the same"
        << " PHSP production whose selected subset is normalization_mc.root.\n";
}

void run_postfit_math_self_test()
{
    IntegratedComponents components;
    for (int term = 0; term < GVV_NTERMS; ++term) {
        const int diagonal = gvv_component_pair_index(term, term);
        components.truth[diagonal] = 1.0 + term;
        components.selected[diagonal] = components.truth[diagonal];
    }
    components.truth[gvv_component_pair_index(0, 1)] = -0.25;
    components.selected[gvv_component_pair_index(0, 1)] = -0.25;
    components.truth[gvv_component_pair_index(2, 3)] = 0.40;
    components.selected[gvv_component_pair_index(2, 3)] = 0.40;
    const ObservableSet result = build_observables(components);
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
            run_postfit_math_self_test();
        } catch (const std::exception& error) {
            std::cerr << "GVV PostFit self-test failed: " << error.what() << '\n';
            return 1;
        }
        return 0;
    }
    if (argc < 5 || argc > 6) {
        usage(argv[0]);
        return 2;
    }
    try {
        const std::string output_prefix =
            argc == 6 ? argv[5] : "results/postfit_result";
        const std::filesystem::path output_path(output_prefix);
        const std::filesystem::path output_directory =
            output_path.has_parent_path()
                ? output_path.parent_path()
                : std::filesystem::path(".");
        std::filesystem::create_directories(output_directory);

        const GVVParsedFitResult fit = gvv_read_fit_result(argv[1]);
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
        GVVPostFitEvaluator evaluator(argv[3], argv[4], branches);
        const IntegratedComponents central_components =
            evaluator.Evaluate(fit.values);
        evaluator.ValidateAgainstTotalPDF(fit.values, central_components);
        ObservableSet central = build_observables(central_components);
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
            evaluator.SelectedEntries());
        write_latex(latex_file, central);

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

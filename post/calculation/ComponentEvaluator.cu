// Process-specific numerical integration used after the fit. Fit/minimizer
// code is deliberately absent from this module.
#include "post/calculation/ComponentEvaluator.h"

#include "process/OmegaWidthTable.h"
#include "process/TermEvaluator.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace {

constexpr int kPostComponentBatchSize = 4096;

void check_cuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

double sum(const std::vector<double>& values)
{
    return std::accumulate(values.begin(), values.end(), 0.0);
}

} // namespace

GVVComponentEvaluator::GVVComponentEvaluator(
    const std::string& truth_file,
    const std::string& selected_file,
    const GVVBranchConfig& branches,
    GVVCompiledModel model,
    std::vector<GVVFitParameterBinding> layout)
    : model_(std::move(model)),
      layout_(std::move(layout)),
      truth_("generated truth MC"),
      selected_("selected normalization MC"),
      device_resonances_(nullptr),
      device_terms_(nullptr),
      device_couplings_(nullptr),
      coefficient_workspace_(nullptr),
      integrated_components_(nullptr),
      component_batch_capacity_(0)
{
    truth_.Load(truth_file, branches);
    selected_.Load(selected_file, branches);
    truth_.UploadAndBuildF(model_.active_wave_types, NumberTerms());
    selected_.UploadAndBuildF(model_.active_wave_types, NumberTerms());
    omega_width_table_.Build();
    omega_width_table_.Upload();

    check_cuda(cudaMallocManaged(
        &device_resonances_,
        model_.resonances.size() * sizeof(ctpwa::PropagatorParameters)),
        "cudaMallocManaged Post resonances");
    check_cuda(cudaMallocManaged(
        &device_terms_, model_.terms.size() * sizeof(TermSpec)),
        "cudaMallocManaged Post terms");
    check_cuda(cudaMallocManaged(
        &device_couplings_,
        model_.initial_couplings.size() * sizeof(DeviceComplex)),
        "cudaMallocManaged Post couplings");
    const int maximum_entries = std::max(truth_.Entries(), selected_.Entries());
    component_batch_capacity_ = std::min(
        kPostComponentBatchSize, maximum_entries);
    check_cuda(cudaMallocManaged(
        &coefficient_workspace_,
        static_cast<std::size_t>(component_batch_capacity_) * NumberTerms()
            * sizeof(DeviceComplex)),
        "cudaMallocManaged Post Term coefficient workspace");
    check_cuda(cudaMallocManaged(
        &integrated_components_,
        static_cast<std::size_t>(NumberPairs()) * sizeof(double)),
        "cudaMallocManaged Post integrated components");
}

GVVComponentEvaluator::~GVVComponentEvaluator()
{
    if (device_resonances_) cudaFree(device_resonances_);
    if (device_terms_) cudaFree(device_terms_);
    if (device_couplings_) cudaFree(device_couplings_);
    if (coefficient_workspace_) cudaFree(coefficient_workspace_);
    if (integrated_components_) cudaFree(integrated_components_);
}

void GVVComponentEvaluator::Upload(const GVVCompiledModel& state)
{
    check_cuda(cudaMemcpy(
        device_resonances_, state.resonances.data(),
        state.resonances.size() * sizeof(ctpwa::PropagatorParameters),
        cudaMemcpyHostToDevice), "cudaMemcpy Post resonances");
    check_cuda(cudaMemcpy(
        device_terms_, state.terms.data(),
        state.terms.size() * sizeof(TermSpec), cudaMemcpyHostToDevice),
        "cudaMemcpy Post terms");
    check_cuda(cudaMemcpy(
        device_couplings_, state.initial_couplings.data(),
        state.initial_couplings.size() * sizeof(DeviceComplex),
        cudaMemcpyHostToDevice), "cudaMemcpy Post couplings");
    truth_.UpdateOmegaFactors(omega_width_table_.DeviceView(), state.omega_resolution_sigma);
    selected_.UpdateOmegaFactors(omega_width_table_.DeviceView(), state.omega_resolution_sigma);
}

std::vector<double> GVVComponentEvaluator::EvaluateSample(GVVSample& sample)
{
    CalGVVComponentIntegrals(
        sample.Momenta(), device_resonances_, device_terms_, device_couplings_,
        omega_width_table_.DeviceView(), sample.FMatrix(),
        coefficient_workspace_, integrated_components_,
        component_batch_capacity_, NumberTerms(),
        static_cast<int>(model_.active_wave_types.size()), sample.Entries(),
        sample.OmegaFactorBuffer());

    std::vector<double> integrated(
        integrated_components_, integrated_components_ + NumberPairs());
    const double total = sum(integrated);
    if (!(total > 0.0) || !std::isfinite(total)) {
        throw std::runtime_error(
            "invalid integrated intensity in " + sample.Label());
    }
    return integrated;
}

GVVIntegratedComponents GVVComponentEvaluator::Evaluate(
    const std::vector<double>& values)
{
    GVVCompiledModel state = model_;
    gvv_apply_fit_parameters(state, layout_, values);
    Upload(state);
    return {EvaluateSample(truth_), EvaluateSample(selected_)};
}

void GVVComponentEvaluator::ValidateSample(
    GVVSample& sample,
    double component_sum)
{
    CalGVVPDF(
        sample.Momenta(), device_resonances_, device_terms_, device_couplings_,
        omega_width_table_.DeviceView(), sample.FMatrix(),
        sample.WaveCoefficientBuffer(), sample.IntensityBuffer(), NumberTerms(),
        static_cast<int>(model_.active_wave_types.size()), sample.Entries(),
        sample.OmegaFactorBuffer());
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
    GVVCompiledModel state = model_;
    gvv_apply_fit_parameters(state, layout_, values);
    Upload(state);
    ValidateSample(truth_, sum(components.truth));
    ValidateSample(selected_, sum(components.selected));
}

int GVVComponentEvaluator::TruthEntries() const { return truth_.Entries(); }
int GVVComponentEvaluator::SelectedEntries() const { return selected_.Entries(); }
int GVVComponentEvaluator::NumberTerms() const
{
    return static_cast<int>(model_.terms.size());
}
int GVVComponentEvaluator::NumberPairs() const
{
    return ctpwa::component_pair_count(NumberTerms());
}
const GVVCompiledModel& GVVComponentEvaluator::Model() const { return model_; }
const std::vector<GVVFitParameterBinding>& GVVComponentEvaluator::Layout() const
{
    return layout_;
}

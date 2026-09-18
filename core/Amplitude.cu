#include "core/Amplitude.h"
#include "core/AmplitudeKernels.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace {
void check_cuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
}

GVVAmplitude::GVVAmplitude(const GVVCompiledModel& model) : model_(model)
{
    try {
        omega_width_.Build();
        omega_width_.Upload();
        check_cuda(cudaMallocManaged(&resonances_, model.initial_parameters.resonances.size()
            * sizeof(ctpwa::PropagatorParameters)), "allocate resonances");
        check_cuda(cudaMallocManaged(&terms_, model.terms.size() * sizeof(TermSpec)), "allocate Terms");
        check_cuda(cudaMallocManaged(&couplings_, model.terms.size() * sizeof(DeviceComplex)), "allocate couplings");
        // Topology is immutable: upload it once, independently of Minuit calls.
        check_cuda(cudaMemcpy(terms_, model.terms.data(), model.terms.size() * sizeof(TermSpec),
            cudaMemcpyHostToDevice), "upload Terms");
        SetParameters(model.initial_parameters);
    } catch (...) {
        Release();
        throw;
    }
}

GVVAmplitude::~GVVAmplitude() { Release(); }

void GVVAmplitude::Release()
{
    if (resonances_) cudaFree(resonances_);
    if (terms_) cudaFree(terms_);
    if (couplings_) cudaFree(couplings_);
    if (coefficients_) cudaFree(coefficients_);
    if (components_) cudaFree(components_);
    if (integrals_) cudaFree(integrals_);
}

void GVVAmplitude::Prepare(GVVSample& sample)
{
    sample.UploadAndBuildF(model_.active_wave_types, static_cast<int>(model_.terms.size()));
}

void GVVAmplitude::SetParameters(const GVVParameterState& state)
{
    if (state.resonances.size() != model_.initial_parameters.resonances.size()
        || state.couplings.size() != model_.terms.size())
        throw std::invalid_argument("amplitude parameter-state dimensions do not match model");
    check_cuda(cudaMemcpy(resonances_, state.resonances.data(), state.resonances.size()
        * sizeof(ctpwa::PropagatorParameters), cudaMemcpyHostToDevice), "upload resonances");
    check_cuda(cudaMemcpy(couplings_, state.couplings.data(), state.couplings.size()
        * sizeof(DeviceComplex), cudaMemcpyHostToDevice), "upload couplings");
    omega_sigma_ = state.omega_resolution_sigma;
}

const double* GVVAmplitude::EvaluateIntensity(GVVSample& sample)
{
    // Sample owns the cache; unchanged sigma does not launch a convolution.
    sample.UpdateOmegaFactors(omega_width_.DeviceView(), omega_sigma_);
    CalGVVPDF(sample.Momenta(), resonances_, terms_, couplings_, omega_width_.DeviceView(),
        sample.FMatrix(), sample.WaveCoefficientBuffer(), sample.IntensityBuffer(),
        static_cast<int>(model_.terms.size()), static_cast<int>(model_.active_wave_types.size()),
        sample.Entries(), sample.OmegaFactorBuffer());
    return sample.IntensityBuffer();
}

void GVVAmplitude::EnsureCoefficientCapacity(int events)
{
    if (events <= coefficient_capacity_) return;
    if (coefficients_) {
        check_cuda(cudaFree(coefficients_), "free Term coefficient workspace");
        coefficients_ = nullptr;
    }
    coefficient_capacity_ = 0;
    check_cuda(cudaMallocManaged(&coefficients_, static_cast<std::size_t>(events)
        * model_.terms.size() * sizeof(DeviceComplex)), "allocate Term coefficient workspace");
    coefficient_capacity_ = events;
}

std::vector<double> GVVAmplitude::EvaluateComponentBatch(
    GVVSample& sample, int first_event, int number_events)
{
    if (first_event < 0 || number_events < 0 || first_event > sample.Entries() - number_events)
        throw std::out_of_range("component batch is outside the sample");
    if (number_events == 0) return {};
    const int terms = static_cast<int>(model_.terms.size());
    const int pairs = ctpwa::component_pair_count(terms);
    EnsureCoefficientCapacity(number_events);
    // Allocate event-by-pair storage only for projections. Integrated Post
    // calculations need just one accumulator per pair.
    if (number_events > component_capacity_) {
        if (components_) {
            check_cuda(cudaFree(components_), "free component batch");
            components_ = nullptr;
        }
        component_capacity_ = 0;
        check_cuda(cudaMallocManaged(&components_, static_cast<std::size_t>(number_events)
            * pairs * sizeof(double)), "allocate component batch");
        component_capacity_ = number_events;
    }
    sample.UpdateOmegaFactors(omega_width_.DeviceView(), omega_sigma_);
    CalGVVComponentBatch(sample.Momenta(), resonances_, terms_, couplings_, omega_width_.DeviceView(),
        sample.FMatrix(), coefficients_, components_, terms,
        static_cast<int>(model_.active_wave_types.size()), first_event, number_events,
        sample.OmegaFactorBuffer());
    return {components_, components_ + static_cast<std::size_t>(number_events) * pairs};
}

std::vector<double> GVVAmplitude::IntegrateComponents(GVVSample& sample, int batch_capacity)
{
    if (batch_capacity <= 0) throw std::invalid_argument("component batch capacity must be positive");
    const int terms = static_cast<int>(model_.terms.size());
    const int pairs = ctpwa::component_pair_count(terms);
    EnsureCoefficientCapacity(batch_capacity);
    if (!integrals_)
        check_cuda(cudaMallocManaged(&integrals_, pairs * sizeof(double)), "allocate component integrals");
    sample.UpdateOmegaFactors(omega_width_.DeviceView(), omega_sigma_);
    CalGVVComponentIntegrals(sample.Momenta(), resonances_, terms_, couplings_, omega_width_.DeviceView(),
        sample.FMatrix(), coefficients_, integrals_, batch_capacity, terms,
        static_cast<int>(model_.active_wave_types.size()), sample.Entries(), sample.OmegaFactorBuffer());
    std::vector<double> result(integrals_, integrals_ + pairs);
    const double total = std::accumulate(result.begin(), result.end(), 0.0);
    if (!(total > 0.0) || !std::isfinite(total))
        throw std::runtime_error("invalid integrated intensity in " + sample.Label());
    return result;
}

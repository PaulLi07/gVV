// GVV sample orchestration and normalized signed likelihood evaluation.
#include "process/FitLikelihood.h"
#include "framework/likelihood/Likelihood.h"
#include "process/PropagatorCompiler.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

} // namespace

FitLikelihood::FitLikelihood(
    GVVCompiledModel model,
    const GVVBranchConfig& branches)
    : branches_(branches),
      model_(std::move(model)),
      device_resonances_(nullptr),
      device_terms_(nullptr),
      device_couplings_(nullptr),
      component_coefficient_buffer_(nullptr),
      component_value_buffer_(nullptr),
      component_batch_capacity_(0),
      prepared_(false)
{
}

FitLikelihood::~FitLikelihood()
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
    if (component_coefficient_buffer_ != nullptr) {
        cudaFree(component_coefficient_buffer_);
    }
    if (component_value_buffer_ != nullptr) {
        cudaFree(component_value_buffer_);
    }
}

std::unique_ptr<GVVSample> FitLikelihood::LoadSample(
    const std::string& file_name,
    const std::string& label) const
{
    std::unique_ptr<GVVSample> sample(new GVVSample(label));
    sample->Load(file_name, branches_);
    return sample;
}

void FitLikelihood::LoadNormalizationMC(const std::string& file_name)
{
    if (prepared_ || normalization_mc_ != nullptr) {
        throw std::runtime_error(
            "normalization MC may be loaded exactly once before Prepare()");
    }
    normalization_mc_ = LoadSample(file_name, "normalization MC");
}

void FitLikelihood::LoadData(const std::string& file_name)
{
    if (prepared_ || data_ != nullptr) {
        throw std::runtime_error(
            "data may be loaded exactly once before Prepare()");
    }
    data_ = LoadSample(file_name, "data");
}

void FitLikelihood::AddBackground(
    const std::string& file_name,
    double likelihood_coefficient,
    const std::string& label)
{
    if (prepared_) {
        throw std::runtime_error(
            "backgrounds must be added before Prepare()");
    }
    BackgroundSample background;
    background.sample = LoadSample(file_name, label);
    background.likelihood_coefficient = likelihood_coefficient;
    backgrounds_.push_back(std::move(background));
}

void FitLikelihood::UploadModel()
{
    const std::size_t number_resonances = model_.resonances.size();
    const std::size_t number_terms = model_.terms.size();
    check_cuda(
        cudaMallocManaged(
            &device_resonances_,
            number_resonances * sizeof(ctpwa::PropagatorParameters)),
        "cudaMallocManaged GVV resonances");
    check_cuda(
        cudaMallocManaged(&device_terms_, number_terms * sizeof(TermSpec)),
        "cudaMallocManaged GVV terms");
    check_cuda(
        cudaMallocManaged(
            &device_couplings_, number_terms * sizeof(DeviceComplex)),
        "cudaMallocManaged GVV couplings");
    SynchronizeModel();
}

void FitLikelihood::SynchronizeModel()
{
    check_cuda(
        cudaMemcpy(
            device_resonances_,
            model_.resonances.data(),
            model_.resonances.size() * sizeof(ctpwa::PropagatorParameters),
            cudaMemcpyHostToDevice),
        "cudaMemcpy GVV resonances");
    check_cuda(
        cudaMemcpy(
            device_terms_,
            model_.terms.data(),
            model_.terms.size() * sizeof(TermSpec),
            cudaMemcpyHostToDevice),
        "cudaMemcpy GVV terms");
    check_cuda(
        cudaMemcpy(
            device_couplings_,
            model_.initial_couplings.data(),
            model_.initial_couplings.size() * sizeof(DeviceComplex),
            cudaMemcpyHostToDevice),
        "cudaMemcpy GVV couplings");
    if (prepared_) {
        normalization_mc_->UpdateOmegaFactors(omega_width_table_.DeviceView(), model_.omega_resolution_sigma);
        data_->UpdateOmegaFactors(omega_width_table_.DeviceView(), model_.omega_resolution_sigma);
        for (auto& background : backgrounds_)
            background.sample->UpdateOmegaFactors(omega_width_table_.DeviceView(), model_.omega_resolution_sigma);
    }
}

void FitLikelihood::Prepare()
{
    if (prepared_) {
        throw std::runtime_error("Prepare() may be called only once");
    }
    if (normalization_mc_ == nullptr || data_ == nullptr) {
        throw std::runtime_error(
            "normalization MC and data must be loaded before Prepare()");
    }

    omega_width_table_.Build();
    omega_width_table_.Upload();
    UploadModel();

    const int number_terms = NumberTerms();
    normalization_mc_->UploadAndBuildF(
        model_.active_wave_types, number_terms);
    data_->UploadAndBuildF(model_.active_wave_types, number_terms);
    for (BackgroundSample& background : backgrounds_) {
        background.sample->UploadAndBuildF(
            model_.active_wave_types, number_terms);
    }
    prepared_ = true;
    const OmegaWidthTableConfig& width_config = omega_width_table_.Config();
    std::cout << "GVV samples, F matrices, and omega width table prepared"
              << " (" << width_config.table_size << " mass points, "
              << width_config.dalitz_bins << "x"
              << width_config.dalitz_bins << " Dalitz midpoint grid, "
              << width_config.minimum_mass << "-"
              << width_config.maximum_mass << " GeV, clamped outside)\n";
}

double FitLikelihood::EvaluateSample(
    GVVSample& sample,
    double likelihood_coefficient,
    double normalization)
{
    CalGVVPDF(
        sample.Momenta(),
        device_resonances_,
        device_terms_,
        device_couplings_,
        omega_width_table_.DeviceView(),
        sample.FMatrix(),
        sample.WaveCoefficientBuffer(),
        sample.IntensityBuffer(),
        NumberTerms(),
        static_cast<int>(model_.active_wave_types.size()),
        sample.Entries(),
        sample.OmegaFactorBuffer());

    return ctpwa::log_likelihood_contribution(
        sample.IntensityBuffer(),
        static_cast<std::size_t>(sample.Entries()),
        normalization,
        likelihood_coefficient);
}

double FitLikelihood::LogLikelihood()
{
    if (!prepared_) {
        throw std::runtime_error("call Prepare() before evaluating likelihood");
    }
    SynchronizeModel();

    CalGVVPDF(
        normalization_mc_->Momenta(),
        device_resonances_,
        device_terms_,
        device_couplings_,
        omega_width_table_.DeviceView(),
        normalization_mc_->FMatrix(),
        normalization_mc_->WaveCoefficientBuffer(),
        normalization_mc_->IntensityBuffer(),
        NumberTerms(),
        static_cast<int>(model_.active_wave_types.size()),
        normalization_mc_->Entries(),
        normalization_mc_->OmegaFactorBuffer());

    const double normalization = ctpwa::monte_carlo_normalization(
        normalization_mc_->IntensityBuffer(),
        static_cast<std::size_t>(normalization_mc_->Entries()));

    double log_likelihood = EvaluateSample(*data_, +1.0, normalization);
    if (!std::isfinite(log_likelihood)) {
        return -1.0e100;
    }
    for (BackgroundSample& background : backgrounds_) {
        const double contribution = EvaluateSample(
            *background.sample,
            background.likelihood_coefficient,
            normalization);
        if (!std::isfinite(contribution)) {
            return -1.0e100;
        }
        log_likelihood += contribution;
    }
    return log_likelihood;
}

std::vector<double> FitLikelihood::EvaluateNormalizationMCIntensity()
{
    if (!prepared_) {
        throw std::runtime_error(
            "call Prepare() before evaluating projection intensities");
    }

    SynchronizeModel();
    CalGVVPDF(
        normalization_mc_->Momenta(),
        device_resonances_,
        device_terms_,
        device_couplings_,
        omega_width_table_.DeviceView(),
        normalization_mc_->FMatrix(),
        normalization_mc_->WaveCoefficientBuffer(),
        normalization_mc_->IntensityBuffer(),
        NumberTerms(),
        static_cast<int>(model_.active_wave_types.size()),
        normalization_mc_->Entries(),
        normalization_mc_->OmegaFactorBuffer());

    std::vector<double> intensity(
        static_cast<std::size_t>(normalization_mc_->Entries()), 0.0);
    std::copy(
        normalization_mc_->IntensityBuffer(),
        normalization_mc_->IntensityBuffer()
            + normalization_mc_->Entries(),
        intensity.begin());
    return intensity;
}

void FitLikelihood::EnsureComponentBatchCapacity(int number_events)
{
    if (number_events <= component_batch_capacity_) {
        return;
    }
    if (component_coefficient_buffer_ != nullptr) {
        check_cuda(
            cudaFree(component_coefficient_buffer_),
            "cudaFree projection Term coefficients");
        component_coefficient_buffer_ = nullptr;
    }
    if (component_value_buffer_ != nullptr) {
        check_cuda(
            cudaFree(component_value_buffer_),
            "cudaFree projection component values");
        component_value_buffer_ = nullptr;
    }
    component_batch_capacity_ = 0;

    const int number_pairs =
        ctpwa::component_pair_count(NumberTerms());
    check_cuda(
        cudaMallocManaged(
            &component_coefficient_buffer_,
            static_cast<std::size_t>(number_events) * NumberTerms()
                * sizeof(DeviceComplex)),
        "cudaMallocManaged projection Term coefficients");
    check_cuda(
        cudaMallocManaged(
            &component_value_buffer_,
            static_cast<std::size_t>(number_events) * number_pairs
                * sizeof(double)),
        "cudaMallocManaged projection component values");
    component_batch_capacity_ = number_events;
}

std::vector<double> FitLikelihood::EvaluateNormalizationMCComponentBatch(
    int first_event,
    int number_events)
{
    if (!prepared_) {
        throw std::runtime_error(
            "call Prepare() before evaluating projection components");
    }
    if (first_event < 0 || number_events < 0
        || first_event > normalization_mc_->Entries() - number_events) {
        throw std::out_of_range(
            "normalization MC component batch is outside the sample");
    }
    if (number_events == 0) {
        return {};
    }

    EnsureComponentBatchCapacity(number_events);
    SynchronizeModel();
    CalGVVComponentBatch(
        normalization_mc_->Momenta(),
        device_resonances_,
        device_terms_,
        device_couplings_,
        omega_width_table_.DeviceView(),
        normalization_mc_->FMatrix(),
        component_coefficient_buffer_,
        component_value_buffer_,
        NumberTerms(),
        static_cast<int>(model_.active_wave_types.size()),
        first_event,
        number_events,
        normalization_mc_->OmegaFactorBuffer());

    const std::size_t value_count =
        static_cast<std::size_t>(number_events)
        * ctpwa::component_pair_count(NumberTerms());
    return std::vector<double>(
        component_value_buffer_,
        component_value_buffer_ + value_count);
}

const GVVSample& FitLikelihood::NormalizationMCSample() const
{
    return *normalization_mc_;
}

const GVVSample& FitLikelihood::DataSample() const
{
    return *data_;
}

std::size_t FitLikelihood::NumberBackgroundSamples() const
{
    return backgrounds_.size();
}

const GVVSample& FitLikelihood::BackgroundSampleAt(
    std::size_t index) const
{
    return *backgrounds_[index].sample;
}

double FitLikelihood::BackgroundLikelihoodCoefficient(
    std::size_t index) const
{
    return backgrounds_[index].likelihood_coefficient;
}

GVVCompiledModel& FitLikelihood::MutableModel()
{
    return model_;
}

const GVVCompiledModel& FitLikelihood::Model() const
{
    return model_;
}

int FitLikelihood::NumberTerms() const
{
    return static_cast<int>(model_.terms.size());
}

int FitLikelihood::NumberResonances() const
{
    return static_cast<int>(model_.resonances.size());
}

int FitLikelihood::DataEntries() const
{
    return data_ == nullptr ? 0 : data_->Entries();
}

int FitLikelihood::NormalizationMCEntries() const
{
    return normalization_mc_ == nullptr ? 0 : normalization_mc_->Entries();
}

void FitLikelihood::PrintModelSummary() const
{
    std::cout << "GVV model '" << model_.definition.name << "': "
              << NumberResonances() << " active Resonances, "
              << NumberTerms() << " active coherent Terms, "
              << model_.active_wave_types.size() << " active Waves\n";
    for (int index = 0; index < NumberResonances(); ++index) {
        const ctpwa::PropagatorParameters& resonance = model_.resonances[index];
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
            for (const GVVPropagatorFitBinding& fit :
                 model_.propagator_fit_bindings) {
                if (fit.resonance_index == index
                    && fit.target == parameter.target) {
                    std::cout << " (fitted as " << fit.fit_name << ')';
                }
            }
        }
        std::cout << '\n';
    }
    for (int term = 0; term < NumberTerms(); ++term) {
        const GVVTermMetadata& metadata = model_.term_metadata[term];
        if (metadata.reference == ctpwa::CouplingReference::ScaleAndPhase) {
            const DeviceComplex coupling = model_.initial_couplings[term];
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

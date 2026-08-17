// GVV sample orchestration and normalized signed likelihood evaluation.
#include "process/FitLikelihood.h"
#include "framework/likelihood/Likelihood.h"

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
    std::cout << "GVV samples, F matrices, and omega width table prepared\n";
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
        sample.TermCoefficientBuffer(),
        sample.IntensityBuffer(),
        NumberTerms(),
        static_cast<int>(model_.active_wave_types.size()),
        sample.Entries());

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
        normalization_mc_->TermCoefficientBuffer(),
        normalization_mc_->IntensityBuffer(),
        NumberTerms(),
        static_cast<int>(model_.active_wave_types.size()),
        normalization_mc_->Entries());

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

std::vector<double> FitLikelihood::EvaluateNormalizationMCIntensity(
    const std::vector<DeviceComplex>& couplings)
{
    if (!prepared_) {
        throw std::runtime_error(
            "call Prepare() before evaluating projection intensities");
    }

    const std::vector<DeviceComplex> fitted_couplings =
        model_.initial_couplings;
    model_.initial_couplings = couplings;
    try {
        SynchronizeModel();
        CalGVVPDF(
            normalization_mc_->Momenta(),
            device_resonances_,
            device_terms_,
            device_couplings_,
            omega_width_table_.DeviceView(),
            normalization_mc_->FMatrix(),
            normalization_mc_->TermCoefficientBuffer(),
            normalization_mc_->IntensityBuffer(),
            NumberTerms(),
            static_cast<int>(model_.active_wave_types.size()),
            normalization_mc_->Entries());

        std::vector<double> intensity(
            static_cast<std::size_t>(normalization_mc_->Entries()), 0.0);
        std::copy(
            normalization_mc_->IntensityBuffer(),
            normalization_mc_->IntensityBuffer()
                + normalization_mc_->Entries(),
            intensity.begin());

        model_.initial_couplings = fitted_couplings;
        SynchronizeModel();
        return intensity;
    } catch (...) {
        model_.initial_couplings = fitted_couplings;
        SynchronizeModel();
        throw;
    }
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
              << NumberResonances() << " resonance definitions, "
              << NumberTerms() << " active coherent Terms, "
              << model_.active_wave_types.size() << " active Waves\n";
    for (int index = 0; index < NumberResonances(); ++index) {
        const ctpwa::PropagatorParameters& resonance = model_.resonances[index];
        std::cout << "  " << std::setw(10)
                  << model_.resonance_metadata[index].id
                  << "  m=" << resonance.mass
                  << "  model=" << ctpwa::propagator_name(
                         resonance.propagator_model);
        if (resonance.propagator_model == ctpwa::PROP_SUBTRACTED_FLATTE) {
            std::cout << "  Gamma_rest=" << resonance.pole_width;
        } else {
            std::cout << "  Gamma=" << resonance.pole_width;
        }
        if (model_.resonance_metadata[index].fit_sd_ratio) {
            std::cout << "  fit r_D/S=" << resonance.sd_ratio;
        }
        if (resonance.propagator_model == ctpwa::PROP_SUBTRACTED_FLATTE) {
            std::cout << "  R_omegaomega=" << resonance.flatte_ratio;
            if (model_.resonance_metadata[index].fit_flatte_ratio) {
                std::cout << " (fitted as log R)";
            } else {
                std::cout << " (fixed)";
            }
        }
        std::cout << '\n';
    }
    for (int term = 0; term < NumberTerms(); ++term) {
        const GVVTermMetadata& metadata = model_.term_metadata[term];
        if (metadata.reference == ctpwa::CouplingReference::ScaleAndPhase) {
            std::cout << "  scale-and-phase reference amplitude: "
                      << metadata.id << " = 1 + 0i\n";
        } else if (metadata.reference == ctpwa::CouplingReference::Phase) {
            std::cout << "  " << metadata.coherence_class
                      << " phase reference amplitude: " << metadata.id
                      << " = rho + 0i, rho > 0 and fitted as log(rho)\n";
        }
    }
}

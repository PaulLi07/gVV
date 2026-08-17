// Process orchestrator for gVV sample preparation and likelihood evaluation.
// Generic minimizer policy and all output serialization live elsewhere.
#ifndef CTPWA_PROCESS_FIT_LIKELIHOOD_H
#define CTPWA_PROCESS_FIT_LIKELIHOOD_H

#include "process/SampleLoader.h"
#include "process/OmegaWidthTable.h"
#include "process/WaveRegistry.cuh"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

// GVV-specific orchestration of samples, cached wave matrices, propagators,
// and the generic likelihood arithmetic. No Minuit policy lives here.
class FitLikelihood {
public:
    explicit FitLikelihood(
        GVVCompiledModel model,
        const GVVBranchConfig& branches = GVVBranchConfig());
    ~FitLikelihood();

    FitLikelihood(const FitLikelihood&) = delete;
    FitLikelihood& operator=(const FitLikelihood&) = delete;

    void LoadNormalizationMC(const std::string& file_name);
    void LoadData(const std::string& file_name);

    // likelihood_coefficient enters exactly as
    //   lnL_eff += coefficient * sum_events ln(P).
    // Therefore SB1=-0.5 and SB2=+0.25 implement the requested 2D sideband
    // subtraction without introducing a background PDF.
    void AddBackground(
        const std::string& file_name,
        double likelihood_coefficient,
        const std::string& label);

    void Prepare();
    double LogLikelihood();

    // ParameterMapping mutates this host state; LogLikelihood synchronizes it
    // to the already allocated device model before evaluating any sample.
    GVVCompiledModel& MutableModel();
    const GVVCompiledModel& Model() const;
    int NumberTerms() const;
    int NumberResonances() const;

    int DataEntries() const;
    int NormalizationMCEntries() const;
    void PrintModelSummary() const;

    // Narrow read/evaluate interface used by ProjectionWriter. The writer
    // sees process samples and intensities, but never device allocations or
    // likelihood-internal synchronization details.
    std::vector<double> EvaluateNormalizationMCIntensity(
        const std::vector<DeviceComplex>& couplings);
    const GVVSample& NormalizationMCSample() const;
    const GVVSample& DataSample() const;
    std::size_t NumberBackgroundSamples() const;
    const GVVSample& BackgroundSampleAt(std::size_t index) const;
    double BackgroundLikelihoodCoefficient(std::size_t index) const;

private:
    struct BackgroundSample {
        std::unique_ptr<GVVSample> sample;
        double likelihood_coefficient;
    };

    std::unique_ptr<GVVSample> LoadSample(
        const std::string& file_name,
        const std::string& label) const;
    void UploadModel();
    void SynchronizeModel();
    double EvaluateSample(
        GVVSample& sample,
        double likelihood_coefficient,
        double normalization);

    GVVBranchConfig branches_;
    GVVCompiledModel model_;

    std::unique_ptr<GVVSample> normalization_mc_;
    std::unique_ptr<GVVSample> data_;
    std::vector<BackgroundSample> backgrounds_;

    OmegaWidthTable omega_width_table_;
    ctpwa::PropagatorParameters* device_resonances_;
    TermSpec* device_terms_;
    DeviceComplex* device_couplings_;
    bool prepared_;
};

#endif // CTPWA_PROCESS_FIT_LIKELIHOOD_H

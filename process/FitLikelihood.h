// Process orchestrator for the gVV likelihood and projection output. Generic
// minimizer policy and output naming remain in framework/fit/.
#ifndef CTPWA_PROCESS_FIT_LIKELIHOOD_H
#define CTPWA_PROCESS_FIT_LIKELIHOOD_H

#include "process/SampleLoader.h"
#include "process/OmegaWidthTable.h"
#include "process/WaveRegistry.cuh"

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

    void SetCoupling(int term_index, double real, double imag);
    void SetLogCouplingMagnitude(int term_index, double log_magnitude);
    DeviceComplex Coupling(int term_index) const;
    void SetLogSDRatio(int resonance_index, double log_ratio);
    void SetLogFlatteRatio(int resonance_index, double log_ratio);
    const ctpwa::PropagatorParameters& Resonance(int resonance_index) const;
    const GVVCompiledModel& Model() const;
    int NumberTerms() const;
    int NumberResonances() const;

    int DataEntries() const;
    int NormalizationMCEntries() const;
    void PrintModelSummary() const;
    void WriteProjection(
        const std::string& save_name,
        int best_start,
        long long best_seed,
        double minimum);

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

#pragma once

#include "core/Model.h"
#include "core/Sample.h"

// Shared numerical engine for Fit, projections, and Post. The caller owns the
// immutable model and samples and must keep them alive. Sample roles, signed
// likelihoods, Minuit policy, and covariance displacements belong to scripts.
class GVVAmplitude {
public:
    explicit GVVAmplitude(const GVVCompiledModel& model);
    ~GVVAmplitude();
    GVVAmplitude(const GVVAmplitude&) = delete;
    GVVAmplitude& operator=(const GVVAmplitude&) = delete;

    void Prepare(GVVSample& sample);
    void SetParameters(const GVVParameterState& state);
    // The returned buffer belongs to the sample, valid until its next evaluation.
    const double* EvaluateIntensity(GVVSample& sample);
    std::vector<double> EvaluateComponentBatch(
        GVVSample& sample, int first_event, int number_events);
    // Sum, not mean: Post efficiency requires matching generated exposures.
    std::vector<double> IntegrateComponents(GVVSample& sample, int batch_capacity = 4096);
    const GVVCompiledModel& Model() const { return model_; }
    double OmegaSigma() const { return omega_sigma_; }
    const OmegaWidthTableConfig& WidthConfig() const { return omega_width_.Config(); }

private:
    void EnsureCoefficientCapacity(int events);
    void Release();

    const GVVCompiledModel& model_;
    OmegaWidthTable omega_width_;
    double omega_sigma_ = 0.0;
    ctpwa::PropagatorParameters* resonances_ = nullptr;
    TermSpec* terms_ = nullptr;
    DeviceComplex* couplings_ = nullptr;
    DeviceComplex* coefficients_ = nullptr;
    double* components_ = nullptr;
    double* integrals_ = nullptr;
    int coefficient_capacity_ = 0;
    int component_capacity_ = 0;
};

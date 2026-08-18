// GPU-backed evaluator shared by all Post Calculation observables. It owns
// truth/selected samples and returns integrated pairwise Term components.
#ifndef GVV_POST_CALCULATION_COMPONENT_EVALUATOR_H
#define GVV_POST_CALCULATION_COMPONENT_EVALUATOR_H

#include "process/ParameterMapping.h"
#include "process/SampleLoader.h"

#include <string>
#include <vector>

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
    ~GVVComponentEvaluator();

    GVVComponentEvaluator(const GVVComponentEvaluator&) = delete;
    GVVComponentEvaluator& operator=(const GVVComponentEvaluator&) = delete;

    GVVIntegratedComponents Evaluate(const std::vector<double>& values);
    void ValidateTotal(
        const std::vector<double>& values,
        const GVVIntegratedComponents& components);

    int TruthEntries() const;
    int SelectedEntries() const;
    int NumberTerms() const;
    int NumberPairs() const;
    const GVVCompiledModel& Model() const;
    const std::vector<GVVFitParameterBinding>& Layout() const;

private:
    void Upload(const GVVCompiledModel& state);
    std::vector<double> EvaluateSample(GVVSample& sample);
    void ValidateSample(GVVSample& sample, double component_sum);

    GVVCompiledModel model_;
    std::vector<GVVFitParameterBinding> layout_;
    GVVSample truth_;
    GVVSample selected_;
    OmegaWidthTable omega_width_table_;
    ctpwa::PropagatorParameters* device_resonances_;
    TermSpec* device_terms_;
    DeviceComplex* device_couplings_;
    double* component_buffer_;
};

#endif // GVV_POST_CALCULATION_COMPONENT_EVALUATOR_H

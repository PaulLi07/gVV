#ifndef GVV_SAMPLE_H
#define GVV_SAMPLE_H

#include "kernel.h"

#include <array>
#include <string>
#include <vector>

enum GVVFourVectorOrder {
    GVV_PX_PY_PZ_E = 0,
    GVV_E_PX_PY_PZ = 1
};

struct GVVBranchConfig {
    std::string tree_name;
    std::array<std::string, 7> branches;
    GVVFourVectorOrder input_order;

    GVVBranchConfig();
};

// A physics-neutral event sample shared by fitting and future post-fit tools.
// It owns ROOT-loaded four-momenta and the corresponding GPU buffers, but it
// does not know whether the sample is data, accepted PHSP, sideband, or truth
// PHSP.  Likelihood coefficients remain the responsibility of NLL_estimator.
class GVVSample {
public:
    explicit GVVSample(const std::string& label = "sample");
    ~GVVSample();

    GVVSample(const GVVSample&) = delete;
    GVVSample& operator=(const GVVSample&) = delete;

    void Load(
        const std::string& file_name,
        const GVVBranchConfig& branches);
    void UploadAndBuildF(
        const std::vector<int>& active_wave_types,
        int number_terms);
    // Transitional overload for legacy post-fit code. Production fitting
    // always supplies the runtime model layout explicitly.
    void UploadAndBuildF();

    const std::string& Label() const;
    int Entries() const;
    const double* HostMomentum(int particle, int event) const;
    GVVDeviceMomenta Momenta() const;
    const double* FMatrix() const;
    DeviceComplex* TermCoefficientBuffer();
    double* IntensityBuffer();
    int NumberActiveWaves() const;
    int NumberTerms() const;

private:
    std::string label_;
    int entries_;
    std::array<std::vector<double>, 7> host_p4_;
    std::array<double*, 7> device_p4_;
    double* F_matrix_;
    DeviceComplex* term_coefficients_;
    double* amp2_;
    int number_active_waves_;
    int number_terms_;
};

#endif // GVV_SAMPLE_H

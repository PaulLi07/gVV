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

// GVV process event sample shared by fitting and PostFit. It owns the fixed
// seven-particle ROOT schema and corresponding GPU buffers, but it does not
// know whether a sample is data, accepted PHSP, sideband, or truth PHSP.
// A future decay topology replaces this class at the process boundary.
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

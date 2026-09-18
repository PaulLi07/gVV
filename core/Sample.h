#pragma once

#include "core/physics/Event.cuh"
#include "core/physics/OmegaWidth.h"
#include <array>
#include <string>
#include <vector>

// ROOT-to-GPU sample boundary for the fixed seven-particle gVV topology.

enum GVVFourVectorOrder {
    GVV_PX_PY_PZ_E = 0,
    GVV_E_PX_PY_PZ = 1
};

// This count is fixed by the gVV final-state topology, not by model.json.
// Resonance, Wave, Term, and fit-parameter counts remain fully dynamic.
constexpr int GVV_NFINAL_PARTICLES = 7;

struct GVVBranchConfig {
    std::string tree_name;
    std::array<std::string, GVV_NFINAL_PARTICLES> branches;
    GVVFourVectorOrder input_order;

    GVVBranchConfig();
};

// GVV process event sample shared by Fit and Post Calculation. It owns the fixed
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

    // The width table is immutable after Prepare. Only sigma changes this cache.
    void UpdateOmegaFactors(ctpwa::TabulatedFunctionView width_table, double sigma);
    const DeviceComplex* OmegaFactorBuffer() const;

    const std::string& Label() const;
    int Entries() const;
    const double* HostMomentum(int particle, int event) const;
    GVVDeviceMomenta Momenta() const;
    const double* FMatrix() const;
    DeviceComplex* WaveCoefficientBuffer();
    double* IntensityBuffer();
    int NumberActiveWaves() const;
    int NumberTerms() const;

private:
    std::string label_;
    int entries_;
    std::array<std::vector<double>, GVV_NFINAL_PARTICLES> host_p4_;
    std::array<double*, GVV_NFINAL_PARTICLES> device_p4_;
    double* F_matrix_;
    DeviceComplex* wave_coefficients_;
    double* amp2_;
    DeviceComplex* omega_factors_;
    double omega_factor_sigma_;
    int number_active_waves_;
    int number_terms_;
};

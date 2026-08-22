// gVV omega running-width lookup table shared by host validation and device
// Term evaluation.
#ifndef CTPWA_PROCESS_OMEGA_WIDTH_TABLE_H
#define CTPWA_PROCESS_OMEGA_WIDTH_TABLE_H

#include "framework/dynamics/Propagators.cuh"
#include "framework/dynamics/TabulatedFunction.cuh"
#include "process/OmegaDecayModel.cuh"

#include <vector>

__host__ __device__ inline DeviceComplex gvv_omega_propagator(
    double s,
    const ctpwa::TabulatedFunctionView& width_table)
{
    return ctpwa::BW_from_width(
        s,
        GVV_OMEGA_MASS,
        width_table.interpolate_clamped(s));
}

// Host-side three-body width builder and optional GPU upload. The table and
// event current share process/OmegaDecayModel.cuh as their rho-isobar source.
class OmegaWidthTable {
public:
    OmegaWidthTable();
    ~OmegaWidthTable();

    OmegaWidthTable(const OmegaWidthTable&) = delete;
    OmegaWidthTable& operator=(const OmegaWidthTable&) = delete;

    void Build(
        int table_size = 512,
        int dalitz_bins = 48,
        double minimum_mass = 0.40,
        double maximum_mass = 1.20);
    void Upload();

    double Width(double s) const;
    ctpwa::TabulatedFunctionView HostView() const;
    ctpwa::TabulatedFunctionView DeviceView() const;

private:
    std::vector<double> values_;
    double s_min_;
    double s_step_;
    double* device_values_;
};

#endif // CTPWA_PROCESS_OMEGA_WIDTH_TABLE_H

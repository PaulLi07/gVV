// gVV omega running-width lookup table shared by host validation and device
// Term evaluation.
#ifndef CTPWA_PROCESS_OMEGA_WIDTH_TABLE_H
#define CTPWA_PROCESS_OMEGA_WIDTH_TABLE_H

#include "framework/math/DeviceComplex.cuh"
#include "process/ProcessKinematics.cuh"

#include <vector>

struct GVVWidthTableView {
    const double* values;
    int size;
    double s_min;
    double s_step;

    __host__ __device__ GVVWidthTableView(
        const double* table = nullptr,
        int table_size = 0,
        double minimum = 0.0,
        double step = 0.0)
        : values(table), size(table_size), s_min(minimum), s_step(step)
    {
    }

    __host__ __device__ double interpolate(double s) const
    {
        if (values == nullptr || size <= 0 || s_step <= 0.0) {
            return 0.0;
        }
        if (s <= s_min) {
            return values[0];
        }
        const double coordinate = (s - s_min) / s_step;
        int lower = static_cast<int>(coordinate);
        if (lower >= size - 1) {
            return values[size - 1];
        }
        if (lower < 0) {
            lower = 0;
        }
        const double fraction = coordinate - lower;
        return values[lower] * (1.0 - fraction)
               + values[lower + 1] * fraction;
    }
};

__host__ __device__ inline DeviceComplex gvv_omega_propagator(
    double s,
    const GVVWidthTableView& width_table)
{
    const double gamma_s = width_table.interpolate(s);
    return 1.0 / DeviceComplex(
        GVV_OMEGA_MASS * GVV_OMEGA_MASS - s,
        -GVV_OMEGA_MASS * gamma_s);
}

// Host-side three-body width builder and optional GPU upload.  The table is
// based on the same coherent rho-isobar current used in process/ProcessKinematics.cuh.
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
    GVVWidthTableView HostView() const;
    GVVWidthTableView DeviceView() const;

private:
    std::vector<double> values_;
    double s_min_;
    double s_step_;
    double* device_values_;
};

#endif // CTPWA_PROCESS_OMEGA_WIDTH_TABLE_H

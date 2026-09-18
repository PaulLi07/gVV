#pragma once

#include "core/physics/Propagators.cuh"
#include "core/physics/OmegaDecay.cuh"
#include <vector>

// Lightweight host/device view of a uniformly sampled scalar function.
// The owner of the sampled values and the physical extrapolation policy stay
// outside this reusable numerical type.

namespace ctpwa {

struct TabulatedFunctionView {
    const double* values;
    int size;
    double x_min;
    double x_step;

    __host__ __device__ TabulatedFunctionView(
        const double* table = nullptr,
        int table_size = 0,
        double minimum = 0.0,
        double step = 0.0)
        : values(table), size(table_size), x_min(minimum), x_step(step)
    {
    }

    // Preserve the project's existing endpoint-clamping behavior explicitly.
    // A process that needs a different extrapolation policy should apply it at
    // its own line-shape boundary rather than hiding it in this view.
    __host__ __device__ double interpolate_clamped(double x) const
    {
        if (values == nullptr || size <= 0 || x_step <= 0.0) {
            return 0.0;
        }
        if (x <= x_min) {
            return values[0];
        }

        const double coordinate = (x - x_min) / x_step;
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

} // namespace ctpwa

// gVV omega running-width lookup table shared by host validation and device
// Term evaluation.

enum class OmegaWidthExtrapolation {
    Clamp
};

struct OmegaWidthTableConfig {
    int table_size = 512;
    int dalitz_bins = 48;
    double minimum_mass = 0.40;
    double maximum_mass = 1.20;
    OmegaWidthExtrapolation extrapolation =
        OmegaWidthExtrapolation::Clamp;
};

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
// event current share core/physics/OmegaDecay.cuh as their rho-isobar source.
class OmegaWidthTable {
public:
    OmegaWidthTable();
    ~OmegaWidthTable();

    OmegaWidthTable(const OmegaWidthTable&) = delete;
    OmegaWidthTable& operator=(const OmegaWidthTable&) = delete;

    void Build(const OmegaWidthTableConfig& config = OmegaWidthTableConfig());
    void Upload();

    double Width(double s) const;
    ctpwa::TabulatedFunctionView HostView() const;
    ctpwa::TabulatedFunctionView DeviceView() const;
    const OmegaWidthTableConfig& Config() const;

private:
    std::vector<double> values_;
    OmegaWidthTableConfig config_;
    double s_min_;
    double s_step_;
    double* device_values_;
};

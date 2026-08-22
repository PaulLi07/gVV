// Numerical three-pion phase-space integration and GPU upload for the omega
// running-width lookup table.
#include "process/OmegaWidthTable.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace {

double kallen(double x, double y, double z)
{
    return x * x + y * y + z * z - 2.0 * (x * y + x * z + y * z);
}

// The common constants in dPhi_3 cancel in the pole normalization.  We retain
// the 1/s part of the Dalitz density here; Build() adds m_omega/sqrt(s), giving
// Gamma_omega(s) = Gamma0*m0/sqrt(s)*I_3(s)/I_3(m0^2).
double omega_phase_integral(double s, int bins)
{
    const double parent_mass = std::sqrt(std::max(s, 0.0));
    if (parent_mass <= GVV_PIP_MASS + GVV_PIM_MASS + GVV_PI0_MASS
        || bins <= 0) {
        return 0.0;
    }

    const double m0_sq = GVV_PI0_MASS * GVV_PI0_MASS;
    const double m1_sq = GVV_PIP_MASS * GVV_PIP_MASS;
    const double m2_sq = GVV_PIM_MASS * GVV_PIM_MASS;
    const double s12_min =
        (GVV_PIP_MASS + GVV_PIM_MASS)
        * (GVV_PIP_MASS + GVV_PIM_MASS);
    const double s12_max = (parent_mass - GVV_PI0_MASS)
                           * (parent_mass - GVV_PI0_MASS);
    const double ds12 = (s12_max - s12_min) / bins;
    double integral = 0.0;

    for (int i = 0; i < bins; ++i) {
        const double s12 = s12_min + (i + 0.5) * ds12;
        const double lambda12 = std::max(kallen(s12, m1_sq, m2_sq), 0.0);
        const double lambda_parent = std::max(kallen(s, s12, m0_sq), 0.0);
        const double center = m1_sq + m0_sq
            + (s12 + m1_sq - m2_sq) * (s - s12 - m0_sq)
              / (2.0 * s12);
        const double half_range = std::sqrt(lambda12 * lambda_parent)
                                  / (2.0 * s12);
        const double s10_min = center - half_range;
        const double s10_max = center + half_range;
        const double ds10 = (s10_max - s10_min) / bins;

        for (int j = 0; j < bins; ++j) {
            const double s10 = s10_min + (j + 0.5) * ds10;
            const double s20 = s + m0_sq + m1_sq + m2_sq - s12 - s10;

            const double energy1 = (s + m1_sq - s20) / (2.0 * parent_mass);
            const double energy2 = (s + m2_sq - s10) / (2.0 * parent_mass);
            const double p1_sq = std::max(energy1 * energy1 - m1_sq, 0.0);
            const double p2_sq = std::max(energy2 * energy2 - m2_sq, 0.0);
            const double minkowski12 = (s12 - m1_sq - m2_sq) / 2.0;
            const double dot3 = energy1 * energy2 - minkowski12;
            const double cross_sq = std::max(
                p1_sq * p2_sq - dot3 * dot3, 0.0);
            const double geometry_sq = s * cross_sq;

            const DeviceComplex coherent_rho = coherent_omega_rho_factor(
                s, s12, s10, s20, m0_sq, m1_sq, m2_sq);

            integral += geometry_sq * coherent_rho.rho2() * ds12 * ds10;
        }
    }
    return integral / s;
}

void check_cuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

} // namespace

OmegaWidthTable::OmegaWidthTable()
    : s_min_(0.0), s_step_(0.0), device_values_(nullptr)
{
}

OmegaWidthTable::~OmegaWidthTable()
{
    if (device_values_ != nullptr) {
        cudaFree(device_values_);
    }
}

void OmegaWidthTable::Build(
    int table_size,
    int dalitz_bins,
    double minimum_mass,
    double maximum_mass)
{
    if (table_size < 2 || dalitz_bins < 2 || minimum_mass <= 0.0
        || maximum_mass <= minimum_mass) {
        throw std::invalid_argument("invalid omega width table configuration");
    }

    s_min_ = minimum_mass * minimum_mass;
    const double s_max = maximum_mass * maximum_mass;
    s_step_ = (s_max - s_min_) / (table_size - 1);
    values_.assign(table_size, 0.0);

    const double pole_s = GVV_OMEGA_MASS * GVV_OMEGA_MASS;
    const double pole_integral = omega_phase_integral(pole_s, dalitz_bins);
    if (!(pole_integral > 0.0)) {
        throw std::runtime_error("omega pole phase-space integral is zero");
    }

    for (int i = 0; i < table_size; ++i) {
        const double s = s_min_ + i * s_step_;
        const double integral = omega_phase_integral(s, dalitz_bins);
        if (integral > 0.0) {
            values_[i] = GVV_OMEGA_WIDTH * GVV_OMEGA_MASS / std::sqrt(s)
                         * integral / pole_integral;
        }
    }
}

void OmegaWidthTable::Upload()
{
    if (values_.empty()) {
        throw std::runtime_error("build omega width table before upload");
    }
    if (device_values_ != nullptr) {
        check_cuda(cudaFree(device_values_), "cudaFree omega width table");
        device_values_ = nullptr;
    }
    check_cuda(
        cudaMallocManaged(&device_values_, values_.size() * sizeof(double)),
        "cudaMallocManaged omega width table");
    check_cuda(
        cudaMemcpy(
            device_values_,
            values_.data(),
            values_.size() * sizeof(double),
            cudaMemcpyHostToDevice),
        "cudaMemcpy omega width table");
}

double OmegaWidthTable::Width(double s) const
{
    return HostView().interpolate_clamped(s);
}

ctpwa::TabulatedFunctionView OmegaWidthTable::HostView() const
{
    return ctpwa::TabulatedFunctionView(
        values_.empty() ? nullptr : values_.data(),
        static_cast<int>(values_.size()),
        s_min_,
        s_step_);
}

ctpwa::TabulatedFunctionView OmegaWidthTable::DeviceView() const
{
    return ctpwa::TabulatedFunctionView(
        device_values_,
        static_cast<int>(values_.size()),
        s_min_,
        s_step_);
}

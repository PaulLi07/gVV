#include "../include/OmegaPropagator.h"

#include "../include/Dynamics.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace {

constexpr double MASS_PIP = 0.13957039;
constexpr double MASS_PIM = 0.13957039;
constexpr double MASS_PI0 = 0.1349768;
constexpr double MASS_RHO = 0.77526;
constexpr double WIDTH_RHO = 0.1474;

double kallen(double x, double y, double z)
{
    return x * x + y * y + z * z - 2.0 * (x * y + x * z + y * z);
}

DeviceComplex rho_factor(
    double s_parent,
    double s_pair,
    double bachelor_mass2,
    double first_mass2,
    double second_mass2)
{
    const double q_parent = ctpwa::two_body_Q(
        s_parent, s_pair, bachelor_mass2);
    const double q_pair = ctpwa::two_body_Q(
        s_pair, first_mass2, second_mass2);
    const double b_parent = ctpwa::blatt_weisskopf(q_parent, 1);
    const double b_pair = ctpwa::blatt_weisskopf(q_pair, 1);
    const DeviceComplex rho = ctpwa::BWR(
        s_pair,
        MASS_RHO,
        WIDTH_RHO,
        1,
        first_mass2,
        second_mass2);
    return b_parent * rho * b_pair;
}

// The common constants in dPhi_3 cancel in the pole normalization.  We retain
// the 1/s part of the Dalitz density here; Build() adds m_omega/sqrt(s), giving
// Gamma_omega(s) = Gamma0*m0/sqrt(s)*I_3(s)/I_3(m0^2).
double omega_phase_integral(double s, int bins)
{
    const double parent_mass = std::sqrt(std::max(s, 0.0));
    if (parent_mass <= MASS_PIP + MASS_PIM + MASS_PI0 || bins <= 0) {
        return 0.0;
    }

    const double m0_sq = MASS_PI0 * MASS_PI0;
    const double m1_sq = MASS_PIP * MASS_PIP;
    const double m2_sq = MASS_PIM * MASS_PIM;
    const double s12_min = (MASS_PIP + MASS_PIM) * (MASS_PIP + MASS_PIM);
    const double s12_max = (parent_mass - MASS_PI0)
                           * (parent_mass - MASS_PI0);
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

            const DeviceComplex f12 = rho_factor(
                s, s12, m0_sq, m1_sq, m2_sq);
            const DeviceComplex f10 = rho_factor(
                s, s10, m2_sq, m1_sq, m0_sq);
            const DeviceComplex f20 = rho_factor(
                s, s20, m1_sq, m2_sq, m0_sq);
            const DeviceComplex coherent_rho = f12 + f10 + f20;

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
    return HostView().interpolate(s);
}

GVVWidthTableView OmegaWidthTable::HostView() const
{
    return GVVWidthTableView(
        values_.empty() ? nullptr : values_.data(),
        static_cast<int>(values_.size()),
        s_min_,
        s_step_);
}

GVVWidthTableView OmegaWidthTable::DeviceView() const
{
    return GVVWidthTableView(
        device_values_,
        static_cast<int>(values_.size()),
        s_min_,
        s_step_);
}

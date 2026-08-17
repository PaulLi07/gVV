// Unit regression for reusable two-body dynamics plus compilation of the
// complete device-side omega decay-current path.
#include "process/ProcessKinematics.cuh"

#include <cmath>
#include <iostream>

namespace {

bool close_to(double lhs, double rhs, double tolerance = 1.0e-12)
{
    return std::abs(lhs - rhs) <= tolerance;
}

} // namespace

// This kernel is compiled (but not launched on the login node) so nvcc checks
// the complete device path through Omega, f^(rho), BWR, B1 and Q.
__global__ void compile_omega_device_path(double* output)
{
    const FV p0(0.25, 0.08, 0.02, 0.18);
    const FV p1(0.30, -0.12, 0.05, -0.20);
    const FV p2(0.24, 0.04, -0.07, 0.02);

    const OmegaDecayCurrent omega =
        build_omega_decay_current(p0, p1, p2);

    output[0] = omega.geometry.Get(0);
    output[1] = omega.rho_factor.real;
    output[2] = omega.rho_factor.imag;
}

int main()
{
    const DeviceComplex lower_half_plane(0.0, -1.0);
    if (!close_to(
            lower_half_plane.phi(),
            1.5 * 3.14159265358979323846)) {
        std::cerr << "DeviceComplex phase convention is wrong\n";
        return 1;
    }

    const double s_a = 2.25;
    const double s_b = 0.36;
    const double s_c = 0.16;
    const double q2_reference =
        (s_a + s_b - s_c) * (s_a + s_b - s_c)
        / (4.0 * s_a) - s_b;

    if (!close_to(ctpwa::two_body_Q2(s_a, s_b, s_c), q2_reference)) {
        std::cerr << "two_body_Q2 does not match the project definition\n";
        return 1;
    }

    const double q = ctpwa::two_body_Q(s_a, s_b, s_c);
    if (!close_to(q * q, q2_reference)) {
        std::cerr << "two_body_Q is inconsistent with two_body_Q2\n";
        return 2;
    }

    const double old_scale = 0.197321 / 0.59;
    const double old_b1 = std::sqrt(2.0 / (q * q + old_scale * old_scale));
    if (!close_to(ctpwa::blatt_weisskopf(q, 1), old_b1)) {
        std::cerr << "B1 no longer matches the legacy CTPWA implementation\n";
        return 3;
    }

    const double rho_mass = 0.77526;
    const double rho_width = 0.1474;
    const double charged_pion_mass = 0.13957039;
    const double pion_mass2 = charged_pion_mass * charged_pion_mass;
    const double gamma_at_pole = ctpwa::running_width(
        rho_mass * rho_mass,
        rho_mass,
        rho_width,
        1,
        pion_mass2,
        pion_mass2,
        0.59);

    if (!close_to(gamma_at_pole, rho_width)) {
        std::cerr << "running width is not normalized at the rho pole\n";
        return 4;
    }

    const double omega_mass = 0.78266;
    const double threshold = 4.0 * omega_mass * omega_mass;
    const DeviceComplex rho_below = ctpwa::two_body_rho_equal_mass(
        0.99 * threshold, omega_mass);
    const DeviceComplex rho_at_threshold = ctpwa::two_body_rho_equal_mass(
        threshold, omega_mass);
    const DeviceComplex rho_above = ctpwa::two_body_rho_equal_mass(
        1.01 * threshold, omega_mass);
    if (!(rho_below.real == 0.0 && rho_below.imag > 0.0
          && rho_at_threshold.real == 0.0
          && rho_at_threshold.imag == 0.0
          && rho_above.real > 0.0 && rho_above.imag == 0.0)) {
        std::cerr << "two-body rho analytic continuation is wrong\n";
        return 5;
    }

    const double f0_mass = 1.522;
    const double f0_width = 0.108;
    const double probe_s = 1.70 * 1.70;
    const DeviceComplex flatte_zero = ctpwa::Flatte_subtracted_effective(
        probe_s, f0_mass, f0_width, 0.0, omega_mass);
    const DeviceComplex fixed_bw = ctpwa::BW_fixed_width(
        probe_s, f0_mass, f0_width);
    if (!close_to(flatte_zero.real, fixed_bw.real)
        || !close_to(flatte_zero.imag, fixed_bw.imag)) {
        std::cerr << "R_omegaomega=0 does not recover the fixed-width BW\n";
        return 6;
    }

    const DeviceComplex flatte_at_mass = ctpwa::Flatte_subtracted_effective(
        f0_mass * f0_mass, f0_mass, f0_width, 3.0, omega_mass);
    const DeviceComplex bw_at_mass = ctpwa::BW_fixed_width(
        f0_mass * f0_mass, f0_mass, f0_width);
    if (!close_to(flatte_at_mass.real, bw_at_mass.real)
        || !close_to(flatte_at_mass.imag, bw_at_mass.imag)) {
        std::cerr << "subtraction does not preserve the fixed mass point\n";
        return 7;
    }

    std::cout << "Dynamics tests passed\n";
    return 0;
}

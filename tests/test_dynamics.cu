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
    const DeviceComplex shared_bw = ctpwa::BW_from_width(
        probe_s, f0_mass, f0_width);
    if (!close_to(shared_bw.real, fixed_bw.real)
        || !close_to(shared_bw.imag, fixed_bw.imag)) {
        std::cerr << "fixed-width BW bypasses the shared denominator\n";
        return 6;
    }
    if (!close_to(flatte_zero.real, fixed_bw.real)
        || !close_to(flatte_zero.imag, fixed_bw.imag)) {
        std::cerr << "R_omegaomega=0 does not recover the fixed-width BW\n";
        return 7;
    }

    const DeviceComplex flatte_at_mass = ctpwa::Flatte_subtracted_effective(
        f0_mass * f0_mass, f0_mass, f0_width, 3.0, omega_mass);
    const DeviceComplex bw_at_mass = ctpwa::BW_fixed_width(
        f0_mass * f0_mass, f0_mass, f0_width);
    if (!close_to(flatte_at_mass.real, bw_at_mass.real)
        || !close_to(flatte_at_mass.imag, bw_at_mass.imag)) {
        std::cerr << "subtraction does not preserve the fixed mass point\n";
        return 8;
    }

    // The event current and omega-width integration must use the same
    // coherent rho-isobar function. Reconstruct the former local expression
    // here as an independent regression reference.
    const double s_omega = GVV_OMEGA_MASS * GVV_OMEGA_MASS;
    const double s0 = GVV_PI0_MASS * GVV_PI0_MASS;
    const double s1 = GVV_PIP_MASS * GVV_PIP_MASS;
    const double s2 = GVV_PIM_MASS * GVV_PIM_MASS;
    const double s12 = 0.220;
    const double s10 = 0.225;
    const double s20 = s_omega + s0 + s1 + s2 - s12 - s10;
    const RhoBWRParameters rho;
    const auto old_isobar = [&](double s_pair,
                                double s_bachelor,
                                double s_first,
                                double s_second) {
        const double q_parent = ctpwa::two_body_Q(
            s_omega, s_pair, s_bachelor);
        const double q_pair = ctpwa::two_body_Q(
            s_pair, s_first, s_second);
        return ctpwa::blatt_weisskopf(
                   q_parent, 1, rho.omega_vertex_radius_fm)
               * ctpwa::BWR(
                   s_pair,
                   rho.mass,
                   rho.width,
                   1,
                   s_first,
                   s_second,
                   rho.rho_vertex_radius_fm)
               * ctpwa::blatt_weisskopf(
                   q_pair, 1, rho.rho_vertex_radius_fm);
    };
    const DeviceComplex old_coherent_rho =
        old_isobar(s12, s0, s1, s2)
        + old_isobar(s10, s2, s1, s0)
        + old_isobar(s20, s1, s2, s0);
    const DeviceComplex shared_coherent_rho = coherent_omega_rho_factor(
        s_omega, s12, s10, s20, s0, s1, s2, rho);
    if (!close_to(shared_coherent_rho.real, old_coherent_rho.real)
        || !close_to(shared_coherent_rho.imag, old_coherent_rho.imag)) {
        std::cerr << "shared omega rho-isobar factor changed the decay model\n";
        return 9;
    }

    std::cout << "Dynamics tests passed\n";
    return 0;
}

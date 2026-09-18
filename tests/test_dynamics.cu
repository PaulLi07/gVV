// Unit regression for reusable two-body dynamics plus compilation of the
// complete device-side omega decay-current path.
#include "core/physics/Event.cuh"

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

    // The higher-L factors use the same normalization: at q=q0, q^L B_L=1.
    // Check both the newly completed consecutive L=3 support and the L=4
    // factor needed by a bare G-wave orbital tensor.
    const double barrier_probe = 0.47;
    const double barrier_q2 = barrier_probe * barrier_probe;
    const double barrier_q4 = barrier_q2 * barrier_q2;
    const double barrier_scale2 = old_scale * old_scale;
    const double barrier_scale4 = barrier_scale2 * barrier_scale2;
    const double expected_b3 = std::sqrt(
        277.0
        / (barrier_q4 * barrier_q2
           + 6.0 * barrier_scale2 * barrier_q4
           + 45.0 * barrier_scale4 * barrier_q2
           + 225.0 * barrier_scale4 * barrier_scale2));
    const double expected_b4 = std::sqrt(
        12746.0
        / (barrier_q4 * barrier_q4
           + 10.0 * barrier_scale2 * barrier_q4 * barrier_q2
           + 135.0 * barrier_scale4 * barrier_q4
           + 1575.0 * barrier_scale4 * barrier_scale2 * barrier_q2
           + 11025.0 * barrier_scale4 * barrier_scale4));
    if (!close_to(ctpwa::blatt_weisskopf(barrier_probe, 3), expected_b3)
        || !close_to(ctpwa::blatt_weisskopf(barrier_probe, 4), expected_b4)) {
        std::cerr << "higher-L Blatt-Weisskopf formula is wrong\n";
        return 11;
    }
    if (!close_to(
            std::pow(old_scale, 3)
                * ctpwa::blatt_weisskopf(old_scale, 3),
            1.0)
        || !close_to(
            std::pow(old_scale, 4)
                * ctpwa::blatt_weisskopf(old_scale, 4),
            1.0)) {
        std::cerr << "higher-L barrier normalization is inconsistent\n";
        return 12;
    }

    const double rho_mass = 0.77526;
    const double rho_width = 0.1474;
    const double charged_pion_mass = 0.13957039;
    const double gamma_at_pole = ctpwa::running_width(
        rho_mass * rho_mass,
        rho_mass,
        rho_width,
        1,
        charged_pion_mass,
        charged_pion_mass,
        0.59);

    if (!close_to(gamma_at_pole, rho_width)) {
        std::cerr << "running width is not normalized at the rho pole\n";
        return 4;
    }

    // The public running-width path now accepts nominal daughter masses and
    // delegates to the single two_body_width_shape implementation. Verify
    // exact equivalence to the former explicit formula away from the pole.
    const double running_probe_s = 0.82 * 0.82;
    const double q_probe = ctpwa::two_body_Q(
        running_probe_s,
        charged_pion_mass * charged_pion_mass,
        charged_pion_mass * charged_pion_mass);
    const double q_pole = ctpwa::two_body_Q(
        rho_mass * rho_mass,
        charged_pion_mass * charged_pion_mass,
        charged_pion_mass * charged_pion_mass);
    const double b_probe = ctpwa::blatt_weisskopf(q_probe, 1, 0.59);
    const double b_pole = ctpwa::blatt_weisskopf(q_pole, 1, 0.59);
    const double explicit_running_width = rho_width * rho_mass
        / std::sqrt(running_probe_s)
        * std::pow(q_probe / q_pole, 3)
        * std::pow(b_probe / b_pole, 2);
    if (!close_to(
            ctpwa::running_width(
                running_probe_s,
                rho_mass,
                rho_width,
                1,
                charged_pion_mass,
                charged_pion_mass,
                0.59),
            explicit_running_width)) {
        std::cerr << "unified running-width core changed the analytic formula\n";
        return 5;
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
        return 6;
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
        return 7;
    }
    if (!close_to(flatte_zero.real, fixed_bw.real)
        || !close_to(flatte_zero.imag, fixed_bw.imag)) {
        std::cerr << "R_omegaomega=0 does not recover the fixed-width BW\n";
        return 8;
    }

    const DeviceComplex flatte_at_mass = ctpwa::Flatte_subtracted_effective(
        f0_mass * f0_mass, f0_mass, f0_width, 3.0, omega_mass);
    const DeviceComplex bw_at_mass = ctpwa::BW_fixed_width(
        f0_mass * f0_mass, f0_mass, f0_width);
    if (!close_to(flatte_at_mass.real, bw_at_mass.real)
        || !close_to(flatte_at_mass.imag, bw_at_mass.imag)) {
        std::cerr << "subtraction does not preserve the fixed mass point\n";
        return 9;
    }

    // The event current and omega-width integration use the same coherent
    // rho-isobar function. Reconstruct it from the nominal pion-mass policy as
    // an independent regression reference.
    const double s_omega = GVV_OMEGA_MASS * GVV_OMEGA_MASS;
    const double s0 = GVV_PI0_MASS * GVV_PI0_MASS;
    const double s1 = GVV_PIP_MASS * GVV_PIP_MASS;
    const double s2 = GVV_PIM_MASS * GVV_PIM_MASS;
    const double s12 = 0.220;
    const double s10 = 0.225;
    const double s20 = s_omega + s0 + s1 + s2 - s12 - s10;
    const RhoBWRParameters rho;
    const auto nominal_isobar = [&](double s_pair,
                                    RhoChargeChannel channel) {
        const RhoIsobarMasses masses = rho_isobar_nominal_masses(channel);
        const double q_parent = ctpwa::two_body_Q(
            s_omega, s_pair, masses.bachelor * masses.bachelor);
        const double q_pair = ctpwa::two_body_Q(
            s_pair,
            masses.first_daughter * masses.first_daughter,
            masses.second_daughter * masses.second_daughter);
        return ctpwa::blatt_weisskopf(
                   q_parent, 1, rho.omega_vertex_radius_fm)
               * ctpwa::BWR(
                   s_pair,
                   rho.mass,
                   rho.width,
                   1,
                   masses.first_daughter,
                   masses.second_daughter,
                   rho.rho_vertex_radius_fm)
               * ctpwa::blatt_weisskopf(
                   q_pair, 1, rho.rho_vertex_radius_fm);
    };
    const DeviceComplex nominal_coherent_rho =
        nominal_isobar(
            s12, RhoChargeChannel::Rho0ToPiPlusPiMinus)
        + nominal_isobar(
            s10, RhoChargeChannel::RhoPlusToPiPlusPi0)
        + nominal_isobar(
            s20, RhoChargeChannel::RhoMinusToPiMinusPi0);
    const DeviceComplex shared_coherent_rho = coherent_omega_rho_factor(
        s_omega, s12, s10, s20, rho);
    if (!close_to(shared_coherent_rho.real, nominal_coherent_rho.real)
        || !close_to(shared_coherent_rho.imag, nominal_coherent_rho.imag)) {
        std::cerr << "rho-isobar factor violates the nominal pion-mass policy\n";
        return 10;
    }

    std::cout << "Dynamics tests passed\n";
    return 0;
}

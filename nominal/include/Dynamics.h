#ifndef DYNAMICS_H
#define DYNAMICS_H

#include "DeviceComplex.h"
#include <cmath>

namespace ctpwa {

// hbar*c in GeV*fm.  Keep the value used by the legacy CTPWA implementation
// so existing amplitudes retain their normalization.
constexpr double HBARC_GEV_FM = 0.197321;
constexpr double DEFAULT_BARRIER_RADIUS_FM = 0.59;

// Q^2_abc = (s_a + s_b - s_c)^2 / (4 s_a) - s_b.
// This is the squared daughter momentum in the rest frame of system a.
__host__ __device__ inline double two_body_Q2(
    double s_a,
    double s_b,
    double s_c)
{
    if (s_a <= 0.0) {
        return 0.0;
    }

    const double numerator = s_a + s_b - s_c;
    return numerator * numerator / (4.0 * s_a) - s_b;
}

__host__ __device__ inline double two_body_Q(
    double s_a,
    double s_b,
    double s_c)
{
    const double q2 = two_body_Q2(s_a, s_b, s_c);

    // Physical events can produce a very small negative q2 through rounding.
    // The present amplitude model is only evaluated in the physical region.
    return sqrt(q2 > 0.0 ? q2 : 0.0);
}

// Blatt-Weisskopf factors in the normalization used by the original project.
// R is expressed in fm and Q in GeV.
__host__ __device__ inline double blatt_weisskopf(
    double Q,
    int L,
    double radius_fm = DEFAULT_BARRIER_RADIUS_FM)
{
    if (radius_fm <= 0.0) {
        return 0.0;
    }

    const double scale = HBARC_GEV_FM / radius_fm;
    const double q2 = Q * Q;
    const double scale2 = scale * scale;

    if (L == 0) {
        return 1.0;
    }
    if (L == 1) {
        return sqrt(2.0 / (q2 + scale2));
    }
    if (L == 2) {
        return sqrt(13.0 /
                    (q2 * q2 + 3.0 * scale2 * q2
                     + 9.0 * scale2 * scale2));
    }
    return 0.0;
}

// Relativistic running width for R -> b c:
// Gamma(s) = Gamma0 * mR/sqrt(s) * (Q/Q0)^(2L+1)
//            * [B_L(Q;R)/B_L(Q0;R)]^2.
__host__ __device__ inline double running_width(
    double s,
    double resonance_mass,
    double pole_width,
    int L,
    double s_b,
    double s_c,
    double radius_fm = DEFAULT_BARRIER_RADIUS_FM)
{
    if (s <= 0.0 || resonance_mass <= 0.0 || pole_width < 0.0 || L < 0) {
        return 0.0;
    }

    const double q = two_body_Q(s, s_b, s_c);
    const double q0 = two_body_Q(
        resonance_mass * resonance_mass, s_b, s_c);

    if (q0 <= 0.0) {
        return 0.0;
    }

    const double barrier = blatt_weisskopf(q, L, radius_fm);
    const double barrier0 = blatt_weisskopf(q0, L, radius_fm);
    if (barrier0 == 0.0) {
        return 0.0;
    }

    const double q_ratio = q / q0;
    const double barrier_ratio = barrier / barrier0;

    return pole_width * resonance_mass / sqrt(s)
           * pow(q_ratio, 2 * L + 1)
           * barrier_ratio * barrier_ratio;
}

// Generic relativistic Breit-Wigner with a running width.  The sign convention
// matches the original CTPWA convention:
//   f(s) = 1 / (m0^2 - s - i m0 Gamma(s)).
__host__ __device__ inline DeviceComplex BWR(
    double s,
    double resonance_mass,
    double pole_width,
    int L,
    double s_b,
    double s_c,
    double radius_fm = DEFAULT_BARRIER_RADIUS_FM)
{
    const double gamma_s = running_width(
        s, resonance_mass, pole_width, L, s_b, s_c, radius_fm);

    return 1.0 / DeviceComplex(
        resonance_mass * resonance_mass - s,
        -resonance_mass * gamma_s);
}

// Constant-width relativistic Breit-Wigner.  This is deliberately kept as a
// separate model for sub-threshold f0(1500), whose omega-omega breakup
// momentum is imaginary at the PDG Breit-Wigner mass.
__host__ __device__ inline DeviceComplex BW_fixed_width(
    double s,
    double resonance_mass,
    double pole_width)
{
    return 1.0 / DeviceComplex(
        resonance_mass * resonance_mass - s,
        -resonance_mass * pole_width);
}

// Analytic two-body S-wave phase-space factor for two equal-mass daughters,
// rho(s)=sqrt(1-4m^2/s).  Below threshold the physical-sheet continuation is
// purely imaginary.  Keeping this as a DeviceComplex is essential for a
// sub-threshold Flatte self-energy: discarding the imaginary rho below
// threshold would incorrectly remove the dispersive mass shift.
__host__ __device__ inline DeviceComplex two_body_rho_equal_mass(
    double s,
    double daughter_mass)
{
    if (s <= 0.0 || daughter_mass < 0.0) {
        return DeviceComplex(0.0, 0.0);
    }

    const double threshold = 4.0 * daughter_mass * daughter_mass;
    if (s >= threshold) {
        return DeviceComplex(sqrt(1.0 - threshold / s), 0.0);
    }
    return DeviceComplex(0.0, sqrt(threshold / s - 1.0));
}

// Subtracted effective Flatte line shape used for the sub-threshold f0(1500):
//
// D(s) = m0^2 - s - i [m0 Gamma_rest
//          + G_omegaomega (rho_omegaomega(s)-rho_omegaomega(m0^2))],
// G_omegaomega = R_omegaomega m0 Gamma_rest, R_omegaomega >= 0.
//
// The subtraction preserves the interpretation of the fixed m0 and
// Gamma_rest: at s=m0^2 the omega-omega correction vanishes.  R=0 therefore
// reproduces BW_fixed_width exactly.  The nominal omega mass is used here;
// finite-width spectral convolution is deliberately outside this first model.
__host__ __device__ inline DeviceComplex Flatte_subtracted_effective(
    double s,
    double resonance_mass,
    double rest_width,
    double omegaomega_ratio,
    double omega_mass)
{
    if (resonance_mass <= 0.0 || rest_width < 0.0
        || omegaomega_ratio < 0.0 || omega_mass < 0.0) {
        return DeviceComplex(0.0, 0.0);
    }

    const DeviceComplex rho_s =
        two_body_rho_equal_mass(s, omega_mass);
    const DeviceComplex rho_at_mass = two_body_rho_equal_mass(
        resonance_mass * resonance_mass, omega_mass);
    const double coupling =
        omegaomega_ratio * resonance_mass * rest_width;
    const DeviceComplex omegaomega_correction =
        DeviceComplex(0.0, -coupling) * (rho_s - rho_at_mass);
    const DeviceComplex denominator(
        resonance_mass * resonance_mass - s,
        -resonance_mass * rest_width);
    return 1.0 / (denominator + omegaomega_correction);
}

// Dimensionless two-body line-shape factor Phi_L(s), normalized to one at the
// pole.  Nominal daughter masses should be used for a physical propagator;
// event-by-event reconstructed masses remain confined to the orbital tensors.
__host__ __device__ inline double two_body_width_shape(
    double s,
    double resonance_mass,
    int L,
    double daughter_mass1,
    double daughter_mass2,
    double radius_fm = DEFAULT_BARRIER_RADIUS_FM)
{
    if (s <= 0.0 || resonance_mass <= 0.0 || L < 0) {
        return 0.0;
    }

    const double m1_sq = daughter_mass1 * daughter_mass1;
    const double m2_sq = daughter_mass2 * daughter_mass2;
    const double q = two_body_Q(s, m1_sq, m2_sq);
    const double q0 = two_body_Q(
        resonance_mass * resonance_mass, m1_sq, m2_sq);
    if (q <= 0.0 || q0 <= 0.0) {
        return 0.0;
    }

    const double barrier = blatt_weisskopf(q, L, radius_fm);
    const double barrier0 = blatt_weisskopf(q0, L, radius_fm);
    if (barrier0 <= 0.0) {
        return 0.0;
    }

    const double q_ratio = q / q0;
    const double barrier_ratio = barrier / barrier0;
    return resonance_mass / sqrt(s)
           * pow(q_ratio, 2 * L + 1)
           * barrier_ratio * barrier_ratio;
}

// Total scalar width made from S- and D-wave omega-omega partial widths.
// sd_ratio is Gamma_D/Gamma_S at s=m0^2.  Since Phi_0=Phi_2=1 at the pole,
// division by (1+r) guarantees Gamma(m0^2)=Gamma0 for every r>=0.
__host__ __device__ inline double scalar_sd_running_width(
    double s,
    double resonance_mass,
    double pole_width,
    double sd_ratio,
    double omega_mass,
    double radius_fm = DEFAULT_BARRIER_RADIUS_FM)
{
    if (sd_ratio < 0.0) {
        sd_ratio = 0.0;
    }
    const double phi_s = two_body_width_shape(
        s, resonance_mass, 0, omega_mass, omega_mass, radius_fm);
    const double phi_d = two_body_width_shape(
        s, resonance_mass, 2, omega_mass, omega_mass, radius_fm);
    return pole_width * (phi_s + sd_ratio * phi_d) / (1.0 + sd_ratio);
}

__host__ __device__ inline DeviceComplex BWR_scalar_sd(
    double s,
    double resonance_mass,
    double pole_width,
    double sd_ratio,
    double omega_mass,
    double radius_fm = DEFAULT_BARRIER_RADIUS_FM)
{
    const double gamma_s = scalar_sd_running_width(
        s,
        resonance_mass,
        pole_width,
        sd_ratio,
        omega_mass,
        radius_fm);
    return 1.0 / DeviceComplex(
        resonance_mass * resonance_mass - s,
        -resonance_mass * gamma_s);
}

__host__ __device__ inline DeviceComplex BWR_two_body_nominal(
    double s,
    double resonance_mass,
    double pole_width,
    int L,
    double daughter_mass1,
    double daughter_mass2,
    double radius_fm = DEFAULT_BARRIER_RADIUS_FM)
{
    const double gamma_s = pole_width * two_body_width_shape(
        s,
        resonance_mass,
        L,
        daughter_mass1,
        daughter_mass2,
        radius_fm);
    return 1.0 / DeviceComplex(
        resonance_mass * resonance_mass - s,
        -resonance_mass * gamma_s);
}

} // namespace ctpwa

#endif // DYNAMICS_H

// Process-independent line-shape library. Functions accept explicit masses,
// widths and orbital momenta and therefore do not know resonance identities.
#ifndef CTPWA_FRAMEWORK_DYNAMICS_PROPAGATORS_CUH
#define CTPWA_FRAMEWORK_DYNAMICS_PROPAGATORS_CUH

#include "framework/math/DeviceComplex.cuh"
#include "framework/dynamics/Kinematics.cuh"
#include "framework/tensors/BarrierFactor.cuh"
#include <cmath>

namespace ctpwa {

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

// Constant-width relativistic Breit-Wigner. This remains a separate option
// for a sub-threshold state whose nominal two-body breakup momentum is not
// real at the Breit-Wigner mass.
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

// Subtracted effective Flatte line shape for one equal-mass threshold channel:
//
// D(s) = m0^2 - s - i [m0 Gamma_rest
//          + G_channel (rho_channel(s)-rho_channel(m0^2))],
// G_channel = R_channel m0 Gamma_rest, R_channel >= 0.
//
// The subtraction preserves the interpretation of the fixed m0 and
// Gamma_rest: at s=m0^2 the channel correction vanishes. R=0 therefore
// reproduces BW_fixed_width exactly. The process chooses the daughter mass;
// finite-width spectral convolution is outside this effective propagator.
__host__ __device__ inline DeviceComplex Flatte_subtracted_effective(
    double s,
    double resonance_mass,
    double rest_width,
    double channel_ratio,
    double daughter_mass)
{
    if (resonance_mass <= 0.0 || rest_width < 0.0
        || channel_ratio < 0.0 || daughter_mass < 0.0) {
        return DeviceComplex(0.0, 0.0);
    }

    const DeviceComplex rho_s =
        two_body_rho_equal_mass(s, daughter_mass);
    const DeviceComplex rho_at_mass = two_body_rho_equal_mass(
        resonance_mass * resonance_mass, daughter_mass);
    const double coupling =
        channel_ratio * resonance_mass * rest_width;
    const DeviceComplex channel_correction =
        DeviceComplex(0.0, -coupling) * (rho_s - rho_at_mass);
    const DeviceComplex denominator(
        resonance_mass * resonance_mass - s,
        -resonance_mass * rest_width);
    return 1.0 / (denominator + channel_correction);
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

// Total scalar width made from S- and D-wave equal-mass partial widths.
// sd_ratio is Gamma_D/Gamma_S at s=m0^2.  Since Phi_0=Phi_2=1 at the pole,
// division by (1+r) guarantees Gamma(m0^2)=Gamma0 for every r>=0.
__host__ __device__ inline double scalar_sd_running_width(
    double s,
    double resonance_mass,
    double pole_width,
    double sd_ratio,
    double daughter_mass,
    double radius_fm = DEFAULT_BARRIER_RADIUS_FM)
{
    const double phi_s = two_body_width_shape(
        s, resonance_mass, 0, daughter_mass, daughter_mass, radius_fm);
    const double phi_d = two_body_width_shape(
        s, resonance_mass, 2, daughter_mass, daughter_mass, radius_fm);
    return pole_width * (phi_s + sd_ratio * phi_d) / (1.0 + sd_ratio);
}

__host__ __device__ inline DeviceComplex BWR_scalar_sd(
    double s,
    double resonance_mass,
    double pole_width,
    double sd_ratio,
    double daughter_mass,
    double radius_fm = DEFAULT_BARRIER_RADIUS_FM)
{
    const double gamma_s = scalar_sd_running_width(
        s,
        resonance_mass,
        pole_width,
        sd_ratio,
        daughter_mass,
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

#endif // CTPWA_FRAMEWORK_DYNAMICS_PROPAGATORS_CUH

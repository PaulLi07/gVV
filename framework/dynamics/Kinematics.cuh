// Process-independent two-body breakup momentum and shared unit constants.
#ifndef CTPWA_FRAMEWORK_DYNAMICS_KINEMATICS_CUH
#define CTPWA_FRAMEWORK_DYNAMICS_KINEMATICS_CUH

#include <cmath>

namespace ctpwa {

// hbar*c in GeV*fm.  The value is kept identical to the original CTPWA
// implementation so the architecture refactor does not change amplitudes.
constexpr double HBARC_GEV_FM = 0.197321;
constexpr double DEFAULT_BARRIER_RADIUS_FM = 0.59;

// Squared daughter momentum in the rest frame of system a.
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
    return sqrt(q2 > 0.0 ? q2 : 0.0);
}

} // namespace ctpwa

#endif // CTPWA_FRAMEWORK_DYNAMICS_KINEMATICS_CUH

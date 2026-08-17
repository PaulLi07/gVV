// Reusable Blatt-Weisskopf barrier factors. Momentum is in GeV and the radius
// is in fm; unsupported orbital angular momentum returns a controlled zero.
#ifndef CTPWA_FRAMEWORK_TENSORS_BARRIER_FACTOR_CUH
#define CTPWA_FRAMEWORK_TENSORS_BARRIER_FACTOR_CUH

#include <cmath>

namespace ctpwa {

// Barrier-factor units and the nominal radius belong to this module. Keeping
// them here prevents tensors/ from depending back on dynamics/.
constexpr double HBARC_GEV_FM = 0.197321;
constexpr double DEFAULT_BARRIER_RADIUS_FM = 0.59;

// Blatt-Weisskopf factors in the normalization used by the original project.
// The radius is in fm and the breakup momentum is in GeV.
__host__ __device__ inline double blatt_weisskopf(
    double momentum,
    int orbital_l,
    double radius_fm = DEFAULT_BARRIER_RADIUS_FM)
{
    if (radius_fm <= 0.0) {
        return 0.0;
    }
    const double scale = HBARC_GEV_FM / radius_fm;
    const double q2 = momentum * momentum;
    const double scale2 = scale * scale;
    if (orbital_l == 0) {
        return 1.0;
    }
    if (orbital_l == 1) {
        return sqrt(2.0 / (q2 + scale2));
    }
    if (orbital_l == 2) {
        return sqrt(
            13.0
            / (q2 * q2 + 3.0 * scale2 * q2
               + 9.0 * scale2 * scale2));
    }
    return 0.0;
}

} // namespace ctpwa

#endif // CTPWA_FRAMEWORK_TENSORS_BARRIER_FACTOR_CUH

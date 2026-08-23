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
    if (orbital_l == 3) {
        const double q4 = q2 * q2;
        const double scale4 = scale2 * scale2;
        return sqrt(
            277.0
            / (q4 * q2 + 6.0 * scale2 * q4
               + 45.0 * scale4 * q2 + 225.0 * scale4 * scale2));
    }
    if (orbital_l == 4) {
        const double q4 = q2 * q2;
        const double q6 = q4 * q2;
        const double scale4 = scale2 * scale2;
        const double scale6 = scale4 * scale2;
        const double scale8 = scale4 * scale4;
        return sqrt(
            12746.0
            / (q4 * q4 + 10.0 * scale2 * q6
               + 135.0 * scale4 * q4
               + 1575.0 * scale6 * q2 + 11025.0 * scale8));
    }
    return 0.0;
}

} // namespace ctpwa

#endif // CTPWA_FRAMEWORK_TENSORS_BARRIER_FACTOR_CUH

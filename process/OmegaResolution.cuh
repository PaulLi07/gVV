// Effective complex omega propagator convolved in mass with a zero-mean Gaussian.
#ifndef GVV_PROCESS_OMEGA_RESOLUTION_CUH
#define GVV_PROCESS_OMEGA_RESOLUTION_CUH

#include "process/OmegaWidthTable.h"
#include <cmath>

inline constexpr double GVV_OMEGA_RESOLUTION_MAX_SIGMA = 0.05; // GeV
inline constexpr char GVV_OMEGA_RESOLUTION_IMPLEMENTATION[] =
    "gvv-amplitude-omega-gauss-v1";

// Integral over positive true mass, retaining the original running-width table
// and its endpoint policy. The Gaussian is cut at +/-8 sigma (tail < 1.3e-15).
// GL16 panels split at every table knot, so interpolation slope changes cannot
// introduce quadrature noise into the fitted sigma. Sigma=0 is exactly legacy.
__host__ __device__ inline DeviceComplex gvv_smeared_omega_propagator(
    double s, const ctpwa::TabulatedFunctionView& width_table, double sigma)
{
    if (sigma == 0.0) return gvv_omega_propagator(s, width_table);
    constexpr double nodes[8] = {
        0.0950125098376374402, 0.281603550779258913,
        0.458016777657227386, 0.617876244402643748,
        0.755404408355003034, 0.865631202387831744,
        0.944575023073232576, 0.989400934991649933};
    constexpr double weights[8] = {
        0.189450610455068496, 0.182603415044923589,
        0.169156519395002538, 0.149595988816576732,
        0.124628971255533872, 0.0951585116824927848,
        0.0622535239386478929, 0.0271524594117540949};
    constexpr double inverse_sqrt_two_pi = 0.398942280401432678;
    const double mass = sqrt(s);
    // A sub-ulp Gaussian is indistinguishable from zero resolution. Avoid
    // panels whose floating-point endpoints cannot advance in this limit.
    if (mass + 2.0 * sigma == mass) return gvv_omega_propagator(s, width_table);
    double low = fmax(0.0, mass - 8.0 * sigma);
    const double end = mass + 8.0 * sigma;
    const double maximum_panel = fmin(2.0 * sigma, GVV_OMEGA_WIDTH);
    int knot = 0;
    if (low * low >= width_table.x_min) {
        const double position = (low * low - width_table.x_min) / width_table.x_step;
        knot = position >= width_table.size - 1
            ? width_table.size : static_cast<int>(position) + 1;
    }
    DeviceComplex result(0.0, 0.0);
    while (low < end) {
        double high = fmin(end, low + maximum_panel);
        if (knot < width_table.size) {
            const double knot_mass = sqrt(width_table.x_min + knot * width_table.x_step);
            if (knot_mass <= low) { ++knot; continue; }
            high = fmin(high, knot_mass);
        }
        const double center = 0.5 * (low + high);
        const double half_width = 0.5 * (high - low);
        DeviceComplex panel(0.0, 0.0);
        for (int point = 0; point < 8; ++point) {
            for (int sign = -1; sign <= 1; sign += 2) {
                const double true_mass = center + sign * half_width * nodes[point];
                const double pull = (mass - true_mass) / sigma;
                panel = panel + (weights[point] * exp(-0.5 * pull * pull))
                         * gvv_omega_propagator(true_mass * true_mass, width_table);
            }
        }
        result = result + (half_width / sigma * inverse_sqrt_two_pi) * panel;
        low = high;
    }
    return result;
}

#endif // GVV_PROCESS_OMEGA_RESOLUTION_CUH

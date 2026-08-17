#include "../include/GVVModel.h"
#include "../include/OmegaPropagator.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace {

bool close_relative(double value, double reference, double tolerance)
{
    return std::abs(value - reference)
           <= tolerance * std::max(std::abs(reference), 1.0e-12);
}

} // namespace

// Compile the complete device propagator dispatch without requiring a GPU on
// the login node.
__global__ void compile_gvv_propagators(double* output)
{
    const GVVResonanceParameters scalar(
        GVV_PROP_SCALAR_SWAVE_BWR, 1.723, 0.149);
    const GVVResonanceParameters pseudoscalar(
        GVV_PROP_PWAVE_BWR, 1.751, 0.240);
    output[0] = gvv_x_propagator(scalar.mass * scalar.mass, scalar).rho2();
    output[1] = gvv_x_propagator(
        pseudoscalar.mass * pseudoscalar.mass, pseudoscalar).rho2();
}

int main()
{
    const GVVResonanceParameters f1500(
        GVV_PROP_SUBTRACTED_FLATTE,
        1.522,
        0.108,
        0.0,
        0,
        1.0,
        1);
    if (f1500.propagator_model != GVV_PROP_SUBTRACTED_FLATTE
        || f1500.fit_flatte_ratio != 1
        || !close_relative(f1500.flatte_ratio, 1.0, 1.0e-12)) {
        std::cerr << "subtracted Flatte device descriptor is wrong\n";
        return 1;
    }
    if (!(gvv_x_propagator(
              f1500.mass * f1500.mass, f1500).rho2() > 0.0)) {
        std::cerr << "subtracted Flatte mass point is not finite\n";
        return 2;
    }

    const GVVResonanceParameters f1710(
        GVV_PROP_SCALAR_SWAVE_BWR, 1.723, 0.149);
    const double scalar_pole_width = f1710.pole_width
        * ctpwa::two_body_width_shape(
            f1710.mass * f1710.mass,
            f1710.mass,
            0,
            GVV_OMEGA_MASS,
            GVV_OMEGA_MASS);
    if (!close_relative(scalar_pole_width, f1710.pole_width, 1.0e-12)) {
        std::cerr << "S-wave width is not pole-normalized\n";
        return 3;
    }

    const GVVResonanceParameters eta1760(
        GVV_PROP_PWAVE_BWR, 1.751, 0.240);
    const double p_wave_width = eta1760.pole_width
        * ctpwa::two_body_width_shape(
            eta1760.mass * eta1760.mass,
            eta1760.mass,
            1,
            GVV_OMEGA_MASS,
            GVV_OMEGA_MASS);
    if (!close_relative(p_wave_width, eta1760.pole_width, 1.0e-12)) {
        std::cerr << "P-wave width is not pole-normalized\n";
        return 4;
    }

    const GVVResonanceParameters nonresonant;
    const DeviceComplex nr = gvv_x_propagator(4.0, nonresonant);
    if (nr.real != 1.0 || nr.imag != 0.0) {
        std::cerr << "nonresonant propagator is not unity\n";
        return 5;
    }

    OmegaWidthTable omega_table;
    omega_table.Build(96, 24, 0.40, 1.20);
    const double omega_width_at_pole = omega_table.Width(
        GVV_OMEGA_MASS * GVV_OMEGA_MASS);
    if (!close_relative(omega_width_at_pole, GVV_OMEGA_WIDTH, 0.02)) {
        std::cerr << "omega three-body width is not normalized at the pole: "
                  << omega_width_at_pole << '\n';
        return 6;
    }
    if (omega_table.Width(0.40 * 0.40) != 0.0) {
        std::cerr << "omega width is nonzero below three-pion threshold\n";
        return 7;
    }

    std::cout << "GVV model and propagator tests passed\n";
    return 0;
}

// Reusable propagator-dispatch regression plus the process-owned omega width
// table normalization check.
#include "framework/dynamics/PropagatorRegistry.cuh"
#include "process/OmegaWidthTable.h"

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
    const ctpwa::PropagatorParameters scalar(
        ctpwa::PROP_TWO_BODY_RUNNING_BW, 1.723, 0.149, 0);
    const ctpwa::PropagatorParameters pseudoscalar(
        ctpwa::PROP_TWO_BODY_RUNNING_BW, 1.751, 0.240, 1);
    output[0] = ctpwa::evaluate_propagator(
        scalar.mass * scalar.mass,
        scalar,
        GVV_OMEGA_MASS,
        GVV_OMEGA_MASS).rho2();
    output[1] = ctpwa::evaluate_propagator(
        pseudoscalar.mass * pseudoscalar.mass,
        pseudoscalar,
        GVV_OMEGA_MASS,
        GVV_OMEGA_MASS).rho2();
}

int main()
{
    const ctpwa::PropagatorParameters f1500(
        ctpwa::PROP_SUBTRACTED_FLATTE,
        1.522,
        0.108,
        0,
        0.0,
        1.0);
    if (f1500.propagator_model != ctpwa::PROP_SUBTRACTED_FLATTE
        || !close_relative(f1500.flatte_ratio, 1.0, 1.0e-12)) {
        std::cerr << "subtracted Flatte device descriptor is wrong\n";
        return 1;
    }
    if (!(ctpwa::evaluate_propagator(
              f1500.mass * f1500.mass,
              f1500,
              GVV_OMEGA_MASS,
              GVV_OMEGA_MASS).rho2() > 0.0)) {
        std::cerr << "subtracted Flatte mass point is not finite\n";
        return 2;
    }

    const ctpwa::PropagatorParameters f1710(
        ctpwa::PROP_TWO_BODY_RUNNING_BW, 1.723, 0.149, 0);
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

    const ctpwa::PropagatorParameters eta1760(
        ctpwa::PROP_TWO_BODY_RUNNING_BW, 1.751, 0.240, 1);
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

    const ctpwa::PropagatorParameters d_wave(
        ctpwa::PROP_TWO_BODY_RUNNING_BW, 2.1, 0.2, 2);
    const double d_wave_width = d_wave.pole_width
        * ctpwa::two_body_width_shape(
            d_wave.mass * d_wave.mass,
            d_wave.mass,
            d_wave.orbital_l,
            GVV_OMEGA_MASS,
            GVV_OMEGA_MASS);
    if (!close_relative(d_wave_width, d_wave.pole_width, 1.0e-12)) {
        std::cerr << "D-wave width is not pole-normalized\n";
        return 5;
    }

    const ctpwa::PropagatorParameters nonresonant;
    const DeviceComplex nr = ctpwa::evaluate_propagator(
        4.0, nonresonant, GVV_OMEGA_MASS, GVV_OMEGA_MASS);
    if (nr.real != 1.0 || nr.imag != 0.0) {
        std::cerr << "nonresonant propagator is not unity\n";
        return 6;
    }

    OmegaWidthTable omega_table;
    omega_table.Build(96, 24, 0.40, 1.20);
    const double omega_width_at_pole = omega_table.Width(
        GVV_OMEGA_MASS * GVV_OMEGA_MASS);
    if (!close_relative(omega_width_at_pole, GVV_OMEGA_WIDTH, 0.02)) {
        std::cerr << "omega three-body width is not normalized at the pole: "
                  << omega_width_at_pole << '\n';
        return 7;
    }
    if (omega_table.Width(0.40 * 0.40) != 0.0) {
        std::cerr << "omega width is nonzero below three-pion threshold\n";
        return 8;
    }

    std::cout << "GVV model and propagator tests passed\n";
    return 0;
}

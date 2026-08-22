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
        ctpwa::PROP_TWO_BODY_RUNNING_BW,
        1.723,
        0.149,
        0,
        0.0,
        0.0,
        GVV_OMEGA_MASS,
        GVV_OMEGA_MASS);
    const ctpwa::PropagatorParameters pseudoscalar(
        ctpwa::PROP_TWO_BODY_RUNNING_BW,
        1.751,
        0.240,
        1,
        0.0,
        0.0,
        GVV_OMEGA_MASS,
        GVV_OMEGA_MASS);
    output[0] = ctpwa::evaluate_propagator(
        scalar.mass * scalar.mass, scalar).rho2();
    output[1] = ctpwa::evaluate_propagator(
        pseudoscalar.mass * pseudoscalar.mass, pseudoscalar).rho2();
}

int main()
{
    const ctpwa::PropagatorParameters f1500(
        ctpwa::PROP_SUBTRACTED_FLATTE,
        1.522,
        0.108,
        0,
        0.0,
        1.0,
        GVV_OMEGA_MASS,
        GVV_OMEGA_MASS);
    if (f1500.propagator_model != ctpwa::PROP_SUBTRACTED_FLATTE
        || !close_relative(f1500.flatte_ratio, 1.0, 1.0e-12)) {
        std::cerr << "subtracted Flatte device descriptor is wrong\n";
        return 1;
    }
    if (!(ctpwa::evaluate_propagator(
              f1500.mass * f1500.mass, f1500).rho2() > 0.0)) {
        std::cerr << "subtracted Flatte mass point is not finite\n";
        return 2;
    }

    const ctpwa::PropagatorParameters f1710(
        ctpwa::PROP_TWO_BODY_RUNNING_BW,
        1.723,
        0.149,
        0,
        0.0,
        0.0,
        GVV_OMEGA_MASS,
        GVV_OMEGA_MASS);
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
        ctpwa::PROP_TWO_BODY_RUNNING_BW,
        1.751,
        0.240,
        1,
        0.0,
        0.0,
        GVV_OMEGA_MASS,
        GVV_OMEGA_MASS);
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
        ctpwa::PROP_TWO_BODY_RUNNING_BW,
        2.1,
        0.2,
        2,
        0.0,
        0.0,
        GVV_OMEGA_MASS,
        GVV_OMEGA_MASS);
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
        4.0, nonresonant);
    if (nr.real != 1.0 || nr.imag != 0.0) {
        std::cerr << "nonresonant propagator is not unity\n";
        return 6;
    }

    OmegaWidthTable omega_table;
    OmegaWidthTableConfig omega_config;
    omega_config.table_size = 96;
    omega_config.dalitz_bins = 24;
    omega_table.Build(omega_config);
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
    const double omega_probe_s = 0.81 * 0.81;
    const DeviceComplex omega_propagator = gvv_omega_propagator(
        omega_probe_s, omega_table.HostView());
    const DeviceComplex shared_denominator = ctpwa::BW_from_width(
        omega_probe_s,
        GVV_OMEGA_MASS,
        omega_table.Width(omega_probe_s));
    if (!close_relative(
            omega_propagator.real, shared_denominator.real, 1.0e-12)
        || !close_relative(
            omega_propagator.imag, shared_denominator.imag, 1.0e-12)) {
        std::cerr << "omega line shape bypasses the shared BW denominator\n";
        return 9;
    }

    OmegaWidthTableConfig refined_config = omega_config;
    refined_config.table_size = 160;
    refined_config.dalitz_bins = 36;
    OmegaWidthTable refined_table;
    refined_table.Build(refined_config);
    for (double mass : {0.74, 0.81, 0.90}) {
        const double s = mass * mass;
        if (!close_relative(
                omega_table.Width(s), refined_table.Width(s), 0.05)) {
            std::cerr << "omega width table is not stable at "
                      << mass << " GeV\n";
            return 10;
        }
    }
    if (omega_table.Config().table_size != 96
        || omega_table.Config().dalitz_bins != 24
        || omega_table.Config().extrapolation
            != OmegaWidthExtrapolation::Clamp) {
        std::cerr << "omega width table did not retain its named config\n";
        return 11;
    }

    std::cout << "GVV model and propagator tests passed\n";
    return 0;
}

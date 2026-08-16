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
    const GVVResonanceParameters scalar =
        gvv_default_resonance(GVV_RES_F0_1710);
    const GVVResonanceParameters pseudoscalar =
        gvv_default_resonance(GVV_RES_ETA_1760);
    output[0] = gvv_x_propagator(scalar.mass * scalar.mass, scalar).rho2();
    output[1] = gvv_x_propagator(
        pseudoscalar.mass * pseudoscalar.mass, pseudoscalar).rho2();
}

int main()
{
    const GVVResonanceParameters f1500 =
        gvv_default_resonance(GVV_RES_F0_1500);
    if (f1500.propagator_model != GVV_PROP_SUBTRACTED_FLATTE
        || f1500.fit_sd_ratio != 0
        || f1500.fit_flatte_ratio != 1
        || !close_relative(f1500.flatte_ratio, 1.0, 1.0e-12)
        || !close_relative(f1500.mass, 1.522, 1.0e-12)
        || !close_relative(f1500.pole_width, 0.108, 1.0e-12)) {
        std::cerr << "f0(1500) subtracted Flatte model is wrong\n";
        return 1;
    }
    const DeviceComplex f1500_at_mass =
        gvv_x_propagator(f1500.mass * f1500.mass, f1500);
    if (!(f1500_at_mass.rho2() > 0.0)) {
        std::cerr << "f0(1500) Flatte mass point is not finite\n";
        return 2;
    }

    const GVVResonanceParameters f1710 =
        gvv_default_resonance(GVV_RES_F0_1710);
    const double scalar_pole_width = f1710.pole_width
        * ctpwa::two_body_width_shape(
        f1710.mass * f1710.mass,
        f1710.mass,
        0,
        GVV_OMEGA_MASS,
        GVV_OMEGA_MASS);
    if (f1710.propagator_model != GVV_PROP_SCALAR_SWAVE_BWR
        || f1710.fit_sd_ratio != 0
        || !close_relative(scalar_pole_width, f1710.pole_width, 1.0e-12)) {
        std::cerr << "f0(1710) pure S-wave model is wrong\n";
        return 3;
    }

    const GVVResonanceParameters eta1760 =
        gvv_default_resonance(GVV_RES_ETA_1760);
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

    const GVVResonanceParameters nonresonant =
        gvv_default_resonance(GVV_RES_NR_0MP);
    const DeviceComplex nr = gvv_x_propagator(4.0, nonresonant);
    if (nr.real != 1.0 || nr.imag != 0.0) {
        std::cerr << "nonresonant 0-+ propagator is not unity\n";
        return 5;
    }

    int number_scalar_22_terms = 0;
    for (int term = 0; term < GVV_NTERMS; ++term) {
        if (gvv_default_term(term).wave_type == GVV_SCALAR_22) {
            ++number_scalar_22_terms;
        }
    }
    if (GVV_NTERMS != 7 || number_scalar_22_terms != 0
        || gvv_default_term(GVV_TERM_NR_0MP_11).wave_type
               != GVV_PSEUDOSCALAR_11) {
        std::cerr << "GVV term-to-wave mapping is wrong\n";
        return 6;
    }
    if (GVV_SCALE_AND_PHASE_REFERENCE_TERM != GVV_TERM_ETA_1760_11
        || GVV_SCALAR_PHASE_REFERENCE_TERM != GVV_TERM_F0_1710_00
        || gvv_coupling_parameterization(GVV_TERM_ETA_1760_11)
               != GVV_COUPLING_FIXED_SCALE_AND_PHASE
        || gvv_coupling_parameterization(GVV_TERM_F0_1710_00)
               != GVV_COUPLING_POSITIVE_REAL
        || gvv_coupling_parameterization(GVV_TERM_F0_1500_00)
               != GVV_COUPLING_COMPLEX
        || gvv_number_coupling_fit_parameters() != 11) {
        std::cerr << "GVV coupling reference policy is wrong\n";
        return 7;
    }

    OmegaWidthTable omega_table;
    omega_table.Build(96, 24, 0.40, 1.20);
    const double omega_width_at_pole = omega_table.Width(
        GVV_OMEGA_MASS * GVV_OMEGA_MASS);
    if (!close_relative(omega_width_at_pole, GVV_OMEGA_WIDTH, 0.02)) {
        std::cerr << "omega three-body width is not normalized at the pole: "
                  << omega_width_at_pole << '\n';
        return 8;
    }
    if (omega_table.Width(0.40 * 0.40) != 0.0) {
        std::cerr << "omega width is nonzero below three-pion threshold\n";
        return 9;
    }

    const double data_log_sum = 10.0;
    const double sb1_log_sum = 4.0;
    const double sb2_log_sum = 2.0;
    const double effective = data_log_sum
        + GVV_SB1_LIKELIHOOD_COEFFICIENT * sb1_log_sum
        + GVV_SB2_LIKELIHOOD_COEFFICIENT * sb2_log_sum;
    if (effective != 8.5) {
        std::cerr << "signed two-dimensional sideband convention is wrong\n";
        return 10;
    }

    int number_fit_parameters = gvv_number_coupling_fit_parameters();
    for (int resonance = 0; resonance < GVV_NRESONANCES; ++resonance) {
        const GVVResonanceParameters state = gvv_default_resonance(resonance);
        number_fit_parameters += state.fit_sd_ratio;
        number_fit_parameters += state.fit_flatte_ratio;
    }
    if (number_fit_parameters != 12) {
        std::cerr << "unexpected number of GVV fit parameters\n";
        return 11;
    }

    std::cout << "GVV model and propagator tests passed\n";
    return 0;
}

#ifndef GVV_MODEL_H
#define GVV_MODEL_H

#include "Dynamics.h"
#include "GVVAmplitude.h"

// Fixed physics content for psi(2S) -> gamma X, X -> omega omega.
// Masses and pole widths are the 2026 PDG central listing values in GeV.
// eta(1760), X(1835), and eta(2370) are PDG listing states omitted from the
// summary table.  X(2370) was renamed eta(2370); the X_2370 code alias is kept
// to match the analysis terminology.

enum GVVPropagatorModel {
    GVV_PROP_FIXED_BW = 0,
    GVV_PROP_SCALAR_SD_BWR = 1,
    GVV_PROP_PWAVE_BWR = 2,
    GVV_PROP_NONRESONANT = 3,
    GVV_PROP_SCALAR_SWAVE_BWR = 4,
    GVV_PROP_SUBTRACTED_FLATTE = 5
};

enum GVVResonanceIndex {
    GVV_RES_F0_1500 = 0,
    GVV_RES_F0_1710 = 1,
    GVV_RES_ETA_1760 = 2,
    GVV_RES_ETA_C_1S = 3,
    GVV_RES_X_1835 = 4,
    GVV_RES_X_2370 = 5,
    GVV_RES_NR_0MP = 6,
    GVV_NRESONANCES = 7
};

enum GVVTermIndex {
    GVV_TERM_F0_1500_00 = 0,
    GVV_TERM_F0_1710_00 = 1,
    GVV_TERM_ETA_1760_11 = 2,
    GVV_TERM_ETA_C_11 = 3,
    GVV_TERM_X_1835_11 = 4,
    GVV_TERM_X_2370_11 = 5,
    GVV_TERM_NR_0MP_11 = 6,
    GVV_NTERMS = 7
};

// Coupling identifiability policy.
//
// The normalized likelihood has one overall scale and phase freedom.  The
// eta(1760) coefficient removes both by remaining exactly 1+0i.  With the
// present polarization-summed tensors, the 0++ and 0-+ sectors have zero
// event-by-event interference, so the scalar sector has one additional
// unobservable common phase.  f0(1710) removes only that phase: its magnitude
// remains fitted and its coefficient is constrained to be positive real.
enum GVVCouplingParameterization {
    GVV_COUPLING_COMPLEX = 0,
    GVV_COUPLING_FIXED_SCALE_AND_PHASE = 1,
    GVV_COUPLING_POSITIVE_REAL = 2
};

constexpr int GVV_SCALE_AND_PHASE_REFERENCE_TERM =
    GVV_TERM_ETA_1760_11;
constexpr int GVV_SCALAR_PHASE_REFERENCE_TERM =
    GVV_TERM_F0_1710_00;

__host__ __device__ inline int gvv_coupling_parameterization(int term)
{
    if (term == GVV_SCALE_AND_PHASE_REFERENCE_TERM) {
        return GVV_COUPLING_FIXED_SCALE_AND_PHASE;
    }
    if (term == GVV_SCALAR_PHASE_REFERENCE_TERM) {
        return GVV_COUPLING_POSITIVE_REAL;
    }
    return GVV_COUPLING_COMPLEX;
}

__host__ __device__ inline int gvv_coupling_parameter_count(int term)
{
    const int parameterization = gvv_coupling_parameterization(term);
    if (parameterization == GVV_COUPLING_FIXED_SCALE_AND_PHASE) {
        return 0;
    }
    if (parameterization == GVV_COUPLING_POSITIVE_REAL) {
        return 1;
    }
    return 2;
}

__host__ __device__ inline int gvv_number_coupling_fit_parameters()
{
    int count = 0;
    for (int term = 0; term < GVV_NTERMS; ++term) {
        count += gvv_coupling_parameter_count(term);
    }
    return count;
}

constexpr double GVV_OMEGA_MASS = 0.78266;
constexpr double GVV_OMEGA_WIDTH = 0.00868;
// User-facing f0(1500) Flatte controls.  Set FIT to 0 and change RATIO for a
// fixed profile scan (including RATIO=0 for the fixed-width BW limit).
constexpr double GVV_F0_1500_FLATTE_RATIO = 1.0;
constexpr int GVV_F0_1500_FIT_FLATTE_RATIO = 1;
constexpr double GVV_FLATTE_LOG_RATIO_MIN = -6.0;
constexpr double GVV_FLATTE_LOG_RATIO_MAX = +3.0;
constexpr double GVV_SB1_LIKELIHOOD_COEFFICIENT = -0.5;
constexpr double GVV_SB2_LIKELIHOOD_COEFFICIENT = +0.25;

struct GVVResonanceParameters {
    int propagator_model;
    double mass;
    double pole_width;
    double sd_ratio;
    int fit_sd_ratio;
    double flatte_ratio;
    int fit_flatte_ratio;

    __host__ __device__ GVVResonanceParameters(
        int model = GVV_PROP_NONRESONANT,
        double m = 0.0,
        double width = 0.0,
        double ratio = 0.0,
        int fit_ratio = 0,
        double omegaomega_ratio = 0.0,
        int fit_omegaomega_ratio = 0)
        : propagator_model(model),
          mass(m),
          pole_width(width),
          sd_ratio(ratio),
          fit_sd_ratio(fit_ratio),
          flatte_ratio(omegaomega_ratio),
          fit_flatte_ratio(fit_omegaomega_ratio)
    {
    }
};

struct GVVTermSpec {
    int resonance_index;
    // wave_slot is a dense, model-local index into the compact F matrix.
    // registered_wave_type is retained for diagnostics and never indexes F.
    int wave_slot;
    int registered_wave_type;

    __host__ __device__ GVVTermSpec(
        int resonance = GVV_RES_NR_0MP,
        int slot = GVV_PSEUDOSCALAR_11,
        int registered_wave = -1)
        : resonance_index(resonance),
          wave_slot(slot),
          registered_wave_type(
              registered_wave < 0 ? slot : registered_wave)
    {
    }
};

__host__ __device__ inline GVVResonanceParameters
gvv_default_resonance(int index)
{
    if (index == GVV_RES_F0_1500) {
        // Pure S-wave subtracted effective Flatte.  pole_width is Gamma_rest;
        // the explicit omega-omega self-energy is controlled by flatte_ratio.
        return GVVResonanceParameters(
            GVV_PROP_SUBTRACTED_FLATTE,
            1.522,
            0.108,
            0.0,
            0,
            GVV_F0_1500_FLATTE_RATIO,
            GVV_F0_1500_FIT_FLATTE_RATIO);
    }
    if (index == GVV_RES_F0_1710) {
        // Pure omega-omega S wave: no 22 amplitude and no D-wave width term.
        return GVVResonanceParameters(
            GVV_PROP_SCALAR_SWAVE_BWR, 1.723, 0.149, 0.0, 0);
    }
    if (index == GVV_RES_ETA_1760) {
        return GVVResonanceParameters(
            GVV_PROP_PWAVE_BWR, 1.751, 0.240, 0.0, 0);
    }
    if (index == GVV_RES_ETA_C_1S) {
        return GVVResonanceParameters(
            GVV_PROP_PWAVE_BWR, 2.98409, 0.0300, 0.0, 0);
    }
    if (index == GVV_RES_X_1835) {
        return GVVResonanceParameters(
            GVV_PROP_PWAVE_BWR, 1.8340, 0.130, 0.0, 0);
    }
    if (index == GVV_RES_X_2370) {
        return GVVResonanceParameters(
            GVV_PROP_PWAVE_BWR, 2.377, 0.148, 0.0, 0);
    }
    return GVVResonanceParameters(
        GVV_PROP_NONRESONANT, 0.0, 0.0, 0.0, 0);
}

__host__ __device__ inline GVVTermSpec gvv_default_term(int index)
{
    if (index == GVV_TERM_F0_1500_00) {
        return GVVTermSpec(
            GVV_RES_F0_1500, GVV_SCALAR_00, GVV_SCALAR_00);
    }
    if (index == GVV_TERM_F0_1710_00) {
        return GVVTermSpec(
            GVV_RES_F0_1710, GVV_SCALAR_00, GVV_SCALAR_00);
    }
    if (index == GVV_TERM_ETA_1760_11) {
        return GVVTermSpec(
            GVV_RES_ETA_1760,
            GVV_PSEUDOSCALAR_11,
            GVV_PSEUDOSCALAR_11);
    }
    if (index == GVV_TERM_ETA_C_11) {
        return GVVTermSpec(
            GVV_RES_ETA_C_1S,
            GVV_PSEUDOSCALAR_11,
            GVV_PSEUDOSCALAR_11);
    }
    if (index == GVV_TERM_X_1835_11) {
        return GVVTermSpec(
            GVV_RES_X_1835,
            GVV_PSEUDOSCALAR_11,
            GVV_PSEUDOSCALAR_11);
    }
    if (index == GVV_TERM_X_2370_11) {
        return GVVTermSpec(
            GVV_RES_X_2370,
            GVV_PSEUDOSCALAR_11,
            GVV_PSEUDOSCALAR_11);
    }
    return GVVTermSpec(
        GVV_RES_NR_0MP,
        GVV_PSEUDOSCALAR_11,
        GVV_PSEUDOSCALAR_11);
}

__host__ __device__ inline DeviceComplex gvv_x_propagator(
    double s,
    const GVVResonanceParameters& resonance,
    double radius_fm = ctpwa::DEFAULT_BARRIER_RADIUS_FM)
{
    if (resonance.propagator_model == GVV_PROP_NONRESONANT) {
        return DeviceComplex(1.0, 0.0);
    }
    if (resonance.propagator_model == GVV_PROP_FIXED_BW) {
        return ctpwa::BW_fixed_width(
            s, resonance.mass, resonance.pole_width);
    }
    if (resonance.propagator_model == GVV_PROP_SUBTRACTED_FLATTE) {
        return ctpwa::Flatte_subtracted_effective(
            s,
            resonance.mass,
            resonance.pole_width,
            resonance.flatte_ratio,
            GVV_OMEGA_MASS);
    }
    if (resonance.propagator_model == GVV_PROP_SCALAR_SD_BWR) {
        return ctpwa::BWR_scalar_sd(
            s,
            resonance.mass,
            resonance.pole_width,
            resonance.sd_ratio,
            GVV_OMEGA_MASS,
            radius_fm);
    }
    if (resonance.propagator_model == GVV_PROP_SCALAR_SWAVE_BWR) {
        return ctpwa::BWR_two_body_nominal(
            s,
            resonance.mass,
            resonance.pole_width,
            0,
            GVV_OMEGA_MASS,
            GVV_OMEGA_MASS,
            radius_fm);
    }
    if (resonance.propagator_model == GVV_PROP_PWAVE_BWR) {
        return ctpwa::BWR_two_body_nominal(
            s,
            resonance.mass,
            resonance.pole_width,
            1,
            GVV_OMEGA_MASS,
            GVV_OMEGA_MASS,
            radius_fm);
    }
    return DeviceComplex(0.0, 0.0);
}

inline const char* gvv_resonance_name(int index)
{
    static const char* names[GVV_NRESONANCES] = {
        "f0_1500", "f0_1710", "eta_1760", "eta_c_1S",
        "X_1835", "X_2370", "NR_0mp"
    };
    return (index >= 0 && index < GVV_NRESONANCES) ? names[index] : "unknown";
}

inline const char* gvv_propagator_name(int model)
{
    if (model == GVV_PROP_FIXED_BW) {
        return "fixed-width BW";
    }
    if (model == GVV_PROP_SCALAR_SD_BWR) {
        return "scalar S+D BWR";
    }
    if (model == GVV_PROP_PWAVE_BWR) {
        return "P-wave BWR";
    }
    if (model == GVV_PROP_NONRESONANT) {
        return "nonresonant";
    }
    if (model == GVV_PROP_SCALAR_SWAVE_BWR) {
        return "scalar S-wave BWR";
    }
    if (model == GVV_PROP_SUBTRACTED_FLATTE) {
        return "subtracted effective Flatte";
    }
    return "unknown";
}

inline const char* gvv_term_name(int index)
{
    static const char* names[GVV_NTERMS] = {
        "f0_1500_00", "f0_1710_00", "eta_1760_11", "eta_c_11",
        "X_1835_11", "X_2370_11", "NR_0mp_11"
    };
    return (index >= 0 && index < GVV_NTERMS) ? names[index] : "unknown";
}

#endif // GVV_MODEL_H

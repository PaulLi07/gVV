#ifndef GVV_MODEL_H
#define GVV_MODEL_H

#include "Dynamics.h"
#include "GVVAmplitude.h"

// Device-level GVV propagator dispatch. Physics content and the number/order
// of resonance instances live exclusively in config/model.json.
enum GVVPropagatorModel {
    GVV_PROP_FIXED_BW = 0,
    GVV_PROP_SCALAR_SD_BWR = 1,
    GVV_PROP_PWAVE_BWR = 2,
    GVV_PROP_NONRESONANT = 3,
    GVV_PROP_SCALAR_SWAVE_BWR = 4,
    GVV_PROP_SUBTRACTED_FLATTE = 5
};

enum GVVCouplingParameterization {
    GVV_COUPLING_COMPLEX = 0,
    GVV_COUPLING_FIXED_SCALE_AND_PHASE = 1,
    GVV_COUPLING_POSITIVE_REAL = 2
};

constexpr double GVV_OMEGA_MASS = 0.78266;
constexpr double GVV_OMEGA_WIDTH = 0.00868;

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
    // Dense model-local slot into the compact active-Wave F matrix.
    int wave_slot;
    // Registered GVV Wave id used only for diagnostics/provenance.
    int registered_wave_type;

    __host__ __device__ GVVTermSpec(
        int resonance = 0,
        int slot = 0,
        int registered_wave = 0)
        : resonance_index(resonance),
          wave_slot(slot),
          registered_wave_type(registered_wave)
    {
    }
};

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

inline const char* gvv_propagator_name(int model)
{
    if (model == GVV_PROP_FIXED_BW) return "fixed-width BW";
    if (model == GVV_PROP_SCALAR_SD_BWR) return "scalar S+D BWR";
    if (model == GVV_PROP_PWAVE_BWR) return "P-wave BWR";
    if (model == GVV_PROP_NONRESONANT) return "nonresonant";
    if (model == GVV_PROP_SCALAR_SWAVE_BWR) return "scalar S-wave BWR";
    if (model == GVV_PROP_SUBTRACTED_FLATTE) {
        return "subtracted effective Flatte";
    }
    return "unknown";
}

#endif // GVV_MODEL_H

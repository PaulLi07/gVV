#ifndef CTPWA_FRAMEWORK_DYNAMICS_PROPAGATOR_REGISTRY_CUH
#define CTPWA_FRAMEWORK_DYNAMICS_PROPAGATOR_REGISTRY_CUH

#include "framework/dynamics/Propagators.cuh"

// Device-level propagator dispatch.  The concrete resonance list and its
// ordering remain exclusively in model.json and the compiled process model.
enum PropagatorModel {
    PROP_FIXED_BW = 0,
    PROP_SCALAR_SD_BWR = 1,
    PROP_PWAVE_BWR = 2,
    PROP_NONRESONANT = 3,
    PROP_SCALAR_SWAVE_BWR = 4,
    PROP_SUBTRACTED_FLATTE = 5
};

struct ResonanceParameters {
    int propagator_model;
    double mass;
    double pole_width;
    double sd_ratio;
    int fit_sd_ratio;
    double flatte_ratio;
    int fit_flatte_ratio;

    __host__ __device__ ResonanceParameters(
        int model = PROP_NONRESONANT,
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

__host__ __device__ inline DeviceComplex evaluate_propagator(
    double s,
    const ResonanceParameters& resonance,
    double daughter_mass1,
    double daughter_mass2,
    double radius_fm = ctpwa::DEFAULT_BARRIER_RADIUS_FM)
{
    if (resonance.propagator_model == PROP_NONRESONANT) {
        return DeviceComplex(1.0, 0.0);
    }
    if (resonance.propagator_model == PROP_FIXED_BW) {
        return ctpwa::BW_fixed_width(
            s, resonance.mass, resonance.pole_width);
    }
    if (resonance.propagator_model == PROP_SUBTRACTED_FLATTE) {
        return ctpwa::Flatte_subtracted_effective(
            s,
            resonance.mass,
            resonance.pole_width,
            resonance.flatte_ratio,
            daughter_mass1);
    }
    if (resonance.propagator_model == PROP_SCALAR_SD_BWR) {
        return ctpwa::BWR_scalar_sd(
            s,
            resonance.mass,
            resonance.pole_width,
            resonance.sd_ratio,
            daughter_mass1,
            radius_fm);
    }
    if (resonance.propagator_model == PROP_SCALAR_SWAVE_BWR) {
        return ctpwa::BWR_two_body_nominal(
            s,
            resonance.mass,
            resonance.pole_width,
            0,
            daughter_mass1,
            daughter_mass2,
            radius_fm);
    }
    if (resonance.propagator_model == PROP_PWAVE_BWR) {
        return ctpwa::BWR_two_body_nominal(
            s,
            resonance.mass,
            resonance.pole_width,
            1,
            daughter_mass1,
            daughter_mass2,
            radius_fm);
    }
    return DeviceComplex(0.0, 0.0);
}

inline const char* propagator_name(int model)
{
    if (model == PROP_FIXED_BW) return "fixed-width BW";
    if (model == PROP_SCALAR_SD_BWR) return "scalar S+D BWR";
    if (model == PROP_PWAVE_BWR) return "P-wave BWR";
    if (model == PROP_NONRESONANT) return "nonresonant";
    if (model == PROP_SCALAR_SWAVE_BWR) return "scalar S-wave BWR";
    if (model == PROP_SUBTRACTED_FLATTE) {
        return "subtracted effective Flatte";
    }
    return "unknown";
}

#endif // CTPWA_FRAMEWORK_DYNAMICS_PROPAGATOR_REGISTRY_CUH

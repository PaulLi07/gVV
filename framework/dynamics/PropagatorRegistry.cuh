// Compact device representation and dispatch for reusable propagator models.
// String-to-enum validation is owned by the active process compiler.
#ifndef CTPWA_FRAMEWORK_DYNAMICS_PROPAGATOR_REGISTRY_CUH
#define CTPWA_FRAMEWORK_DYNAMICS_PROPAGATOR_REGISTRY_CUH

#include "framework/dynamics/Propagators.cuh"

namespace ctpwa {

// Device-level propagator dispatch.  The concrete resonance list and its
// ordering remain exclusively in model.json and the compiled process model.
enum PropagatorModel {
    PROP_FIXED_BW = 0,
    PROP_SCALAR_SD_BWR = 1,
    PROP_TWO_BODY_RUNNING_BW = 2,
    PROP_NONRESONANT = 3,
    PROP_SUBTRACTED_FLATTE = 4
};

// Pure numerical device descriptor. Whether a field is fixed or fitted is a
// process/model-binding concern and must not be stored in this framework type.
struct PropagatorParameters {
    int propagator_model;
    int orbital_l;
    double mass;
    double pole_width;
    double sd_ratio;
    double flatte_ratio;

    __host__ __device__ PropagatorParameters(
        int model = PROP_NONRESONANT,
        double m = 0.0,
        double width = 0.0,
        int l = 0,
        double ratio = 0.0,
        double effective_channel_ratio = 0.0)
        : propagator_model(model),
          orbital_l(l),
          mass(m),
          pole_width(width),
          sd_ratio(ratio),
          flatte_ratio(effective_channel_ratio)
    {
    }
};

__host__ __device__ inline DeviceComplex evaluate_propagator(
    double s,
    const PropagatorParameters& resonance,
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
    if (resonance.propagator_model == PROP_TWO_BODY_RUNNING_BW) {
        return ctpwa::BWR_two_body_nominal(
            s,
            resonance.mass,
            resonance.pole_width,
            resonance.orbital_l,
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
    if (model == PROP_TWO_BODY_RUNNING_BW) {
        return "two-body running-width BW";
    }
    if (model == PROP_NONRESONANT) return "nonresonant";
    if (model == PROP_SUBTRACTED_FLATTE) {
        return "subtracted effective Flatte";
    }
    return "unknown";
}

} // namespace ctpwa

#endif // CTPWA_FRAMEWORK_DYNAMICS_PROPAGATOR_REGISTRY_CUH

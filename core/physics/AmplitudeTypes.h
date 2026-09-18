#pragma once

#include "core/math/Complex.cuh"
#include "core/math/BarrierFactor.cuh"

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
    double daughter_mass1;
    double daughter_mass2;
    double barrier_radius_fm;

    __host__ __device__ PropagatorParameters(
        int model = PROP_NONRESONANT,
        double m = 0.0,
        double width = 0.0,
        int l = 0,
        double ratio = 0.0,
        double effective_channel_ratio = 0.0,
        double first_daughter_mass = 0.0,
        double second_daughter_mass = 0.0,
        double radius_fm = ctpwa::DEFAULT_BARRIER_RADIUS_FM)
        : propagator_model(model),
          orbital_l(l),
          mass(m),
          pole_width(width),
          sd_ratio(ratio),
          flatte_ratio(effective_channel_ratio),
          daughter_mass1(first_daughter_mass),
          daughter_mass2(second_daughter_mass),
          barrier_radius_fm(radius_fm)
    {
    }
};

} // namespace ctpwa

struct TermSpec {
    int resonance_index = 0;
    int wave_slot = 0;
    int registered_wave_type = 0;
};

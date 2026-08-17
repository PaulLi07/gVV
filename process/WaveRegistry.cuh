// gVV Wave registration boundary: device enum/dispatch plus the compiled
// process-model data structures shared with kernels.
#ifndef CTPWA_PROCESS_WAVE_REGISTRY_CUH
#define CTPWA_PROCESS_WAVE_REGISTRY_CUH

#include "framework/math/DeviceComplex.cuh"
#include "framework/dynamics/PropagatorRegistry.cuh"
#include "framework/model/Model.h"
#include "process/waves/Pseudoscalar11.cuh"
#include "process/waves/Scalar00.cuh"
#include "process/waves/Scalar22.cuh"

#include <string>
#include <vector>

enum GVVWaveType {
    GVV_SCALAR_00 = 0,
    GVV_SCALAR_22 = 1,
    GVV_PSEUDOSCALAR_11 = 2,
    GVV_NBASIS = 3
};

// Dense process runtime objects. Stable user-facing ids remain in the host
// metadata below; kernels need only integer slots and coupling policy codes.
enum CouplingParameterization {
    COUPLING_COMPLEX = 0,
    COUPLING_FIXED_SCALE_AND_PHASE = 1,
    COUPLING_POSITIVE_REAL = 2
};

struct TermSpec {
    int resonance_index = 0;
    int wave_slot = 0;
    int registered_wave_type = 0;
};

// This is the only device dispatch point for complete process Waves.
__device__ inline tensor gvv_wave_tensor(
    const GVVEventKinematics& event,
    int wave_type,
    const GVVBarrierParameters& barrier = GVVBarrierParameters())
{
    if (wave_type == GVV_SCALAR_00) {
        return gvv_scalar_00_tensor(event);
    }
    if (wave_type == GVV_SCALAR_22) {
        return gvv_scalar_22_tensor(event, barrier);
    }
    if (wave_type == GVV_PSEUDOSCALAR_11) {
        return gvv_pseudoscalar_11_tensor(event, barrier);
    }
    return tensor();
}

__device__ inline tensor gvv_photon_projector(
    const FV& psi,
    const FV& gamma)
{
    const FV recoil = psi - gamma;
    return tensor::Gnormal()
           - (tensor(gamma, recoil) + tensor(recoil, gamma))
                 / (gamma * recoil)
           + tensor(gamma, gamma) * (recoil * recoil)
                 / (gamma * recoil) / (gamma * recoil);
}

__device__ inline double gvv_wave_contraction(
    const tensor& first,
    const tensor& second,
    const tensor& photon_projector)
{
    double result = 0.0;
    const tensor metric = tensor::Gnormal();
    for (int mu = 1; mu < 3; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            for (int nup = 0; nup < 4; ++nup) {
                result += first._matrix[mu][nu]
                          * metric._matrix[nu][nu]
                          * photon_projector._matrix[nu][nup]
                          * metric._matrix[nup][nup]
                          * second._matrix[mu][nup];
            }
        }
    }
    return -0.5 * result;
}

__device__ inline double gvv_cal_F(
    const GVVEventKinematics& event,
    int first_wave,
    int second_wave,
    const GVVBarrierParameters& barrier = GVVBarrierParameters())
{
    const tensor photon_projector =
        gvv_photon_projector(event.psi, event.gamma);
    return gvv_wave_contraction(
        gvv_wave_tensor(event, first_wave, barrier),
        gvv_wave_tensor(event, second_wave, barrier),
        photon_projector);
}

struct GVVWaveMetadata {
    std::string id;
    std::string jpc;
    std::string latex;
    std::string coherence_class;
    int wave_type = -1;
};

struct GVVResonanceMetadata {
    std::string id;
    std::string label;
    std::string propagator_id;
    bool fit_sd_ratio = false;
    bool fit_flatte_ratio = false;
};

struct GVVTermMetadata {
    std::string id;
    std::string label;
    std::string wave_id;
    std::string jpc;
    std::string latex;
    std::string coherence_class;
    int registered_wave_type = -1;
    int coupling_parameterization = COUPLING_COMPLEX;
    ctpwa::CouplingReference reference = ctpwa::CouplingReference::None;
};

// Host-side process model compiled from the generic JSON definition.  Numeric
// indices are dense, runtime-only device layout; stable ids live in metadata.
struct GVVCompiledModel {
    ctpwa::ModelDefinition definition;
    std::vector<ctpwa::PropagatorParameters> resonances;
    std::vector<TermSpec> terms;
    std::vector<DeviceComplex> initial_couplings;
    std::vector<int> active_wave_types;
    std::vector<GVVResonanceMetadata> resonance_metadata;
    std::vector<GVVTermMetadata> term_metadata;

    int find_resonance(const std::string& id) const;
    int find_term(const std::string& id) const;
};

const std::vector<GVVWaveMetadata>& gvv_wave_registry();
const GVVWaveMetadata& gvv_registered_wave(const std::string& id);

GVVCompiledModel gvv_compile_model(
    const ctpwa::ModelDefinition& definition);

GVVCompiledModel gvv_load_compiled_model(const std::string& file_name);

#endif // CTPWA_PROCESS_WAVE_REGISTRY_CUH

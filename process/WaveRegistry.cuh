// Complete GVV Wave registration boundary: device enum/dispatch and the small
// host catalogue. Resonance and Term compilation live in ModelCompiler.
#ifndef CTPWA_PROCESS_WAVE_REGISTRY_CUH
#define CTPWA_PROCESS_WAVE_REGISTRY_CUH

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

struct GVVWaveMetadata {
    std::string id;
    std::string jpc;
    std::string latex;
    std::string coherence_class;
    int wave_type = -1;
};

const std::vector<GVVWaveMetadata>& gvv_wave_registry();
const GVVWaveMetadata& gvv_registered_wave(const std::string& id);

#endif // CTPWA_PROCESS_WAVE_REGISTRY_CUH

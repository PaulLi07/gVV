// Complete GVV Wave registration boundary: device enum/dispatch and the small
// host catalogue. Resonance and Term compilation live in ModelCompiler.
#ifndef CTPWA_PROCESS_WAVE_REGISTRY_CUH
#define CTPWA_PROCESS_WAVE_REGISTRY_CUH

#include "process/waves/Pseudoscalar11.cuh"
#include "process/waves/Scalar00.cuh"
#include "process/waves/Scalar22.cuh"
#include "process/waves/Tensor02U1.cuh"
#include "process/waves/Tensor02U2.cuh"
#include "process/waves/Tensor02U3.cuh"
#include "process/waves/Tensor20U1.cuh"
#include "process/waves/Tensor20U2.cuh"
#include "process/waves/Tensor20U3.cuh"
#include "process/waves/Tensor22U1.cuh"
#include "process/waves/Tensor22U2.cuh"
#include "process/waves/Tensor22U3.cuh"
#include "process/waves/Tensor42U1.cuh"
#include "process/waves/Tensor42U2.cuh"
#include "process/waves/Tensor42U3.cuh"

#include <string>
#include <vector>

enum GVVWaveType {
    GVV_SCALAR_00 = 0,
    GVV_SCALAR_22 = 1,
    GVV_PSEUDOSCALAR_11 = 2,

    GVV_TENSOR_02_U1 = 3,
    GVV_TENSOR_02_U2 = 4,
    GVV_TENSOR_02_U3 = 5,

    GVV_TENSOR_20_U1 = 6,
    GVV_TENSOR_20_U2 = 7,
    GVV_TENSOR_20_U3 = 8,

    GVV_TENSOR_22_U1 = 9,
    GVV_TENSOR_22_U2 = 10,
    GVV_TENSOR_22_U3 = 11,

    GVV_TENSOR_42_U1 = 12,
    GVV_TENSOR_42_U2 = 13,
    GVV_TENSOR_42_U3 = 14,

    GVV_NBASIS = 15
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

    if (wave_type == GVV_TENSOR_02_U1) {
        return gvv_tensor_02_u1_tensor(event);
    }
    if (wave_type == GVV_TENSOR_02_U2) {
        return gvv_tensor_02_u2_tensor(event, barrier);
    }
    if (wave_type == GVV_TENSOR_02_U3) {
        return gvv_tensor_02_u3_tensor(event, barrier);
    }
    if (wave_type == GVV_TENSOR_20_U1) {
        return gvv_tensor_20_u1_tensor(event, barrier);
    }
    if (wave_type == GVV_TENSOR_20_U2) {
        return gvv_tensor_20_u2_tensor(event, barrier);
    }
    if (wave_type == GVV_TENSOR_20_U3) {
        return gvv_tensor_20_u3_tensor(event, barrier);
    }
    if (wave_type == GVV_TENSOR_22_U1) {
        return gvv_tensor_22_u1_tensor(event, barrier);
    }
    if (wave_type == GVV_TENSOR_22_U2) {
        return gvv_tensor_22_u2_tensor(event, barrier);
    }
    if (wave_type == GVV_TENSOR_22_U3) {
        return gvv_tensor_22_u3_tensor(event, barrier);
    }
    if (wave_type == GVV_TENSOR_42_U1) {
        return gvv_tensor_42_u1_tensor(event, barrier);
    }
    if (wave_type == GVV_TENSOR_42_U2) {
        return gvv_tensor_42_u2_tensor(event, barrier);
    }
    if (wave_type == GVV_TENSOR_42_U3) {
        return gvv_tensor_42_u3_tensor(event, barrier);
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

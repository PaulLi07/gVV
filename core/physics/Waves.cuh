#pragma once

#include "core/physics/waves/Pseudoscalar.cuh"
#include "core/physics/waves/Scalar.cuh"
#include "core/physics/waves/Tensor02.cuh"
#include "core/physics/waves/Tensor20.cuh"
#include "core/physics/waves/Tensor22.cuh"
#include "core/physics/waves/Tensor42.cuh"
#include <string>
#include <vector>

// Complete GVV Wave registration boundary: device enum/dispatch and the small
// host catalogue. Resonance and Term compilation live in ModelCompiler.

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
    // Phase-reference block metadata; never used to mask Gram-matrix entries.
    std::string coherence_class;
    int wave_type = -1;
};

const std::vector<GVVWaveMetadata>& gvv_wave_registry();
const GVVWaveMetadata& gvv_registered_wave(const std::string& id);

// GVV-specific polarization sum and Wave-pair contraction. Complete Wave
// construction/registration stays in WaveRegistry; reusable tensor building
// blocks stay in core/math.

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

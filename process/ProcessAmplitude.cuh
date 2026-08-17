// GVV-specific polarization sum and Wave-pair contraction. Complete Wave
// construction/registration stays in WaveRegistry; reusable tensor building
// blocks stay in framework/tensors.
#ifndef CTPWA_PROCESS_AMPLITUDE_CUH
#define CTPWA_PROCESS_AMPLITUDE_CUH

#include "process/WaveRegistry.cuh"

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

#endif // CTPWA_PROCESS_AMPLITUDE_CUH

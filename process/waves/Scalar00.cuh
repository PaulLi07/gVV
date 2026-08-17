#ifndef CTPWA_PROCESS_WAVES_SCALAR00_CUH
#define CTPWA_PROCESS_WAVES_SCALAR00_CUH

#include "process/ProcessEvent.cuh"

// Uhat_00^{mu nu} = g^{mu nu} E1^alpha E2_alpha.
__device__ inline tensor gvv_scalar_00_tensor(
    const GVVEventKinematics& event)
{
    const double omega_spin0 =
        event.omega_current1.geometry * event.omega_current2.geometry;
    return tensor::Gnormal() * omega_spin0;
}

#endif // CTPWA_PROCESS_WAVES_SCALAR00_CUH

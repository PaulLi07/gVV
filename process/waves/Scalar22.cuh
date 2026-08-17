#ifndef CTPWA_PROCESS_WAVES_SCALAR22_CUH
#define CTPWA_PROCESS_WAVES_SCALAR22_CUH

#include "framework/tensors/BarrierFactor.cuh"
#include "framework/tensors/OrbitalTensor.cuh"
#include "process/ProcessEvent.cuh"

// Uhat_22^{mu nu} = g^{mu nu} t^(2)_alpha_beta E1^alpha E2^beta.
__device__ inline tensor gvv_scalar_22_tensor(
    const GVVEventKinematics& event,
    const GVVBarrierParameters& barrier = GVVBarrierParameters())
{
    const double q_x_omega_omega = ctpwa::two_body_Q(
        event.X * event.X,
        event.omega1 * event.omega1,
        event.omega2 * event.omega2);
    const tensor t2 = ctpwa::orbital_dwave(
        event.X, event.relative_omega_momentum)
        * ctpwa::blatt_weisskopf(
            q_x_omega_omega, 2, barrier.x_decay_radius_fm);
    const double omega_spin2 =
        event.omega_current1.geometry
        * (t2 * event.omega_current2.geometry);
    return tensor::Gnormal() * omega_spin2;
}

#endif // CTPWA_PROCESS_WAVES_SCALAR22_CUH

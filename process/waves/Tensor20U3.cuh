// Complete registered GVV 2++(20), U3 covariant numerator.
// Resonance and omega/rho propagators are applied later by TermEvaluator.
#ifndef CTPWA_PROCESS_WAVES_TENSOR20U3_CUH
#define CTPWA_PROCESS_WAVES_TENSOR20U3_CUH

#include "framework/tensors/BarrierFactor.cuh"
#include "framework/tensors/OrbitalTensor.cuh"
#include "framework/tensors/TensorContraction.cuh"
#include "process/ProcessEvent.cuh"

#include <cmath>

// U_20^(3)^{mu nu}
// = q_gamma^mu D_20^{nu alpha} p_psi,alpha
//   B_2(Q_psi-gamma-X).
__device__ inline tensor gvv_tensor_20_u3_tensor(
    const GVVEventKinematics& event,
    const GVVBarrierParameters& barrier = GVVBarrierParameters())
{
    const tensor orbital_d2 = ctpwa::orbital_dwave(
        event.X,
        event.relative_omega_momentum);
    const double omega_spin0 =
        event.omega_current1.geometry
        * event.omega_current2.geometry;

    const double q_x_omega_omega = ctpwa::two_body_Q(
        event.X * event.X,
        event.omega1 * event.omega1,
        event.omega2 * event.omega2);
    const double decay_barrier = ctpwa::blatt_weisskopf(
        q_x_omega_omega,
        2,
        barrier.x_decay_radius_fm);

    const double orbital_normalization = sqrt(3.0 / 2.0);
    const double spin_normalization = 1.0 / sqrt(3.0);
    const tensor decay = orbital_d2
        * (omega_spin0
           * orbital_normalization
           * spin_normalization
           * decay_barrier);

    const FV decay_times_psi = ctpwa::contract_second_index(
        decay,
        event.psi);

    const double q_psi_gamma_x = ctpwa::two_body_Q(
        event.psi * event.psi,
        event.gamma * event.gamma,
        event.X * event.X);
    const double production_barrier = ctpwa::blatt_weisskopf(
        q_psi_gamma_x,
        2,
        barrier.production_radius_fm);

    return tensor(event.gamma, decay_times_psi)
           * production_barrier;
}

#endif // CTPWA_PROCESS_WAVES_TENSOR20U3_CUH

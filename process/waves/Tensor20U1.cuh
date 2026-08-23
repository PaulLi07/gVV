// Complete registered GVV 2++(20), U1 covariant numerator.
// Resonance and omega/rho propagators are applied later by TermEvaluator.
#ifndef CTPWA_PROCESS_WAVES_TENSOR20U1_CUH
#define CTPWA_PROCESS_WAVES_TENSOR20U1_CUH

#include "framework/tensors/BarrierFactor.cuh"
#include "framework/tensors/OrbitalTensor.cuh"
#include "process/ProcessEvent.cuh"

#include <cmath>

// U_20^(1)^{mu nu} = D_20^{mu nu},
// D_20^{mu nu} = t_bare^(2)^{mu nu} (Omega1 . Omega2)
//                  [sqrt(3/2)] [1/sqrt(3)] B_2(Q_X-omega-omega).
__device__ inline tensor gvv_tensor_20_u1_tensor(
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

    return orbital_d2
           * (omega_spin0
              * orbital_normalization
              * spin_normalization
              * decay_barrier);
}

#endif // CTPWA_PROCESS_WAVES_TENSOR20U1_CUH

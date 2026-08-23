// Complete registered GVV 2++(42), U1 covariant numerator.
// Resonance and omega/rho propagators are applied later by TermEvaluator.
#ifndef CTPWA_PROCESS_WAVES_TENSOR42U1_CUH
#define CTPWA_PROCESS_WAVES_TENSOR42U1_CUH

#include "framework/tensors/BarrierFactor.cuh"
#include "framework/tensors/OrbitalTensor.cuh"
#include "process/ProcessEvent.cuh"

#include <cmath>

// U_42^(1)^{mu nu} = D_42^{mu nu}.
// contract_orbital_gwave() evaluates the bare rank-four orbital contraction
// with the spin-two omega source. Its result is already spin two, so no outer
// P^(2) is applied here.
__device__ inline tensor gvv_tensor_42_u1_tensor(
    const GVVEventKinematics& event,
    const GVVBarrierParameters& barrier = GVVBarrierParameters())
{
    const tensor omega_product(
        event.omega_current1.geometry,
        event.omega_current2.geometry);
    const tensor coupled_bare = ctpwa::contract_orbital_gwave(
        event.X,
        event.relative_omega_momentum,
        omega_product);

    const double q_x_omega_omega = ctpwa::two_body_Q(
        event.X * event.X,
        event.omega1 * event.omega1,
        event.omega2 * event.omega2);
    const double decay_barrier = ctpwa::blatt_weisskopf(
        q_x_omega_omega,
        4,
        barrier.x_decay_radius_fm);

    const double orbital_normalization = sqrt(35.0 / 8.0);
    const double ls_coupling = sqrt(5.0) / 3.0;

    return coupled_bare
           * (orbital_normalization
              * ls_coupling
              * decay_barrier);
}

#endif // CTPWA_PROCESS_WAVES_TENSOR42U1_CUH

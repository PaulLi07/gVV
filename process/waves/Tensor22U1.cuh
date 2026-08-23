// Complete registered GVV 2++(22), U1 covariant numerator.
// Resonance and omega/rho propagators are applied later by TermEvaluator.
#ifndef CTPWA_PROCESS_WAVES_TENSOR22U1_CUH
#define CTPWA_PROCESS_WAVES_TENSOR22U1_CUH

#include "framework/tensors/BarrierFactor.cuh"
#include "framework/tensors/OrbitalTensor.cuh"
#include "framework/tensors/SpinProjector.cuh"
#include "framework/tensors/TensorContraction.cuh"
#include "process/ProcessEvent.cuh"

#include <cmath>

// U_22^(1)^{mu nu} = D_22^{mu nu}.
// The inner projector couples the omega currents to spin two. The named
// contraction lowers the shared lambda index with the (+---) metric, and the
// outer projector selects the total-J=2 part.
__device__ inline tensor gvv_tensor_22_u1_tensor(
    const GVVEventKinematics& event,
    const GVVBarrierParameters& barrier = GVVBarrierParameters())
{
    const tensor orbital_d2 = ctpwa::orbital_dwave(
        event.X,
        event.relative_omega_momentum);
    const tensor omega_spin2 = ctpwa::spin2_project(
        event.X,
        tensor(
            event.omega_current1.geometry,
            event.omega_current2.geometry));
    const tensor coupled_bare = ctpwa::spin2_project(
        event.X,
        ctpwa::contract_second_indices(
            orbital_d2,
            omega_spin2));

    const double q_x_omega_omega = ctpwa::two_body_Q(
        event.X * event.X,
        event.omega1 * event.omega1,
        event.omega2 * event.omega2);
    const double decay_barrier = ctpwa::blatt_weisskopf(
        q_x_omega_omega,
        2,
        barrier.x_decay_radius_fm);

    const double orbital_normalization = sqrt(3.0 / 2.0);
    const double ls_coupling = sqrt(12.0 / 7.0);

    return coupled_bare
           * (orbital_normalization
              * ls_coupling
              * decay_barrier);
}

#endif // CTPWA_PROCESS_WAVES_TENSOR22U1_CUH

// Complete registered GVV 2++(02), U2 covariant numerator.
// Resonance and omega/rho propagators are applied later by TermEvaluator.
#ifndef CTPWA_PROCESS_WAVES_TENSOR02U2_CUH
#define CTPWA_PROCESS_WAVES_TENSOR02U2_CUH

#include "framework/tensors/BarrierFactor.cuh"
#include "framework/tensors/SpinProjector.cuh"
#include "framework/tensors/TensorContraction.cuh"
#include "process/ProcessEvent.cuh"

// U_02^(2)^{mu nu}
// = g^{mu nu} p_psi^alpha p_psi^beta D_02,alpha beta
//   B_2(Q_psi-gamma-X).
__device__ inline tensor gvv_tensor_02_u2_tensor(
    const GVVEventKinematics& event,
    const GVVBarrierParameters& barrier = GVVBarrierParameters())
{
    const tensor omega_product(
        event.omega_current1.geometry,
        event.omega_current2.geometry);
    const tensor decay = ctpwa::spin2_project(
        event.X,
        omega_product);

    const FV decay_times_psi = ctpwa::contract_second_index(
        decay,
        event.psi);
    const double production_scalar = event.psi * decay_times_psi;

    const double q_psi_gamma_x = ctpwa::two_body_Q(
        event.psi * event.psi,
        event.gamma * event.gamma,
        event.X * event.X);
    const double production_barrier = ctpwa::blatt_weisskopf(
        q_psi_gamma_x,
        2,
        barrier.production_radius_fm);

    return tensor::Gnormal()
           * (production_scalar * production_barrier);
}

#endif // CTPWA_PROCESS_WAVES_TENSOR02U2_CUH

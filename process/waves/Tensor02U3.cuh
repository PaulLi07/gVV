// Complete registered GVV 2++(02), U3 covariant numerator.
// Resonance and omega/rho propagators are applied later by TermEvaluator.
#ifndef CTPWA_PROCESS_WAVES_TENSOR02U3_CUH
#define CTPWA_PROCESS_WAVES_TENSOR02U3_CUH

#include "framework/tensors/BarrierFactor.cuh"
#include "framework/tensors/SpinProjector.cuh"
#include "framework/tensors/TensorContraction.cuh"
#include "process/ProcessEvent.cuh"

// U_02^(3)^{mu nu}
// = q_gamma^mu D_02^{nu alpha} p_psi,alpha
//   B_2(Q_psi-gamma-X).
// The first index is the psi-polarization index and the second is the photon
// index, matching the process-wide polarization contraction.
__device__ inline tensor gvv_tensor_02_u3_tensor(
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

#endif // CTPWA_PROCESS_WAVES_TENSOR02U3_CUH

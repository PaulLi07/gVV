// Complete registered GVV 2++(02), U1 covariant numerator.
// Resonance and omega/rho propagators are applied later by TermEvaluator.
#ifndef CTPWA_PROCESS_WAVES_TENSOR02U1_CUH
#define CTPWA_PROCESS_WAVES_TENSOR02U1_CUH

#include "framework/tensors/SpinProjector.cuh"
#include "process/ProcessEvent.cuh"

// U_02^(1)^{mu nu} = D_02^{mu nu},
// D_02^{mu nu} = P^(2)^{mu nu}_{rho sigma}(X)
//                  Omega1^rho Omega2^sigma.
__device__ inline tensor gvv_tensor_02_u1_tensor(
    const GVVEventKinematics& event)
{
    const tensor omega_product(
        event.omega_current1.geometry,
        event.omega_current2.geometry);

    return ctpwa::spin2_project(event.X, omega_product);
}

#endif // CTPWA_PROCESS_WAVES_TENSOR02U1_CUH

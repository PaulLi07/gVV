#pragma once

#include "core/math/TensorOps.cuh"
#include "core/physics/Event.cuh"
#include "core/math/BarrierFactor.cuh"

// Complete registered GVV 2++(02), U1 covariant numerator.
// Resonance and omega/rho propagators are applied later by TermEvaluator.

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

// Complete registered GVV 2++(02), U2 covariant numerator.
// Resonance and omega/rho propagators are applied later by TermEvaluator.

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

// Complete registered GVV 2++(02), U3 covariant numerator.
// Resonance and omega/rho propagators are applied later by TermEvaluator.

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

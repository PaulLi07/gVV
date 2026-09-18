#pragma once

#include "core/math/BarrierFactor.cuh"
#include "core/math/OrbitalTensor.cuh"
#include "core/math/TensorOps.cuh"
#include "core/physics/Event.cuh"
#include <cmath>

// Complete registered GVV 2++(22), U1 covariant numerator.
// Resonance and omega/rho propagators are applied later by TermEvaluator.

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

// Complete registered GVV 2++(22), U2 covariant numerator.
// Resonance and omega/rho propagators are applied later by TermEvaluator.

// U_22^(2)^{mu nu}
// = g^{mu nu} p_psi^alpha p_psi^beta D_22,alpha beta
//   B_2(Q_psi-gamma-X).
__device__ inline tensor gvv_tensor_22_u2_tensor(
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
    const tensor decay = coupled_bare
        * (orbital_normalization
           * ls_coupling
           * decay_barrier);

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

// Complete registered GVV 2++(22), U3 covariant numerator.
// Resonance and omega/rho propagators are applied later by TermEvaluator.

// U_22^(3)^{mu nu}
// = q_gamma^mu D_22^{nu alpha} p_psi,alpha
//   B_2(Q_psi-gamma-X).
__device__ inline tensor gvv_tensor_22_u3_tensor(
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
    const tensor decay = coupled_bare
        * (orbital_normalization
           * ls_coupling
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

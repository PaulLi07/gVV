#pragma once

#include "core/math/BarrierFactor.cuh"
#include "core/math/OrbitalTensor.cuh"
#include "core/physics/Event.cuh"

// Complete registered gVV 0-+(11) covariant tensor basis.

// Uhat_11^{mu nu} for psi -> gamma X, X -> omega omega.
__device__ inline tensor gvv_pseudoscalar_11_tensor(
    const GVVEventKinematics& event,
    const GVVBarrierParameters& barrier = GVVBarrierParameters())
{
    const double q_psi_gamma_x = ctpwa::two_body_Q(
        event.psi * event.psi,
        event.gamma * event.gamma,
        event.X * event.X);
    const tensor production =
        tensor::Epsilon(event.psi, event.gamma)
        * ctpwa::blatt_weisskopf(
            q_psi_gamma_x, 1, barrier.production_radius_fm);

    const double q_x_omega_omega = ctpwa::two_body_Q(
        event.X * event.X,
        event.omega1 * event.omega1,
        event.omega2 * event.omega2);
    const FV t1 = ctpwa::orbital_pwave(
        event.X, event.relative_omega_momentum)
        * ctpwa::blatt_weisskopf(
            q_x_omega_omega, 1, barrier.x_decay_radius_fm);
    const tensor omega_spin1 = tensor::Epsilon(
        event.omega_current1.geometry,
        event.omega_current2.geometry);
    const double decay_scalar = t1 * (omega_spin1 * event.X);
    return production * decay_scalar;
}

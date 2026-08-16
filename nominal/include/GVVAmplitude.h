#ifndef GVV_AMPLITUDE_H
#define GVV_AMPLITUDE_H

#include "Omega.h"

enum GVVWaveType {
    GVV_SCALAR_00 = 0,
    GVV_SCALAR_22 = 1,
    GVV_PSEUDOSCALAR_11 = 2,
    GVV_NBASIS = 3
};

struct GVVBarrierParameters {
    double production_radius_fm;
    double x_decay_radius_fm;

    __host__ __device__ GVVBarrierParameters(
        double production_radius = ctpwa::DEFAULT_BARRIER_RADIUS_FM,
        double x_decay_radius = ctpwa::DEFAULT_BARRIER_RADIUS_FM)
        : production_radius_fm(production_radius),
          x_decay_radius_fm(x_decay_radius)
    {
    }
};

// Event kinematics and the two omega -> 3pi currents.  For each omega the pion
// convention is p0=pi0, p1=pi+, p2=pi-, matching Omega.h.
struct GVVEventKinematics {
    FV gamma;
    FV omega1;
    FV omega2;
    FV X;
    FV psi;
    FV relative_omega_momentum;
    OmegaDecayCurrent omega_current1;
    OmegaDecayCurrent omega_current2;

    __device__ GVVEventKinematics(
        const FV& pi01,
        const FV& pip1,
        const FV& pim1,
        const FV& pi02,
        const FV& pip2,
        const FV& pim2,
        const FV& bachelor_gamma,
        const RhoBWRParameters& rho = RhoBWRParameters())
        : gamma(bachelor_gamma),
          omega1(pi01 + pip1 + pim1),
          omega2(pi02 + pip2 + pim2),
          X(omega1 + omega2),
          psi(X + gamma),
          relative_omega_momentum(omega1 - omega2),
          omega_current1(build_omega_decay_current(pi01, pip1, pim1, rho)),
          omega_current2(build_omega_decay_current(pi02, pip2, pim2, rho))
    {
    }
};

// Values supplied by the independent propagator layer.  No omega running-width
// model is assumed here; the caller evaluates f_omega(k_i^2) first.
struct GVVEventPropagators {
    DeviceComplex f_omega1;
    DeviceComplex f_omega2;

    __device__ GVVEventPropagators(
        const DeviceComplex& omega1_propagator,
        const DeviceComplex& omega2_propagator)
        : f_omega1(omega1_propagator),
          f_omega2(omega2_propagator)
    {
    }
};

// One resonance/coupling contribution evaluated for the current event.  f_X is
// deliberately an input so the tensor code is independent of its width model.
struct GVVAmplitudeTerm {
    int wave_type;
    DeviceComplex coupling;
    DeviceComplex f_X;

    __device__ GVVAmplitudeTerm(
        int wave,
        const DeviceComplex& complex_coupling,
        const DeviceComplex& x_propagator)
        : wave_type(wave), coupling(complex_coupling), f_X(x_propagator)
    {
    }
};

// t^(1)_delta = r_tilde_delta B1(Q_X,omega,omega).
// B1 is part of t^(1) and must not be multiplied again downstream.
__device__ inline FV gvv_orbital_t1(
    const GVVEventKinematics& event,
    const GVVBarrierParameters& barrier = GVVBarrierParameters())
{
    const double q_x_omega_omega = ctpwa::two_body_Q(
        event.X * event.X,
        event.omega1 * event.omega1,
        event.omega2 * event.omega2);

    return tensor::Projection_Pwave(
               event.X, event.relative_omega_momentum)
           * ctpwa::blatt_weisskopf(
               q_x_omega_omega, 1, barrier.x_decay_radius_fm);
}

// t^(2)_alpha_beta is the symmetric traceless D-wave tensor including B2.
__device__ inline tensor gvv_orbital_t2(
    const GVVEventKinematics& event,
    const GVVBarrierParameters& barrier = GVVBarrierParameters())
{
    const double q_x_omega_omega = ctpwa::two_body_Q(
        event.X * event.X,
        event.omega1 * event.omega1,
        event.omega2 * event.omega2);

    return tensor::Projection_Dwave(
               event.X, event.relative_omega_momentum)
           * ctpwa::blatt_weisskopf(
               q_x_omega_omega, 2, barrier.x_decay_radius_fm);
}

// Uhat_00^{mu nu} = g^{mu nu} E1^alpha E2_alpha.
// This is the single independent radiative spin-0 production tensor in the
// Coulomb-gauge convention used below: e.gamma=e.psi=0.  The photon projector
// supplies the corresponding physical transverse-polarization sum.
__device__ inline tensor gvv_scalar_00_tensor(
    const GVVEventKinematics& event)
{
    const double omega_spin0 =
        event.omega_current1.geometry * event.omega_current2.geometry;
    return tensor::Gnormal() * omega_spin0;
}

// Uhat_22^{mu nu} = g^{mu nu} t^(2)_alpha_beta E1^alpha E2^beta.
__device__ inline tensor gvv_scalar_22_tensor(
    const GVVEventKinematics& event,
    const GVVBarrierParameters& barrier = GVVBarrierParameters())
{
    const tensor t2 = gvv_orbital_t2(event, barrier);
    const double omega_spin2 =
        event.omega_current1.geometry
        * (t2 * event.omega_current2.geometry);
    return tensor::Gnormal() * omega_spin2;
}

// Uhat_11^{mu nu} = epsilon^{mu nu rho sigma} p_psi,rho p_gamma,sigma
//                    B1(Q_psi,gamma,X)
//                  * epsilon^{delta lambda}_{alpha beta} K_lambda
//                    E1^alpha E2^beta t^(1)_delta.
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

    const tensor omega_spin1 = tensor::Epsilon(
        event.omega_current1.geometry,
        event.omega_current2.geometry);
    const FV t1 = gvv_orbital_t1(event, barrier);

    // First contract lambda with K, then delta with t^(1).
    const double decay_scalar = t1 * (omega_spin1 * event.X);
    return production * decay_scalar;
}

__device__ inline tensor gvv_wave_tensor(
    const GVVEventKinematics& event,
    int wave_type,
    const GVVBarrierParameters& barrier = GVVBarrierParameters())
{
    if (wave_type == GVV_SCALAR_00) {
        return gvv_scalar_00_tensor(event);
    }
    if (wave_type == GVV_SCALAR_22) {
        return gvv_scalar_22_tensor(event, barrier);
    }
    if (wave_type == GVV_PSEUDOSCALAR_11) {
        return gvv_pseudoscalar_11_tensor(event, barrier);
    }
    return tensor();
}

__device__ inline tensor gvv_photon_projector(
    const FV& psi,
    const FV& gamma)
{
    const FV K = psi - gamma;
    const tensor part1 = tensor::Gnormal();
    const tensor part2 =
        (tensor(gamma, K) + tensor(K, gamma)) / (gamma * K);
    const tensor part3 =
        tensor(gamma, gamma) * (K * K)
        / (gamma * K) / (gamma * K);
    return part1 - part2 + part3;
}

// Polarization-summed real kinematic bilinear F_ab.  psi indices 1 and 2 are
// the transverse polarizations from e+e- production, as in the original code.
// For the present spin-0 production tensors, the scalar-pseudoscalar entries
// vanish event by event.  Consequently those two coherent sectors have
// independent unobservable common phases; the coupling policy in GVVModel.h
// fixes one phase reference in each sector.
__device__ inline double gvv_F_contract(
    const tensor& U1,
    const tensor& U2,
    const tensor& photon_projector)
{
    double result = 0.0;
    const tensor metric = tensor::Gnormal();

    for (int mu = 1; mu < 3; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            for (int nup = 0; nup < 4; ++nup) {
                result += U1._matrix[mu][nu]
                          * metric._matrix[nu][nu]
                          * photon_projector._matrix[nu][nup]
                          * metric._matrix[nup][nup]
                          * U2._matrix[mu][nup];
            }
        }
    }
    return -0.5 * result;
}

__device__ inline double gvv_cal_F(
    const GVVEventKinematics& event,
    int wave1,
    int wave2,
    const GVVBarrierParameters& barrier = GVVBarrierParameters())
{
    const tensor photon_projector =
        gvv_photon_projector(event.psi, event.gamma);
    const tensor U1 = gvv_wave_tensor(event, wave1, barrier);
    const tensor U2 = gvv_wave_tensor(event, wave2, barrier);
    return gvv_F_contract(U1, U2, photon_projector);
}

__host__ __device__ inline int gvv_F_index(int wave1, int wave2)
{
    return wave1 * GVV_NBASIS + wave2;
}

__device__ inline DeviceComplex gvv_common_omega_factor(
    const GVVEventKinematics& event,
    const GVVEventPropagators& propagators)
{
    return event.omega_current1.rho_factor
           * event.omega_current2.rho_factor
           * propagators.f_omega1
           * propagators.f_omega2;
}

// Complete polarization-summed |A|^2.  Each term carries its own Lambda and
// already-evaluated f_X(K^2); omega/rho dynamics is common to every term.
__device__ inline double gvv_cross_section(
    const GVVEventKinematics& event,
    const GVVEventPropagators& propagators,
    const GVVAmplitudeTerm* terms,
    int nterms,
    const double* F_matrix)
{
    const DeviceComplex common_omega =
        gvv_common_omega_factor(event, propagators);
    double result = 0.0;

    for (int i = 0; i < nterms; ++i) {
        const DeviceComplex zi =
            terms[i].coupling * terms[i].f_X * common_omega;

        for (int j = 0; j < nterms; ++j) {
            const DeviceComplex zj =
                terms[j].coupling * terms[j].f_X * common_omega;
            const double F = F_matrix[
                gvv_F_index(terms[i].wave_type, terms[j].wave_type)];
            result += (zi * zj.conjugate() * F).real;
        }
    }

    if (result < -1.0e-10) {
        printf("WARNING! NEGATIVE GVV PDF: %e\n", result);
    }
    return result;
}

// Array interface used by the PDF kernel. f_X_values contains one value per
// event and amplitude term; couplings and wave types are common to all events.
__device__ inline double gvv_cross_section_from_arrays(
    const GVVEventKinematics& event,
    const GVVEventPropagators& propagators,
    const int* wave_types,
    const DeviceComplex* couplings,
    const DeviceComplex* f_X_values,
    int nterms,
    const double* F_matrix)
{
    const DeviceComplex common_omega =
        gvv_common_omega_factor(event, propagators);
    double result = 0.0;

    for (int i = 0; i < nterms; ++i) {
        const DeviceComplex zi =
            couplings[i] * f_X_values[i] * common_omega;

        for (int j = 0; j < nterms; ++j) {
            const DeviceComplex zj =
                couplings[j] * f_X_values[j] * common_omega;
            const double F = F_matrix[
                gvv_F_index(wave_types[i], wave_types[j])];
            result += (zi * zj.conjugate() * F).real;
        }
    }

    if (result < -1.0e-10) {
        printf("WARNING! NEGATIVE GVV PDF: %e\n", result);
    }
    return result;
}

#endif // GVV_AMPLITUDE_H

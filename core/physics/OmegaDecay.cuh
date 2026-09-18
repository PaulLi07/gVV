#pragma once

#include "core/physics/Propagators.cuh"

// Shared omega -> rho pi -> 3pi dynamics used by both event amplitudes and
// the omega running-width integration. This remains process-specific while
// calling only reusable framework propagator and barrier functions.

constexpr double GVV_OMEGA_MASS = 0.78266;
constexpr double GVV_OMEGA_WIDTH = 0.00868;
constexpr double GVV_RHO_MASS = 0.77526;
constexpr double GVV_RHO_WIDTH = 0.1474;
constexpr double GVV_PIP_MASS = 0.13957039;
constexpr double GVV_PIM_MASS = 0.13957039;
constexpr double GVV_PI0_MASS = 0.1349768;

struct RhoBWRParameters {
    double mass;
    double width;
    double omega_vertex_radius_fm;
    double rho_vertex_radius_fm;

    __host__ __device__ RhoBWRParameters(
        double rho_mass = GVV_RHO_MASS,
        double rho_width = GVV_RHO_WIDTH,
        double omega_radius_fm = ctpwa::DEFAULT_BARRIER_RADIUS_FM,
        double rho_radius_fm = ctpwa::DEFAULT_BARRIER_RADIUS_FM)
        : mass(rho_mass),
          width(rho_width),
          omega_vertex_radius_fm(omega_radius_fm),
          rho_vertex_radius_fm(rho_radius_fm)
    {
    }
};

enum class RhoChargeChannel {
    Rho0ToPiPlusPiMinus,
    RhoPlusToPiPlusPi0,
    RhoMinusToPiMinusPi0
};

struct RhoIsobarMasses {
    double first_daughter;
    double second_daughter;
    double bachelor;
};

// Particle identities, rather than reconstructed p_i^2 values, define all
// pion masses entering the scalar rho-isobar dynamics. Event dependence stays
// in s_omega and s_pair.
__host__ __device__ inline RhoIsobarMasses rho_isobar_nominal_masses(
    RhoChargeChannel channel)
{
    if (channel == RhoChargeChannel::Rho0ToPiPlusPiMinus) {
        return {GVV_PIP_MASS, GVV_PIM_MASS, GVV_PI0_MASS};
    }
    if (channel == RhoChargeChannel::RhoPlusToPiPlusPi0) {
        return {GVV_PIP_MASS, GVV_PI0_MASS, GVV_PIM_MASS};
    }
    return {GVV_PIM_MASS, GVV_PI0_MASS, GVV_PIP_MASS};
}

// f^(rho)_ij = B1(Q_omega,rho(ij),k) * BWR_rho(s_ij)
//              * B1(Q_rho(ij),i,j).
__host__ __device__ inline DeviceComplex omega_rho_isobar_factor(
    double s_omega,
    double s_pair,
    RhoChargeChannel channel,
    const RhoBWRParameters& rho = RhoBWRParameters())
{
    const RhoIsobarMasses masses = rho_isobar_nominal_masses(channel);
    const double q_omega_rho =
        ctpwa::two_body_Q(
            s_omega, s_pair, masses.bachelor * masses.bachelor);
    const double q_rho_pipi =
        ctpwa::two_body_Q(
            s_pair,
            masses.first_daughter * masses.first_daughter,
            masses.second_daughter * masses.second_daughter);

    const double omega_barrier = ctpwa::blatt_weisskopf(
        q_omega_rho, 1, rho.omega_vertex_radius_fm);
    const double rho_barrier = ctpwa::blatt_weisskopf(
        q_rho_pipi, 1, rho.rho_vertex_radius_fm);
    const DeviceComplex rho_propagator = ctpwa::BWR(
        s_pair,
        rho.mass,
        rho.width,
        1,
        masses.first_daughter,
        masses.second_daughter,
        rho.rho_vertex_radius_fm);

    return omega_barrier * rho_propagator * rho_barrier;
}

// The pion convention is p0=pi0, p1=pi+, p2=pi-. The three pair invariants
// s12, s10 and s20 therefore select rho0, rho+ and rho-, respectively.
__host__ __device__ inline DeviceComplex coherent_omega_rho_factor(
    double s_omega,
    double s12,
    double s10,
    double s20,
    const RhoBWRParameters& rho = RhoBWRParameters())
{
    const DeviceComplex f12 = omega_rho_isobar_factor(
        s_omega,
        s12,
        RhoChargeChannel::Rho0ToPiPlusPiMinus,
        rho);
    const DeviceComplex f10 = omega_rho_isobar_factor(
        s_omega,
        s10,
        RhoChargeChannel::RhoPlusToPiPlusPi0,
        rho);
    const DeviceComplex f20 = omega_rho_isobar_factor(
        s_omega,
        s20,
        RhoChargeChannel::RhoMinusToPiMinusPi0,
        rho);
    return f12 + f10 + f20;
}

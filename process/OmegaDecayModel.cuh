// Shared omega -> rho pi -> 3pi dynamics used by both event amplitudes and
// the omega running-width integration. This remains process-specific while
// calling only reusable framework propagator and barrier functions.
#ifndef CTPWA_PROCESS_OMEGA_DECAY_MODEL_CUH
#define CTPWA_PROCESS_OMEGA_DECAY_MODEL_CUH

#include "framework/dynamics/Propagators.cuh"

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

// f^(rho)_ij = B1(Q_omega,rho(ij),k) * BWR_rho(s_ij)
//              * B1(Q_rho(ij),i,j).
__host__ __device__ inline DeviceComplex omega_rho_isobar_factor(
    double s_omega,
    double s_pair,
    double s_bachelor,
    double s_first,
    double s_second,
    const RhoBWRParameters& rho = RhoBWRParameters())
{
    const double q_omega_rho =
        ctpwa::two_body_Q(s_omega, s_pair, s_bachelor);
    const double q_rho_pipi =
        ctpwa::two_body_Q(s_pair, s_first, s_second);

    const double omega_barrier = ctpwa::blatt_weisskopf(
        q_omega_rho, 1, rho.omega_vertex_radius_fm);
    const double rho_barrier = ctpwa::blatt_weisskopf(
        q_rho_pipi, 1, rho.rho_vertex_radius_fm);
    const DeviceComplex rho_propagator = ctpwa::BWR(
        s_pair,
        rho.mass,
        rho.width,
        1,
        s_first,
        s_second,
        rho.rho_vertex_radius_fm);

    return omega_barrier * rho_propagator * rho_barrier;
}

// The pion convention is p0=pi0, p1=pi+, p2=pi-. The three arguments s12,
// s10 and s20 therefore represent pi+pi-, pi+pi0 and pi-pi0, respectively.
__host__ __device__ inline DeviceComplex coherent_omega_rho_factor(
    double s_omega,
    double s12,
    double s10,
    double s20,
    double s0,
    double s1,
    double s2,
    const RhoBWRParameters& rho = RhoBWRParameters())
{
    const DeviceComplex f12 = omega_rho_isobar_factor(
        s_omega, s12, s0, s1, s2, rho);
    const DeviceComplex f10 = omega_rho_isobar_factor(
        s_omega, s10, s2, s1, s0, rho);
    const DeviceComplex f20 = omega_rho_isobar_factor(
        s_omega, s20, s1, s2, s0, rho);
    return f12 + f10 + f20;
}

#endif // CTPWA_PROCESS_OMEGA_DECAY_MODEL_CUH

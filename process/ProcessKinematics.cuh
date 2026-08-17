#ifndef CTPWA_PROCESS_KINEMATICS_CUH
#define CTPWA_PROCESS_KINEMATICS_CUH

#include "framework/dynamics/Propagators.cuh"
#include "framework/tensors/Tensor.cuh"

constexpr double GVV_OMEGA_MASS = 0.78266;
constexpr double GVV_OMEGA_WIDTH = 0.00868;

// The pion convention used throughout this file is
// p0 = pi0, p1 = pi+, p2 = pi-.
struct RhoBWRParameters {
    double mass;
    double width;
    double omega_vertex_radius_fm;
    double rho_vertex_radius_fm;

    __host__ __device__ RhoBWRParameters(
        double rho_mass = 0.77526,
        double rho_width = 0.1474,
        double omega_radius_fm = ctpwa::DEFAULT_BARRIER_RADIUS_FM,
        double rho_radius_fm = ctpwa::DEFAULT_BARRIER_RADIUS_FM)
        : mass(rho_mass),
          width(rho_width),
          omega_vertex_radius_fm(omega_radius_fm),
          rho_vertex_radius_fm(rho_radius_fm)
    {
    }
};

// Omega^mu = geometry^mu * rho_factor.  Keeping the real geometric current
// separate from the complex rho factor allows the existing real F matrix to be
// retained when the same omega decay model is common to all production waves.
struct OmegaDecayCurrent {
    FV geometry;
    DeviceComplex rho_factor;

    __device__ OmegaDecayCurrent(
        const FV& geometric_current,
        const DeviceComplex& dynamic_factor)
        : geometry(geometric_current), rho_factor(dynamic_factor)
    {
    }
};

__device__ inline double omega_metric_sign(int index)
{
    return ctpwa::metric_sign(index);
}

// E^mu = epsilon^mu_{ nu lambda sigma }
//        p1^nu p2^lambda p0^sigma.
// tensor::epsilon stores epsilon^{0123}=+1.  The three metric factors lower
// the contracted indices for the (+---) metric used by FV.
__device__ inline FV omega_geometric_current(
    const FV& p0,
    const FV& p1,
    const FV& p2)
{
    double component[4] = {0.0, 0.0, 0.0, 0.0};

    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            for (int lambda = 0; lambda < 4; ++lambda) {
                for (int sigma = 0; sigma < 4; ++sigma) {
                    const double lowering_sign =
                        omega_metric_sign(nu)
                        * omega_metric_sign(lambda)
                        * omega_metric_sign(sigma);

                    component[mu] +=
                        lowering_sign
                        * tensor::epsilon(mu, nu, lambda, sigma)
                        * p1.Get(nu)
                        * p2.Get(lambda)
                        * p0.Get(sigma);
                }
            }
        }
    }

    return FV(component[0], component[1], component[2], component[3]);
}

// f^(rho)_ij = B1(Q_omega,rho(ij),k) * BWR_rho(s_ij)
//              * B1(Q_rho(ij),i,j).
__device__ inline DeviceComplex rho_isobar_factor(
    double s_omega,
    double s_pair,
    double s_bachelor,
    double s_first,
    double s_second,
    const RhoBWRParameters& rho)
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

__device__ inline OmegaDecayCurrent build_omega_decay_current(
    const FV& p0,
    const FV& p1,
    const FV& p2,
    const RhoBWRParameters& rho = RhoBWRParameters())
{
    const FV p_omega = p0 + p1 + p2;
    const FV p12 = p1 + p2;
    const FV p10 = p1 + p0;
    const FV p20 = p2 + p0;

    const double s_omega = p_omega * p_omega;
    const double s0 = p0 * p0;
    const double s1 = p1 * p1;
    const double s2 = p2 * p2;
    const double s12 = p12 * p12;
    const double s10 = p10 * p10;
    const double s20 = p20 * p20;

    // rho(12) with bachelor p0
    const DeviceComplex f12 = rho_isobar_factor(
        s_omega, s12, s0, s1, s2, rho);

    // rho(10) with bachelor p2
    const DeviceComplex f10 = rho_isobar_factor(
        s_omega, s10, s2, s1, s0, rho);

    // rho(20) with bachelor p1
    const DeviceComplex f20 = rho_isobar_factor(
        s_omega, s20, s1, s2, s0, rho);

    return OmegaDecayCurrent(
        omega_geometric_current(p0, p1, p2),
        f12 + f10 + f20);
}

#endif // CTPWA_PROCESS_KINEMATICS_CUH

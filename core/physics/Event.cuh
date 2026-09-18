#pragma once

#include "core/math/Tensor.cuh"
#include "core/physics/OmegaDecay.cuh"

// gVV-specific omega->3pi event currents. Scalar subdecay dynamics and nominal
// particle constants live in OmegaDecayModel.cuh so the width integration can
// use the same implementation.

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
    const double s12 = p12 * p12;
    const double s10 = p10 * p10;
    const double s20 = p20 * p20;

    return OmegaDecayCurrent(
        omega_geometric_current(p0, p1, p2),
        coherent_omega_rho_factor(
            s_omega, s12, s10, s20, rho));
}

// Complete device event and barrier-parameter views consumed by every gVV
// Wave. Input ROOT branch details are isolated in Sample.

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

// Complete process event used by every registered GVV Wave.  The pion order
// is pi0, pi+, pi- for each omega, matching Event.cuh.
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

// Every input four-vector is stored contiguously as [event][px,py,pz,E].
struct GVVDeviceMomenta {
    const double* pip1;
    const double* pim1;
    const double* pi01;
    const double* pip2;
    const double* pim2;
    const double* pi02;
    const double* gamma;

    __host__ GVVDeviceMomenta(
        const double* p_pip1 = nullptr,
        const double* p_pim1 = nullptr,
        const double* p_pi01 = nullptr,
        const double* p_pip2 = nullptr,
        const double* p_pim2 = nullptr,
        const double* p_pi02 = nullptr,
        const double* p_gamma = nullptr)
        : pip1(p_pip1), pim1(p_pim1), pi01(p_pi01),
          pip2(p_pip2), pim2(p_pim2), pi02(p_pi02), gamma(p_gamma)
    {
    }
};

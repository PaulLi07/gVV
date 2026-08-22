// gVV-specific omega->3pi event currents. Scalar subdecay dynamics and nominal
// particle constants live in OmegaDecayModel.cuh so the width integration can
// use the same implementation.
#ifndef CTPWA_PROCESS_KINEMATICS_CUH
#define CTPWA_PROCESS_KINEMATICS_CUH

#include "framework/tensors/Tensor.cuh"
#include "process/OmegaDecayModel.cuh"

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

#endif // CTPWA_PROCESS_KINEMATICS_CUH

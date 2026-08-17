#ifndef CTPWA_PROCESS_EVENT_CUH
#define CTPWA_PROCESS_EVENT_CUH

#include "process/ProcessKinematics.cuh"

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
// is pi0, pi+, pi- for each omega, matching ProcessKinematics.cuh.
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

#endif // CTPWA_PROCESS_EVENT_CUH

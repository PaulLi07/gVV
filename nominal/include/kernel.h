#ifndef KERNEL_H
#define KERNEL_H

#include "GVVAmplitude.h"
#include "GVVModel.h"
#include "OmegaPropagator.h"

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

// F is flat [event][GVV_NBASIS][GVV_NBASIS].
__global__ void CalGVVFmatrix_device(
    GVVDeviceMomenta momenta,
    double* F_matrix,
    int nevt);

void CalGVVFmatrix(
    GVVDeviceMomenta momenta,
    double* F_matrix,
    int nevt);

__global__ void CalGVVPDF_device(
    GVVDeviceMomenta momenta,
    const GVVResonanceParameters* resonances,
    const GVVTermSpec* terms,
    const DeviceComplex* couplings,
    GVVWidthTableView omega_width_table,
    const double* F_matrix,
    double* amp2,
    int nevt);

void CalGVVPDF(
    GVVDeviceMomenta momenta,
    const GVVResonanceParameters* resonances,
    const GVVTermSpec* terms,
    const DeviceComplex* couplings,
    GVVWidthTableView omega_width_table,
    const double* F_matrix,
    double* amp2,
    int nevt);

constexpr int GVV_NCOMPONENT_PAIRS =
    GVV_NTERMS * (GVV_NTERMS + 1) / 2;

// Compact upper-triangle ordering used by PostFit:
//   diagonal: |A_i|^2
//   i<j:      2 Re(A_i A_j*) including both ordered F contractions.
__host__ __device__ inline int gvv_component_pair_index(int first, int second)
{
    if (first > second) {
        const int temporary = first;
        first = second;
        second = temporary;
    }
    return first * GVV_NTERMS
           - first * (first - 1) / 2
           + (second - first);
}

__global__ void CalGVVComponentMatrix_device(
    GVVDeviceMomenta momenta,
    const GVVResonanceParameters* resonances,
    const GVVTermSpec* terms,
    const DeviceComplex* couplings,
    GVVWidthTableView omega_width_table,
    const double* F_matrix,
    double* component_matrix,
    int nevt);

void CalGVVComponentMatrix(
    GVVDeviceMomenta momenta,
    const GVVResonanceParameters* resonances,
    const GVVTermSpec* terms,
    const DeviceComplex* couplings,
    GVVWidthTableView omega_width_table,
    const double* F_matrix,
    double* component_matrix,
    int nevt);

#endif // KERNEL_H

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

// F is compact and model-dependent: [event][active wave][active wave].
// active_wave_types maps each dense slot back to the registered GVV wave.
void CalGVVFmatrix(
    GVVDeviceMomenta momenta,
    const int* active_wave_types,
    int number_active_waves,
    double* F_matrix,
    int number_events);

// Process-specific Term construction. The output is the complex dynamical
// coefficient for every [event][term], before Wave contraction. This is the
// boundary a future process (for example GPPP) replaces.
void CalGVVTermCoefficients(
    GVVDeviceMomenta momenta,
    const GVVResonanceParameters* resonances,
    const GVVTermSpec* terms,
    const DeviceComplex* couplings,
    GVVWidthTableView omega_width_table,
    DeviceComplex* coefficients,
    int number_terms,
    int number_events);

// Process-neutral coherent contraction of Term coefficients with the Wave
// Gram matrix. TermSpec supplies only the dense wave slot.
void CalCoherentIntensity(
    const GVVTermSpec* terms,
    const DeviceComplex* coefficients,
    const double* F_matrix,
    double* intensity,
    int number_terms,
    int number_active_waves,
    int number_events);

void CalGVVPDF(
    GVVDeviceMomenta momenta,
    const GVVResonanceParameters* resonances,
    const GVVTermSpec* terms,
    const DeviceComplex* couplings,
    GVVWidthTableView omega_width_table,
    const double* F_matrix,
    DeviceComplex* coefficient_workspace,
    double* intensity,
    int number_terms,
    int number_active_waves,
    int number_events);

__host__ __device__ inline int gvv_number_component_pairs(int number_terms)
{
    return number_terms * (number_terms + 1) / 2;
}

// Removed with the legacy fixed-layout PostFit migration.
constexpr int GVV_NCOMPONENT_PAIRS =
    GVV_NTERMS * (GVV_NTERMS + 1) / 2;

// Compact upper-triangle ordering:
//   diagonal: |A_i|^2
//   i<j:      2 Re(A_i A_j*) including both ordered F contractions.
__host__ __device__ inline int gvv_component_pair_index(
    int first,
    int second,
    int number_terms)
{
    if (first > second) {
        const int temporary = first;
        first = second;
        second = temporary;
    }
    return first * number_terms
           - first * (first - 1) / 2
           + (second - first);
}

__host__ __device__ inline int gvv_component_pair_index(
    int first,
    int second)
{
    return gvv_component_pair_index(first, second, GVV_NTERMS);
}

void CalGVVComponentMatrix(
    GVVDeviceMomenta momenta,
    const GVVResonanceParameters* resonances,
    const GVVTermSpec* terms,
    const DeviceComplex* couplings,
    GVVWidthTableView omega_width_table,
    const double* F_matrix,
    DeviceComplex* coefficient_workspace,
    double* component_matrix,
    int number_terms,
    int number_active_waves,
    int number_events);

#endif // KERNEL_H

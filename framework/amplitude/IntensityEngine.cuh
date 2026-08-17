// Process-neutral coherent-intensity algebra. The process supplies dense Term
// descriptors, complex coefficients, and the event Wave Gram matrix.
#ifndef CTPWA_FRAMEWORK_AMPLITUDE_INTENSITY_ENGINE_CUH
#define CTPWA_FRAMEWORK_AMPLITUDE_INTENSITY_ENGINE_CUH

#include "framework/math/DeviceComplex.cuh"

namespace ctpwa {

__host__ __device__ inline int component_pair_count(int number_terms)
{
    return number_terms * (number_terms + 1) / 2;
}

// Compact upper-triangle ordering used for diagonal components and pairwise
// interference.  This utility is independent of the active decay process.
__host__ __device__ inline int component_pair_index(
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

template <typename TermSpec>
__device__ inline double coherent_intensity(
    const TermSpec* terms,
    const DeviceComplex* coefficients,
    const double* wave_matrix,
    int number_terms,
    int number_active_waves)
{
    double value = 0.0;
    for (int first = 0; first < number_terms; ++first) {
        for (int second = 0; second < number_terms; ++second) {
            const double wave_factor = wave_matrix[
                terms[first].wave_slot * number_active_waves
                + terms[second].wave_slot];
            value += (
                coefficients[first]
                * coefficients[second].conjugate()
                * wave_factor).real;
        }
    }
    if (value < 0.0 && value > -1.0e-10) {
        value = 0.0;
    }
    return value;
}

} // namespace ctpwa

#endif // CTPWA_FRAMEWORK_AMPLITUDE_INTENSITY_ENGINE_CUH

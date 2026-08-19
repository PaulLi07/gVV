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

// Reference implementation retained for numerical closure tests and
// Term-level component decomposition. The Fit hot path uses the Wave-level
// overload below after all Term-specific propagators have been evaluated.
template <typename TermSpec>
__host__ __device__ inline double coherent_intensity_from_terms(
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

// Terms that share exactly the same complete Wave tensor may be summed after
// their resonance propagators and couplings have been evaluated:
//   B_w = sum_{t: wave(t)=w} C_t,
//   I   = sum_{w,v} Re[B_w B_v^* F_wv].
// This preserves every cross-Wave term while changing the Fit contraction
// from O(T^2) to O(T + W^2).
__host__ __device__ inline double coherent_intensity_from_waves(
    const DeviceComplex* wave_coefficients,
    const double* wave_matrix,
    int number_active_waves)
{
    double value = 0.0;
    for (int first = 0; first < number_active_waves; ++first) {
        for (int second = 0; second < number_active_waves; ++second) {
            value += (
                wave_coefficients[first]
                * wave_coefficients[second].conjugate()
                * wave_matrix[first * number_active_waves + second]).real;
        }
    }
    if (value < 0.0 && value > -1.0e-10) {
        value = 0.0;
    }
    return value;
}

template <typename TermSpec>
__host__ __device__ inline double term_pair_component(
    const TermSpec* terms,
    const DeviceComplex* coefficients,
    const double* wave_matrix,
    int first,
    int second,
    int number_active_waves)
{
    const double forward_wave_factor = wave_matrix[
        terms[first].wave_slot * number_active_waves
        + terms[second].wave_slot];
    double value = (
        coefficients[first]
        * coefficients[second].conjugate()
        * forward_wave_factor).real;
    if (first != second) {
        const double reverse_wave_factor = wave_matrix[
            terms[second].wave_slot * number_active_waves
            + terms[first].wave_slot];
        value += (
            coefficients[second]
            * coefficients[first].conjugate()
            * reverse_wave_factor).real;
    }
    return value;
}

// Keep the original public spelling for small compile-only clients. New code
// should name the representation explicitly through one of the functions
// above.
template <typename TermSpec>
__host__ __device__ inline double coherent_intensity(
    const TermSpec* terms,
    const DeviceComplex* coefficients,
    const double* wave_matrix,
    int number_terms,
    int number_active_waves)
{
    return coherent_intensity_from_terms(
        terms,
        coefficients,
        wave_matrix,
        number_terms,
        number_active_waves);
}

} // namespace ctpwa

#endif // CTPWA_FRAMEWORK_AMPLITUDE_INTENSITY_ENGINE_CUH

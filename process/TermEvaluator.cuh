#ifndef CTPWA_PROCESS_TERM_EVALUATOR_CUH
#define CTPWA_PROCESS_TERM_EVALUATOR_CUH

#include "framework/amplitude/IntensityEngine.cuh"
#include "framework/dynamics/PropagatorRegistry.cuh"
#include "process/OmegaWidthTable.h"
#include "process/ProcessEvent.cuh"
#include "process/WaveRegistry.cuh"

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
    const ResonanceParameters* resonances,
    const TermSpec* terms,
    const DeviceComplex* couplings,
    GVVWidthTableView omega_width_table,
    DeviceComplex* coefficients,
    int number_terms,
    int number_events);

// Process-neutral coherent contraction of Term coefficients with the Wave
// Gram matrix. TermSpec supplies only the dense wave slot.
void CalCoherentIntensity(
    const TermSpec* terms,
    const DeviceComplex* coefficients,
    const double* F_matrix,
    double* intensity,
    int number_terms,
    int number_active_waves,
    int number_events);

void CalGVVPDF(
    GVVDeviceMomenta momenta,
    const ResonanceParameters* resonances,
    const TermSpec* terms,
    const DeviceComplex* couplings,
    GVVWidthTableView omega_width_table,
    const double* F_matrix,
    DeviceComplex* coefficient_workspace,
    double* intensity,
    int number_terms,
    int number_active_waves,
    int number_events);

void CalGVVComponentMatrix(
    GVVDeviceMomenta momenta,
    const ResonanceParameters* resonances,
    const TermSpec* terms,
    const DeviceComplex* couplings,
    GVVWidthTableView omega_width_table,
    const double* F_matrix,
    DeviceComplex* coefficient_workspace,
    double* component_matrix,
    int number_terms,
    int number_active_waves,
    int number_events);

#endif // CTPWA_PROCESS_TERM_EVALUATOR_CUH

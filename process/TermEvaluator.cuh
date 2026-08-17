// CUDA entry points for cached Wave contractions, process Term coefficients,
// total intensity, and optional component decomposition.
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

// These public composite operations validate their dimensions once. Their
// coefficient/contraction kernel stages are private implementation details.
void CalGVVPDF(
    GVVDeviceMomenta momenta,
    const ctpwa::PropagatorParameters* resonances,
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
    const ctpwa::PropagatorParameters* resonances,
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

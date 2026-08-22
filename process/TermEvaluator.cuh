// CUDA entry points for cached Wave contractions, the optimized total-PDF
// path, and the Term-level component path used by Projection and Post.
#ifndef CTPWA_PROCESS_TERM_EVALUATOR_CUH
#define CTPWA_PROCESS_TERM_EVALUATOR_CUH

#include "framework/amplitude/IntensityEngine.cuh"
#include "framework/dynamics/PropagatorRegistry.cuh"
#include "process/OmegaWidthTable.h"
#include "process/ProcessEvent.cuh"
#include "process/ProcessModel.h"

// F is compact and model-dependent: [event][active wave][active wave].
// active_wave_types maps each dense slot back to the registered GVV wave.
void CalGVVFmatrix(
    GVVDeviceMomenta momenta,
    const int* active_wave_types,
    int number_active_waves,
    double* F_matrix,
    int number_events);

// The Fit workspace is [event][active Wave]. Every Term-specific propagator
// is evaluated before coefficients are aggregated by its exact Wave slot.
void CalGVVPDF(
    GVVDeviceMomenta momenta,
    const ctpwa::PropagatorParameters* resonances,
    const TermSpec* terms,
    const DeviceComplex* couplings,
    ctpwa::TabulatedFunctionView omega_width_table,
    const double* F_matrix,
    DeviceComplex* wave_coefficient_workspace,
    double* intensity,
    int number_terms,
    int number_active_waves,
    int number_events);

// Evaluate a contiguous event batch into packed upper-triangle Term-pair
// components. coefficient_workspace is [batch event][Term], while
// packed_components is [batch event][component pair]. F_matrix and momenta
// refer to the complete sample; first_event selects the requested slice.
void CalGVVComponentBatch(
    GVVDeviceMomenta momenta,
    const ctpwa::PropagatorParameters* resonances,
    const TermSpec* terms,
    const DeviceComplex* couplings,
    ctpwa::TabulatedFunctionView omega_width_table,
    const double* F_matrix,
    DeviceComplex* coefficient_workspace,
    double* packed_components,
    int number_terms,
    int number_active_waves,
    int first_event,
    int number_batch_events);

// Integrate all packed Term-pair components without materializing an
// event-by-pair matrix. coefficient_workspace has capacity
// [batch_capacity][Term], and integrated_components has one value per pair.
void CalGVVComponentIntegrals(
    GVVDeviceMomenta momenta,
    const ctpwa::PropagatorParameters* resonances,
    const TermSpec* terms,
    const DeviceComplex* couplings,
    ctpwa::TabulatedFunctionView omega_width_table,
    const double* F_matrix,
    DeviceComplex* coefficient_workspace,
    double* integrated_components,
    int batch_capacity,
    int number_terms,
    int number_active_waves,
    int number_events);

#endif // CTPWA_PROCESS_TERM_EVALUATOR_CUH

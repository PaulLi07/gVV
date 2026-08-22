// CUDA kernels implementing the gVV event -> Term -> coherent intensity path.
// Runtime dimensions come from the compiled model, never nominal model counts.
#include "process/TermEvaluator.cuh"
#include "process/ProcessAmplitude.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdexcept>
#include <string>

namespace {

__device__ FV read_four_vector(const double* values, int event_index)
{
    const int offset = 4 * event_index;
    return FV(
        values[offset + 3],
        values[offset + 0],
        values[offset + 1],
        values[offset + 2]);
}

__device__ GVVEventKinematics read_event(
    const GVVDeviceMomenta& momenta,
    int index)
{
    return GVVEventKinematics(
        read_four_vector(momenta.pi01, index),
        read_four_vector(momenta.pip1, index),
        read_four_vector(momenta.pim1, index),
        read_four_vector(momenta.pi02, index),
        read_four_vector(momenta.pip2, index),
        read_four_vector(momenta.pim2, index),
        read_four_vector(momenta.gamma, index));
}

void check_cuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void require_layout(
    int number_terms,
    int number_active_waves,
    int number_events)
{
    if (number_terms <= 0 || number_active_waves <= 0
        || number_events < 0) {
        throw std::invalid_argument("invalid runtime amplitude dimensions");
    }
}

} // namespace

__global__ void CalGVVFmatrix_device(
    GVVDeviceMomenta momenta,
    const int* active_wave_types,
    int number_active_waves,
    double* F_matrix,
    int number_events)
{
    const int event_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (event_index >= number_events) {
        return;
    }

    const GVVEventKinematics event = read_event(momenta, event_index);
    const int offset =
        event_index * number_active_waves * number_active_waves;
    for (int first_slot = 0;
         first_slot < number_active_waves;
         ++first_slot) {
        for (int second_slot = 0;
             second_slot < number_active_waves;
             ++second_slot) {
            F_matrix[
                offset + first_slot * number_active_waves + second_slot] =
                gvv_cal_F(
                    event,
                    active_wave_types[first_slot],
                    active_wave_types[second_slot]);
        }
    }
}

void CalGVVFmatrix(
    GVVDeviceMomenta momenta,
    const int* active_wave_types,
    int number_active_waves,
    double* F_matrix,
    int number_events)
{
    if (number_active_waves <= 0 || number_events < 0) {
        throw std::invalid_argument("invalid F-matrix dimensions");
    }
    if (number_events == 0) {
        return;
    }
    const int threads = 256;
    const int blocks = (number_events + threads - 1) / threads;
    CalGVVFmatrix_device<<<blocks, threads>>>(
        momenta,
        active_wave_types,
        number_active_waves,
        F_matrix,
        number_events);
    check_cuda(cudaGetLastError(), "launch CalGVVFmatrix_device");
    check_cuda(cudaDeviceSynchronize(), "synchronize CalGVVFmatrix_device");
}

__device__ DeviceComplex gvv_common_omega_factor(
    const GVVEventKinematics& event,
    ctpwa::TabulatedFunctionView omega_width_table)
{
    const double s_omega1 = event.omega1 * event.omega1;
    const double s_omega2 = event.omega2 * event.omega2;
    return event.omega_current1.rho_factor
           * event.omega_current2.rho_factor
           * gvv_omega_propagator(s_omega1, omega_width_table)
           * gvv_omega_propagator(s_omega2, omega_width_table);
}

__device__ DeviceComplex gvv_term_coefficient(
    double s_x,
    const DeviceComplex& common_omega,
    const ctpwa::PropagatorParameters* resonances,
    const TermSpec& term,
    const DeviceComplex& coupling)
{
    return coupling
           * ctpwa::evaluate_propagator(
               s_x,
               resonances[term.resonance_index],
               GVV_OMEGA_MASS,
               GVV_OMEGA_MASS)
           * common_omega;
}

__global__ void CalGVVWaveCoefficients_device(
    GVVDeviceMomenta momenta,
    const ctpwa::PropagatorParameters* resonances,
    const TermSpec* terms,
    const DeviceComplex* couplings,
    ctpwa::TabulatedFunctionView omega_width_table,
    DeviceComplex* wave_coefficients,
    int number_terms,
    int number_active_waves,
    int number_events)
{
    const int event_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (event_index >= number_events) {
        return;
    }

    const GVVEventKinematics event = read_event(momenta, event_index);
    const double s_x = event.X * event.X;
    const DeviceComplex common_omega =
        gvv_common_omega_factor(event, omega_width_table);

    const int output_offset = event_index * number_active_waves;
    for (int wave = 0; wave < number_active_waves; ++wave) {
        wave_coefficients[output_offset + wave] = DeviceComplex(0.0, 0.0);
    }
    for (int term = 0; term < number_terms; ++term) {
        const int wave_slot = terms[term].wave_slot;
        wave_coefficients[output_offset + wave_slot] =
            wave_coefficients[output_offset + wave_slot]
            + gvv_term_coefficient(
                s_x,
                common_omega,
                resonances,
                terms[term],
                couplings[term]);
    }
}

__global__ void CalGVVTermCoefficients_device(
    GVVDeviceMomenta momenta,
    const ctpwa::PropagatorParameters* resonances,
    const TermSpec* terms,
    const DeviceComplex* couplings,
    ctpwa::TabulatedFunctionView omega_width_table,
    DeviceComplex* coefficients,
    int number_terms,
    int first_event,
    int number_batch_events)
{
    const int local_event = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_event >= number_batch_events) {
        return;
    }

    const int event_index = first_event + local_event;
    const GVVEventKinematics event = read_event(momenta, event_index);
    const double s_x = event.X * event.X;
    const DeviceComplex common_omega =
        gvv_common_omega_factor(event, omega_width_table);
    const int output_offset = local_event * number_terms;
    for (int term = 0; term < number_terms; ++term) {
        coefficients[output_offset + term] = gvv_term_coefficient(
            s_x,
            common_omega,
            resonances,
            terms[term],
            couplings[term]);
    }
}

__global__ void CalWaveCoherentIntensity_device(
    const DeviceComplex* wave_coefficients,
    const double* F_matrix,
    double* intensity,
    int number_active_waves,
    int number_events)
{
    const int event_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (event_index >= number_events) {
        return;
    }

    const int coefficient_offset = event_index * number_active_waves;
    const int F_offset =
        event_index * number_active_waves * number_active_waves;
    intensity[event_index] = ctpwa::coherent_intensity_from_waves(
        wave_coefficients + coefficient_offset,
        F_matrix + F_offset,
        number_active_waves);
}

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
    int number_events)
{
    require_layout(number_terms, number_active_waves, number_events);
    if (number_events == 0) {
        return;
    }
    const int threads = 256;
    const int blocks = (number_events + threads - 1) / threads;
    CalGVVWaveCoefficients_device<<<blocks, threads>>>(
        momenta,
        resonances,
        terms,
        couplings,
        omega_width_table,
        wave_coefficient_workspace,
        number_terms,
        number_active_waves,
        number_events);
    check_cuda(cudaGetLastError(), "launch CalGVVWaveCoefficients_device");
    CalWaveCoherentIntensity_device<<<blocks, threads>>>(
        wave_coefficient_workspace,
        F_matrix,
        intensity,
        number_active_waves,
        number_events);
    check_cuda(
        cudaGetLastError(), "launch CalWaveCoherentIntensity_device");
    check_cuda(
        cudaDeviceSynchronize(),
        "synchronize optimized GVV PDF evaluation");
}

__global__ void CalGVVComponentBatch_device(
    const TermSpec* terms,
    const DeviceComplex* coefficients,
    const double* F_matrix,
    double* packed_components,
    int number_terms,
    int number_active_waves,
    int first_event,
    int number_batch_events)
{
    const int local_event = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_event >= number_batch_events) {
        return;
    }

    const int event_index = first_event + local_event;
    const int coefficient_offset = local_event * number_terms;
    const int F_offset =
        event_index * number_active_waves * number_active_waves;
    const int number_pairs = ctpwa::component_pair_count(number_terms);
    const int output_offset = local_event * number_pairs;
    for (int first = 0; first < number_terms; ++first) {
        for (int second = first; second < number_terms; ++second) {
            packed_components[
                output_offset
                + ctpwa::component_pair_index(
                    first, second, number_terms)] =
                ctpwa::term_pair_component(
                    terms,
                    coefficients + coefficient_offset,
                    F_matrix + F_offset,
                    first,
                    second,
                    number_active_waves);
        }
    }
}

__device__ void component_pair_coordinates(
    int pair_index,
    int number_terms,
    int& first,
    int& second)
{
    first = 0;
    int row_size = number_terms;
    while (pair_index >= row_size) {
        pair_index -= row_size;
        ++first;
        --row_size;
    }
    second = first + pair_index;
}

__global__ void CalGVVComponentIntegrals_device(
    const TermSpec* terms,
    const DeviceComplex* coefficients,
    const double* F_matrix,
    double* integrated_components,
    int number_terms,
    int number_active_waves,
    int first_event,
    int number_batch_events)
{
    constexpr int kThreads = 256;
    __shared__ double partial[kThreads];

    const int pair_index = blockIdx.x;
    int first = 0;
    int second = 0;
    component_pair_coordinates(pair_index, number_terms, first, second);

    double local_sum = 0.0;
    for (int local_event = threadIdx.x;
         local_event < number_batch_events;
         local_event += blockDim.x) {
        const int coefficient_offset = local_event * number_terms;
        const int F_offset =
            (first_event + local_event)
            * number_active_waves * number_active_waves;
        local_sum += ctpwa::term_pair_component(
            terms,
            coefficients + coefficient_offset,
            F_matrix + F_offset,
            first,
            second,
            number_active_waves);
    }
    partial[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // One block owns each pair. Batch launches are serialized in the
        // default stream, so no atomic update is required.
        integrated_components[pair_index] += partial[0];
    }
}

static void launch_term_coefficients(
    GVVDeviceMomenta momenta,
    const ctpwa::PropagatorParameters* resonances,
    const TermSpec* terms,
    const DeviceComplex* couplings,
    ctpwa::TabulatedFunctionView omega_width_table,
    DeviceComplex* coefficient_workspace,
    int number_terms,
    int first_event,
    int number_batch_events)
{
    if (number_batch_events == 0) {
        return;
    }
    const int threads = 256;
    const int blocks = (number_batch_events + threads - 1) / threads;
    CalGVVTermCoefficients_device<<<blocks, threads>>>(
        momenta,
        resonances,
        terms,
        couplings,
        omega_width_table,
        coefficient_workspace,
        number_terms,
        first_event,
        number_batch_events);
    check_cuda(cudaGetLastError(), "launch CalGVVTermCoefficients_device");
}

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
    int number_batch_events)
{
    require_layout(number_terms, number_active_waves, number_batch_events);
    if (first_event < 0) {
        throw std::invalid_argument("negative component batch offset");
    }
    if (number_batch_events == 0) {
        return;
    }
    launch_term_coefficients(
        momenta,
        resonances,
        terms,
        couplings,
        omega_width_table,
        coefficient_workspace,
        number_terms,
        first_event,
        number_batch_events);
    const int threads = 256;
    const int blocks = (number_batch_events + threads - 1) / threads;
    CalGVVComponentBatch_device<<<blocks, threads>>>(
        terms,
        coefficient_workspace,
        F_matrix,
        packed_components,
        number_terms,
        number_active_waves,
        first_event,
        number_batch_events);
    check_cuda(cudaGetLastError(), "launch CalGVVComponentBatch_device");
    check_cuda(
        cudaDeviceSynchronize(),
        "synchronize CalGVVComponentBatch_device");
}

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
    int number_events)
{
    require_layout(number_terms, number_active_waves, number_events);
    if (batch_capacity <= 0) {
        throw std::invalid_argument(
            "component integration batch capacity must be positive");
    }
    const int number_pairs = ctpwa::component_pair_count(number_terms);
    check_cuda(
        cudaMemset(
            integrated_components,
            0,
            static_cast<std::size_t>(number_pairs) * sizeof(double)),
        "zero integrated GVV components");
    for (int first_event = 0;
         first_event < number_events;
         first_event += batch_capacity) {
        const int remaining = number_events - first_event;
        const int batch_events =
            remaining < batch_capacity ? remaining : batch_capacity;
        launch_term_coefficients(
            momenta,
            resonances,
            terms,
            couplings,
            omega_width_table,
            coefficient_workspace,
            number_terms,
            first_event,
            batch_events);
        CalGVVComponentIntegrals_device<<<number_pairs, 256>>>(
            terms,
            coefficient_workspace,
            F_matrix,
            integrated_components,
            number_terms,
            number_active_waves,
            first_event,
            batch_events);
        check_cuda(
            cudaGetLastError(),
            "launch CalGVVComponentIntegrals_device");
    }
    check_cuda(
        cudaDeviceSynchronize(),
        "synchronize CalGVVComponentIntegrals_device");
}

#include "process/TermEvaluator.cuh"

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

__global__ void CalGVVTermCoefficients_device(
    GVVDeviceMomenta momenta,
    const ResonanceParameters* resonances,
    const TermSpec* terms,
    const DeviceComplex* couplings,
    GVVWidthTableView omega_width_table,
    DeviceComplex* coefficients,
    int number_terms,
    int number_events)
{
    const int event_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (event_index >= number_events) {
        return;
    }

    const GVVEventKinematics event = read_event(momenta, event_index);
    const double s_x = event.X * event.X;
    const double s_omega1 = event.omega1 * event.omega1;
    const double s_omega2 = event.omega2 * event.omega2;
    const DeviceComplex common_omega =
        event.omega_current1.rho_factor
        * event.omega_current2.rho_factor
        * gvv_omega_propagator(s_omega1, omega_width_table)
        * gvv_omega_propagator(s_omega2, omega_width_table);

    const int output_offset = event_index * number_terms;
    for (int term = 0; term < number_terms; ++term) {
        coefficients[output_offset + term] =
            couplings[term]
            * evaluate_propagator(
                s_x,
                resonances[terms[term].resonance_index],
                GVV_OMEGA_MASS,
                GVV_OMEGA_MASS)
            * common_omega;
    }
}

void CalGVVTermCoefficients(
    GVVDeviceMomenta momenta,
    const ResonanceParameters* resonances,
    const TermSpec* terms,
    const DeviceComplex* couplings,
    GVVWidthTableView omega_width_table,
    DeviceComplex* coefficients,
    int number_terms,
    int number_events)
{
    if (number_terms <= 0 || number_events < 0) {
        throw std::invalid_argument("invalid GVV Term dimensions");
    }
    if (number_events == 0) {
        return;
    }
    const int threads = 256;
    const int blocks = (number_events + threads - 1) / threads;
    CalGVVTermCoefficients_device<<<blocks, threads>>>(
        momenta,
        resonances,
        terms,
        couplings,
        omega_width_table,
        coefficients,
        number_terms,
        number_events);
    check_cuda(cudaGetLastError(), "launch CalGVVTermCoefficients_device");
    check_cuda(
        cudaDeviceSynchronize(),
        "synchronize CalGVVTermCoefficients_device");
}

__global__ void CalCoherentIntensity_device(
    const TermSpec* terms,
    const DeviceComplex* coefficients,
    const double* F_matrix,
    double* intensity,
    int number_terms,
    int number_active_waves,
    int number_events)
{
    const int event_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (event_index >= number_events) {
        return;
    }

    const int coefficient_offset = event_index * number_terms;
    const int F_offset =
        event_index * number_active_waves * number_active_waves;
    intensity[event_index] = ctpwa::coherent_intensity(
        terms,
        coefficients + coefficient_offset,
        F_matrix + F_offset,
        number_terms,
        number_active_waves);
}

void CalCoherentIntensity(
    const TermSpec* terms,
    const DeviceComplex* coefficients,
    const double* F_matrix,
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
    CalCoherentIntensity_device<<<blocks, threads>>>(
        terms,
        coefficients,
        F_matrix,
        intensity,
        number_terms,
        number_active_waves,
        number_events);
    check_cuda(cudaGetLastError(), "launch CalCoherentIntensity_device");
    check_cuda(
        cudaDeviceSynchronize(),
        "synchronize CalCoherentIntensity_device");
}

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
    int number_events)
{
    require_layout(number_terms, number_active_waves, number_events);
    CalGVVTermCoefficients(
        momenta,
        resonances,
        terms,
        couplings,
        omega_width_table,
        coefficient_workspace,
        number_terms,
        number_events);
    CalCoherentIntensity(
        terms,
        coefficient_workspace,
        F_matrix,
        intensity,
        number_terms,
        number_active_waves,
        number_events);
}

__global__ void CalGVVComponentMatrix_device(
    const TermSpec* terms,
    const DeviceComplex* coefficients,
    const double* F_matrix,
    double* component_matrix,
    int number_terms,
    int number_active_waves,
    int number_events)
{
    const int event_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (event_index >= number_events) {
        return;
    }

    const int coefficient_offset = event_index * number_terms;
    const int F_offset =
        event_index * number_active_waves * number_active_waves;
    const int number_pairs = ctpwa::component_pair_count(number_terms);
    const int output_offset = event_index * number_pairs;
    for (int first = 0; first < number_terms; ++first) {
        for (int second = first; second < number_terms; ++second) {
            const DeviceComplex first_coefficient =
                coefficients[coefficient_offset + first];
            const DeviceComplex second_coefficient =
                coefficients[coefficient_offset + second];
            const double forward_F = F_matrix[
                F_offset
                + terms[first].wave_slot * number_active_waves
                + terms[second].wave_slot];
            double contribution = (
                first_coefficient
                * second_coefficient.conjugate()
                * forward_F).real;
            if (first != second) {
                const double reverse_F = F_matrix[
                    F_offset
                    + terms[second].wave_slot * number_active_waves
                    + terms[first].wave_slot];
                contribution += (
                    second_coefficient
                    * first_coefficient.conjugate()
                    * reverse_F).real;
            }
            component_matrix[
                output_offset
                + ctpwa::component_pair_index(
                    first, second, number_terms)] =
                contribution;
        }
    }
}

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
    int number_events)
{
    require_layout(number_terms, number_active_waves, number_events);
    if (number_events == 0) {
        return;
    }
    CalGVVTermCoefficients(
        momenta,
        resonances,
        terms,
        couplings,
        omega_width_table,
        coefficient_workspace,
        number_terms,
        number_events);
    const int threads = 256;
    const int blocks = (number_events + threads - 1) / threads;
    CalGVVComponentMatrix_device<<<blocks, threads>>>(
        terms,
        coefficient_workspace,
        F_matrix,
        component_matrix,
        number_terms,
        number_active_waves,
        number_events);
    check_cuda(cudaGetLastError(), "launch CalGVVComponentMatrix_device");
    check_cuda(
        cudaDeviceSynchronize(),
        "synchronize CalGVVComponentMatrix_device");
}

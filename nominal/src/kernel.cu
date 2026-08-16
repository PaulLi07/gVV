#include "../include/kernel.h"

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

} // namespace

__global__ void CalGVVFmatrix_device(
    GVVDeviceMomenta momenta,
    double* F_matrix,
    int nevt)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nevt) {
        return;
    }

    const GVVEventKinematics event = read_event(momenta, index);
    const int offset = index * GVV_NBASIS * GVV_NBASIS;
    for (int wave1 = 0; wave1 < GVV_NBASIS; ++wave1) {
        for (int wave2 = 0; wave2 < GVV_NBASIS; ++wave2) {
            F_matrix[offset + gvv_F_index(wave1, wave2)] =
                gvv_cal_F(event, wave1, wave2);
        }
    }
}

void CalGVVFmatrix(
    GVVDeviceMomenta momenta,
    double* F_matrix,
    int nevt)
{
    if (nevt <= 0) {
        return;
    }
    const int threads = 256;
    const int blocks = (nevt + threads - 1) / threads;
    CalGVVFmatrix_device<<<blocks, threads>>>(momenta, F_matrix, nevt);
    check_cuda(cudaGetLastError(), "launch CalGVVFmatrix_device");
    check_cuda(cudaDeviceSynchronize(), "synchronize CalGVVFmatrix_device");
}

__global__ void CalGVVPDF_device(
    GVVDeviceMomenta momenta,
    const GVVResonanceParameters* resonances,
    const GVVTermSpec* terms,
    const DeviceComplex* couplings,
    GVVWidthTableView omega_width_table,
    const double* F_matrix,
    double* amp2,
    int nevt)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nevt) {
        return;
    }

    const GVVEventKinematics event = read_event(momenta, index);
    const double s_x = event.X * event.X;
    const double s_omega1 = event.omega1 * event.omega1;
    const double s_omega2 = event.omega2 * event.omega2;
    const DeviceComplex common_omega =
        event.omega_current1.rho_factor
        * event.omega_current2.rho_factor
        * gvv_omega_propagator(s_omega1, omega_width_table)
        * gvv_omega_propagator(s_omega2, omega_width_table);

    DeviceComplex coefficient[GVV_NTERMS];
    for (int term = 0; term < GVV_NTERMS; ++term) {
        const int resonance_index = terms[term].resonance_index;
        coefficient[term] = couplings[term]
                            * gvv_x_propagator(
                                s_x, resonances[resonance_index])
                            * common_omega;
    }

    const int F_offset = index * GVV_NBASIS * GVV_NBASIS;
    double intensity = 0.0;
    for (int i = 0; i < GVV_NTERMS; ++i) {
        for (int j = 0; j < GVV_NTERMS; ++j) {
            const double F = F_matrix[
                F_offset
                + gvv_F_index(terms[i].wave_type, terms[j].wave_type)];
            intensity += (
                coefficient[i] * coefficient[j].conjugate() * F).real;
        }
    }

    // Roundoff can produce a tiny negative number although the contracted
    // polarization sum is non-negative.  Preserve genuine failures as a
    // negative value so the likelihood can reject them explicitly.
    if (intensity < 0.0 && intensity > -1.0e-10) {
        intensity = 0.0;
    }
    amp2[index] = intensity;
}

void CalGVVPDF(
    GVVDeviceMomenta momenta,
    const GVVResonanceParameters* resonances,
    const GVVTermSpec* terms,
    const DeviceComplex* couplings,
    GVVWidthTableView omega_width_table,
    const double* F_matrix,
    double* amp2,
    int nevt)
{
    if (nevt <= 0) {
        return;
    }
    const int threads = 256;
    const int blocks = (nevt + threads - 1) / threads;
    CalGVVPDF_device<<<blocks, threads>>>(
        momenta,
        resonances,
        terms,
        couplings,
        omega_width_table,
        F_matrix,
        amp2,
        nevt);
    check_cuda(cudaGetLastError(), "launch CalGVVPDF_device");
    check_cuda(cudaDeviceSynchronize(), "synchronize CalGVVPDF_device");
}

__global__ void CalGVVComponentMatrix_device(
    GVVDeviceMomenta momenta,
    const GVVResonanceParameters* resonances,
    const GVVTermSpec* terms,
    const DeviceComplex* couplings,
    GVVWidthTableView omega_width_table,
    const double* F_matrix,
    double* component_matrix,
    int nevt)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nevt) {
        return;
    }

    const GVVEventKinematics event = read_event(momenta, index);
    const double s_x = event.X * event.X;
    const double s_omega1 = event.omega1 * event.omega1;
    const double s_omega2 = event.omega2 * event.omega2;
    const DeviceComplex common_omega =
        event.omega_current1.rho_factor
        * event.omega_current2.rho_factor
        * gvv_omega_propagator(s_omega1, omega_width_table)
        * gvv_omega_propagator(s_omega2, omega_width_table);

    DeviceComplex coefficient[GVV_NTERMS];
    for (int term = 0; term < GVV_NTERMS; ++term) {
        coefficient[term] = couplings[term]
                            * gvv_x_propagator(
                                s_x,
                                resonances[terms[term].resonance_index])
                            * common_omega;
    }

    const int F_offset = index * GVV_NBASIS * GVV_NBASIS;
    const int output_offset = index * GVV_NCOMPONENT_PAIRS;
    for (int first = 0; first < GVV_NTERMS; ++first) {
        for (int second = first; second < GVV_NTERMS; ++second) {
            const double forward_F = F_matrix[
                F_offset
                + gvv_F_index(
                    terms[first].wave_type, terms[second].wave_type)];
            double contribution = (
                coefficient[first]
                * coefficient[second].conjugate()
                * forward_F).real;
            if (first != second) {
                const double reverse_F = F_matrix[
                    F_offset
                    + gvv_F_index(
                        terms[second].wave_type, terms[first].wave_type)];
                contribution += (
                    coefficient[second]
                    * coefficient[first].conjugate()
                    * reverse_F).real;
            }
            component_matrix[
                output_offset + gvv_component_pair_index(first, second)] =
                contribution;
        }
    }
}

void CalGVVComponentMatrix(
    GVVDeviceMomenta momenta,
    const GVVResonanceParameters* resonances,
    const GVVTermSpec* terms,
    const DeviceComplex* couplings,
    GVVWidthTableView omega_width_table,
    const double* F_matrix,
    double* component_matrix,
    int nevt)
{
    if (nevt <= 0) {
        return;
    }
    const int threads = 256;
    const int blocks = (nevt + threads - 1) / threads;
    CalGVVComponentMatrix_device<<<blocks, threads>>>(
        momenta,
        resonances,
        terms,
        couplings,
        omega_width_table,
        F_matrix,
        component_matrix,
        nevt);
    check_cuda(cudaGetLastError(), "launch CalGVVComponentMatrix_device");
    check_cuda(
        cudaDeviceSynchronize(),
        "synchronize CalGVVComponentMatrix_device");
}

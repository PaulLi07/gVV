// Runtime GPU closure checks for the optimized Fit intensity and the batched
// Projection/Post component interfaces.
#include "framework/amplitude/IntensityEngine.cuh"
#include "process/ProcessAmplitude.cuh"
#include "process/TermEvaluator.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kEvents = 4;
constexpr int kWaves = 3;
constexpr int kMaximumTerms = 20;

void check_cuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool close_relative(double first, double second, double tolerance = 2.0e-9)
{
    return std::fabs(first - second)
           <= tolerance * std::max({1.0, std::fabs(first), std::fabs(second)});
}

__device__ FV read_p4(const double* values, int event)
{
    const int offset = 4 * event;
    return FV(
        values[offset + 3],
        values[offset + 0],
        values[offset + 1],
        values[offset + 2]);
}

__global__ void term_reference_intensity(
    GVVDeviceMomenta momenta,
    const ctpwa::PropagatorParameters* resonances,
    const TermSpec* terms,
    const DeviceComplex* couplings,
    ctpwa::TabulatedFunctionView omega_width_table,
    const double* F_matrix,
    double* output,
    int number_terms,
    int number_events)
{
    const int event_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (event_index >= number_events) {
        return;
    }
    const GVVEventKinematics event(
        read_p4(momenta.pi01, event_index),
        read_p4(momenta.pip1, event_index),
        read_p4(momenta.pim1, event_index),
        read_p4(momenta.pi02, event_index),
        read_p4(momenta.pip2, event_index),
        read_p4(momenta.pim2, event_index),
        read_p4(momenta.gamma, event_index));
    const double s_x = event.X * event.X;
    const double s_omega1 = event.omega1 * event.omega1;
    const double s_omega2 = event.omega2 * event.omega2;
    const DeviceComplex common_omega =
        event.omega_current1.rho_factor
        * event.omega_current2.rho_factor
        * gvv_omega_propagator(s_omega1, omega_width_table)
        * gvv_omega_propagator(s_omega2, omega_width_table);

    DeviceComplex coefficients[kMaximumTerms];
    for (int term = 0; term < number_terms; ++term) {
        coefficients[term] = couplings[term]
            * ctpwa::evaluate_propagator(
                s_x, resonances[terms[term].resonance_index])
            * common_omega;
    }
    output[event_index] = ctpwa::coherent_intensity_from_terms(
        terms,
        coefficients,
        F_matrix + event_index * kWaves * kWaves,
        number_terms,
        kWaves);
}

struct ManagedBuffers {
    std::array<double*, 7> p4{{nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr}};
    int* wave_types = nullptr;
    double* F = nullptr;
    ctpwa::PropagatorParameters* resonances = nullptr;
    TermSpec* terms = nullptr;
    DeviceComplex* couplings = nullptr;
    double* width_values = nullptr;
    DeviceComplex* wave_workspace = nullptr;
    double* wave_intensity = nullptr;
    double* term_intensity = nullptr;

    ~ManagedBuffers()
    {
        for (double* pointer : p4) {
            if (pointer != nullptr) cudaFree(pointer);
        }
        if (wave_types != nullptr) cudaFree(wave_types);
        if (F != nullptr) cudaFree(F);
        if (resonances != nullptr) cudaFree(resonances);
        if (terms != nullptr) cudaFree(terms);
        if (couplings != nullptr) cudaFree(couplings);
        if (width_values != nullptr) cudaFree(width_values);
        if (wave_workspace != nullptr) cudaFree(wave_workspace);
        if (wave_intensity != nullptr) cudaFree(wave_intensity);
        if (term_intensity != nullptr) cudaFree(term_intensity);
    }
};

void prepare_momenta(ManagedBuffers& buffers)
{
    // [particle][px, py, pz, E], following the process input convention.
    const double base[7][4] = {
        { 0.03,  0.02,  0.16, 0.22},
        { 0.12, -0.03, -0.20, 0.30},
        {-0.15,  0.01,  0.04, 0.27},
        {-0.02,  0.04, -0.15, 0.23},
        {-0.10, -0.02,  0.18, 0.29},
        { 0.12, -0.02, -0.03, 0.28},
        { 0.61, -0.43,  1.92, 2.060097}};
    const double angles[kEvents] = {0.0, 0.37, -0.82, 1.11};
    const double spatial_scales[kEvents] = {1.0, 0.985, 1.018, 1.006};

    for (int particle = 0; particle < 7; ++particle) {
        check_cuda(
            cudaMallocManaged(
                &buffers.p4[particle], kEvents * 4 * sizeof(double)),
            "allocate test momenta");
        const double mass2 = std::max(
            0.0,
            base[particle][3] * base[particle][3]
                - base[particle][0] * base[particle][0]
                - base[particle][1] * base[particle][1]
                - base[particle][2] * base[particle][2]);
        for (int event = 0; event < kEvents; ++event) {
            const double cosine = std::cos(angles[event]);
            const double sine = std::sin(angles[event]);
            const double scale =
                particle == 6 ? 1.0 : spatial_scales[event];
            const double px = scale * (
                cosine * base[particle][0] - sine * base[particle][1]);
            const double py = scale * (
                sine * base[particle][0] + cosine * base[particle][1]);
            const double pz = scale * base[particle][2];
            const int offset = event * 4;
            buffers.p4[particle][offset + 0] = px;
            buffers.p4[particle][offset + 1] = py;
            buffers.p4[particle][offset + 2] = pz;
            buffers.p4[particle][offset + 3] =
                std::sqrt(mass2 + px * px + py * py + pz * pz);
        }
    }
}

GVVDeviceMomenta momenta_view(const ManagedBuffers& buffers)
{
    return GVVDeviceMomenta(
        buffers.p4[1], buffers.p4[2], buffers.p4[0],
        buffers.p4[4], buffers.p4[5], buffers.p4[3], buffers.p4[6]);
}

void prepare_model(ManagedBuffers& buffers)
{
    check_cuda(cudaMallocManaged(
        &buffers.wave_types, kWaves * sizeof(int)), "allocate Wave ids");
    buffers.wave_types[0] = GVV_SCALAR_00;
    buffers.wave_types[1] = GVV_SCALAR_22;
    buffers.wave_types[2] = GVV_PSEUDOSCALAR_11;

    check_cuda(cudaMallocManaged(
        &buffers.resonances,
        4 * sizeof(ctpwa::PropagatorParameters)),
        "allocate test Resonances");
    buffers.resonances[0] = ctpwa::PropagatorParameters();
    buffers.resonances[1] = ctpwa::PropagatorParameters(
        ctpwa::PROP_FIXED_BW, 1.72, 0.14, 0);
    buffers.resonances[2] = ctpwa::PropagatorParameters(
        ctpwa::PROP_TWO_BODY_RUNNING_BW,
        1.78,
        0.18,
        0,
        0.0,
        0.0,
        GVV_OMEGA_MASS,
        GVV_OMEGA_MASS);
    buffers.resonances[3] = ctpwa::PropagatorParameters(
        ctpwa::PROP_TWO_BODY_RUNNING_BW,
        1.84,
        0.20,
        1,
        0.0,
        0.0,
        GVV_OMEGA_MASS,
        GVV_OMEGA_MASS);

    check_cuda(cudaMallocManaged(
        &buffers.terms, kMaximumTerms * sizeof(TermSpec)),
        "allocate test Terms");
    check_cuda(cudaMallocManaged(
        &buffers.couplings, kMaximumTerms * sizeof(DeviceComplex)),
        "allocate test couplings");
    for (int term = 0; term < kMaximumTerms; ++term) {
        buffers.terms[term] = TermSpec{
            term % 4,
            term % kWaves,
            buffers.wave_types[term % kWaves]};
        buffers.couplings[term] = DeviceComplex(
            0.08 * std::cos(0.43 * term),
            0.06 * std::sin(0.71 * term));
    }
    // Terms 0 and 1 deliberately share both dynamics and Wave slot and nearly
    // cancel. This exercises aggregation after the propagator rather than a
    // premature grouping by resonance or quantum-number label.
    buffers.terms[0] = TermSpec{0, 0, GVV_SCALAR_00};
    buffers.terms[1] = TermSpec{0, 0, GVV_SCALAR_00};
    buffers.couplings[0] = DeviceComplex(1.0, 0.2);
    buffers.couplings[1] = DeviceComplex(-0.999999, -0.1999997);

    check_cuda(cudaMallocManaged(
        &buffers.width_values, 2 * sizeof(double)),
        "allocate omega-width fixture");
    buffers.width_values[0] = GVV_OMEGA_WIDTH;
    buffers.width_values[1] = GVV_OMEGA_WIDTH;

    check_cuda(cudaMallocManaged(
        &buffers.F, kEvents * kWaves * kWaves * sizeof(double)),
        "allocate test F matrix");
    check_cuda(cudaMallocManaged(
        &buffers.wave_workspace,
        kEvents * kWaves * sizeof(DeviceComplex)),
        "allocate test Wave workspace");
    check_cuda(cudaMallocManaged(
        &buffers.wave_intensity, kEvents * sizeof(double)),
        "allocate Wave intensity");
    check_cuda(cudaMallocManaged(
        &buffers.term_intensity, kEvents * sizeof(double)),
        "allocate Term-reference intensity");
}

void check_term_count(ManagedBuffers& buffers, int number_terms)
{
    const int number_pairs = ctpwa::component_pair_count(number_terms);
    const ctpwa::TabulatedFunctionView width_table(
        buffers.width_values, 2, 0.0, 10.0);
    const GVVDeviceMomenta momenta = momenta_view(buffers);

    CalGVVPDF(
        momenta,
        buffers.resonances,
        buffers.terms,
        buffers.couplings,
        width_table,
        buffers.F,
        buffers.wave_workspace,
        buffers.wave_intensity,
        number_terms,
        kWaves,
        kEvents);
    term_reference_intensity<<<1, 32>>>(
        momenta,
        buffers.resonances,
        buffers.terms,
        buffers.couplings,
        width_table,
        buffers.F,
        buffers.term_intensity,
        number_terms,
        kEvents);
    check_cuda(cudaGetLastError(), "launch Term-reference intensity");
    check_cuda(cudaDeviceSynchronize(), "run Term-reference intensity");

    DeviceComplex* coefficients = nullptr;
    double* packed = nullptr;
    double* offset_packed = nullptr;
    DeviceComplex* integral_coefficients = nullptr;
    double* integrated = nullptr;
    check_cuda(cudaMallocManaged(
        &coefficients, kEvents * number_terms * sizeof(DeviceComplex)),
        "allocate component coefficients");
    check_cuda(cudaMallocManaged(
        &packed, kEvents * number_pairs * sizeof(double)),
        "allocate packed components");
    check_cuda(cudaMallocManaged(
        &offset_packed, (kEvents - 1) * number_pairs * sizeof(double)),
        "allocate offset packed components");
    check_cuda(cudaMallocManaged(
        &integral_coefficients, 2 * number_terms * sizeof(DeviceComplex)),
        "allocate integral coefficients");
    check_cuda(cudaMallocManaged(
        &integrated, number_pairs * sizeof(double)),
        "allocate integrated components");

    CalGVVComponentBatch(
        momenta,
        buffers.resonances,
        buffers.terms,
        buffers.couplings,
        width_table,
        buffers.F,
        coefficients,
        packed,
        number_terms,
        kWaves,
        0,
        kEvents);
    CalGVVComponentBatch(
        momenta,
        buffers.resonances,
        buffers.terms,
        buffers.couplings,
        width_table,
        buffers.F,
        coefficients,
        offset_packed,
        number_terms,
        kWaves,
        1,
        kEvents - 1);
    CalGVVComponentIntegrals(
        momenta,
        buffers.resonances,
        buffers.terms,
        buffers.couplings,
        width_table,
        buffers.F,
        integral_coefficients,
        integrated,
        2,
        number_terms,
        kWaves,
        kEvents);

    std::vector<double> host_integrated(number_pairs, 0.0);
    for (int event = 0; event < kEvents; ++event) {
        double component_sum = 0.0;
        for (int pair = 0; pair < number_pairs; ++pair) {
            const double value = packed[event * number_pairs + pair];
            component_sum += value;
            host_integrated[pair] += value;
            if (event > 0) {
                require(
                    close_relative(
                        value,
                        offset_packed[(event - 1) * number_pairs + pair]),
                    "component batch depends on its first-event offset");
            }
        }
        require(
            close_relative(
                buffers.term_intensity[event],
                buffers.wave_intensity[event]),
            "Term and Wave intensity contractions disagree");
        require(
            close_relative(component_sum, buffers.term_intensity[event]),
            "packed Term components do not close to total intensity");
    }
    for (int pair = 0; pair < number_pairs; ++pair) {
        require(
            close_relative(host_integrated[pair], integrated[pair]),
            "GPU component reduction disagrees with host integration");
    }

    check_cuda(cudaFree(coefficients), "free component coefficients");
    check_cuda(cudaFree(packed), "free packed components");
    check_cuda(cudaFree(offset_packed), "free offset packed components");
    check_cuda(cudaFree(integral_coefficients), "free integral coefficients");
    check_cuda(cudaFree(integrated), "free integrated components");
}

} // namespace

int main()
{
    try {
        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "query CUDA devices");
        require(device_count > 0, "no CUDA device is available");

        ManagedBuffers buffers;
        prepare_momenta(buffers);
        prepare_model(buffers);
        CalGVVFmatrix(
            momenta_view(buffers),
            buffers.wave_types,
            kWaves,
            buffers.F,
            kEvents);

        for (int number_terms : {1, 7, 20}) {
            check_term_count(buffers, number_terms);
        }
        std::cout << "GVV Term/Wave/component GPU equivalence tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << "GVV intensity-equivalence GPU test failed: "
                  << error.what() << '\n';
        return 1;
    }
    return 0;
}

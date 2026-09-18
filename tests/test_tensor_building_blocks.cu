// Runtime CUDA regression for process-neutral Lorentz tensor building blocks.
// It is separate from `make check` because FV and tensor algebra is device-only.
#include "core/math/OrbitalTensor.cuh"
#include "core/math/TensorOps.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct TensorChecks {
    double named_contraction_error;
    double dwave_rest_frame_error;
    double spin2_reference_error;
    double spin2_rest_frame_error;
    double spin2_symmetry_error;
    double spin2_transversality_error;
    double spin2_trace_error;
    double spin2_idempotence_error;
    double gwave_reference_error;
    double gwave_symmetry_error;
    double gwave_transversality_error;
    double gwave_trace_error;
    double gwave_spin2_error;
    double gwave_even_error;
    double gwave_scaling_error;
    double gwave_m0_error;
    double gwave_m2_error;
    int all_finite;
};

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

__device__ double vector_scale(const FV& value)
{
    double result = 0.0;
    for (int mu = 0; mu < 4; ++mu) {
        result = fmax(result, fabs(value.Get(mu)));
    }
    return result;
}

__device__ double tensor_scale(const tensor& value)
{
    double result = 0.0;
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            result = fmax(result, fabs(value._matrix[mu][nu]));
        }
    }
    return result;
}

__device__ double relative_tensor_error(
    const tensor& value,
    const tensor& reference)
{
    double difference = 0.0;
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            difference = fmax(
                difference,
                fabs(value._matrix[mu][nu] - reference._matrix[mu][nu]));
        }
    }
    return difference / fmax(
        1.0, fmax(tensor_scale(value), tensor_scale(reference)));
}

__device__ double symmetry_error(const tensor& value)
{
    double difference = 0.0;
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            difference = fmax(
                difference,
                fabs(value._matrix[mu][nu] - value._matrix[nu][mu]));
        }
    }
    return difference / fmax(1.0, tensor_scale(value));
}

__device__ double transversality_error(
    const tensor& value,
    const FV& parent)
{
    return vector_scale(value * parent)
           / fmax(1.0, 4.0 * tensor_scale(value) * vector_scale(parent));
}

// Independent explicit P^(2) reference. Both source indices are lowered by
// metric_sign before contraction with the rank-four projector.
__device__ tensor explicit_spin2_project(
    const FV& parent,
    const tensor& source)
{
    const tensor transverse_metric = ctpwa::transverse_projector(parent);
    tensor result;
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            for (int rho = 0; rho < 4; ++rho) {
                for (int sigma = 0; sigma < 4; ++sigma) {
                    const double projector_component = 0.5
                        * (transverse_metric._matrix[mu][rho]
                               * transverse_metric._matrix[nu][sigma]
                           + transverse_metric._matrix[mu][sigma]
                               * transverse_metric._matrix[nu][rho])
                        - transverse_metric._matrix[mu][nu]
                              * transverse_metric._matrix[rho][sigma] / 3.0;
                    result._matrix[mu][nu] += projector_component
                        * ctpwa::metric_sign(rho)
                        * ctpwa::metric_sign(sigma)
                        * source._matrix[rho][sigma];
                }
            }
        }
    }
    return result;
}

__device__ double explicit_gwave_component(
    const tensor& transverse_metric,
    const FV& orbital,
    double orbital2,
    int mu,
    int nu,
    int lambda,
    int tau)
{
    const double a_mu = orbital.Get(mu);
    const double a_nu = orbital.Get(nu);
    const double a_lambda = orbital.Get(lambda);
    const double a_tau = orbital.Get(tau);

    const double one_trace =
        transverse_metric._matrix[mu][nu] * a_lambda * a_tau
        + transverse_metric._matrix[mu][lambda] * a_nu * a_tau
        + transverse_metric._matrix[mu][tau] * a_nu * a_lambda
        + transverse_metric._matrix[nu][lambda] * a_mu * a_tau
        + transverse_metric._matrix[nu][tau] * a_mu * a_lambda
        + transverse_metric._matrix[lambda][tau] * a_mu * a_nu;
    const double two_traces =
        transverse_metric._matrix[mu][nu]
            * transverse_metric._matrix[lambda][tau]
        + transverse_metric._matrix[mu][lambda]
            * transverse_metric._matrix[nu][tau]
        + transverse_metric._matrix[mu][tau]
            * transverse_metric._matrix[nu][lambda];

    return a_mu * a_nu * a_lambda * a_tau
           - orbital2 * one_trace / 7.0
           + orbital2 * orbital2 * two_traces / 35.0;
}

// Explicit 4^4 construction is intentionally confined to this one-thread
// reference test. Production code uses the algebraically reduced O(4^2) path.
__device__ tensor explicit_gwave_contract(
    const FV& parent,
    const FV& relative_momentum,
    const tensor& source)
{
    const tensor transverse_metric = ctpwa::transverse_projector(parent);
    const FV orbital = transverse_metric * relative_momentum;
    const double orbital2 = orbital * orbital;
    tensor result;
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            for (int lambda = 0; lambda < 4; ++lambda) {
                for (int tau = 0; tau < 4; ++tau) {
                    result._matrix[mu][nu] += explicit_gwave_component(
                        transverse_metric,
                        orbital,
                        orbital2,
                        mu,
                        nu,
                        lambda,
                        tau)
                        * ctpwa::metric_sign(lambda)
                        * ctpwa::metric_sign(tau)
                        * source._matrix[lambda][tau];
                }
            }
        }
    }
    return result;
}

__global__ void evaluate_tensor_checks(TensorChecks* output)
{
    // Lock the metric sign in every named contraction with a sparse fixture.
    tensor first;
    tensor second;
    first._matrix[0][1] = 5.0;
    first._matrix[1][1] = 2.0;
    second._matrix[0][1] = 7.0;
    second._matrix[1][1] = 3.0;
    second._matrix[2][1] = 3.0;
    const tensor one_index = ctpwa::contract_second_indices(first, second);
    const FV vector(0.0, 4.0, 0.0, 0.0);
    const FV one_vector = ctpwa::contract_second_index(first, vector);
    output->named_contraction_error = fmax(
        fabs(one_index._matrix[1][2] + 6.0),
        fmax(
            fabs(one_vector.Get(1) + 8.0),
            fmax(
                fabs(ctpwa::lorentz_trace(first) + 2.0),
                fabs(ctpwa::double_contract(first, second) + 29.0))));

    // A non-rest parent and nonsymmetric source exercise every projector term.
    const FV parent(3.2, 0.43, -0.31, 0.72);
    const FV relative(0.27, 0.81, -0.54, 0.39);
    const FV source_left(0.34, -0.28, 0.67, 0.41);
    const FV source_right(0.19, 0.73, -0.16, 0.52);
    const FV source_extra(-0.11, 0.26, 0.37, -0.64);
    const tensor source = tensor(source_left, source_right)
                          + tensor(source_extra, source_left);

    const tensor spin2 = ctpwa::spin2_project(parent, source);
    const tensor spin2_reference = explicit_spin2_project(parent, source);
    output->spin2_reference_error = relative_tensor_error(
        spin2, spin2_reference);
    output->spin2_symmetry_error = symmetry_error(spin2);
    output->spin2_transversality_error = transversality_error(spin2, parent);
    output->spin2_trace_error = fabs(ctpwa::lorentz_trace(spin2))
                                / fmax(1.0, tensor_scale(spin2));
    output->spin2_idempotence_error = relative_tensor_error(
        ctpwa::spin2_project(parent, spin2), spin2);

    const FV rest_parent(2.4, 0.0, 0.0, 0.0);
    const FV x_axis(0.0, 1.0, 0.0, 0.0);
    const tensor rest_spin2 = ctpwa::spin2_project(
        rest_parent, tensor(x_axis, x_axis));
    output->spin2_rest_frame_error = 0.0;
    const double expected_rest[4] = {0.0, 2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0};
    for (int index = 0; index < 4; ++index) {
        output->spin2_rest_frame_error = fmax(
            output->spin2_rest_frame_error,
            fabs(rest_spin2._matrix[index][index] - expected_rest[index]));
    }

    const FV z_axis(0.0, 0.0, 0.0, 1.0);
    const tensor rest_dwave = ctpwa::orbital_dwave(rest_parent, z_axis);
    const double expected_dwave[4] = {
        0.0, -1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0};
    output->dwave_rest_frame_error = 0.0;
    for (int index = 0; index < 4; ++index) {
        output->dwave_rest_frame_error = fmax(
            output->dwave_rest_frame_error,
            fabs(rest_dwave._matrix[index][index]
                 - expected_dwave[index]));
    }

    const tensor gwave = ctpwa::contract_orbital_gwave(
        parent, relative, source);
    const tensor gwave_reference = explicit_gwave_contract(
        parent, relative, source);
    output->gwave_reference_error = relative_tensor_error(
        gwave, gwave_reference);
    output->gwave_symmetry_error = symmetry_error(gwave);
    output->gwave_transversality_error = transversality_error(gwave, parent);
    output->gwave_trace_error = fabs(ctpwa::lorentz_trace(gwave))
                                / fmax(1.0, tensor_scale(gwave));
    output->gwave_spin2_error = relative_tensor_error(
        ctpwa::spin2_project(parent, gwave), gwave);
    output->gwave_even_error = relative_tensor_error(
        ctpwa::contract_orbital_gwave(
            parent, relative * -1.0, source),
        gwave);

    constexpr double scale = 1.7;
    output->gwave_scaling_error = relative_tensor_error(
        ctpwa::contract_orbital_gwave(
            parent, relative * scale, source),
        gwave * (scale * scale * scale * scale));

    // Rest-frame magnetic probes provide simple analytic G-wave contractions.
    constexpr double q = 0.63;
    const FV z_orbital(0.0, 0.0, 0.0, q);
    tensor spin_m0;
    spin_m0._matrix[1][1] = -1.0 / sqrt(6.0);
    spin_m0._matrix[2][2] = -1.0 / sqrt(6.0);
    spin_m0._matrix[3][3] = 2.0 / sqrt(6.0);
    tensor spin_m2;
    spin_m2._matrix[1][1] = 1.0 / sqrt(2.0);
    spin_m2._matrix[2][2] = -1.0 / sqrt(2.0);
    const double q4 = q * q * q * q;
    output->gwave_m0_error = relative_tensor_error(
        ctpwa::contract_orbital_gwave(
            rest_parent, z_orbital, spin_m0),
        spin_m0 * (12.0 * q4 / 35.0));
    output->gwave_m2_error = relative_tensor_error(
        ctpwa::contract_orbital_gwave(
            rest_parent, z_orbital, spin_m2),
        spin_m2 * (2.0 * q4 / 35.0));

    const double values[] = {
        output->named_contraction_error,
        output->dwave_rest_frame_error,
        output->spin2_reference_error,
        output->spin2_rest_frame_error,
        output->spin2_symmetry_error,
        output->spin2_transversality_error,
        output->spin2_trace_error,
        output->spin2_idempotence_error,
        output->gwave_reference_error,
        output->gwave_symmetry_error,
        output->gwave_transversality_error,
        output->gwave_trace_error,
        output->gwave_spin2_error,
        output->gwave_even_error,
        output->gwave_scaling_error,
        output->gwave_m0_error,
        output->gwave_m2_error};
    output->all_finite = 1;
    for (double value : values) {
        if (!isfinite(value)) {
            output->all_finite = 0;
        }
    }
}

} // namespace

int main()
{
    try {
        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "query CUDA devices");
        require(device_count > 0, "no CUDA device is available");

        TensorChecks* checks = nullptr;
        check_cuda(
            cudaMallocManaged(&checks, sizeof(TensorChecks)),
            "allocate tensor-check output");
        evaluate_tensor_checks<<<1, 1>>>(checks);
        check_cuda(cudaGetLastError(), "launch tensor-building-block test");
        check_cuda(cudaDeviceSynchronize(), "run tensor-building-block test");

        constexpr double tolerance = 2.0e-10;
        require(checks->all_finite != 0, "tensor checks produced a non-finite value");
        require(checks->named_contraction_error < tolerance,
                "named Lorentz contraction has the wrong metric sign");
        require(checks->dwave_rest_frame_error < tolerance,
                "bare D-wave has the wrong rest-frame STF form");
        require(checks->spin2_reference_error < tolerance,
                "spin-two projection differs from the explicit projector");
        require(checks->spin2_rest_frame_error < tolerance,
                "spin-two projection has the wrong rest-frame STF form");
        require(checks->spin2_symmetry_error < tolerance,
                "spin-two projection is not symmetric");
        require(checks->spin2_transversality_error < tolerance,
                "spin-two projection is not transverse");
        require(checks->spin2_trace_error < tolerance,
                "spin-two projection is not traceless");
        require(checks->spin2_idempotence_error < tolerance,
                "spin-two projector is not idempotent");
        require(checks->gwave_reference_error < tolerance,
                "G-wave contraction differs from the explicit rank-four reference");
        require(checks->gwave_symmetry_error < tolerance,
                "G-wave contraction is not symmetric");
        require(checks->gwave_transversality_error < tolerance,
                "G-wave contraction is not transverse");
        require(checks->gwave_trace_error < tolerance,
                "G-wave contraction is not traceless");
        require(checks->gwave_spin2_error < tolerance,
                "G-wave contraction needs a redundant outer spin-two projector");
        require(checks->gwave_even_error < tolerance,
                "G-wave contraction is not even under r -> -r");
        require(checks->gwave_scaling_error < tolerance,
                "G-wave contraction does not scale as r^4");
        require(checks->gwave_m0_error < tolerance,
                "G-wave m=0 rest-frame reference is wrong");
        require(checks->gwave_m2_error < tolerance,
                "G-wave m=2 rest-frame reference is wrong");

        check_cuda(cudaFree(checks), "free tensor-check output");
        std::cout << "Tensor building-block GPU tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << "Tensor building-block GPU test failed: "
                  << error.what() << '\n';
        return 1;
    }
    return 0;
}

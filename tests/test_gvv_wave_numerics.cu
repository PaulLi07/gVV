// Runtime GPU checks for the complete registered gVV Wave tensors. This test
// is intentionally separate from login-node compile checks.
#include "framework/tensors/OrbitalTensor.cuh"
#include "framework/tensors/SpinProjector.cuh"
#include "process/ProcessAmplitude.cuh"
#include "process/WaveRegistry.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct FixtureMomenta {
    FV pi01;
    FV pip1;
    FV pim1;
    FV pi02;
    FV pip2;
    FV pim2;
    FV gamma;
};

struct WaveNumerics {
    double F[GVV_NBASIS * GVV_NBASIS];
    double rotation_error;
    double bose_error;
    double omega_current_transversality;
    double projector_transversality;
    double p_wave_transversality;
    double d_wave_transversality;
    double d_wave_trace;
    double photon_transversality;
    int all_finite;
};

void check_cuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__device__ double square(double value)
{
    return value * value;
}

__device__ FV boost_to_frame(const FV& input, const FV& frame)
{
    const double bx = frame.Get(1) / frame.Get(0);
    const double by = frame.Get(2) / frame.Get(0);
    const double bz = frame.Get(3) / frame.Get(0);
    const double beta2 = bx * bx + by * by + bz * bz;
    const double gamma = 1.0 / sqrt(1.0 - beta2);
    const double beta_dot_p =
        bx * input.Get(1) + by * input.Get(2) + bz * input.Get(3);
    const double spatial_factor =
        beta2 > 0.0
            ? (gamma - 1.0) * beta_dot_p / beta2
              + gamma * input.Get(0)
            : 0.0;
    return FV(
        gamma * (input.Get(0) + beta_dot_p),
        input.Get(1) + spatial_factor * bx,
        input.Get(2) + spatial_factor * by,
        input.Get(3) + spatial_factor * bz);
}

__device__ FV rotate_z(const FV& input, double angle)
{
    const double cosine = cos(angle);
    const double sine = sin(angle);
    return FV(
        input.Get(0),
        cosine * input.Get(1) - sine * input.Get(2),
        sine * input.Get(1) + cosine * input.Get(2),
        input.Get(3));
}

__device__ FixtureMomenta rotate_z(
    const FixtureMomenta& input,
    double angle)
{
    FixtureMomenta result;
    result.pi01 = rotate_z(input.pi01, angle);
    result.pip1 = rotate_z(input.pip1, angle);
    result.pim1 = rotate_z(input.pim1, angle);
    result.pi02 = rotate_z(input.pi02, angle);
    result.pip2 = rotate_z(input.pip2, angle);
    result.pim2 = rotate_z(input.pim2, angle);
    result.gamma = rotate_z(input.gamma, angle);
    return result;
}

__device__ FixtureMomenta make_physical_fixture()
{
    constexpr double psi_mass = 3.686097;
    constexpr double x_mass = 2.0;
    constexpr double pion0_mass = 0.1349768;
    constexpr double charged_pion_mass = 0.13957039;
    constexpr double pion_momentum = 0.2213676740364851;
    constexpr double sqrt_three_over_two = 0.8660254037844386;

    const double photon_momentum =
        (psi_mass * psi_mass - x_mass * x_mass) / (2.0 * psi_mass);
    const double gamma_x = 0.31;
    const double gamma_y = -0.27;
    const double gamma_z = sqrt(1.0 - gamma_x * gamma_x - gamma_y * gamma_y);
    const FV x_in_psi(
        sqrt(x_mass * x_mass + photon_momentum * photon_momentum),
        -photon_momentum * gamma_x,
        -photon_momentum * gamma_y,
        -photon_momentum * gamma_z);
    const double omega_momentum = sqrt(
        x_mass * x_mass / 4.0 - GVV_OMEGA_MASS * GVV_OMEGA_MASS);
    const double dx = 0.45;
    const double dy = 0.35;
    const double dz = sqrt(1.0 - dx * dx - dy * dy);
    const FV omega1_in_x(
        x_mass / 2.0,
        omega_momentum * dx,
        omega_momentum * dy,
        omega_momentum * dz);
    const FV omega2_in_x(
        x_mass / 2.0,
        -omega_momentum * dx,
        -omega_momentum * dy,
        -omega_momentum * dz);

    const double neutral_energy = sqrt(
        square(pion_momentum) + square(pion0_mass));
    const double charged_energy = sqrt(
        square(pion_momentum) + square(charged_pion_mass));
    const FV pi01_rest(
        neutral_energy, pion_momentum, 0.0, 0.0);
    const FV pip1_rest(
        charged_energy,
        -0.5 * pion_momentum,
        sqrt_three_over_two * pion_momentum,
        0.0);
    const FV pim1_rest(
        charged_energy,
        -0.5 * pion_momentum,
        -sqrt_three_over_two * pion_momentum,
        0.0);

    // Use a different decay-plane orientation for the second omega while
    // keeping its three-body rest-frame momentum sum exactly zero.
    const FV pi02_rest(
        neutral_energy, 0.0, pion_momentum, 0.0);
    const FV pip2_rest(
        charged_energy,
        0.0,
        -0.5 * pion_momentum,
        sqrt_three_over_two * pion_momentum);
    const FV pim2_rest(
        charged_energy,
        0.0,
        -0.5 * pion_momentum,
        -sqrt_three_over_two * pion_momentum);

    FixtureMomenta result;
    result.pi01 = boost_to_frame(
        boost_to_frame(pi01_rest, omega1_in_x), x_in_psi);
    result.pip1 = boost_to_frame(
        boost_to_frame(pip1_rest, omega1_in_x), x_in_psi);
    result.pim1 = boost_to_frame(
        boost_to_frame(pim1_rest, omega1_in_x), x_in_psi);
    result.pi02 = boost_to_frame(
        boost_to_frame(pi02_rest, omega2_in_x), x_in_psi);
    result.pip2 = boost_to_frame(
        boost_to_frame(pip2_rest, omega2_in_x), x_in_psi);
    result.pim2 = boost_to_frame(
        boost_to_frame(pim2_rest, omega2_in_x), x_in_psi);
    result.gamma = FV(
        photon_momentum,
        photon_momentum * gamma_x,
        photon_momentum * gamma_y,
        photon_momentum * gamma_z);
    return result;
}

__device__ GVVEventKinematics make_event(const FixtureMomenta& p)
{
    return GVVEventKinematics(
        p.pi01, p.pip1, p.pim1,
        p.pi02, p.pip2, p.pim2, p.gamma);
}

__device__ GVVEventKinematics make_swapped_event(const FixtureMomenta& p)
{
    return GVVEventKinematics(
        p.pi02, p.pip2, p.pim2,
        p.pi01, p.pip1, p.pim1, p.gamma);
}

__device__ double fv_scale(const FV& value)
{
    double scale = 0.0;
    for (int component = 0; component < 4; ++component) {
        scale = fmax(scale, fabs(value.Get(component)));
    }
    return scale;
}

__device__ double tensor_scale(const tensor& value)
{
    double scale = 0.0;
    for (int first = 0; first < 4; ++first) {
        for (int second = 0; second < 4; ++second) {
            scale = fmax(scale, fabs(value._matrix[first][second]));
        }
    }
    return scale;
}

__device__ double relative_vector_scale(
    const FV& residual,
    const tensor& source,
    const FV& vector)
{
    const double denominator = fmax(
        1.0,
        4.0 * tensor_scale(source) * fv_scale(vector));
    return fv_scale(residual) / denominator;
}

__device__ double relative_tensor_difference(
    const tensor& first,
    const tensor& second)
{
    double difference = 0.0;
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            difference = fmax(
                difference,
                fabs(first._matrix[row][column]
                     - second._matrix[row][column]));
        }
    }
    return difference / fmax(
        1.0, fmax(tensor_scale(first), tensor_scale(second)));
}

__global__ void evaluate_wave_numerics(WaveNumerics* output)
{
    const FixtureMomenta momenta = make_physical_fixture();
    const GVVEventKinematics event = make_event(momenta);
    const GVVEventKinematics swapped = make_swapped_event(momenta);
    const GVVEventKinematics rotated = make_event(rotate_z(momenta, 0.71));

    output->all_finite = 1;
    output->rotation_error = 0.0;
    output->bose_error = 0.0;
    for (int first = 0; first < GVV_NBASIS; ++first) {
        const tensor wave = gvv_wave_tensor(event, first);
        const tensor swapped_wave = gvv_wave_tensor(swapped, first);
        output->bose_error = fmax(
            output->bose_error,
            relative_tensor_difference(wave, swapped_wave));
        for (int second = 0; second < GVV_NBASIS; ++second) {
            const int index = first * GVV_NBASIS + second;
            output->F[index] = gvv_cal_F(event, first, second);
            const double rotated_F = gvv_cal_F(rotated, first, second);
            output->rotation_error = fmax(
                output->rotation_error,
                fabs(output->F[index] - rotated_F)
                    / fmax(1.0, fabs(output->F[index])));
            if (!isfinite(output->F[index]) || !isfinite(rotated_F)) {
                output->all_finite = 0;
            }
        }
        if (!isfinite(tensor_scale(wave))) {
            output->all_finite = 0;
        }
    }

    const double current1_scale = fmax(
        1.0,
        4.0 * fv_scale(event.omega_current1.geometry)
            * fv_scale(event.omega1));
    const double current2_scale = fmax(
        1.0,
        4.0 * fv_scale(event.omega_current2.geometry)
            * fv_scale(event.omega2));
    output->omega_current_transversality = fmax(
        fabs(event.omega_current1.geometry * event.omega1) / current1_scale,
        fabs(event.omega_current2.geometry * event.omega2) / current2_scale);

    const tensor projector = ctpwa::transverse_projector(event.X);
    output->projector_transversality = relative_vector_scale(
        projector * event.X, projector, event.X);
    const FV p_wave = ctpwa::orbital_pwave(
        event.X, event.relative_omega_momentum);
    output->p_wave_transversality = fabs(p_wave * event.X) / fmax(
        1.0, 4.0 * fv_scale(p_wave) * fv_scale(event.X));
    const tensor d_wave = ctpwa::orbital_dwave(
        event.X, event.relative_omega_momentum);
    output->d_wave_transversality = relative_vector_scale(
        d_wave * event.X, d_wave, event.X);
    double d_trace = 0.0;
    for (int index = 0; index < 4; ++index) {
        d_trace += ctpwa::metric_sign(index) * d_wave._matrix[index][index];
    }
    output->d_wave_trace = fabs(d_trace) / fmax(1.0, tensor_scale(d_wave));

    const tensor photon_projector =
        gvv_photon_projector(event.psi, event.gamma);
    output->photon_transversality = relative_vector_scale(
        photon_projector * event.gamma,
        photon_projector,
        event.gamma);
}

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

} // namespace

int main()
{
    try {
        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "query CUDA devices");
        require(device_count > 0, "no CUDA device is available");

        WaveNumerics* values = nullptr;
        check_cuda(
            cudaMallocManaged(&values, sizeof(WaveNumerics)),
            "allocate Wave numerical output");
        evaluate_wave_numerics<<<1, 1>>>(values);
        check_cuda(cudaGetLastError(), "launch Wave numerical test");
        check_cuda(cudaDeviceSynchronize(), "run Wave numerical test");

        constexpr double tolerance = 2.0e-10;
        require(values->all_finite != 0, "Wave evaluation produced non-finite values");
        require(values->rotation_error < tolerance,
                "Wave Gram matrix is not invariant under a z rotation");
        require(values->bose_error < tolerance,
                "complete Wave tensor violates omega Bose symmetry");
        require(values->omega_current_transversality < tolerance,
                "omega geometric current is not transverse");
        require(values->projector_transversality < tolerance,
                "spin projector is not transverse");
        require(values->p_wave_transversality < tolerance,
                "P-wave orbital tensor is not transverse");
        require(values->d_wave_transversality < tolerance,
                "D-wave orbital tensor is not transverse");
        require(values->d_wave_trace < tolerance,
                "D-wave orbital tensor is not traceless");
        require(values->photon_transversality < tolerance,
                "photon projector is not transverse");

        for (int first = 0; first < GVV_NBASIS; ++first) {
            require(
                values->F[first * GVV_NBASIS + first] >= -tolerance,
                "Wave Gram matrix has a negative diagonal");
            for (int second = 0; second < GVV_NBASIS; ++second) {
                const double forward =
                    values->F[first * GVV_NBASIS + second];
                const double reverse =
                    values->F[second * GVV_NBASIS + first];
                require(
                    std::fabs(forward - reverse)
                        <= tolerance * std::max(1.0, std::fabs(forward)),
                    "Wave Gram matrix is not symmetric");
            }
        }
        double matrix_scale = 1.0;
        for (double value : values->F) {
            matrix_scale = std::max(matrix_scale, std::fabs(value));
        }
        for (int wave = 0; wave < GVV_NBASIS; ++wave) {
            require(
                values->F[wave * GVV_NBASIS + wave]
                    > 1.0e-14 * matrix_scale,
                "registered Wave is identically zero on the generic fixture");
        }
        for (int first = 0; first < GVV_NBASIS; ++first) {
            for (int second = first + 1;
                 second < GVV_NBASIS;
                 ++second) {
                const double principal_minor =
                    values->F[first * GVV_NBASIS + first]
                    * values->F[second * GVV_NBASIS + second]
                    - values->F[first * GVV_NBASIS + second]
                      * values->F[second * GVV_NBASIS + first];
                require(
                    principal_minor >=
                        -tolerance * matrix_scale * matrix_scale,
                    "Wave Gram matrix has a negative second-order principal minor");
            }
        }
        const double* F = values->F;
        const double determinant =
            F[0] * (F[4] * F[8] - F[5] * F[7])
            - F[1] * (F[3] * F[8] - F[5] * F[6])
            + F[2] * (F[3] * F[7] - F[4] * F[6]);
        require(
            determinant >=
                -tolerance * matrix_scale * matrix_scale * matrix_scale,
            "Wave Gram matrix has a negative determinant");
        for (int scalar = GVV_SCALAR_00;
             scalar <= GVV_SCALAR_22;
             ++scalar) {
            require(
                std::fabs(values->F[
                    scalar * GVV_NBASIS + GVV_PSEUDOSCALAR_11])
                    <= tolerance * matrix_scale,
                "scalar and pseudoscalar coherence classes are not orthogonal");
        }

        check_cuda(cudaFree(values), "free Wave numerical output");
        std::cout << "GVV complete-Wave GPU numerical tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << "GVV complete-Wave GPU test failed: "
                  << error.what() << '\n';
        return 1;
    }
    return 0;
}

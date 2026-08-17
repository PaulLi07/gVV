// Device compile test for all registered gVV Waves and coherent contraction.
#include "framework/amplitude/IntensityEngine.cuh"
#include "process/WaveRegistry.cuh"

#include <iostream>

// Compiling this kernel validates the complete registered-Wave and generic
// coherent-intensity device path without launching a GPU on the login node.
__global__ void compile_gvv_amplitude_path(double* output)
{
    const FV pi01(0.22, 0.03, 0.02, 0.16);
    const FV pip1(0.30, 0.12, -0.03, -0.20);
    const FV pim1(0.27, -0.15, 0.01, 0.04);
    const FV pi02(0.23, -0.02, 0.04, -0.15);
    const FV pip2(0.29, -0.10, -0.02, 0.18);
    const FV pim2(0.28, 0.12, -0.02, -0.03);
    const FV gamma(2.06, 0.0, 0.0, 2.06);
    const GVVEventKinematics event(
        pi01, pip1, pim1, pi02, pip2, pim2, gamma);

    double wave_matrix[GVV_NBASIS * GVV_NBASIS];
    for (int first = 0; first < GVV_NBASIS; ++first) {
        for (int second = 0; second < GVV_NBASIS; ++second) {
            wave_matrix[first * GVV_NBASIS + second] =
                gvv_cal_F(event, first, second);
        }
    }

    TermSpec terms[GVV_NBASIS] = {
        {0, GVV_SCALAR_00, GVV_SCALAR_00},
        {1, GVV_SCALAR_22, GVV_SCALAR_22},
        {2, GVV_PSEUDOSCALAR_11, GVV_PSEUDOSCALAR_11}};
    DeviceComplex coefficients[GVV_NBASIS] = {
        DeviceComplex(1.0, 0.0),
        DeviceComplex(0.5, 0.2),
        DeviceComplex(0.3, -0.1)};

    output[0] = wave_matrix[
        GVV_SCALAR_00 * GVV_NBASIS + GVV_SCALAR_00];
    output[1] = wave_matrix[
        GVV_SCALAR_22 * GVV_NBASIS + GVV_SCALAR_22];
    output[2] = wave_matrix[
        GVV_PSEUDOSCALAR_11 * GVV_NBASIS + GVV_PSEUDOSCALAR_11];
    output[3] = ctpwa::coherent_intensity(
        terms,
        coefficients,
        wave_matrix,
        GVV_NBASIS,
        GVV_NBASIS);
}

int main()
{
    std::cout << "GVV registered-Wave amplitude path compiled\n";
    return 0;
}

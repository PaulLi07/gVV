#include "../include/GVVAmplitude.h"

#include <iostream>

// Compiling this kernel makes nvcc validate the complete GVV device path.
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
        pi01, pip1, pim1,
        pi02, pip2, pim2,
        gamma);

    double F_matrix[GVV_NBASIS * GVV_NBASIS];
    for (int wave1 = 0; wave1 < GVV_NBASIS; ++wave1) {
        for (int wave2 = 0; wave2 < GVV_NBASIS; ++wave2) {
            F_matrix[gvv_F_index(wave1, wave2)] =
                gvv_cal_F(event, wave1, wave2);
        }
    }

    const GVVEventPropagators omega_propagators(
        DeviceComplex(1.0, 0.0),
        DeviceComplex(1.0, 0.0));
    GVVAmplitudeTerm terms[GVV_NBASIS] = {
        GVVAmplitudeTerm(GVV_SCALAR_00,
                         DeviceComplex(1.0, 0.0),
                         DeviceComplex(1.0, 0.0)),
        GVVAmplitudeTerm(GVV_SCALAR_22,
                         DeviceComplex(0.5, 0.2),
                         DeviceComplex(1.0, 0.0)),
        GVVAmplitudeTerm(GVV_PSEUDOSCALAR_11,
                         DeviceComplex(0.3, -0.1),
                         DeviceComplex(1.0, 0.0))
    };

    output[0] = F_matrix[gvv_F_index(GVV_SCALAR_00, GVV_SCALAR_00)];
    output[1] = F_matrix[gvv_F_index(GVV_SCALAR_22, GVV_SCALAR_22)];
    output[2] = F_matrix[gvv_F_index(
        GVV_PSEUDOSCALAR_11, GVV_PSEUDOSCALAR_11)];
    output[3] = gvv_cross_section(
        event, omega_propagators, terms, GVV_NBASIS, F_matrix);
}

int main()
{
    std::cout << "GVV amplitude device path compiled\n";
    return 0;
}

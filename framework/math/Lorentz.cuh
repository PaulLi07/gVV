#ifndef CTPWA_FRAMEWORK_MATH_LORENTZ_CUH
#define CTPWA_FRAMEWORK_MATH_LORENTZ_CUH

namespace ctpwa {

__host__ __device__ inline double metric_sign(int index)
{
    return index == 0 ? 1.0 : -1.0;
}

// Levi-Civita convention used throughout CTPWA: epsilon^{0123}=+1.
__host__ __device__ inline double levi_civita(
    int mu,
    int nu,
    int rho,
    int sigma)
{
    const int values[4] = {mu, nu, rho, sigma};
    for (int first = 0; first < 4; ++first) {
        if (values[first] < 0 || values[first] > 3) {
            return 0.0;
        }
        for (int second = first + 1; second < 4; ++second) {
            if (values[first] == values[second]) {
                return 0.0;
            }
        }
    }
    int inversions = 0;
    for (int first = 0; first < 4; ++first) {
        for (int second = first + 1; second < 4; ++second) {
            if (values[first] > values[second]) {
                ++inversions;
            }
        }
    }
    return inversions % 2 == 0 ? 1.0 : -1.0;
}

} // namespace ctpwa

#endif // CTPWA_FRAMEWORK_MATH_LORENTZ_CUH

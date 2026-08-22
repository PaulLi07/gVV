// Lightweight host/device view of a uniformly sampled scalar function.
// The owner of the sampled values and the physical extrapolation policy stay
// outside this reusable numerical type.
#ifndef CTPWA_FRAMEWORK_DYNAMICS_TABULATED_FUNCTION_CUH
#define CTPWA_FRAMEWORK_DYNAMICS_TABULATED_FUNCTION_CUH

namespace ctpwa {

struct TabulatedFunctionView {
    const double* values;
    int size;
    double x_min;
    double x_step;

    __host__ __device__ TabulatedFunctionView(
        const double* table = nullptr,
        int table_size = 0,
        double minimum = 0.0,
        double step = 0.0)
        : values(table), size(table_size), x_min(minimum), x_step(step)
    {
    }

    // Preserve the project's existing endpoint-clamping behavior explicitly.
    // A process that needs a different extrapolation policy should apply it at
    // its own line-shape boundary rather than hiding it in this view.
    __host__ __device__ double interpolate_clamped(double x) const
    {
        if (values == nullptr || size <= 0 || x_step <= 0.0) {
            return 0.0;
        }
        if (x <= x_min) {
            return values[0];
        }

        const double coordinate = (x - x_min) / x_step;
        int lower = static_cast<int>(coordinate);
        if (lower >= size - 1) {
            return values[size - 1];
        }
        if (lower < 0) {
            lower = 0;
        }

        const double fraction = coordinate - lower;
        return values[lower] * (1.0 - fraction)
               + values[lower + 1] * fraction;
    }
};

} // namespace ctpwa

#endif // CTPWA_FRAMEWORK_DYNAMICS_TABULATED_FUNCTION_CUH

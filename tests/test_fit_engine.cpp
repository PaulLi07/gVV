#include "framework/fit/FitEngine.h"

#include <cmath>
#include <iostream>
#include <vector>

int main()
{
    ctpwa::FitParameterSpec x;
    x.name = "x";
    x.initial_value = 3.0;
    x.step = 0.05;
    ctpwa::FitParameterSpec y;
    y.name = "y";
    y.initial_value = -2.0;
    y.step = 0.05;

    ctpwa::FitOptions options;
    options.number_starts = 1;
    const ctpwa::FitSummary result = ctpwa::run_multistart_fit(
        {x, y},
        options,
        [](const std::vector<double>& values) {
            const double dx = values[0] - 1.0;
            const double dy = values[1] + 0.5;
            return dx * dx + dy * dy;
        });
    if (!result.best.valid || result.best.values.size() != 2
        || std::fabs(result.best.values[0] - 1.0) > 1.0e-4
        || std::fabs(result.best.values[1] + 0.5) > 1.0e-4
        || result.best.minimum > 1.0e-8) {
        std::cerr << "generic fit engine failed its quadratic test\n";
        return 1;
    }
    std::cout << "Generic fit engine tests passed\n";
    return 0;
}

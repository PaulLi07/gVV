// Unit test for generic MC normalization and signed sample contributions.
#include "framework/likelihood/Likelihood.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

int main()
{
    const double normalization_sample[] = {1.0, 2.0, 3.0, 4.0};
    const double normalization = ctpwa::monte_carlo_normalization(
        normalization_sample, 4);
    if (std::fabs(normalization - 2.5) > 1.0e-14) {
        std::cerr << "Monte-Carlo normalization is wrong\n";
        return 1;
    }
    const double data[] = {2.5, 5.0};
    const double signal = ctpwa::log_likelihood_contribution(
        data, 2, normalization, +1.0);
    const double sideband = ctpwa::log_likelihood_contribution(
        data, 2, normalization, -0.5);
    if (std::fabs(signal - std::log(2.0)) > 1.0e-14
        || std::fabs(sideband + 0.5 * std::log(2.0)) > 1.0e-14) {
        std::cerr << "signed likelihood composition is wrong\n";
        return 2;
    }
    bool rejected = false;
    try {
        const double invalid[] = {0.0};
        (void)ctpwa::log_likelihood_contribution(
            invalid, 1, normalization, 1.0);
    } catch (const std::domain_error&) {
        rejected = true;
    }
    if (!rejected) {
        std::cerr << "invalid PDF was not rejected\n";
        return 3;
    }
    std::cout << "Likelihood tests passed\n";
    return 0;
}

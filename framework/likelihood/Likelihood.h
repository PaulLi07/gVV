// Process-neutral Monte-Carlo normalization and signed unbinned
// log-likelihood arithmetic. Sample loading and intensity evaluation are
// intentionally outside this header.
#ifndef CTPWA_FRAMEWORK_LIKELIHOOD_H
#define CTPWA_FRAMEWORK_LIKELIHOOD_H

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace ctpwa {

// Monte-Carlo normalization shared by decay processes. The process supplies
// only event intensities; this layer owns probability/likelihood semantics.
inline double monte_carlo_normalization(
    const double* intensity,
    std::size_t number_events)
{
    if (intensity == nullptr || number_events == 0) {
        throw std::invalid_argument("normalization sample is empty");
    }
    double sum = 0.0;
    for (std::size_t event = 0; event < number_events; ++event) {
        if (!(intensity[event] >= 0.0) || !std::isfinite(intensity[event])) {
            throw std::domain_error(
                "normalization contains an invalid intensity");
        }
        sum += intensity[event];
    }
    const double result = sum / static_cast<double>(number_events);
    if (!(result > 0.0) || !std::isfinite(result)) {
        throw std::domain_error("normalization is not finite and positive");
    }
    return result;
}

inline double log_likelihood_contribution(
    const double* intensity,
    std::size_t number_events,
    double normalization,
    double coefficient)
{
    if (intensity == nullptr || !std::isfinite(coefficient)
        || !(normalization > 0.0) || !std::isfinite(normalization)) {
        throw std::invalid_argument("invalid likelihood input");
    }
    double logarithm_sum = 0.0;
    for (std::size_t event = 0; event < number_events; ++event) {
        const double pdf = intensity[event] / normalization;
        if (!(pdf > 0.0) || !std::isfinite(pdf)) {
            throw std::domain_error("likelihood contains an invalid PDF");
        }
        logarithm_sum += std::log(pdf);
    }
    return coefficient * logarithm_sum;
}

} // namespace ctpwa

#endif // CTPWA_FRAMEWORK_LIKELIHOOD_H

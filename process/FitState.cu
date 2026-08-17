#include "process/ParameterMapping.h"

#include "process/FitLikelihood.h"

#include <cmath>
#include <ostream>
#include <stdexcept>

void gvv_apply_fit_parameters(
    FitLikelihood& likelihood,
    const std::vector<GVVFitParameterBinding>& layout,
    const std::vector<double>& values)
{
    if (layout.size() != values.size()) {
        throw std::invalid_argument("incorrect number of GVV fit parameters");
    }
    for (std::size_t cursor = 0; cursor < layout.size(); ++cursor) {
        if (!std::isfinite(values[cursor])) {
            throw std::invalid_argument("non-finite GVV fit parameter");
        }
        const GVVFitParameterBinding& parameter = layout[cursor];
        if (parameter.target == GVVFitParameterTarget::CouplingReal) {
            const DeviceComplex current = likelihood.Coupling(
                parameter.target_index);
            likelihood.SetCoupling(
                parameter.target_index, values[cursor], current.imag);
        } else if (
            parameter.target == GVVFitParameterTarget::CouplingImaginary) {
            const DeviceComplex current = likelihood.Coupling(
                parameter.target_index);
            likelihood.SetCoupling(
                parameter.target_index, current.real, values[cursor]);
        } else if (
            parameter.target == GVVFitParameterTarget::CouplingLogMagnitude) {
            likelihood.SetLogCouplingMagnitude(
                parameter.target_index, values[cursor]);
        } else if (
            parameter.target == GVVFitParameterTarget::ResonanceLogSDRatio) {
            likelihood.SetLogSDRatio(
                parameter.target_index, values[cursor]);
        } else {
            likelihood.SetLogFlatteRatio(
                parameter.target_index, values[cursor]);
        }
    }
}

void gvv_write_fit_details(
    std::ostream& output,
    const FitLikelihood& likelihood,
    const std::vector<GVVFitParameterBinding>& layout,
    const ctpwa::FitAttempt& best)
{
    if (layout.size() != best.values.size()
        || layout.size() != best.errors.size()) {
        throw std::invalid_argument("GVV fit detail layout mismatch");
    }
    const GVVCompiledModel& model = likelihood.Model();
    output << "# process-specific physical model state\n";
    std::size_t parameter = 0;
    for (int term = 0; term < likelihood.NumberTerms(); ++term) {
        const GVVTermMetadata& metadata = model.term_metadata[term];
        const int parameterization = metadata.coupling_parameterization;
        if (parameterization == COUPLING_FIXED_SCALE_AND_PHASE) {
            const DeviceComplex value = likelihood.Coupling(term);
            output << "coupling " << metadata.id << ' '
                   << value.real << ' ' << value.imag << " fixed\n";
            continue;
        }
        if (parameterization == COUPLING_POSITIVE_REAL) {
            const double log_magnitude = best.values.at(parameter);
            const double magnitude = std::exp(log_magnitude);
            output << "coupling " << metadata.id << ' '
                   << magnitude << " 0 "
                   << magnitude * best.errors.at(parameter)
                   << " 0 phase_fixed log_rho " << log_magnitude
                   << " log_error " << best.errors.at(parameter) << '\n';
            ++parameter;
            continue;
        }
        output << "coupling " << metadata.id << ' '
               << best.values.at(parameter) << ' '
               << best.values.at(parameter + 1) << ' '
               << best.errors.at(parameter) << ' '
               << best.errors.at(parameter + 1) << '\n';
        parameter += 2;
    }
    for (int resonance = 0;
         resonance < likelihood.NumberResonances();
         ++resonance) {
        const ResonanceParameters& state = likelihood.Resonance(resonance);
        output << "resonance " << model.resonance_metadata[resonance].id
               << " model " << propagator_name(state.propagator_model)
               << " mass " << state.mass << " fixed";
        if (state.propagator_model == PROP_SUBTRACTED_FLATTE) {
            output << " Gamma_rest " << state.pole_width << " fixed";
        } else {
            output << " width " << state.pole_width << " fixed";
        }
        if (state.fit_sd_ratio) {
            output << " r_D_over_S " << state.sd_ratio
                   << " log_error " << best.errors.at(parameter++);
        }
        if (state.propagator_model == PROP_SUBTRACTED_FLATTE) {
            output << " R_omegaomega " << state.flatte_ratio;
            if (state.fit_flatte_ratio) {
                output << " log_error " << best.errors.at(parameter++);
            } else {
                output << " fixed";
            }
        }
        output << '\n';
    }
    if (parameter != best.values.size()) {
        throw std::runtime_error("GVV fit detail parameter count mismatch");
    }
}

#include "process/ParameterMapping.h"

#include <cmath>
#include <stdexcept>

namespace {

ctpwa::FitParameterSpec make_fit_spec(
    const std::string& name,
    double initial,
    double step)
{
    ctpwa::FitParameterSpec result;
    result.name = name;
    result.initial_value = initial;
    result.step = step;
    return result;
}

} // namespace

std::vector<GVVFitParameterBinding> gvv_fit_parameter_layout(
    const GVVCompiledModel& model)
{
    if (model.terms.size() != model.initial_couplings.size()
        || model.terms.size() != model.term_metadata.size()
        || model.resonances.size() != model.resonance_metadata.size()) {
        throw std::invalid_argument("incomplete compiled GVV model layout");
    }

    std::vector<GVVFitParameterBinding> layout;
    for (std::size_t term = 0; term < model.terms.size(); ++term) {
        const GVVTermMetadata& metadata = model.term_metadata[term];
        const DeviceComplex coupling = model.initial_couplings[term];
        if (metadata.coupling_parameterization
            == COUPLING_FIXED_SCALE_AND_PHASE) {
            continue;
        }
        if (metadata.coupling_parameterization
            == COUPLING_POSITIVE_REAL) {
            if (!(coupling.real > 0.0) || coupling.imag != 0.0) {
                throw std::runtime_error(
                    "positive-real coupling '" + metadata.id
                    + "' is invalid");
            }
            GVVFitParameterBinding parameter;
            parameter.fit = make_fit_spec(
                "log_rho_" + metadata.id, std::log(coupling.real), 0.10);
            parameter.fit.randomization =
                ctpwa::ParameterRandomization::LogMagnitude;
            parameter.fit.randomization_group = static_cast<int>(term);
            parameter.target = GVVFitParameterTarget::CouplingLogMagnitude;
            parameter.target_index = static_cast<int>(term);
            layout.push_back(parameter);
            continue;
        }

        GVVFitParameterBinding real;
        real.fit = make_fit_spec(
            "Re_" + metadata.id, coupling.real, 0.05);
        real.fit.randomization =
            ctpwa::ParameterRandomization::ComplexReal;
        real.fit.randomization_group = static_cast<int>(term);
        real.target = GVVFitParameterTarget::CouplingReal;
        real.target_index = static_cast<int>(term);
        layout.push_back(real);

        GVVFitParameterBinding imaginary = real;
        imaginary.fit.name = "Im_" + metadata.id;
        imaginary.fit.initial_value = coupling.imag;
        imaginary.fit.randomization =
            ctpwa::ParameterRandomization::ComplexImaginary;
        imaginary.target = GVVFitParameterTarget::CouplingImaginary;
        layout.push_back(imaginary);
    }

    for (std::size_t resonance = 0;
         resonance < model.resonances.size();
         ++resonance) {
        const ResonanceParameters& values = model.resonances[resonance];
        const GVVResonanceMetadata& metadata =
            model.resonance_metadata[resonance];
        const ctpwa::ResonanceDefinition& definition =
            model.definition.resonance(metadata.id);

        auto append_log_parameter = [&](const std::string& source_name,
                                        const std::string& fit_name,
                                        GVVFitParameterTarget target,
                                        double physical_value) {
            const auto found = definition.parameters.find(source_name);
            if (found == definition.parameters.end() || found->second.fixed
                || found->second.transform != "log") {
                throw std::runtime_error(
                    "compiled fitted parameter '" + source_name
                    + "' is inconsistent for resonance '" + metadata.id
                    + "'");
            }
            if (!(physical_value > 0.0)) {
                throw std::runtime_error(
                    "non-positive fitted parameter for resonance '"
                    + metadata.id + "'");
            }
            GVVFitParameterBinding parameter;
            parameter.fit = make_fit_spec(
                fit_name + metadata.id,
                std::log(physical_value),
                found->second.step);
            parameter.fit.has_lower_bound = found->second.has_lower_bound;
            parameter.fit.has_upper_bound = found->second.has_upper_bound;
            parameter.fit.lower_bound = found->second.lower_bound;
            parameter.fit.upper_bound = found->second.upper_bound;
            parameter.target = target;
            parameter.target_index = static_cast<int>(resonance);
            layout.push_back(parameter);
        };

        if (values.fit_sd_ratio) {
            append_log_parameter(
                "sd_ratio",
                "log_rDS_",
                GVVFitParameterTarget::ResonanceLogSDRatio,
                values.sd_ratio);
        }
        if (values.fit_flatte_ratio) {
            append_log_parameter(
                "omegaomega_ratio",
                "log_Romega_",
                GVVFitParameterTarget::ResonanceLogFlatteRatio,
                values.flatte_ratio);
        }
    }
    return layout;
}

std::vector<ctpwa::FitParameterSpec> gvv_fit_parameter_specs(
    const std::vector<GVVFitParameterBinding>& layout)
{
    std::vector<ctpwa::FitParameterSpec> result;
    result.reserve(layout.size());
    for (const GVVFitParameterBinding& parameter : layout) {
        result.push_back(parameter.fit);
    }
    return result;
}

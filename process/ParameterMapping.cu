// Complete translation layer between compiled gVV model state and the flat
// Minuit vector: layout construction, state application, and TXT details.
#include "process/ParameterMapping.h"

#include <algorithm>
#include <cmath>
#include <ostream>
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

double positive_from_log(double value, const char* name)
{
    const double physical = std::exp(value);
    if (!(physical > 0.0) || !std::isfinite(physical)) {
        throw std::invalid_argument(
            std::string(name) + " is outside the numerical range");
    }
    return physical;
}

} // namespace

std::vector<GVVFitParameterBinding> gvv_fit_parameter_layout(
    const GVVCompiledModel& model)
{
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
        const ctpwa::PropagatorParameters& values = model.resonances[resonance];
        const GVVResonanceMetadata& metadata =
            model.resonance_metadata[resonance];
        const ctpwa::ResonanceDefinition& definition =
            model.definition.resonance(metadata.id);

        auto append_log_parameter = [&](const std::string& source_name,
                                        const std::string& fit_name,
                                        GVVFitParameterTarget target,
                                        double physical_value) {
            // WaveRegistry has already validated the process contract and
            // marked only free log-transformed parameters as fitted.
            const ctpwa::ParameterDefinition& source =
                definition.parameters.at(source_name);
            GVVFitParameterBinding parameter;
            parameter.fit = make_fit_spec(
                fit_name + metadata.id,
                std::log(physical_value),
                source.step);
            parameter.fit.has_lower_bound = source.has_lower_bound;
            parameter.fit.has_upper_bound = source.has_upper_bound;
            parameter.fit.lower_bound = source.lower_bound;
            parameter.fit.upper_bound = source.upper_bound;
            parameter.target = target;
            parameter.target_index = static_cast<int>(resonance);
            layout.push_back(parameter);
        };

        if (metadata.fit_sd_ratio) {
            append_log_parameter(
                "sd_ratio",
                "log_rDS_",
                GVVFitParameterTarget::ResonanceLogSDRatio,
                values.sd_ratio);
        }
        if (metadata.fit_flatte_ratio) {
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

void gvv_apply_fit_parameters(
    GVVCompiledModel& model,
    const std::vector<GVVFitParameterBinding>& layout,
    const std::vector<double>& values)
{
    if (layout.size() != values.size()) {
        throw std::invalid_argument("incorrect number of GVV fit parameters");
    }
    for (std::size_t cursor = 0; cursor < layout.size(); ++cursor) {
        const GVVFitParameterBinding& parameter = layout[cursor];
        if (parameter.target == GVVFitParameterTarget::CouplingReal) {
            model.initial_couplings[parameter.target_index].real =
                values[cursor];
        } else if (
            parameter.target == GVVFitParameterTarget::CouplingImaginary) {
            model.initial_couplings[parameter.target_index].imag =
                values[cursor];
        } else if (
            parameter.target == GVVFitParameterTarget::CouplingLogMagnitude) {
            model.initial_couplings[parameter.target_index] = DeviceComplex(
                positive_from_log(values[cursor], "log coupling magnitude"),
                0.0);
        } else if (
            parameter.target == GVVFitParameterTarget::ResonanceLogSDRatio) {
            model.resonances[parameter.target_index].sd_ratio =
                positive_from_log(values[cursor], "log S/D ratio");
        } else {
            model.resonances[parameter.target_index].flatte_ratio =
                positive_from_log(
                    values[cursor], "log Flatte omega-omega ratio");
        }
    }
}

void gvv_write_fit_details(
    std::ostream& output,
    const GVVCompiledModel& model,
    const std::vector<GVVFitParameterBinding>& layout,
    const ctpwa::FitAttempt& best)
{
    if (layout.size() != best.values.size()
        || layout.size() != best.errors.size()) {
        throw std::invalid_argument("GVV fit detail layout mismatch");
    }
    output << "model: " << model.definition.name << '\n'
           << "active_resonances: " << model.resonances.size() << '\n'
           << "active_terms: " << model.terms.size() << '\n'
           << "active_waves: " << model.active_wave_types.size() << "\n\n"
           << "# WAVE REGISTRY USED BY THE ACTIVE MODEL\n"
           << "# wave_id JPC coherence_class latex\n";
    for (const GVVWaveMetadata& wave : gvv_wave_registry()) {
        if (std::find(
                model.active_wave_types.begin(),
                model.active_wave_types.end(),
                wave.wave_type) != model.active_wave_types.end()) {
            output << "wave " << wave.id << ' ' << wave.jpc << ' '
                   << wave.coherence_class << ' ' << wave.latex << '\n';
        }
    }

    output << "\n# ACTIVE TERMS AND COUPLINGS\n"
           << "# term id label wave resonance JPC coherence coupling policy\n";
    std::size_t parameter = 0;
    for (std::size_t term = 0; term < model.terms.size(); ++term) {
        const GVVTermMetadata& metadata = model.term_metadata[term];
        const GVVResonanceMetadata& resonance = model.resonance_metadata[
            model.terms[term].resonance_index];
        const int parameterization = metadata.coupling_parameterization;
        output << "term " << metadata.id << " label=\"" << metadata.label
               << "\" wave=" << metadata.wave_id
               << " resonance=" << resonance.id
               << " JPC=" << metadata.jpc
               << " coherence=" << metadata.coherence_class
               << " coupling_mode=";
        if (parameterization == COUPLING_FIXED_SCALE_AND_PHASE) {
            const DeviceComplex value = model.initial_couplings[term];
            output << "fixed_complex reference="
                   << ctpwa::coupling_reference_name(metadata.reference)
                   << " value=(" << value.real << ',' << value.imag
                   << ") status=fixed_reference\n";
            continue;
        }
        if (parameterization == COUPLING_POSITIVE_REAL) {
            const double log_magnitude = best.values.at(parameter);
            const double magnitude = std::exp(log_magnitude);
            output << "positive_real reference="
                   << ctpwa::coupling_reference_name(metadata.reference)
                   << " value=(" << magnitude << ",0)"
                   << " magnitude_error="
                   << magnitude * best.errors.at(parameter)
                   << " fitted_as=" << layout.at(parameter).fit.name
                   << " fitted_value=" << log_magnitude
                   << " fitted_error=" << best.errors.at(parameter) << '\n';
            ++parameter;
            continue;
        }
        output << "complex_cartesian reference=none value=("
               << best.values.at(parameter) << ','
               << best.values.at(parameter + 1) << ')'
               << " error=(" << best.errors.at(parameter) << ','
               << best.errors.at(parameter + 1) << ')'
               << " fitted_as=(" << layout.at(parameter).fit.name << ','
               << layout.at(parameter + 1).fit.name << ")\n";
        parameter += 2;
    }

    output << "\n# ACTIVE RESONANCES\n"
           << "# Values are the final physical values. The source status is "
              "taken from model.json.\n";
    for (std::size_t resonance = 0;
         resonance < model.resonances.size();
         ++resonance) {
        const ctpwa::PropagatorParameters& state = model.resonances[resonance];
        const GVVResonanceMetadata& metadata =
            model.resonance_metadata[resonance];
        output << "resonance " << metadata.id << " label=\""
               << metadata.label << "\" propagator=" << metadata.propagator_id
               << " compiled_model=\""
               << ctpwa::propagator_name(state.propagator_model) << "\"\n";
        const ctpwa::ResonanceDefinition& definition =
            model.definition.resonance(metadata.id);
        std::vector<std::string> names;
        names.reserve(definition.parameters.size());
        for (const auto& item : definition.parameters) names.push_back(item.first);
        std::sort(names.begin(), names.end());
        for (const std::string& name : names) {
            const ctpwa::ParameterDefinition& source =
                definition.parameters.at(name);
            double value = source.value;
            if (name == "mass") value = state.mass;
            else if (name == "width") value = state.pole_width;
            else if (name == "sd_ratio") value = state.sd_ratio;
            else if (name == "omegaomega_ratio") value = state.flatte_ratio;
            output << "  parameter " << name << " value=" << value
                   << " status=" << (source.fixed ? "fixed" : "free")
                   << " transform=" << source.transform;
            if (source.has_lower_bound) {
                output << " bounds=[" << source.lower_bound << ','
                       << source.upper_bound << ']';
            }
            output << '\n';
        }
        if (metadata.fit_sd_ratio) {
            output << "  fitted_parameter " << layout.at(parameter).fit.name
                   << " value=" << best.values.at(parameter)
                   << " error=" << best.errors.at(parameter) << '\n';
            ++parameter;
        }
        if (metadata.fit_flatte_ratio) {
            output << "  fitted_parameter " << layout.at(parameter).fit.name
                   << " value=" << best.values.at(parameter)
                   << " error=" << best.errors.at(parameter) << '\n';
            ++parameter;
        }
    }
}

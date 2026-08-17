#ifndef MINUIT_FCN_H
#define MINUIT_FCN_H

#include "framework/likelihood/GVVLikelihood.h"
#include "framework/fit/FitParameters.h"

#include "TMinuit.h"

#include <exception>
#include <iostream>

inline void apply_gvv_fit_parameters(
    NLL_estimator& fitter,
    const Double_t* parameters)
{
    const std::vector<GVVFitParameterSpec> layout =
        gvv_fit_parameter_layout(fitter.Model());
    for (std::size_t cursor = 0; cursor < layout.size(); ++cursor) {
        const GVVFitParameterSpec& parameter = layout[cursor];
        if (parameter.target == GVVFitParameterTarget::CouplingReal) {
            const DeviceComplex current = fitter.Coupling(
                parameter.target_index);
            fitter.SetCoupling(
                parameter.target_index, parameters[cursor], current.imag);
        } else if (
            parameter.target == GVVFitParameterTarget::CouplingImaginary) {
            const DeviceComplex current = fitter.Coupling(
                parameter.target_index);
            fitter.SetCoupling(
                parameter.target_index, current.real, parameters[cursor]);
        } else if (
            parameter.target == GVVFitParameterTarget::CouplingLogMagnitude) {
            fitter.SetLogCouplingMagnitude(
                parameter.target_index, parameters[cursor]);
        } else if (
            parameter.target == GVVFitParameterTarget::ResonanceLogSDRatio) {
            fitter.SetLogSDRatio(parameter.target_index, parameters[cursor]);
        } else if (
            parameter.target
            == GVVFitParameterTarget::ResonanceLogFlatteRatio) {
            fitter.SetLogFlatteRatio(
                parameter.target_index, parameters[cursor]);
        }
    }
}

inline void objective_function(
    Int_t& number_parameters,
    Double_t* gradient,
    Double_t& result,
    Double_t* parameters,
    Int_t flag)
{
    (void)gradient;
    (void)flag;
    NLL_estimator* fitter =
        static_cast<NLL_estimator*>(gMinuit->GetObjectFit());
    if (fitter == nullptr
        || number_parameters != fitter->NumberFitParameters()) {
        result = 1.0e100;
        return;
    }

    try {
        apply_gvv_fit_parameters(*fitter, parameters);
        result = -fitter->Cal_log_likelihood();
    } catch (const std::exception& error) {
        std::cerr << "GVV likelihood error: " << error.what() << '\n';
        result = 1.0e100;
    }
}

#endif // MINUIT_FCN_H

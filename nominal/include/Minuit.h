#ifndef MINUIT_FCN_H
#define MINUIT_FCN_H

#include "NLL_estimator.h"

#include "TMinuit.h"

#include <exception>
#include <iostream>

inline void apply_gvv_fit_parameters(
    NLL_estimator& fitter,
    const Double_t* parameters)
{
    int cursor = 0;
    for (int term = 0; term < GVV_NTERMS; ++term) {
        const int parameterization = gvv_coupling_parameterization(term);
        if (parameterization == GVV_COUPLING_FIXED_SCALE_AND_PHASE) {
            continue;
        }
        if (parameterization == GVV_COUPLING_POSITIVE_REAL) {
            fitter.SetLogCouplingMagnitude(term, parameters[cursor]);
            ++cursor;
            continue;
        }
        fitter.SetCoupling(term, parameters[cursor], parameters[cursor + 1]);
        cursor += 2;
    }
    for (int resonance = 0; resonance < GVV_NRESONANCES; ++resonance) {
        if (fitter.Resonance(resonance).fit_sd_ratio) {
            fitter.SetLogSDRatio(resonance, parameters[cursor]);
            ++cursor;
        }
        if (fitter.Resonance(resonance).fit_flatte_ratio) {
            fitter.SetLogFlatteRatio(resonance, parameters[cursor]);
            ++cursor;
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

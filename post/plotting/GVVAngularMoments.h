// Shared numerical construction for Projection angular moments.
// Orders, mass binning, axes, curve styles, canvas layout, annotations, and
// output names belong in the individual even/odd ROOT macros.
#ifndef GVV_ANGULAR_MOMENTS_H
#define GVV_ANGULAR_MOMENTS_H

#include "GVVPlotUtils.h"

#include "TH1D.h"

#include <utility>

namespace gvvplot {

inline double LegendrePolynomial(int order, double x)
{
    if (order == 0) return 1.0;
    if (order == 1) return x;
    double previous = 1.0;
    double current = x;
    for (int degree = 2; degree <= order; ++degree) {
        const double next =
            ((2.0 * degree - 1.0) * x * current
             - (degree - 1.0) * previous)
            / degree;
        previous = current;
        current = next;
    }
    return current;
}

inline double MomentWeight(int order, double cosine, bool odd_diagnostic)
{
    if (odd_diagnostic) return LegendrePolynomial(order, cosine);
    return 0.5
           * (LegendrePolynomial(order, cosine)
              + LegendrePolynomial(order, -cosine));
}

inline std::pair<double, int> MomentChiSquare(
    const TH1D* data,
    const TH1D* model)
{
    double chi_square = 0.0;
    int bins = 0;
    for (int bin = 1; bin <= data->GetNbinsX(); ++bin) {
        const double variance =
            data->GetBinError(bin) * data->GetBinError(bin)
            + model->GetBinError(bin) * model->GetBinError(bin);
        if (!(variance > 0.0)) continue;
        const double residual =
            data->GetBinContent(bin) - model->GetBinContent(bin);
        chi_square += residual * residual / variance;
        ++bins;
    }
    return std::make_pair(chi_square, bins);
}

struct MomentHistograms {
    TH1D* data = nullptr;
    TH1D* background = nullptr;
    TH1D* model = nullptr;
};

// Build unstyled data-minus-background and fitted-MC moment histograms for one
// Legendre order. The macro owns all presentation choices.
inline MomentHistograms BuildMomentHistograms(
    const ProjectionInput& input,
    int order,
    bool odd_diagnostic,
    int bins,
    double lower,
    double upper,
    const char* name_prefix)
{
    MomentHistograms result;
    result.data = new TH1D(
        Form("%s_data_%d", name_prefix, order), "", bins, lower, upper);
    result.background = new TH1D(
        Form("%s_bg_%d", name_prefix, order), "", bins, lower, upper);
    result.model = new TH1D(
        Form("%s_mc_%d", name_prefix, order), "", bins, lower, upper);
    result.data->SetDirectory(nullptr);
    result.background->SetDirectory(nullptr);
    result.model->SetDirectory(nullptr);
    result.data->Sumw2();
    result.background->Sumw2();
    result.model->Sumw2();

    Branches data_values;
    data_values.Bind(input.data, false, false);
    for (Long64_t event = 0; event < input.data->GetEntries(); ++event) {
        input.data->GetEntry(event);
        result.data->Fill(
            data_values.m_omegaomega,
            MomentWeight(
                order, data_values.cos_theta_omega1, odd_diagnostic));
    }

    Branches background_values;
    background_values.Bind(input.background, false, true);
    for (Long64_t event = 0;
         event < input.background->GetEntries();
         ++event) {
        input.background->GetEntry(event);
        result.background->Fill(
            background_values.m_omegaomega,
            MomentWeight(
                order,
                background_values.cos_theta_omega1,
                odd_diagnostic)
                * background_values.weight_bg);
    }

    Branches mc_values;
    mc_values.Bind(input.mc, true, false);
    for (Long64_t event = 0; event < input.mc->GetEntries(); ++event) {
        input.mc->GetEntry(event);
        result.model->Fill(
            mc_values.m_omegaomega,
            MomentWeight(order, mc_values.cos_theta_omega1, odd_diagnostic)
                * mc_values.weight);
    }

    result.data->Add(result.background, -1.0);
    return result;
}

} // namespace gvvplot

#endif // GVV_ANGULAR_MOMENTS_H

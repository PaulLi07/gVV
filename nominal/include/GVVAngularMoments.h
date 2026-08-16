#ifndef GVV_ANGULAR_MOMENTS_H
#define GVV_ANGULAR_MOMENTS_H

#include "GVVPlotUtils.h"

#include "TLine.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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

inline void DrawAngularMoments(
    const char* input_file,
    const char* output_prefix,
    bool odd_diagnostic)
{
    SetBESIIIStyle();
    std::unique_ptr<TFile> input(TFile::Open(input_file, "READ"));
    if (!input || input->IsZombie()) {
        throw std::runtime_error(
            std::string("cannot open projection file ") + input_file);
    }
    TTree* data = nullptr;
    TTree* mc = nullptr;
    TTree* background = nullptr;
    input->GetObject("data", data);
    input->GetObject("MC", mc);
    input->GetObject("bg", background);
    if (data == nullptr || mc == nullptr || background == nullptr) {
        throw std::runtime_error("projection file is missing data/MC/bg tree");
    }

    const std::vector<int> orders = odd_diagnostic
                                        ? std::vector<int>{1, 3, 5}
                                        : std::vector<int>{0, 2, 4, 6};
    TCanvas* canvas = new TCanvas(
        odd_diagnostic ? "gvv_odd_moments" : "gvv_even_moments",
        "GVV angular moments",
        odd_diagnostic ? 1500 : 1200,
        odd_diagnostic ? 500 : 900);
    canvas->Divide(
        odd_diagnostic ? 3 : 2,
        odd_diagnostic ? 1 : 2,
        0.002,
        0.002);

    for (std::size_t panel = 0; panel < orders.size(); ++panel) {
        canvas->cd(static_cast<int>(panel) + 1);
        const int order = orders[panel];
        TH1D* data_moment = new TH1D(
            Form("gvv_moment_data_%d_%d", odd_diagnostic, order),
            "", 34, 1.50, 3.20);
        TH1D* background_moment = new TH1D(
            Form("gvv_moment_bg_%d_%d", odd_diagnostic, order),
            "", 34, 1.50, 3.20);
        TH1D* mc_moment = new TH1D(
            Form("gvv_moment_mc_%d_%d", odd_diagnostic, order),
            "", 34, 1.50, 3.20);
        data_moment->Sumw2();
        background_moment->Sumw2();
        mc_moment->Sumw2();
        data_moment->SetDirectory(nullptr);
        background_moment->SetDirectory(nullptr);
        mc_moment->SetDirectory(nullptr);

        Branches data_values;
        data_values.Bind(data, false, false);
        for (Long64_t event = 0; event < data->GetEntries(); ++event) {
            data->GetEntry(event);
            const double moment = odd_diagnostic
                                      ? LegendrePolynomial(
                                            order, data_values.cos_theta_omega)
                                      : 0.5
                                            * (LegendrePolynomial(
                                                   order,
                                                   data_values.cos_theta_omega)
                                               + LegendrePolynomial(
                                                   order,
                                                   -data_values.cos_theta_omega));
            data_moment->Fill(data_values.m_omegaomega, moment);
        }

        Branches background_values;
        background_values.Bind(background, false, true);
        for (Long64_t event = 0; event < background->GetEntries(); ++event) {
            background->GetEntry(event);
            const double moment = odd_diagnostic
                                      ? LegendrePolynomial(
                                            order,
                                            background_values.cos_theta_omega)
                                      : 0.5
                                            * (LegendrePolynomial(
                                                   order,
                                                   background_values.cos_theta_omega)
                                               + LegendrePolynomial(
                                                   order,
                                                   -background_values.cos_theta_omega));
            background_moment->Fill(
                background_values.m_omegaomega,
                moment * background_values.weight_bg);
        }

        Branches mc_values;
        mc_values.Bind(mc, true, false);
        for (Long64_t event = 0; event < mc->GetEntries(); ++event) {
            mc->GetEntry(event);
            const double moment = odd_diagnostic
                                      ? LegendrePolynomial(
                                            order, mc_values.cos_theta_omega)
                                      : 0.5
                                            * (LegendrePolynomial(
                                                   order,
                                                   mc_values.cos_theta_omega)
                                               + LegendrePolynomial(
                                                   order,
                                                   -mc_values.cos_theta_omega));
            mc_moment->Fill(
                mc_values.m_omegaomega,
                moment * mc_values.weight);
        }

        data_moment->Add(background_moment, -1.0);
        data_moment->SetMarkerStyle(8);
        data_moment->SetMarkerSize(0.70);
        data_moment->SetLineColor(kBlack);
        data_moment->SetLineWidth(1);
        mc_moment->SetLineColor(kBlue + 1);
        mc_moment->SetLineWidth(2);
        data_moment->GetXaxis()->SetTitle(
            "M(#omega#omega) (GeV/#font[12]{c}^{2})");
        data_moment->GetYaxis()->SetTitle(
            Form("#LT P_{%d}(cos#theta_{#omega}) #GT / 50 MeV", order));
        data_moment->GetXaxis()->CenterTitle(kTRUE);
        data_moment->GetYaxis()->CenterTitle(kTRUE);
        const double maximum = std::max(
            data_moment->GetMaximum(), mc_moment->GetMaximum());
        const double minimum = std::min(
            data_moment->GetMinimum(), mc_moment->GetMinimum());
        data_moment->GetYaxis()->SetRangeUser(
            minimum < 0.0 ? 1.35 * minimum : 0.0,
            maximum > 0.0 ? 1.35 * maximum : 1.0);
        data_moment->Draw("E1");
        mc_moment->Draw("HIST SAME");
        data_moment->Draw("E1 SAME");

        const std::pair<double, int> chi_square =
            MomentChiSquare(data_moment, mc_moment);
        TLatex label;
        label.SetNDC();
        label.SetTextFont(22);
        label.SetTextSize(0.050);
        label.DrawLatex(
            0.18, 0.84,
            Form("P_{%d}: #chi^{2}/N_{bin}=%.1f/%d",
                 order, chi_square.first, chi_square.second));
        if (odd_diagnostic) {
            label.SetTextSize(0.040);
            label.DrawLatex(0.18, 0.76, "ordered-#omega diagnostic only");
        }
    }
    canvas->Print((std::string(output_prefix) + ".pdf").c_str());
    canvas->Print((std::string(output_prefix) + ".eps").c_str());
}

} // namespace gvvplot

#endif // GVV_ANGULAR_MOMENTS_H

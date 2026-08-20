#include "../GVVPlotUtils.h"

#include "TCanvas.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TMath.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

// Detailed 4x2 diagnostic. The seven observable definitions and all visual
// settings are local to this macro for independent adjustment.
namespace projection_detailed {

const std::vector<gvvplot::VariableSpec> kVariables = {
    {gvvplot::kMassOmegaOmega, 60, 1.50, 3.20,
     "M(#omega#omega) (GeV/#font[12]{c}^{2})", true},
    {gvvplot::kMassGammaOmega, 60, 0.85, 2.95,
     "M(#gamma#omega) (GeV/#font[12]{c}^{2})", true},
    {gvvplot::kCosThetaGamma, 40, -1.0, 1.0,
     "cos#theta_{#gamma}", false},
    {gvvplot::kCosThetaOmega, 40, -1.0, 1.0,
     "sym. cos#theta_{#omega}", false},
    {gvvplot::kOmegaDecayPlane, 40, -TMath::Pi(), TMath::Pi(),
     "#phi_{#omega} (rad)", false},
    {gvvplot::kDeltaPhiDecayPlanes, 40, -TMath::Pi(), TMath::Pi(),
     "sym. #Delta#phi_{planes} (rad)", false},
    {gvvplot::kMassOmega, 42, 0.740, 0.824,
     "M(#pi^{+}#pi^{-}#pi^{0}) (GeV/#font[12]{c}^{2})", true}};

constexpr int kCanvasWidth = 1320;
constexpr int kCanvasHeight = 720;
constexpr int kCanvasColumns = 4;
constexpr int kCanvasRows = 2;
constexpr int kLegendPad = 8;

const int kGroupLineStyles[] = {2, 7, 9, 3, 5};
const int kGroupLineColors[] = {
    kRed + 1, kGreen + 2, kMagenta + 1, kOrange + 7, kCyan + 2};

void FormatPanel(
    gvvplot::PanelHistograms& panel,
    const gvvplot::VariableSpec& variable)
{
    panel.background->SetFillStyle(3004);
    panel.background->SetFillColor(kBlue);
    panel.background->SetLineColor(kBlue);
    panel.total->SetLineColor(kBlue + 1);
    panel.total->SetLineWidth(2);
    for (std::size_t group = 0; group < panel.groups.size(); ++group) {
        panel.groups[group]->SetLineColor(kGroupLineColors[group % 5]);
        panel.groups[group]->SetLineStyle(kGroupLineStyles[group % 5]);
        panel.groups[group]->SetLineWidth(2);
    }

    const double bin_width =
        (variable.upper - variable.lower) / variable.bins;
    panel.data->GetXaxis()->SetTitle(variable.x_title);
    panel.data->GetYaxis()->SetTitle(
        variable.mass_axis
            ? Form("Events / (%.1f MeV/#font[12]{c}^{2})",
                   1000.0 * bin_width)
            : Form("Events / %.3g", bin_width));
    panel.data->GetXaxis()->CenterTitle(kTRUE);
    panel.data->GetYaxis()->CenterTitle(kTRUE);
    panel.data->GetXaxis()->SetNdivisions(505);
    panel.data->GetYaxis()->SetNdivisions(505);
    panel.data->SetMarkerStyle(8);
    panel.data->SetMarkerSize(0.55);
    panel.data->SetLineColor(kBlack);
    panel.data->SetLineWidth(1);

    const double maximum =
        std::max(panel.data->GetMaximum(), panel.total->GetMaximum());
    const double minimum = std::min(0.0, panel.total->GetMinimum());
    panel.data->GetYaxis()->SetRangeUser(
        minimum < 0.0 ? 1.25 * minimum : 0.0,
        maximum > 0.0 ? 1.45 * maximum : 1.0);
}

void DrawPanel(
    gvvplot::PanelHistograms& panel,
    const gvvplot::VariableSpec& variable,
    std::size_t panel_index)
{
    FormatPanel(panel, variable);
    panel.data->Draw("E1");
    panel.background->Draw("HIST SAME");
    for (TH1D* group : panel.groups) group->Draw("HIST SAME");
    panel.total->Draw("HIST SAME");
    panel.data->Draw("E1 SAME");

    const std::pair<double, int> chi_square =
        gvvplot::PearsonChiSquare(panel.data, panel.total);
    TLatex label;
    label.SetNDC();
    label.SetTextFont(22);
    label.SetTextSize(0.047);
    label.DrawLatex(
        0.18,
        0.84,
        Form("(%c) #chi^{2}/N_{bin}=%.1f/%d",
             static_cast<char>('a' + panel_index),
             chi_square.first,
             chi_square.second));
    std::cout << variable.x_title << "  chi2/Nbin=" << chi_square.first
              << '/' << chi_square.second << '\n';
}

} // namespace projection_detailed

void Draw_projection(
    const char* input_file = nullptr,
    const char* output_prefix = nullptr)
{
    const std::string input_path = gvvplot::ResolveMacroArgument(
        input_file,
        __FILE__,
        "../../../results/projection-initial.root");
    const std::string output_path = gvvplot::ResolveMacroArgument(
        output_prefix,
        __FILE__,
        "../results/projection_detailed-initial");

    gvvplot::SetBESIIIStyle();
    gvvplot::ProjectionInput input = gvvplot::LoadProjection(input_path.c_str());
    TCanvas* canvas = new TCanvas(
        "gvv_projection_detailed",
        "GVV detailed projections",
        projection_detailed::kCanvasWidth,
        projection_detailed::kCanvasHeight);
    canvas->Divide(
        projection_detailed::kCanvasColumns,
        projection_detailed::kCanvasRows,
        0.002,
        0.002);

    std::vector<gvvplot::PanelHistograms> panels;
    for (std::size_t index = 0;
         index < projection_detailed::kVariables.size();
         ++index) {
        canvas->cd(static_cast<int>(index) + 1);
        panels.push_back(gvvplot::BuildPanel(
            input,
            projection_detailed::kVariables[index],
            static_cast<int>(index),
            false));
        projection_detailed::DrawPanel(
            panels.back(), projection_detailed::kVariables[index], index);
    }

    canvas->cd(projection_detailed::kLegendPad);
    TLegend* legend = new TLegend(0.54, 0.60, 0.93, 0.86);
    legend->SetBorderSize(0);
    legend->SetFillStyle(0);
    legend->SetTextFont(22);
    legend->SetTextSize(0.050);
    legend->AddEntry(panels[0].data, "Data", "lep");
    legend->AddEntry(panels[0].background, "Background", "f");
    legend->AddEntry(panels[0].total, "Total fit", "l");
    for (std::size_t group = 0; group < input.groups.size(); ++group) {
        const std::string label = "coherent " + input.groups[group].label;
        legend->AddEntry(panels[0].groups[group], label.c_str(), "l");
    }
    legend->Draw();

    canvas->Print((output_path + ".pdf").c_str());
    canvas->Print((output_path + ".eps").c_str());
}

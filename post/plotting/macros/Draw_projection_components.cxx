#include "../GVVPlotUtils.h"

#include "TCanvas.h"
#include "TColor.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TMath.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

// Diagonal |A_i|^2 diagnostic. Interference is intentionally omitted, so the
// component curves are not expected to sum to the coherent total.
namespace projection_components {

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
     "sym. #Delta#phi_{planes} (rad)", false}};

constexpr int kCanvasWidth = 1320;
constexpr int kCanvasHeight = 720;
constexpr int kCanvasColumns = 4;
constexpr int kCanvasRows = 2;
constexpr int kLegendPad = 7;

std::array<int, 7> ComponentColors()
{
    return {
        TColor::GetColor("#08306B"),
        TColor::GetColor("#2171B5"),
        TColor::GetColor("#6BAED6"),
        TColor::GetColor("#9ECAE1"),
        TColor::GetColor("#41AB5D"),
        TColor::GetColor("#A1D76A"),
        TColor::GetColor("#FDE725")};
}

void FormatPanel(
    gvvplot::PanelHistograms& panel,
    const gvvplot::VariableSpec& variable,
    const std::array<int, 7>& component_colors)
{
    panel.background->SetFillStyle(3004);
    panel.background->SetFillColor(kBlue);
    panel.background->SetLineColor(kBlue);
    panel.total->SetLineColor(kBlue + 1);
    panel.total->SetLineWidth(2);
    for (std::size_t component = 0;
         component < panel.components.size();
         ++component) {
        panel.components[component]->SetLineColor(
            component_colors[component % component_colors.size()]);
        panel.components[component]->SetLineStyle(1);
        panel.components[component]->SetLineWidth(1);
        panel.components[component]->SetMarkerStyle(0);
        panel.components[component]->SetMarkerSize(0);
        panel.components[component]->SetFillStyle(0);
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
    std::size_t panel_index,
    const std::array<int, 7>& component_colors)
{
    FormatPanel(panel, variable, component_colors);
    panel.data->Draw("E1");
    panel.background->Draw("HIST SAME");
    for (TH1D* component : panel.components) {
        component->Draw("HIST C SAME");
    }
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

} // namespace projection_components

void Draw_projection_components(
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
        "../results/projection_components-initial");

    gvvplot::SetBESIIIStyle();
    gvvplot::ProjectionInput input = gvvplot::LoadProjection(input_path.c_str());
    const std::array<int, 7> component_colors =
        projection_components::ComponentColors();
    TCanvas* canvas = new TCanvas(
        "gvv_projection_components",
        "GVV component projections",
        projection_components::kCanvasWidth,
        projection_components::kCanvasHeight);
    canvas->Divide(
        projection_components::kCanvasColumns,
        projection_components::kCanvasRows,
        0.002,
        0.002);

    std::vector<gvvplot::PanelHistograms> panels;
    for (std::size_t index = 0;
         index < projection_components::kVariables.size();
         ++index) {
        canvas->cd(static_cast<int>(index) + 1);
        panels.push_back(gvvplot::BuildPanel(
            input,
            projection_components::kVariables[index],
            static_cast<int>(index),
            true));
        projection_components::DrawPanel(
            panels.back(),
            projection_components::kVariables[index],
            index,
            component_colors);
    }

    canvas->cd(projection_components::kLegendPad);
    TLegend* legend = new TLegend(0.10, 0.08, 0.94, 0.92);
    legend->SetBorderSize(0);
    legend->SetFillStyle(0);
    legend->SetTextFont(22);
    legend->SetTextSize(0.055);
    legend->AddEntry(panels[0].data, "Data", "lep");
    legend->AddEntry(panels[0].background, "Background", "f");
    legend->AddEntry(panels[0].total, "Total fit", "l");
    for (std::size_t component = 0;
         component < input.components.size();
         ++component) {
        legend->AddEntry(
            panels[0].components[component],
            gvvplot::RootLabel(input.components[component].label).c_str(),
            "l");
    }
    legend->Draw();

    canvas->Print((output_path + ".pdf").c_str());
    canvas->Print((output_path + ".eps").c_str());
}

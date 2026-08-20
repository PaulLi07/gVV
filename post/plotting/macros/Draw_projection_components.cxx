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

namespace projection_components {

// ============================================================================
// User configuration
// ============================================================================
// This figure shows diagonal |A_i|^2 terms only. Interference is intentionally
// omitted, so the component curves are not expected to sum to the coherent
// total. Relative paths are interpreted from the project root.
//
// Run with defaults:
//   root post/plotting/macros/Draw_projection_components.cxx
// Override at runtime:
//   root -l -b -q 'post/plotting/macros/Draw_projection_components.cxx(
//     "results/projection-TAG.root",
//     "post/plotting/results/projection_components-TAG")'
constexpr const char* kDefaultInput = "results/projection-initial.root";
constexpr const char* kDefaultOutput =
    "post/plotting/results/projection_components-initial";

// Observable order, binning, numerical ranges, and x-axis titles.
const std::vector<gvvplot::VariableSpec> kVariables = {
    {gvvplot::kMassOmegaOmega, 60, 1.50, 3.20,
     "M(#omega#omega) (GeV/#font[12]{c}^{2})", true},
    {gvvplot::kMassGammaOmega, 60, 0.85, 2.95,
     "M(#gamma#omega) (GeV/#font[12]{c}^{2})", true},
    {gvvplot::kCosThetaGamma, 40, -1.0, 1.0,
     "cos#theta_{#gamma}", false},
    {gvvplot::kCosThetaOmega1, 40, -1.0, 1.0,
     "sym. cos#theta_{#omega}", false},
    {gvvplot::kPhiDecayPlaneOmega, 40, -TMath::Pi(), TMath::Pi(),
     "#phi_{decay plane}^{#omega} (rad)", false},
    {gvvplot::kDeltaPhiDecayPlanes, 40, -TMath::Pi(), TMath::Pi(),
     "sym. #Delta#phi_{planes} (rad)", false}};

constexpr const char* kCanvasName = "gvv_projection_components";
constexpr const char* kCanvasTitle = "GVV component projections";
constexpr int kCanvasWidth = 1320;
constexpr int kCanvasHeight = 720;
constexpr int kCanvasColumns = 4;
constexpr int kCanvasRows = 2;
constexpr double kPadGap = 0.002;
constexpr int kLegendPad = 7;

// Data, background, total-fit, and component appearance.
constexpr int kDataMarkerStyle = 8;
constexpr double kDataMarkerSize = 0.55;
constexpr int kDataColor = kBlack;
constexpr int kDataLineWidth = 1;
constexpr int kBackgroundFillStyle = 3004;
constexpr int kBackgroundColor = kBlue;
constexpr int kTotalColor = kBlue + 1;
constexpr int kTotalLineWidth = 2;
constexpr int kComponentLineStyle = 1;
constexpr int kComponentLineWidth = 1;
constexpr int kComponentMarkerStyle = 0;
constexpr double kComponentMarkerSize = 0.0;
constexpr int kComponentFillStyle = 0;
const std::array<const char*, 7> kComponentColorHex = {
    "#08306B", "#2171B5", "#6BAED6", "#9ECAE1",
    "#41AB5D", "#A1D76A", "#FDE725"};

// Axes and automatic vertical range.
constexpr int kAxisDivisions = 505;
constexpr const char* kMassYAxisFormat =
    "Events / (%.1f MeV/#font[12]{c}^{2})";
constexpr const char* kDimensionlessYAxisFormat = "Events / %.3g";
constexpr double kGeVToMeV = 1000.0;
constexpr bool kCenterAxisTitles = true;
constexpr double kNegativeRangeScale = 1.25;
constexpr double kPositiveRangeScale = 1.45;

// Per-panel chi-square annotation.
constexpr int kAnnotationFont = 22;
constexpr double kAnnotationSize = 0.047;
constexpr double kAnnotationX = 0.18;
constexpr double kAnnotationY = 0.84;
constexpr const char* kAnnotationFormat =
    "(%c) #chi^{2}/N_{bin}=%.1f/%d";

// ROOT draw options and layer order used by DrawPanel().
constexpr const char* kDataDrawOption = "E1";
constexpr const char* kBackgroundDrawOption = "HIST SAME";
constexpr const char* kComponentDrawOption = "HIST C SAME";
constexpr const char* kTotalDrawOption = "HIST SAME";
constexpr const char* kDataRedrawOption = "E1 SAME";

// Legend box and text.
constexpr double kLegendX1 = 0.10;
constexpr double kLegendY1 = 0.08;
constexpr double kLegendX2 = 0.94;
constexpr double kLegendY2 = 0.92;
constexpr int kLegendFont = 22;
constexpr double kLegendTextSize = 0.055;
constexpr int kLegendBorderSize = 0;
constexpr int kLegendFillStyle = 0;
constexpr const char* kDataLegendLabel = "Data";
constexpr const char* kBackgroundLegendLabel = "Background";
constexpr const char* kTotalLegendLabel = "Total fit";
constexpr const char* kDataLegendOption = "lep";
constexpr const char* kBackgroundLegendOption = "f";
constexpr const char* kLineLegendOption = "l";

// ============================================================================
// Implementation below. Normal figure changes should only require the block
// above.
// ============================================================================

std::array<int, 7> ComponentColors()
{
    std::array<int, 7> colors = {};
    for (std::size_t index = 0; index < colors.size(); ++index) {
        colors[index] = TColor::GetColor(kComponentColorHex[index]);
    }
    return colors;
}

void FormatPanel(
    gvvplot::PanelHistograms& panel,
    const gvvplot::VariableSpec& variable,
    const std::array<int, 7>& component_colors)
{
    panel.background->SetFillStyle(kBackgroundFillStyle);
    panel.background->SetFillColor(kBackgroundColor);
    panel.background->SetLineColor(kBackgroundColor);
    panel.total->SetLineColor(kTotalColor);
    panel.total->SetLineWidth(kTotalLineWidth);
    for (std::size_t component = 0;
         component < panel.components.size();
         ++component) {
        panel.components[component]->SetLineColor(
            component_colors[component % component_colors.size()]);
        panel.components[component]->SetLineStyle(kComponentLineStyle);
        panel.components[component]->SetLineWidth(kComponentLineWidth);
        panel.components[component]->SetMarkerStyle(kComponentMarkerStyle);
        panel.components[component]->SetMarkerSize(kComponentMarkerSize);
        panel.components[component]->SetFillStyle(kComponentFillStyle);
    }

    const double bin_width =
        (variable.upper - variable.lower) / variable.bins;
    panel.data->GetXaxis()->SetTitle(variable.x_title);
    panel.data->GetYaxis()->SetTitle(
        variable.mass_axis
            ? Form(kMassYAxisFormat, kGeVToMeV * bin_width)
            : Form(kDimensionlessYAxisFormat, bin_width));
    panel.data->GetXaxis()->CenterTitle(kCenterAxisTitles);
    panel.data->GetYaxis()->CenterTitle(kCenterAxisTitles);
    panel.data->GetXaxis()->SetNdivisions(kAxisDivisions);
    panel.data->GetYaxis()->SetNdivisions(kAxisDivisions);
    panel.data->SetMarkerStyle(kDataMarkerStyle);
    panel.data->SetMarkerSize(kDataMarkerSize);
    panel.data->SetLineColor(kDataColor);
    panel.data->SetLineWidth(kDataLineWidth);

    const double maximum =
        std::max(panel.data->GetMaximum(), panel.total->GetMaximum());
    const double minimum = std::min(0.0, panel.total->GetMinimum());
    panel.data->GetYaxis()->SetRangeUser(
        minimum < 0.0 ? kNegativeRangeScale * minimum : 0.0,
        maximum > 0.0 ? kPositiveRangeScale * maximum : 1.0);
}

void DrawPanel(
    gvvplot::PanelHistograms& panel,
    const gvvplot::VariableSpec& variable,
    std::size_t panel_index,
    const std::array<int, 7>& component_colors)
{
    FormatPanel(panel, variable, component_colors);
    panel.data->Draw(kDataDrawOption);
    panel.background->Draw(kBackgroundDrawOption);
    for (TH1D* component : panel.components) {
        component->Draw(kComponentDrawOption);
    }
    panel.total->Draw(kTotalDrawOption);
    panel.data->Draw(kDataRedrawOption);

    const std::pair<double, int> chi_square =
        gvvplot::PearsonChiSquare(panel.data, panel.total);
    TLatex label;
    label.SetNDC();
    label.SetTextFont(kAnnotationFont);
    label.SetTextSize(kAnnotationSize);
    label.DrawLatex(
        kAnnotationX,
        kAnnotationY,
        Form(kAnnotationFormat,
             static_cast<char>('a' + panel_index),
             chi_square.first,
             chi_square.second));
    std::cout << variable.x_title << "  chi2/Nbin=" << chi_square.first
              << '/' << chi_square.second << '\n';
}

} // namespace projection_components

void Draw_projection_components(
    const char* input_file = projection_components::kDefaultInput,
    const char* output_prefix = projection_components::kDefaultOutput)
{
    const std::string input_path =
        gvvplot::ResolveProjectPath(input_file, __FILE__);
    const std::string output_path =
        gvvplot::ResolveProjectPath(output_prefix, __FILE__);

    gvvplot::SetBESIIIStyle();
    gvvplot::ProjectionInput input = gvvplot::LoadProjection(input_path.c_str());
    const std::array<int, 7> component_colors =
        projection_components::ComponentColors();
    TCanvas* canvas = new TCanvas(
        projection_components::kCanvasName,
        projection_components::kCanvasTitle,
        projection_components::kCanvasWidth,
        projection_components::kCanvasHeight);
    canvas->Divide(
        projection_components::kCanvasColumns,
        projection_components::kCanvasRows,
        projection_components::kPadGap,
        projection_components::kPadGap);

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
    TLegend* legend = new TLegend(
        projection_components::kLegendX1,
        projection_components::kLegendY1,
        projection_components::kLegendX2,
        projection_components::kLegendY2);
    legend->SetBorderSize(projection_components::kLegendBorderSize);
    legend->SetFillStyle(projection_components::kLegendFillStyle);
    legend->SetTextFont(projection_components::kLegendFont);
    legend->SetTextSize(projection_components::kLegendTextSize);
    legend->AddEntry(
        panels[0].data,
        projection_components::kDataLegendLabel,
        projection_components::kDataLegendOption);
    legend->AddEntry(
        panels[0].background,
        projection_components::kBackgroundLegendLabel,
        projection_components::kBackgroundLegendOption);
    legend->AddEntry(
        panels[0].total,
        projection_components::kTotalLegendLabel,
        projection_components::kLineLegendOption);
    for (std::size_t component = 0;
         component < input.components.size();
         ++component) {
        legend->AddEntry(
            panels[0].components[component],
            gvvplot::RootLabel(input.components[component].label).c_str(),
            projection_components::kLineLegendOption);
    }
    legend->Draw();

    canvas->Print((output_path + ".pdf").c_str());
    canvas->Print((output_path + ".eps").c_str());
}

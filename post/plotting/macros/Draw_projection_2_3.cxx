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

namespace projection_main {

// ============================================================================
// User configuration
// ============================================================================
// Relative paths are interpreted from the project root. Run with these
// defaults using:
//   root post/plotting/macros/Draw_projection_2_3.cxx
// Override them at runtime using:
//   root -l -b -q 'post/plotting/macros/Draw_projection_2_3.cxx(
//     "results/projection-TAG.root",
//     "post/plotting/results/projection-TAG")'
constexpr const char* kDefaultInput = "results/projection-initial.root";
constexpr const char* kDefaultOutput =
    "post/plotting/results/projection-initial";

// Observable order, binning, numerical ranges, and x-axis titles.
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

constexpr const char* kCanvasName = "gvv_projection_main";
constexpr const char* kCanvasTitle = "GVV main projections";
constexpr int kCanvasWidth = 1080;
constexpr int kCanvasHeight = 720;
constexpr int kCanvasColumns = 3;
constexpr int kCanvasRows = 2;
constexpr double kPadGap = 0.002;
constexpr int kLegendPad = 1;

// Data, background, total-fit, and coherent-group appearance.
constexpr int kDataMarkerStyle = 8;
constexpr double kDataMarkerSize = 0.55;
constexpr int kDataColor = kBlack;
constexpr int kDataLineWidth = 1;
constexpr int kBackgroundFillStyle = 3004;
constexpr int kBackgroundColor = kBlue;
constexpr int kTotalColor = kBlue + 1;
constexpr int kTotalLineWidth = 2;
constexpr int kGroupLineWidth = 2;
const int kGroupLineStyles[] = {2, 7, 9, 3, 5};
const int kGroupLineColors[] = {
    kRed + 1, kGreen + 2, kMagenta + 1, kOrange + 7, kCyan + 2};
constexpr std::size_t kGroupStyleCount =
    sizeof(kGroupLineStyles) / sizeof(kGroupLineStyles[0]);

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
constexpr const char* kGroupDrawOption = "HIST SAME";
constexpr const char* kTotalDrawOption = "HIST SAME";
constexpr const char* kDataRedrawOption = "E1 SAME";

// Legend box and text.
constexpr double kLegendX1 = 0.54;
constexpr double kLegendY1 = 0.60;
constexpr double kLegendX2 = 0.93;
constexpr double kLegendY2 = 0.86;
constexpr int kLegendFont = 22;
constexpr double kLegendTextSize = 0.050;
constexpr int kLegendBorderSize = 0;
constexpr int kLegendFillStyle = 0;
constexpr const char* kDataLegendLabel = "Data";
constexpr const char* kBackgroundLegendLabel = "Background";
constexpr const char* kTotalLegendLabel = "Total fit";
constexpr const char* kDataLegendOption = "lep";
constexpr const char* kBackgroundLegendOption = "f";
constexpr const char* kLineLegendOption = "l";
constexpr const char* kGroupLegendPrefix = "coherent ";

// ============================================================================
// Implementation below. Normal figure changes should only require the block
// above.
// ============================================================================

void FormatPanel(
    gvvplot::PanelHistograms& panel,
    const gvvplot::VariableSpec& variable)
{
    panel.background->SetFillStyle(kBackgroundFillStyle);
    panel.background->SetFillColor(kBackgroundColor);
    panel.background->SetLineColor(kBackgroundColor);
    panel.total->SetLineColor(kTotalColor);
    panel.total->SetLineWidth(kTotalLineWidth);
    for (std::size_t group = 0; group < panel.groups.size(); ++group) {
        panel.groups[group]->SetLineColor(
            kGroupLineColors[group % kGroupStyleCount]);
        panel.groups[group]->SetLineStyle(
            kGroupLineStyles[group % kGroupStyleCount]);
        panel.groups[group]->SetLineWidth(kGroupLineWidth);
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
    std::size_t panel_index)
{
    FormatPanel(panel, variable);
    panel.data->Draw(kDataDrawOption);
    panel.background->Draw(kBackgroundDrawOption);
    for (TH1D* group : panel.groups) group->Draw(kGroupDrawOption);
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

} // namespace projection_main

void Draw_projection_2_3(
    const char* input_file = projection_main::kDefaultInput,
    const char* output_prefix = projection_main::kDefaultOutput)
{
    const std::string input_path =
        gvvplot::ResolveProjectPath(input_file, __FILE__);
    const std::string output_path =
        gvvplot::ResolveProjectPath(output_prefix, __FILE__);

    gvvplot::SetBESIIIStyle();
    gvvplot::ProjectionInput input = gvvplot::LoadProjection(input_path.c_str());
    TCanvas* canvas = new TCanvas(
        projection_main::kCanvasName,
        projection_main::kCanvasTitle,
        projection_main::kCanvasWidth,
        projection_main::kCanvasHeight);
    canvas->Divide(
        projection_main::kCanvasColumns,
        projection_main::kCanvasRows,
        projection_main::kPadGap,
        projection_main::kPadGap);

    std::vector<gvvplot::PanelHistograms> panels;
    for (std::size_t index = 0;
         index < projection_main::kVariables.size();
         ++index) {
        canvas->cd(static_cast<int>(index) + 1);
        panels.push_back(gvvplot::BuildPanel(
            input,
            projection_main::kVariables[index],
            static_cast<int>(index),
            false));
        projection_main::DrawPanel(
            panels.back(), projection_main::kVariables[index], index);
    }

    canvas->cd(projection_main::kLegendPad);
    TLegend* legend = new TLegend(
        projection_main::kLegendX1,
        projection_main::kLegendY1,
        projection_main::kLegendX2,
        projection_main::kLegendY2);
    legend->SetBorderSize(projection_main::kLegendBorderSize);
    legend->SetFillStyle(projection_main::kLegendFillStyle);
    legend->SetTextFont(projection_main::kLegendFont);
    legend->SetTextSize(projection_main::kLegendTextSize);
    legend->AddEntry(
        panels[0].data,
        projection_main::kDataLegendLabel,
        projection_main::kDataLegendOption);
    legend->AddEntry(
        panels[0].background,
        projection_main::kBackgroundLegendLabel,
        projection_main::kBackgroundLegendOption);
    legend->AddEntry(
        panels[0].total,
        projection_main::kTotalLegendLabel,
        projection_main::kLineLegendOption);
    for (std::size_t group = 0; group < input.groups.size(); ++group) {
        const std::string label =
            projection_main::kGroupLegendPrefix + input.groups[group].label;
        legend->AddEntry(
            panels[0].groups[group],
            label.c_str(),
            projection_main::kLineLegendOption);
    }
    legend->Draw();

    canvas->Print((output_path + ".pdf").c_str());
    canvas->Print((output_path + ".eps").c_str());
}

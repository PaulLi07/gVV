#include "../GVVPlotUtils.h"

#include "TCanvas.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TMath.h"
#include "TPad.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace polarization {

// ============================================================================
// User configuration
// ============================================================================
// For omega_i, the helicity-frame z axis follows omega_i in the X rest frame,
// y is normal to the X -> omega omega decay plane, and x completes a
// right-handed basis. The oriented decay-plane normal is
// n_i = unit[p(pi+_i) cross p(pi-_i)] in the omega_i rest frame. The two omega
// candidates are combined with half weight per candidate; Delta phi is filled
// together with its exchange image. Relative paths are interpreted from the
// project root.
//
// Run with defaults:
//   root post/plotting/macros/Draw_polarization.cxx
// Override at runtime:
//   root -l -b -q 'post/plotting/macros/Draw_polarization.cxx(
//     "results/projection-TAG.root",
//     "post/plotting/results/polarization-TAG")'
constexpr const char* kDefaultInput = "results/projection-initial.root";
constexpr const char* kDefaultOutput =
    "post/plotting/results/polarization-initial";

// Polarization observables, binning, ranges, and x-axis titles.
const std::vector<gvvplot::VariableSpec> kVariables = {
    {gvvplot::kCosThetaDecayPlaneOmega, 40, -1.0, 1.0,
     "cos#theta_{#hat{n}_{#omega}}^{(#omega hel.)}", false},
    {gvvplot::kPhiDecayPlaneOmega, 40, -TMath::Pi(), TMath::Pi(),
     "#phi_{#hat{n}_{#omega}}^{(#omega hel.)} (rad)", false},
    {gvvplot::kDeltaPhiDecayPlanes, 40, -TMath::Pi(), TMath::Pi(),
     "#Delta#phi(#hat{n}_{1},#hat{n}_{2}) (rad)", false}};

constexpr const char* kCanvasName = "gvv_polarization";
constexpr const char* kCanvasTitle = "GVV polarization observables";
constexpr int kCanvasWidth = 1500;
constexpr int kCanvasHeight = 560;
constexpr int kCanvasColumns = 3;
constexpr int kCanvasRows = 1;
constexpr double kPadGap = 0.002;
constexpr double kPlotPadY1 = 0.00;
constexpr double kPlotPadY2 = 0.86;
constexpr double kLegendPadY1 = 0.86;
constexpr double kLegendPadY2 = 1.00;

// Data, background, total-fit, and coherent-group appearance.
constexpr int kDataMarkerStyle = 8;
constexpr double kDataMarkerSize = 0.55;
constexpr int kDataColor = kBlack;
constexpr int kDataLineWidth = 1;
constexpr int kBackgroundFillStyle = 3004;
constexpr int kBackgroundColor = kGray + 1;
constexpr int kTotalColor = kBlue + 1;
constexpr int kTotalLineWidth = 2;
constexpr int kGroupLineWidth = 2;
constexpr int kGroupLineStyle = 2;
const int kGroupLineColors[] = {
    kRed + 1, kGreen + 2, kMagenta + 1, kOrange + 7,
    kCyan + 2, kViolet + 1, kTeal + 3, kPink + 7};
constexpr std::size_t kGroupStyleCount =
    sizeof(kGroupLineColors) / sizeof(kGroupLineColors[0]);

// Axes and automatic vertical range.
constexpr int kAxisDivisions = 505;
constexpr const char* kDimensionlessYAxisFormat = "Events / %.3g";
constexpr const char* kAzimuthYAxisFormat = "Events / (%.3g rad)";
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

// ROOT draw options and layer order.
constexpr const char* kDataDrawOption = "E1";
constexpr const char* kBackgroundDrawOption = "HIST SAME";
constexpr const char* kGroupDrawOption = "HIST SAME";
constexpr const char* kTotalDrawOption = "HIST SAME";
constexpr const char* kDataRedrawOption = "E1 SAME";

// Legend box and text.
constexpr double kLegendX1 = 0.03;
constexpr double kLegendY1 = 0.05;
constexpr double kLegendX2 = 0.97;
constexpr double kLegendY2 = 0.95;
constexpr int kLegendColumns = 3;
constexpr int kLegendFont = 22;
constexpr double kLegendTextSize = 0.28;
constexpr int kLegendBorderSize = 0;
constexpr int kLegendFillStyle = 0;
constexpr const char* kDataLegendLabel = "Data";
constexpr const char* kBackgroundLegendLabel = "Background";
constexpr const char* kTotalLegendLabel = "Total fit";
constexpr const char* kDataLegendOption = "lep";
constexpr const char* kBackgroundLegendOption = "f";
constexpr const char* kLineLegendOption = "l";
constexpr const char* kGroupLegendSuffix = " coherence class";

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
        panel.groups[group]->SetLineStyle(kGroupLineStyle);
        panel.groups[group]->SetLineWidth(kGroupLineWidth);
    }

    const double bin_width =
        (variable.upper - variable.lower) / variable.bins;
    panel.data->GetXaxis()->SetTitle(variable.x_title);
    const bool azimuth = variable.variable == gvvplot::kPhiDecayPlaneOmega
                         || variable.variable
                                == gvvplot::kDeltaPhiDecayPlanes;
    panel.data->GetYaxis()->SetTitle(Form(
        azimuth ? kAzimuthYAxisFormat : kDimensionlessYAxisFormat,
        bin_width));
    panel.data->GetXaxis()->CenterTitle(kCenterAxisTitles);
    panel.data->GetYaxis()->CenterTitle(kCenterAxisTitles);
    panel.data->GetXaxis()->SetNdivisions(kAxisDivisions);
    panel.data->GetYaxis()->SetNdivisions(kAxisDivisions);
    panel.data->SetMarkerStyle(kDataMarkerStyle);
    panel.data->SetMarkerSize(kDataMarkerSize);
    panel.data->SetLineColor(kDataColor);
    panel.data->SetLineWidth(kDataLineWidth);

    std::vector<const TH1D*> curves = {panel.background, panel.total};
    curves.insert(curves.end(), panel.groups.begin(), panel.groups.end());
    const gvvplot::VerticalRange range =
        gvvplot::FindVerticalRange(panel.data, curves);
    panel.data->GetYaxis()->SetRangeUser(
        range.minimum < 0.0 ? kNegativeRangeScale * range.minimum : 0.0,
        range.maximum > 0.0 ? kPositiveRangeScale * range.maximum : 1.0);
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

} // namespace polarization

void Draw_polarization(
    const char* input_file = polarization::kDefaultInput,
    const char* output_prefix = polarization::kDefaultOutput)
{
    const std::string input_path =
        gvvplot::ResolveProjectPath(input_file, __FILE__);
    const std::string output_path =
        gvvplot::ResolveProjectPath(output_prefix, __FILE__);

    gvvplot::SetBESIIIStyle();
    gvvplot::ProjectionInput input = gvvplot::LoadProjection(input_path.c_str());
    TCanvas* canvas = new TCanvas(
        polarization::kCanvasName,
        polarization::kCanvasTitle,
        polarization::kCanvasWidth,
        polarization::kCanvasHeight);
    TPad* plot_pad = new TPad(
        "gvv_polarization_plots", "", 0.0, polarization::kPlotPadY1,
        1.0, polarization::kPlotPadY2);
    plot_pad->SetFillStyle(0);
    plot_pad->Draw();
    plot_pad->Divide(
        polarization::kCanvasColumns,
        polarization::kCanvasRows,
        polarization::kPadGap,
        polarization::kPadGap);

    canvas->cd();
    TPad* legend_pad = new TPad(
        "gvv_polarization_legend", "", 0.0,
        polarization::kLegendPadY1, 1.0, polarization::kLegendPadY2);
    legend_pad->SetFillStyle(0);
    legend_pad->SetMargin(0.0, 0.0, 0.0, 0.0);
    legend_pad->Draw();

    std::vector<gvvplot::PanelHistograms> panels;
    for (std::size_t index = 0;
         index < polarization::kVariables.size();
         ++index) {
        plot_pad->cd(static_cast<int>(index) + 1);
        panels.push_back(gvvplot::BuildPanel(
            input,
            polarization::kVariables[index],
            static_cast<int>(index),
            false));
        polarization::DrawPanel(
            panels.back(), polarization::kVariables[index], index);
    }

    legend_pad->cd();
    TLegend* legend = new TLegend(
        polarization::kLegendX1,
        polarization::kLegendY1,
        polarization::kLegendX2,
        polarization::kLegendY2);
    legend->SetBorderSize(polarization::kLegendBorderSize);
    legend->SetFillStyle(polarization::kLegendFillStyle);
    legend->SetTextFont(polarization::kLegendFont);
    legend->SetTextSize(polarization::kLegendTextSize);
    legend->SetNColumns(polarization::kLegendColumns);
    legend->AddEntry(
        panels[0].data,
        polarization::kDataLegendLabel,
        polarization::kDataLegendOption);
    legend->AddEntry(
        panels[0].background,
        polarization::kBackgroundLegendLabel,
        polarization::kBackgroundLegendOption);
    legend->AddEntry(
        panels[0].total,
        polarization::kTotalLegendLabel,
        polarization::kLineLegendOption);
    for (std::size_t group = 0; group < input.groups.size(); ++group) {
        const std::string label =
            input.groups[group].label + polarization::kGroupLegendSuffix;
        legend->AddEntry(
            panels[0].groups[group],
            label.c_str(),
            polarization::kLineLegendOption);
    }
    legend->Draw();

    gvvplot::EnsureOutputDirectory(output_path);
    canvas->Print((output_path + ".pdf").c_str());
    canvas->Print((output_path + ".eps").c_str());
}

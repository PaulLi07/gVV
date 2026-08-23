#include "../GVVPlotUtils.h"

#include "TCanvas.h"
#include "TLatex.h"
#include "TLegend.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace omega_decay_checks {

// ============================================================================
// User configuration
// ============================================================================
// Each distribution combines the omega1 and omega2 candidates with half
// weight per candidate. A pion polar angle is measured in the corresponding
// omega_i helicity frame: z_i follows omega_i in the X rest frame. No
// additional cosine reflection is applied. Relative paths are interpreted
// from the project root.
//
// Run with defaults:
//   root post/plotting/macros/Draw_omega_decay_checks.cxx
// Override at runtime:
//   root -l -b -q 'post/plotting/macros/Draw_omega_decay_checks.cxx(
//     "results/projection-TAG.root",
//     "post/plotting/results/omega_decay_checks-TAG")'
constexpr const char* kDefaultInput = "results/projection-initial.root";
constexpr const char* kDefaultOutput =
    "post/plotting/results/omega_decay_checks-initial";

// Check observables, binning, ranges, and x-axis titles.
const std::vector<gvvplot::VariableSpec> kVariables = {
    {gvvplot::kCosThetaPipOmega, 40, -1.0, 1.0,
     "cos#theta_{#pi^{+}}^{(#omega hel.)}", false},
    {gvvplot::kCosThetaPimOmega, 40, -1.0, 1.0,
     "cos#theta_{#pi^{-}}^{(#omega hel.)}", false},
    {gvvplot::kCosThetaPi0Omega, 40, -1.0, 1.0,
     "cos#theta_{#pi^{0}}^{(#omega hel.)}", false},
    {gvvplot::kMassPipPimOmega, 45, 0.25, 0.70,
     "M(#pi^{+}#pi^{-}) (GeV/#font[12]{c}^{2})", true},
    {gvvplot::kMassPipPi0Omega, 45, 0.25, 0.70,
     "M(#pi^{+}#pi^{0}) (GeV/#font[12]{c}^{2})", true},
    {gvvplot::kMassPimPi0Omega, 45, 0.25, 0.70,
     "M(#pi^{-}#pi^{0}) (GeV/#font[12]{c}^{2})", true}};

constexpr const char* kCanvasName = "gvv_omega_decay_checks";
constexpr const char* kCanvasTitle = "GVV omega decay checks";
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
constexpr const char* kMassYAxisFormat =
    "Events / (%.1f MeV/#font[12]{c}^{2})";
constexpr const char* kDimensionlessYAxisFormat = "Events / %.3g";
constexpr double kGeVToMeV = 1000.0;
constexpr bool kCenterAxisTitles = true;
constexpr double kNegativeRangeScale = 1.25;
constexpr double kPositiveRangeScale = 1.45;
// The shared legend is drawn in the first panel. Its larger headroom keeps
// the legend clear of the distributions without compressing the other five.
constexpr double kLegendPanelPositiveRangeScale = 1.95;

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
constexpr const char* kGroupLegendSuffix = " coherence class";

// ============================================================================
// Implementation below. Normal figure changes should only require the block
// above.
// ============================================================================

void FormatPanel(
    gvvplot::PanelHistograms& panel,
    const gvvplot::VariableSpec& variable,
    std::size_t panel_index)
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

    std::vector<const TH1D*> curves = {panel.background, panel.total};
    curves.insert(curves.end(), panel.groups.begin(), panel.groups.end());
    const gvvplot::VerticalRange range =
        gvvplot::FindVerticalRange(panel.data, curves);
    const double positive_scale =
        panel_index + 1 == static_cast<std::size_t>(kLegendPad)
            ? kLegendPanelPositiveRangeScale
            : kPositiveRangeScale;
    panel.data->GetYaxis()->SetRangeUser(
        range.minimum < 0.0 ? kNegativeRangeScale * range.minimum : 0.0,
        range.maximum > 0.0 ? positive_scale * range.maximum : 1.0);
}

void DrawPanel(
    gvvplot::PanelHistograms& panel,
    const gvvplot::VariableSpec& variable,
    std::size_t panel_index)
{
    FormatPanel(panel, variable, panel_index);
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

} // namespace omega_decay_checks

void Draw_omega_decay_checks(
    const char* input_file = omega_decay_checks::kDefaultInput,
    const char* output_prefix = omega_decay_checks::kDefaultOutput)
{
    const std::string input_path =
        gvvplot::ResolveProjectPath(input_file, __FILE__);
    const std::string output_path =
        gvvplot::ResolveProjectPath(output_prefix, __FILE__);

    gvvplot::SetBESIIIStyle();
    gvvplot::ProjectionInput input = gvvplot::LoadProjection(input_path.c_str());
    TCanvas* canvas = new TCanvas(
        omega_decay_checks::kCanvasName,
        omega_decay_checks::kCanvasTitle,
        omega_decay_checks::kCanvasWidth,
        omega_decay_checks::kCanvasHeight);
    canvas->Divide(
        omega_decay_checks::kCanvasColumns,
        omega_decay_checks::kCanvasRows,
        omega_decay_checks::kPadGap,
        omega_decay_checks::kPadGap);

    std::vector<gvvplot::PanelHistograms> panels;
    for (std::size_t index = 0;
         index < omega_decay_checks::kVariables.size();
         ++index) {
        canvas->cd(static_cast<int>(index) + 1);
        panels.push_back(gvvplot::BuildPanel(
            input,
            omega_decay_checks::kVariables[index],
            static_cast<int>(index),
            false));
        omega_decay_checks::DrawPanel(
            panels.back(), omega_decay_checks::kVariables[index], index);
    }

    canvas->cd(omega_decay_checks::kLegendPad);
    TLegend* legend = new TLegend(
        omega_decay_checks::kLegendX1,
        omega_decay_checks::kLegendY1,
        omega_decay_checks::kLegendX2,
        omega_decay_checks::kLegendY2);
    legend->SetBorderSize(omega_decay_checks::kLegendBorderSize);
    legend->SetFillStyle(omega_decay_checks::kLegendFillStyle);
    legend->SetTextFont(omega_decay_checks::kLegendFont);
    legend->SetTextSize(omega_decay_checks::kLegendTextSize);
    legend->AddEntry(
        panels[0].data,
        omega_decay_checks::kDataLegendLabel,
        omega_decay_checks::kDataLegendOption);
    legend->AddEntry(
        panels[0].background,
        omega_decay_checks::kBackgroundLegendLabel,
        omega_decay_checks::kBackgroundLegendOption);
    legend->AddEntry(
        panels[0].total,
        omega_decay_checks::kTotalLegendLabel,
        omega_decay_checks::kLineLegendOption);
    for (std::size_t group = 0; group < input.groups.size(); ++group) {
        const std::string label =
            input.groups[group].label
            + omega_decay_checks::kGroupLegendSuffix;
        legend->AddEntry(
            panels[0].groups[group],
            label.c_str(),
            omega_decay_checks::kLineLegendOption);
    }
    legend->Draw();

    gvvplot::EnsureOutputDirectory(output_path);
    canvas->Print((output_path + ".pdf").c_str());
    canvas->Print((output_path + ".eps").c_str());
}

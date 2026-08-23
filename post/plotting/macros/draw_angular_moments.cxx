#include "../GVVAngularMoments.h"

#include "TCanvas.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TLine.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace even_moments {

// ============================================================================
// User configuration
// ============================================================================
// This macro draws physical even moments with omega-exchange symmetrization.
// Relative paths are interpreted from the project root.
//
// Run with defaults:
//   root post/plotting/macros/draw_angular_moments.cxx
// Override at runtime:
//   root -l -b -q 'post/plotting/macros/draw_angular_moments.cxx(
//     "results/projection-TAG.root",
//     "post/plotting/results/angular_moments-TAG")'
constexpr const char* kDefaultInput = "results/projection-initial.root";
constexpr const char* kDefaultOutput =
    "post/plotting/results/angular_moments-initial";

// Moment orders and M(omega omega) binning.
const std::vector<int> kOrders = {0, 2, 4, 6};
constexpr int kMassBins = 34;
constexpr double kMassLower = 1.50;
constexpr double kMassUpper = 3.20;
constexpr bool kOddDiagnostic = false;

// Canvas layout.
constexpr const char* kCanvasName = "gvv_even_moments";
constexpr const char* kCanvasTitle = "GVV even angular moments";
constexpr int kCanvasWidth = 1200;
constexpr int kCanvasHeight = 900;
constexpr int kCanvasColumns = 2;
constexpr int kCanvasRows = 2;
constexpr double kPadGap = 0.002;
constexpr int kLegendPad = 1;

// Axes and automatic vertical range.
constexpr const char* kXAxisTitle =
    "M(#omega#omega) (GeV/#font[12]{c}^{2})";
constexpr const char* kYAxisTitleFormat =
    "#sum P_{%d}(cos#theta_{#omega}^{(X hel.)}) / "
    "(%.1f MeV/#font[12]{c}^{2})";
constexpr double kGeVToMeV = 1000.0;
constexpr bool kCenterAxisTitles = true;
constexpr double kNegativeRangeScale = 1.35;
constexpr double kPositiveRangeScale = 1.35;
// The shared legend remains inside the first subplot. Its panel alone gets
// extra vertical headroom so the other moment scales are not compressed.
constexpr double kLegendPanelPositiveRangeScale = 1.55;

// Data and fitted-model appearance.
constexpr int kDataMarkerStyle = 8;
constexpr double kDataMarkerSize = 0.70;
constexpr int kDataColor = kBlack;
constexpr int kDataLineWidth = 1;
constexpr int kModelColor = kBlue + 1;
constexpr int kModelLineWidth = 2;

// A single legend in the first panel applies to all moment orders.
constexpr double kLegendX1 = 0.49;
constexpr double kLegendY1 = 0.74;
constexpr double kLegendX2 = 0.93;
constexpr double kLegendY2 = 0.89;
constexpr int kLegendFont = 22;
constexpr double kLegendTextSize = 0.043;
constexpr int kLegendBorderSize = 0;
constexpr int kLegendFillStyle = 0;
constexpr const char* kDataLegendLabel = "Data - signed background";
constexpr const char* kModelLegendLabel = "Fitted signal MC";

// Zero reference for nonzero Legendre moments.
constexpr int kZeroLineColor = kGray + 1;
constexpr int kZeroLineStyle = 3;
constexpr int kZeroLineWidth = 1;

// Per-panel chi-square annotation.
constexpr int kAnnotationFont = 22;
constexpr double kAnnotationSize = 0.050;
constexpr double kAnnotationX = 0.18;
constexpr double kAnnotationY = 0.84;
constexpr const char* kAnnotationFormat =
    "P_{%d}: #chi^{2}/N_{bin}=%.1f/%d";

// ROOT draw options and layer order used by DrawPanel().
constexpr const char* kDataDrawOption = "E1";
constexpr const char* kModelDrawOption = "HIST SAME";
constexpr const char* kDataRedrawOption = "E1 SAME";

// ============================================================================
// Implementation below. Normal figure changes should only require the block
// above.
// ============================================================================

void FormatPanel(
    gvvplot::MomentHistograms& histograms,
    int order,
    std::size_t panel_index)
{
    histograms.data->SetMarkerStyle(kDataMarkerStyle);
    histograms.data->SetMarkerSize(kDataMarkerSize);
    histograms.data->SetLineColor(kDataColor);
    histograms.data->SetLineWidth(kDataLineWidth);
    histograms.model->SetLineColor(kModelColor);
    histograms.model->SetLineWidth(kModelLineWidth);
    histograms.data->GetXaxis()->SetTitle(kXAxisTitle);
    const double mass_bin_width_mev =
        kGeVToMeV * (kMassUpper - kMassLower) / kMassBins;
    histograms.data->GetYaxis()->SetTitle(
        Form(kYAxisTitleFormat, order, mass_bin_width_mev));
    histograms.data->GetXaxis()->CenterTitle(kCenterAxisTitles);
    histograms.data->GetYaxis()->CenterTitle(kCenterAxisTitles);

    const std::vector<const TH1D*> curves = {histograms.model};
    const gvvplot::VerticalRange range =
        gvvplot::FindVerticalRange(histograms.data, curves);
    const double positive_scale =
        panel_index + 1 == static_cast<std::size_t>(kLegendPad)
            ? kLegendPanelPositiveRangeScale
            : kPositiveRangeScale;
    histograms.data->GetYaxis()->SetRangeUser(
        range.minimum < 0.0 ? kNegativeRangeScale * range.minimum : 0.0,
        range.maximum > 0.0 ? positive_scale * range.maximum : 1.0);
}

void DrawPanel(
    gvvplot::MomentHistograms& histograms,
    int order,
    std::size_t panel_index)
{
    FormatPanel(histograms, order, panel_index);
    histograms.data->Draw(kDataDrawOption);
    histograms.model->Draw(kModelDrawOption);
    if (order != 0) {
        TLine* zero = new TLine(kMassLower, 0.0, kMassUpper, 0.0);
        zero->SetLineColor(kZeroLineColor);
        zero->SetLineStyle(kZeroLineStyle);
        zero->SetLineWidth(kZeroLineWidth);
        zero->Draw("SAME");
    }
    histograms.data->Draw(kDataRedrawOption);

    const std::pair<double, int> chi_square =
        gvvplot::MomentChiSquare(histograms.data, histograms.model);
    TLatex label;
    label.SetNDC();
    label.SetTextFont(kAnnotationFont);
    label.SetTextSize(kAnnotationSize);
    label.DrawLatex(
        kAnnotationX,
        kAnnotationY,
        Form(kAnnotationFormat,
             order,
             chi_square.first,
             chi_square.second));
}

} // namespace even_moments

void draw_angular_moments(
    const char* input_file = even_moments::kDefaultInput,
    const char* output_prefix = even_moments::kDefaultOutput)
{
    const std::string input_path =
        gvvplot::ResolveProjectPath(input_file, __FILE__);
    const std::string output_path =
        gvvplot::ResolveProjectPath(output_prefix, __FILE__);

    gvvplot::SetBESIIIStyle();
    gvvplot::ProjectionInput input = gvvplot::LoadProjection(input_path.c_str());
    TCanvas* canvas = new TCanvas(
        even_moments::kCanvasName,
        even_moments::kCanvasTitle,
        even_moments::kCanvasWidth,
        even_moments::kCanvasHeight);
    canvas->Divide(
        even_moments::kCanvasColumns,
        even_moments::kCanvasRows,
        even_moments::kPadGap,
        even_moments::kPadGap);

    for (std::size_t panel = 0;
         panel < even_moments::kOrders.size();
         ++panel) {
        canvas->cd(static_cast<int>(panel) + 1);
        const int order = even_moments::kOrders[panel];
        gvvplot::MomentHistograms histograms =
            gvvplot::BuildMomentHistograms(
                input,
                order,
                even_moments::kOddDiagnostic,
                even_moments::kMassBins,
                even_moments::kMassLower,
                even_moments::kMassUpper,
                "gvv_even_moment");
        even_moments::DrawPanel(histograms, order, panel);

        if (panel + 1 == static_cast<std::size_t>(even_moments::kLegendPad)) {
            TLegend* legend = new TLegend(
                even_moments::kLegendX1,
                even_moments::kLegendY1,
                even_moments::kLegendX2,
                even_moments::kLegendY2);
            legend->SetBorderSize(even_moments::kLegendBorderSize);
            legend->SetFillStyle(even_moments::kLegendFillStyle);
            legend->SetTextFont(even_moments::kLegendFont);
            legend->SetTextSize(even_moments::kLegendTextSize);
            legend->AddEntry(
                histograms.data,
                even_moments::kDataLegendLabel,
                "lep");
            legend->AddEntry(
                histograms.model,
                even_moments::kModelLegendLabel,
                "l");
            legend->Draw();
        }
    }

    gvvplot::EnsureOutputDirectory(output_path);
    canvas->Print((output_path + ".pdf").c_str());
    canvas->Print((output_path + ".eps").c_str());
}

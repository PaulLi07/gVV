#include "../GVVAngularMoments.h"

#include "TCanvas.h"
#include "TLatex.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace odd_moments {

// ============================================================================
// User configuration
// ============================================================================
// These ordered-omega odd moments diagnose pairing/order bias; they are not
// label-independent observables of the identical-omega final state. Relative
// paths are interpreted from the project root.
//
// Run with defaults:
//   root post/plotting/macros/draw_angular_moments_odd.cxx
// Override at runtime:
//   root -l -b -q 'post/plotting/macros/draw_angular_moments_odd.cxx(
//     "results/projection-TAG.root",
//     "post/plotting/results/angular_moments_odd_diagnostic-TAG")'
constexpr const char* kDefaultInput = "results/projection-initial.root";
constexpr const char* kDefaultOutput =
    "post/plotting/results/angular_moments_odd_diagnostic-initial";

// Moment orders and M(omega omega) binning.
const std::vector<int> kOrders = {1, 3, 5};
constexpr int kMassBins = 34;
constexpr double kMassLower = 1.50;
constexpr double kMassUpper = 3.20;
constexpr bool kOddDiagnostic = true;

// Canvas layout.
constexpr const char* kCanvasName = "gvv_odd_moments";
constexpr const char* kCanvasTitle = "GVV odd angular-moment diagnostic";
constexpr int kCanvasWidth = 1500;
constexpr int kCanvasHeight = 500;
constexpr int kCanvasColumns = 3;
constexpr int kCanvasRows = 1;
constexpr double kPadGap = 0.002;

// Axes and automatic vertical range.
constexpr const char* kXAxisTitle =
    "M(#omega#omega) (GeV/#font[12]{c}^{2})";
constexpr const char* kYAxisTitleFormat =
    "#LT P_{%d}(cos#theta_{#omega}) #GT / 50 MeV";
constexpr bool kCenterAxisTitles = true;
constexpr double kNegativeRangeScale = 1.35;
constexpr double kPositiveRangeScale = 1.35;

// Data and fitted-model appearance.
constexpr int kDataMarkerStyle = 8;
constexpr double kDataMarkerSize = 0.70;
constexpr int kDataColor = kBlack;
constexpr int kDataLineWidth = 1;
constexpr int kModelColor = kBlue + 1;
constexpr int kModelLineWidth = 2;

// Per-panel chi-square and ordered-omega diagnostic annotations.
constexpr int kAnnotationFont = 22;
constexpr double kAnnotationSize = 0.050;
constexpr double kAnnotationX = 0.18;
constexpr double kAnnotationY = 0.84;
constexpr const char* kAnnotationFormat =
    "P_{%d}: #chi^{2}/N_{bin}=%.1f/%d";
constexpr double kDiagnosticSize = 0.040;
constexpr double kDiagnosticX = 0.18;
constexpr double kDiagnosticY = 0.76;
constexpr const char* kDiagnosticText = "ordered-#omega diagnostic only";

// ROOT draw options and layer order used by DrawPanel().
constexpr const char* kDataDrawOption = "E1";
constexpr const char* kModelDrawOption = "HIST SAME";
constexpr const char* kDataRedrawOption = "E1 SAME";

// ============================================================================
// Implementation below. Normal figure changes should only require the block
// above.
// ============================================================================

void FormatPanel(gvvplot::MomentHistograms& histograms, int order)
{
    histograms.data->SetMarkerStyle(kDataMarkerStyle);
    histograms.data->SetMarkerSize(kDataMarkerSize);
    histograms.data->SetLineColor(kDataColor);
    histograms.data->SetLineWidth(kDataLineWidth);
    histograms.model->SetLineColor(kModelColor);
    histograms.model->SetLineWidth(kModelLineWidth);
    histograms.data->GetXaxis()->SetTitle(kXAxisTitle);
    histograms.data->GetYaxis()->SetTitle(
        Form(kYAxisTitleFormat, order));
    histograms.data->GetXaxis()->CenterTitle(kCenterAxisTitles);
    histograms.data->GetYaxis()->CenterTitle(kCenterAxisTitles);

    const double maximum = std::max(
        histograms.data->GetMaximum(), histograms.model->GetMaximum());
    const double minimum = std::min(
        histograms.data->GetMinimum(), histograms.model->GetMinimum());
    histograms.data->GetYaxis()->SetRangeUser(
        minimum < 0.0 ? kNegativeRangeScale * minimum : 0.0,
        maximum > 0.0 ? kPositiveRangeScale * maximum : 1.0);
}

void DrawPanel(gvvplot::MomentHistograms& histograms, int order)
{
    FormatPanel(histograms, order);
    histograms.data->Draw(kDataDrawOption);
    histograms.model->Draw(kModelDrawOption);
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
    label.SetTextSize(kDiagnosticSize);
    label.DrawLatex(
        kDiagnosticX, kDiagnosticY, kDiagnosticText);
}

} // namespace odd_moments

void draw_angular_moments_odd(
    const char* input_file = odd_moments::kDefaultInput,
    const char* output_prefix = odd_moments::kDefaultOutput)
{
    const std::string input_path =
        gvvplot::ResolveProjectPath(input_file, __FILE__);
    const std::string output_path =
        gvvplot::ResolveProjectPath(output_prefix, __FILE__);

    gvvplot::SetBESIIIStyle();
    gvvplot::ProjectionInput input = gvvplot::LoadProjection(input_path.c_str());
    TCanvas* canvas = new TCanvas(
        odd_moments::kCanvasName,
        odd_moments::kCanvasTitle,
        odd_moments::kCanvasWidth,
        odd_moments::kCanvasHeight);
    canvas->Divide(
        odd_moments::kCanvasColumns,
        odd_moments::kCanvasRows,
        odd_moments::kPadGap,
        odd_moments::kPadGap);

    for (std::size_t panel = 0;
         panel < odd_moments::kOrders.size();
         ++panel) {
        canvas->cd(static_cast<int>(panel) + 1);
        const int order = odd_moments::kOrders[panel];
        gvvplot::MomentHistograms histograms =
            gvvplot::BuildMomentHistograms(
                input,
                order,
                odd_moments::kOddDiagnostic,
                odd_moments::kMassBins,
                odd_moments::kMassLower,
                odd_moments::kMassUpper,
                "gvv_odd_moment");
        odd_moments::DrawPanel(histograms, order);
    }

    canvas->Print((output_path + ".pdf").c_str());
    canvas->Print((output_path + ".eps").c_str());
}

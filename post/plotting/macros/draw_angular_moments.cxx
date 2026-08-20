#include "../GVVAngularMoments.h"

#include "TCanvas.h"
#include "TLatex.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

// Physical even moments. The omega-exchange symmetrization is performed by
// the shared numerical helper; all presentation settings remain here.
namespace even_moments {

const std::vector<int> kOrders = {0, 2, 4, 6};
constexpr int kMassBins = 34;
constexpr double kMassLower = 1.50;
constexpr double kMassUpper = 3.20;
constexpr int kCanvasWidth = 1200;
constexpr int kCanvasHeight = 900;
constexpr int kCanvasColumns = 2;
constexpr int kCanvasRows = 2;

void FormatPanel(gvvplot::MomentHistograms& histograms, int order)
{
    histograms.data->SetMarkerStyle(8);
    histograms.data->SetMarkerSize(0.70);
    histograms.data->SetLineColor(kBlack);
    histograms.data->SetLineWidth(1);
    histograms.model->SetLineColor(kBlue + 1);
    histograms.model->SetLineWidth(2);
    histograms.data->GetXaxis()->SetTitle(
        "M(#omega#omega) (GeV/#font[12]{c}^{2})");
    histograms.data->GetYaxis()->SetTitle(
        Form("#LT P_{%d}(cos#theta_{#omega}) #GT / 50 MeV", order));
    histograms.data->GetXaxis()->CenterTitle(kTRUE);
    histograms.data->GetYaxis()->CenterTitle(kTRUE);

    const double maximum = std::max(
        histograms.data->GetMaximum(), histograms.model->GetMaximum());
    const double minimum = std::min(
        histograms.data->GetMinimum(), histograms.model->GetMinimum());
    histograms.data->GetYaxis()->SetRangeUser(
        minimum < 0.0 ? 1.35 * minimum : 0.0,
        maximum > 0.0 ? 1.35 * maximum : 1.0);
}

void DrawPanel(gvvplot::MomentHistograms& histograms, int order)
{
    FormatPanel(histograms, order);
    histograms.data->Draw("E1");
    histograms.model->Draw("HIST SAME");
    histograms.data->Draw("E1 SAME");

    const std::pair<double, int> chi_square =
        gvvplot::MomentChiSquare(histograms.data, histograms.model);
    TLatex label;
    label.SetNDC();
    label.SetTextFont(22);
    label.SetTextSize(0.050);
    label.DrawLatex(
        0.18,
        0.84,
        Form("P_{%d}: #chi^{2}/N_{bin}=%.1f/%d",
             order,
             chi_square.first,
             chi_square.second));
}

} // namespace even_moments

void draw_angular_moments(
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
        "../results/angular_moments-initial");

    gvvplot::SetBESIIIStyle();
    gvvplot::ProjectionInput input = gvvplot::LoadProjection(input_path.c_str());
    TCanvas* canvas = new TCanvas(
        "gvv_even_moments",
        "GVV even angular moments",
        even_moments::kCanvasWidth,
        even_moments::kCanvasHeight);
    canvas->Divide(
        even_moments::kCanvasColumns,
        even_moments::kCanvasRows,
        0.002,
        0.002);

    for (std::size_t panel = 0;
         panel < even_moments::kOrders.size();
         ++panel) {
        canvas->cd(static_cast<int>(panel) + 1);
        const int order = even_moments::kOrders[panel];
        gvvplot::MomentHistograms histograms =
            gvvplot::BuildMomentHistograms(
                input,
                order,
                false,
                even_moments::kMassBins,
                even_moments::kMassLower,
                even_moments::kMassUpper,
                "gvv_even_moment");
        even_moments::DrawPanel(histograms, order);
    }

    canvas->Print((output_path + ".pdf").c_str());
    canvas->Print((output_path + ".eps").c_str());
}

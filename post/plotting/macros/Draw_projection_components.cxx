#include "../GVVPlotUtils.h"

#include "TCanvas.h"
#include "TColor.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TMath.h"
#include "TPad.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
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
     "M(#gamma#omega_{i}) (GeV/#font[12]{c}^{2}), i=1,2", true},
    {gvvplot::kCosThetaGamma, 40, -1.0, 1.0,
     "cos#theta_{#gamma}^{(#psi(2S) rest)}", false},
    {gvvplot::kCosThetaOmega, 40, -1.0, 1.0,
     "cos#theta_{#omega}^{(X hel.; sym.)}", false},
    {gvvplot::kPhiOmega, 40, -TMath::Pi(), TMath::Pi(),
     "#phi_{#omega}^{(X hel.; sym.)} (rad)", false},
    {gvvplot::kMassOmega, 42, 0.740, 0.824,
     "M(#pi^{+}_{i}#pi^{-}_{i}#pi^{0}_{i}) "
     "(GeV/#font[12]{c}^{2}), i=1,2", true}};

constexpr const char* kCanvasName = "gvv_projection_components";
constexpr const char* kCanvasTitle = "GVV component projections";
constexpr int kCanvasWidth = 1800;
constexpr int kCanvasHeight = 900;
constexpr int kCanvasColumns = 3;
constexpr int kCanvasRows = 2;
constexpr double kPadGap = 0.002;
constexpr double kPlotRegionXMax = 0.78;
constexpr double kLegendRegionXMin = 0.78;
constexpr const char* kPlotPadName = "gvv_component_plot_pad";
constexpr const char* kLegendPadName = "gvv_component_legend_pad";

// Data, background, total-fit, and component appearance.
constexpr int kDataMarkerStyle = 8;
constexpr double kDataMarkerSize = 0.55;
constexpr int kDataColor = kBlack;
constexpr int kDataLineWidth = 1;
constexpr int kBackgroundFillStyle = 3004;
constexpr int kBackgroundFillColor = kGray + 1;
constexpr int kBackgroundLineColor = kGray + 2;
constexpr int kTotalColor = kBlue + 1;
constexpr int kTotalLineWidth = 2;
constexpr int kComponentLineWidth = 2;
constexpr int kComponentMarkerStyle = 0;
constexpr double kComponentMarkerSize = 0.0;
constexpr int kComponentFillStyle = 0;
// This qualitative palette deliberately excludes the Total-fit blue, the
// neutral Background gray, and Data black. A stable metadata key chooses the
// starting color; probing is used only to keep the active Terms distinct.
const std::array<const char*, 20> kComponentColorHex = {
    "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#7B2CBF",
    "#8C564B", "#6B8E23", "#00A6A6", "#E41A1C", "#4DAF4A",
    "#984EA3", "#FF7F00", "#A65628", "#F781BF", "#8A8A00",
    "#00A087", "#DC0000", "#7E6148", "#B09C85", "#6A3D9A"};
const std::array<int, 4> kComponentLineStyles = {1, 7, 9, 10};

// Axes and automatic vertical range.
constexpr int kAxisDivisions = 505;
constexpr const char* kMassYAxisFormat =
    "Events / (%.1f MeV/#font[12]{c}^{2})";
constexpr const char* kDimensionlessYAxisFormat = "Events / %.3g";
constexpr const char* kAzimuthYAxisFormat = "Events / (%.3g rad)";
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
constexpr const char* kComponentDrawOption = "HIST SAME";
constexpr const char* kTotalDrawOption = "HIST SAME";
constexpr const char* kDataRedrawOption = "E1 SAME";

// Legend box and text.
constexpr double kLegendX1 = 0.04;
constexpr double kLegendY1 = 0.03;
constexpr double kLegendX2 = 0.98;
constexpr double kLegendY2 = 0.88;
constexpr int kLegendColumns = 1;
constexpr int kLegendFont = 22;
constexpr double kLegendTextSize = 0.035;
constexpr int kLegendBorderSize = 0;
constexpr int kLegendFillStyle = 0;
constexpr int kLegendNoteFont = 22;
constexpr double kLegendNoteSize = 0.042;
constexpr double kLegendNoteX = 0.50;
constexpr double kLegendNoteLine1Y = 0.965;
constexpr double kLegendNoteLine2Y = 0.920;
constexpr const char* kLegendNoteLine1 =
    "Diagonal |A_{i}|^{2} components";
constexpr const char* kLegendNoteLine2 =
    "Interference terms are omitted";
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

struct ComponentStyle {
    int color = kBlack;
    int line_style = 1;
};

std::uint64_t StableHash(const std::string& text)
{
    // FNV-1a makes the mapping independent of the order in model.json.
    std::uint64_t hash = 14695981039346656037ULL;
    for (unsigned char character : text) {
        hash ^= character;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string StyleKey(const gvvplot::ComponentInfo& component)
{
    return component.resonance_id + "\x1f" + component.wave_id
        + "\x1f" + component.name;
}

int PreferredLineStyle(const gvvplot::ComponentInfo& component)
{
    if (component.wave_id.find("_u2") != std::string::npos) return 7;
    if (component.wave_id.find("_u3") != std::string::npos) return 9;
    if (component.wave_id.find("_u4") != std::string::npos) return 10;
    return 1;
}

std::vector<int> ComponentColors()
{
    std::vector<int> colors(kComponentColorHex.size(), kBlack);
    for (std::size_t index = 0; index < colors.size(); ++index) {
        colors[index] = TColor::GetColor(kComponentColorHex[index]);
    }
    return colors;
}

std::vector<ComponentStyle> BuildComponentStyles(
    const std::vector<gvvplot::ComponentInfo>& components)
{
    const std::vector<int> colors = ComponentColors();
    std::vector<ComponentStyle> styles(components.size());
    std::vector<std::size_t> ordered_indices(components.size());
    std::iota(ordered_indices.begin(), ordered_indices.end(), 0);
    std::sort(
        ordered_indices.begin(), ordered_indices.end(),
        [&components](std::size_t first, std::size_t second) {
            return StyleKey(components[first]) < StyleKey(components[second]);
        });

    std::vector<int> color_use_count(colors.size(), 0);
    std::vector<std::pair<int, int>> used_pairs;
    for (std::size_t component_index : ordered_indices) {
        const gvvplot::ComponentInfo& component = components[component_index];
        const std::uint64_t hash = StableHash(StyleKey(component));
        const std::size_t first_color = hash % colors.size();

        // Prefer an unused color. If the model outgrows the palette, reuse the
        // least-used available color with a distinct line style.
        std::size_t color_index = first_color;
        int best_use_count = color_use_count[color_index];
        for (std::size_t step = 0; step < colors.size(); ++step) {
            const std::size_t candidate = (first_color + step) % colors.size();
            if (color_use_count[candidate] == 0) {
                color_index = candidate;
                best_use_count = 0;
                break;
            }
            if (color_use_count[candidate] < best_use_count) {
                color_index = candidate;
                best_use_count = color_use_count[candidate];
            }
        }

        int line_style = PreferredLineStyle(component);
        for (std::size_t offset = 0;
             offset < kComponentLineStyles.size();
             ++offset) {
            const int candidate = kComponentLineStyles[
                (hash + offset) % kComponentLineStyles.size()];
            const int requested = offset == 0 ? line_style : candidate;
            const std::pair<int, int> style_pair = {
                colors[color_index], requested};
            if (std::find(
                    used_pairs.begin(), used_pairs.end(), style_pair)
                == used_pairs.end()) {
                line_style = requested;
                break;
            }
        }

        styles[component_index] = {colors[color_index], line_style};
        ++color_use_count[color_index];
        used_pairs.push_back({colors[color_index], line_style});
    }
    return styles;
}

void FormatPanel(
    gvvplot::PanelHistograms& panel,
    const gvvplot::VariableSpec& variable,
    const std::vector<ComponentStyle>& component_styles)
{
    panel.background->SetFillStyle(kBackgroundFillStyle);
    panel.background->SetFillColor(kBackgroundFillColor);
    panel.background->SetLineColor(kBackgroundLineColor);
    panel.total->SetLineColor(kTotalColor);
    panel.total->SetLineWidth(kTotalLineWidth);
    for (std::size_t component = 0;
         component < panel.components.size();
         ++component) {
        panel.components[component]->SetLineColor(
            component_styles[component].color);
        panel.components[component]->SetLineStyle(
            component_styles[component].line_style);
        panel.components[component]->SetLineWidth(kComponentLineWidth);
        panel.components[component]->SetMarkerStyle(kComponentMarkerStyle);
        panel.components[component]->SetMarkerSize(kComponentMarkerSize);
        panel.components[component]->SetFillStyle(kComponentFillStyle);
    }

    const double bin_width =
        (variable.upper - variable.lower) / variable.bins;
    panel.data->GetXaxis()->SetTitle(variable.x_title);
    if (variable.mass_axis) {
        panel.data->GetYaxis()->SetTitle(
            Form(kMassYAxisFormat, kGeVToMeV * bin_width));
    } else if (variable.variable == gvvplot::kPhiOmega) {
        panel.data->GetYaxis()->SetTitle(
            Form(kAzimuthYAxisFormat, bin_width));
    } else {
        panel.data->GetYaxis()->SetTitle(
            Form(kDimensionlessYAxisFormat, bin_width));
    }
    panel.data->GetXaxis()->CenterTitle(kCenterAxisTitles);
    panel.data->GetYaxis()->CenterTitle(kCenterAxisTitles);
    panel.data->GetXaxis()->SetNdivisions(kAxisDivisions);
    panel.data->GetYaxis()->SetNdivisions(kAxisDivisions);
    panel.data->SetMarkerStyle(kDataMarkerStyle);
    panel.data->SetMarkerSize(kDataMarkerSize);
    panel.data->SetLineColor(kDataColor);
    panel.data->SetLineWidth(kDataLineWidth);

    std::vector<const TH1D*> curves = {panel.background, panel.total};
    curves.insert(
        curves.end(), panel.components.begin(), panel.components.end());
    const gvvplot::VerticalRange range =
        gvvplot::FindVerticalRange(panel.data, curves);
    const double minimum = std::min(0.0, range.minimum);
    const double maximum = std::max(0.0, range.maximum);
    panel.data->GetYaxis()->SetRangeUser(
        minimum < 0.0 ? kNegativeRangeScale * minimum : 0.0,
        maximum > 0.0 ? kPositiveRangeScale * maximum : 1.0);
}

void DrawPanel(
    gvvplot::PanelHistograms& panel,
    const gvvplot::VariableSpec& variable,
    std::size_t panel_index,
    const std::vector<ComponentStyle>& component_styles)
{
    FormatPanel(panel, variable, component_styles);
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
    const std::vector<projection_components::ComponentStyle>
        component_styles =
            projection_components::BuildComponentStyles(input.components);
    TCanvas* canvas = new TCanvas(
        projection_components::kCanvasName,
        projection_components::kCanvasTitle,
        projection_components::kCanvasWidth,
        projection_components::kCanvasHeight);

    // Keep a 3 x 2 physics grid on the left and reserve a full-height strip on
    // the right for the model-dependent component legend.
    TPad* plot_pad = new TPad(
        projection_components::kPlotPadName,
        "",
        0.0,
        0.0,
        projection_components::kPlotRegionXMax,
        1.0);
    plot_pad->Draw();
    plot_pad->cd();
    plot_pad->Divide(
        projection_components::kCanvasColumns,
        projection_components::kCanvasRows,
        projection_components::kPadGap,
        projection_components::kPadGap);

    std::vector<gvvplot::PanelHistograms> panels;
    for (std::size_t index = 0;
         index < projection_components::kVariables.size();
         ++index) {
        plot_pad->cd(static_cast<int>(index) + 1);
        panels.push_back(gvvplot::BuildPanel(
            input,
            projection_components::kVariables[index],
            static_cast<int>(index),
            true));
        projection_components::DrawPanel(
            panels.back(),
            projection_components::kVariables[index],
            index,
            component_styles);
    }

    canvas->cd();
    TPad* legend_pad = new TPad(
        projection_components::kLegendPadName,
        "",
        projection_components::kLegendRegionXMin,
        0.0,
        1.0,
        1.0);
    legend_pad->SetTopMargin(0.0);
    legend_pad->SetBottomMargin(0.0);
    legend_pad->SetLeftMargin(0.0);
    legend_pad->SetRightMargin(0.0);
    legend_pad->Draw();
    legend_pad->cd();

    TLatex legend_note;
    legend_note.SetNDC();
    legend_note.SetTextAlign(22);
    legend_note.SetTextFont(projection_components::kLegendNoteFont);
    legend_note.SetTextSize(projection_components::kLegendNoteSize);
    legend_note.DrawLatex(
        projection_components::kLegendNoteX,
        projection_components::kLegendNoteLine1Y,
        projection_components::kLegendNoteLine1);
    legend_note.DrawLatex(
        projection_components::kLegendNoteX,
        projection_components::kLegendNoteLine2Y,
        projection_components::kLegendNoteLine2);

    TLegend* legend = new TLegend(
        projection_components::kLegendX1,
        projection_components::kLegendY1,
        projection_components::kLegendX2,
        projection_components::kLegendY2);
    legend->SetBorderSize(projection_components::kLegendBorderSize);
    legend->SetFillStyle(projection_components::kLegendFillStyle);
    legend->SetTextFont(projection_components::kLegendFont);
    legend->SetTextSize(projection_components::kLegendTextSize);
    legend->SetNColumns(projection_components::kLegendColumns);
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

    gvvplot::EnsureOutputDirectory(output_path);
    canvas->Modified();
    canvas->Update();
    canvas->Print((output_path + ".pdf").c_str());
    canvas->Print((output_path + ".eps").c_str());
}

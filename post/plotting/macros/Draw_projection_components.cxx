#include "../GVVPlotUtils.h"

#include "TCanvas.h"
#include "TColor.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TMath.h"
#include "TPad.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace projection_components {

struct RgbColor {
    int red;
    int green;
    int blue;
};

// ============================================================================
// User configuration
// ============================================================================
// This figure shows individual component contributions. Interference between
// distinct resonances is intentionally omitted, so the component curves are
// not expected to sum to the coherent total. Each Resonance is shown once as
// the coherent sum of all of its active Wave Terms. Relative paths are
// interpreted from the project root.
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
constexpr int kCanvasWidth = 1500;
constexpr int kCanvasHeight = 800;
constexpr int kCanvasColumns = 3;
constexpr int kCanvasRows = 2;
constexpr double kPadGap = 0.002;
constexpr double kPlotRegionXMax = 0.90;
constexpr double kLegendRegionXMin = 0.90;
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
constexpr int kComponentLineWidth = 1;
constexpr int kComponentLineStyle = 1;
constexpr int kComponentMarkerStyle = 0;
constexpr double kComponentMarkerSize = 0.0;
constexpr int kComponentFillStyle = 0;
// Components are listed in the legend from blue through green to yellow.
// The continuous gradient remains ordered when the active model changes size.
constexpr RgbColor kComponentColorStart = {53, 88, 177};
constexpr RgbColor kComponentColorMiddle = {0, 158, 115};
constexpr RgbColor kComponentColorEnd = {253, 231, 37};

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
constexpr double kLegendX1 = 0.02;
constexpr double kLegendY1 = 0.50;
constexpr double kLegendX2 = 0.98;
constexpr double kLegendY2 = 0.85;
constexpr int kLegendColumns = 1;
constexpr int kLegendFont = 42;
constexpr double kLegendTextSize = 0.092;
constexpr double kLegendMargin = 0.34;
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

struct ComponentStyle {
    int color = kBlack;
};

struct DisplayComponent {
    std::string label;
    std::string resonance_id;
    std::vector<std::size_t> source_indices;
};

std::string ResonanceLabel(
    const gvvplot::ComponentInfo& component)
{
    // Multi-Wave Term labels append their Wave identifier after '~'. The
    // combined curve represents the Resonance, so retain only that prefix.
    const std::size_t separator = component.label.find('~');
    return separator == std::string::npos
        ? component.label
        : component.label.substr(0, separator);
}

std::vector<DisplayComponent> BuildDisplayComponents(
    const std::vector<gvvplot::ComponentInfo>& components)
{
    std::vector<DisplayComponent> display_components;
    for (std::size_t source = 0; source < components.size(); ++source) {
        const gvvplot::ComponentInfo& component = components[source];
        // A Resonance can contain several Wave Terms. Keep one display curve
        // and one legend entry for the Resonance rather than one for each
        // individual basis amplitude.
        const auto existing = std::find_if(
            display_components.begin(),
            display_components.end(),
            [&component](const DisplayComponent& candidate) {
                return candidate.resonance_id == component.resonance_id;
            });
        if (existing == display_components.end()) {
            display_components.push_back({
                ResonanceLabel(component),
                component.resonance_id,
                {source}});
        } else {
            existing->source_indices.push_back(source);
        }
    }
    return display_components;
}

int InterpolateChannel(int first, int second, double fraction)
{
    return static_cast<int>(
        first + (second - first) * fraction + 0.5);
}

int GradientColor(double fraction)
{
    const RgbColor& first = fraction <= 0.5
        ? kComponentColorStart
        : kComponentColorMiddle;
    const RgbColor& second = fraction <= 0.5
        ? kComponentColorMiddle
        : kComponentColorEnd;
    const double local_fraction = fraction <= 0.5
        ? 2.0 * fraction
        : 2.0 * fraction - 1.0;
    return TColor::GetColor(
        InterpolateChannel(first.red, second.red, local_fraction),
        InterpolateChannel(first.green, second.green, local_fraction),
        InterpolateChannel(first.blue, second.blue, local_fraction));
}

std::vector<ComponentStyle> BuildComponentStyles(
    std::size_t number_components)
{
    std::vector<ComponentStyle> styles(number_components);
    for (std::size_t index = 0; index < number_components; ++index) {
        const double fraction = number_components <= 1
            ? 0.0
            : static_cast<double>(index)
                / static_cast<double>(number_components - 1);
        styles[index] = {GradientColor(fraction)};
    }
    return styles;
}

void BuildDisplayHistograms(
    gvvplot::PanelHistograms& panel,
    const std::vector<DisplayComponent>& display_components,
    int serial)
{
    std::vector<TH1D*> display_histograms;
    display_histograms.reserve(display_components.size());
    for (std::size_t display = 0;
         display < display_components.size();
         ++display) {
        const std::vector<std::size_t>& sources =
            display_components[display].source_indices;
        TH1D* histogram = dynamic_cast<TH1D*>(
            panel.components[sources.front()]->Clone(
                Form("gvv_display_component_%d_%zu", serial, display)));
        histogram->SetDirectory(nullptr);
        histogram->Reset("ICES");
        for (std::size_t source : sources) {
            histogram->Add(panel.components[source]);
        }
        display_histograms.push_back(histogram);
    }
    panel.components = std::move(display_histograms);
}

void AddInternalWaveInterference(
    gvvplot::PanelHistograms& panel,
    const gvvplot::ProjectionInput& input,
    const gvvplot::VariableSpec& variable,
    const std::vector<DisplayComponent>& display_components)
{
    const std::size_t matrix_size = input.components.size();
    gvvplot::Branches mc_values;
    mc_values.Bind(input.mc, true, false);
    for (Long64_t event = 0; event < input.mc->GetEntries(); ++event) {
        input.mc->GetEntry(event);
        for (std::size_t display = 0;
             display < display_components.size();
             ++display) {
            const DisplayComponent& component = display_components[display];
            const std::vector<std::size_t>& sources =
                component.source_indices;
            if (sources.size() < 2) {
                continue;
            }

            // ProjectionWriter stores a symmetric matrix. Each non-diagonal
            // entry already contains the complete K_ij + K_ji interference.
            // Visit each unordered pair of Terms within this Resonance once.
            double internal_interference = 0.0;
            for (std::size_t first = 0; first < sources.size(); ++first) {
                const int first_index = input.components[sources[first]].index;
                for (std::size_t second = first + 1;
                     second < sources.size();
                     ++second) {
                    const int second_index =
                        input.components[sources[second]].index;
                    internal_interference += mc_values.weight_component->at(
                        static_cast<std::size_t>(first_index) * matrix_size
                        + second_index);
                }
            }
            gvvplot::FillObservable(
                panel.components[display],
                mc_values,
                variable.variable,
                internal_interference);
        }
    }
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
        panel.components[component]->SetLineStyle(kComponentLineStyle);
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
    const std::vector<projection_components::DisplayComponent>
        display_components =
            projection_components::BuildDisplayComponents(input.components);
    const std::vector<projection_components::ComponentStyle>
        component_styles =
            projection_components::BuildComponentStyles(display_components.size());
    TCanvas* canvas = new TCanvas(
        projection_components::kCanvasName,
        projection_components::kCanvasTitle,
        projection_components::kCanvasWidth,
        projection_components::kCanvasHeight);

    // Keep a 3 x 2 physics grid on the left and a compact external legend in
    // the right margin, following the conventional projection-plot layout.
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
        projection_components::BuildDisplayHistograms(
            panels.back(), display_components, static_cast<int>(index));
        projection_components::AddInternalWaveInterference(
            panels.back(),
            input,
            projection_components::kVariables[index],
            display_components);
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
    legend->SetMargin(projection_components::kLegendMargin);
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
         component < display_components.size();
         ++component) {
        legend->AddEntry(
            panels[0].components[component],
            gvvplot::RootLabel(display_components[component].label).c_str(),
            projection_components::kLineLegendOption);
    }
    legend->Draw();

    gvvplot::EnsureOutputDirectory(output_path);
    canvas->Modified();
    canvas->Update();
    canvas->Print((output_path + ".pdf").c_str());
    canvas->Print((output_path + ".eps").c_str());
}

// Projection plotting utilities. All component/group identities are read from
// the projection ROOT contract; this file contains no nominal resonance list.
#ifndef GVV_PLOT_UTILS_H
#define GVV_PLOT_UTILS_H

#include "TCanvas.h"
#include "TColor.h"
#include "TFile.h"
#include "TH1D.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TMath.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TTree.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gvvplot {

enum Variable {
    kMassOmegaOmega = 0,
    kMassGammaOmega = 1,
    kCosThetaGamma = 2,
    kCosThetaOmega = 3,
    kOmegaDecayPlane = 4,
    kDeltaPhiDecayPlanes = 5,
    kMassOmega = 6
};

struct VariableSpec {
    Variable variable;
    int bins;
    double lower;
    double upper;
    const char* x_title;
    bool mass_axis;
};

inline std::vector<VariableSpec> MainVariables()
{
    return {
        {kMassOmegaOmega, 60, 1.50, 3.20,
         "M(#omega#omega) (GeV/#font[12]{c}^{2})", true},
        {kMassGammaOmega, 60, 0.85, 2.95,
         "M(#gamma#omega) (GeV/#font[12]{c}^{2})", true},
        {kCosThetaGamma, 40, -1.0, 1.0,
         "cos#theta_{#gamma}", false},
        {kCosThetaOmega, 40, -1.0, 1.0,
         "sym. cos#theta_{#omega}", false},
        {kOmegaDecayPlane, 40, -TMath::Pi(), TMath::Pi(),
         "#phi_{#omega} (rad)", false},
        {kDeltaPhiDecayPlanes, 40, -TMath::Pi(), TMath::Pi(),
         "sym. #Delta#phi_{planes} (rad)", false}};
}

inline VariableSpec OmegaMassVariable()
{
    return {kMassOmega, 42, 0.740, 0.824,
            "M(#pi^{+}#pi^{-}#pi^{0}) (GeV/#font[12]{c}^{2})", true};
}

inline void SetBESIIIStyle()
{
    TStyle* style = dynamic_cast<TStyle*>(
        gROOT->GetListOfStyles()->FindObject("gvv_bes3"));
    if (style == nullptr) {
        style = new TStyle("gvv_bes3", "GVV BESIII style");
    }
    style->SetFrameBorderMode(0);
    style->SetCanvasBorderMode(0);
    style->SetPadBorderMode(0);
    style->SetPadColor(0);
    style->SetCanvasColor(0);
    style->SetStatColor(0);
    style->SetTitleFillColor(0);
    style->SetPalette(1);
    style->SetPaperSize(TStyle::kUSLetter);
    style->SetPadTopMargin(0.12);
    style->SetPadLeftMargin(0.15);
    style->SetPadRightMargin(0.06);
    style->SetPadBottomMargin(0.15);
    style->SetTextFont(22);
    style->SetTextSize(0.055);
    style->SetLabelSize(0.050, "xyz");
    style->SetLabelOffset(0.010, "xyz");
    style->SetTitleFont(22, "xyz");
    style->SetLabelFont(22, "xyz");
    style->SetTitleSize(0.055, "xyz");
    style->SetTitleXOffset(1.05);
    style->SetTitleYOffset(1.30);
    style->SetFrameLineWidth(1.0);
    style->SetLineWidth(2.0);
    style->SetHistLineWidth(1.0);
    style->SetErrorX(0.001);
    style->SetOptTitle(0);
    style->SetOptStat(0);
    style->SetOptDate(0);
    style->SetStripDecimals(kFALSE);
    style->SetEndErrorSize(0.0);
    gROOT->SetStyle("gvv_bes3");
    gROOT->ForceStyle();
}

inline void RequireBranch(TTree* tree, const char* name)
{
    if (tree->GetBranch(name) == nullptr) {
        throw std::runtime_error(
            std::string("missing branch '") + name + "' in tree "
            + tree->GetName());
    }
}

struct Branches {
    double m_omega1 = 0.0;
    double m_omega2 = 0.0;
    double m_omegaomega = 0.0;
    double m_gammaomega1 = 0.0;
    double m_gammaomega2 = 0.0;
    double cos_theta_gamma = 0.0;
    double cos_theta_omega = 0.0;
    double omega1_decay_plane_angle = 0.0;
    double omega2_decay_plane_angle = 0.0;
    double delta_phi_decay_planes = 0.0;
    double weight = 1.0;
    double weight_bg = 1.0;
    int background_index = -1;
    std::vector<double>* weight_group = nullptr;
    std::vector<double>* weight_component = nullptr;

    void Bind(TTree* tree, bool is_mc, bool is_background)
    {
        const char* required[] = {
            "m_omega1", "m_omega2", "m_omegaomega",
            "m_gammaomega1", "m_gammaomega2",
            "cos_theta_gamma", "cos_theta_omega",
            "omega1_decay_plane_angle", "omega2_decay_plane_angle",
            "delta_phi_decay_planes"};
        for (const char* name : required) RequireBranch(tree, name);
        tree->SetBranchAddress("m_omega1", &m_omega1);
        tree->SetBranchAddress("m_omega2", &m_omega2);
        tree->SetBranchAddress("m_omegaomega", &m_omegaomega);
        tree->SetBranchAddress("m_gammaomega1", &m_gammaomega1);
        tree->SetBranchAddress("m_gammaomega2", &m_gammaomega2);
        tree->SetBranchAddress("cos_theta_gamma", &cos_theta_gamma);
        tree->SetBranchAddress("cos_theta_omega", &cos_theta_omega);
        tree->SetBranchAddress(
            "omega1_decay_plane_angle", &omega1_decay_plane_angle);
        tree->SetBranchAddress(
            "omega2_decay_plane_angle", &omega2_decay_plane_angle);
        tree->SetBranchAddress(
            "delta_phi_decay_planes", &delta_phi_decay_planes);
        if (is_mc) {
            const char* weights[] = {
                "weight", "weight_group", "weight_component"};
            for (const char* name : weights) RequireBranch(tree, name);
            tree->SetBranchAddress("weight", &weight);
            tree->SetBranchAddress("weight_group", &weight_group);
            tree->SetBranchAddress("weight_component", &weight_component);
        }
        if (is_background) {
            RequireBranch(tree, "background_index");
            RequireBranch(tree, "weight_bg");
            tree->SetBranchAddress("background_index", &background_index);
            tree->SetBranchAddress("weight_bg", &weight_bg);
        }
    }
};

inline void FillObservable(
    TH1D* histogram,
    const Branches& values,
    Variable variable,
    double weight)
{
    if (variable == kMassOmegaOmega) {
        histogram->Fill(values.m_omegaomega, weight);
    } else if (variable == kMassGammaOmega) {
        histogram->Fill(values.m_gammaomega1, 0.5 * weight);
        histogram->Fill(values.m_gammaomega2, 0.5 * weight);
    } else if (variable == kCosThetaGamma) {
        histogram->Fill(values.cos_theta_gamma, weight);
    } else if (variable == kCosThetaOmega) {
        histogram->Fill(values.cos_theta_omega, 0.5 * weight);
        histogram->Fill(-values.cos_theta_omega, 0.5 * weight);
    } else if (variable == kOmegaDecayPlane) {
        histogram->Fill(values.omega1_decay_plane_angle, 0.5 * weight);
        histogram->Fill(values.omega2_decay_plane_angle, 0.5 * weight);
    } else if (variable == kDeltaPhiDecayPlanes) {
        histogram->Fill(values.delta_phi_decay_planes, 0.5 * weight);
        histogram->Fill(-values.delta_phi_decay_planes, 0.5 * weight);
    } else if (variable == kMassOmega) {
        histogram->Fill(values.m_omega1, 0.5 * weight);
        histogram->Fill(values.m_omega2, 0.5 * weight);
    }
}

struct ComponentInfo {
    int index = -1;
    std::string name;
    std::string label;
    std::string jpc;
};

struct GroupInfo {
    int index = -1;
    std::string jpc;
    std::string label;
};

inline std::string RootLabel(std::string label)
{
    std::replace(label.begin(), label.end(), '\\', '#');
    std::replace(label.begin(), label.end(), '~', ' ');
    return label;
}

inline std::vector<ComponentInfo> ReadComponentMap(TFile& input)
{
    TTree* tree = nullptr;
    input.GetObject("component_map", tree);
    if (tree == nullptr) {
        throw std::runtime_error("projection file has no component_map tree");
    }
    RequireBranch(tree, "component_index");
    RequireBranch(tree, "name");
    RequireBranch(tree, "label");
    RequireBranch(tree, "jpc");
    int component_index = -1;
    char name[64] = {0};
    char label[128] = {0};
    char jpc[16] = {0};
    tree->SetBranchAddress("component_index", &component_index);
    tree->SetBranchAddress("name", name);
    tree->SetBranchAddress("label", label);
    tree->SetBranchAddress("jpc", jpc);
    std::vector<ComponentInfo> result;
    for (Long64_t row = 0; row < tree->GetEntries(); ++row) {
        tree->GetEntry(row);
        if (component_index < 0) {
            throw std::runtime_error("invalid component index in component_map");
        }
        result.push_back({component_index, name, label, jpc});
    }
    std::sort(
        result.begin(), result.end(),
        [](const ComponentInfo& first, const ComponentInfo& second) {
            return first.index < second.index;
        });
    return result;
}

inline std::vector<GroupInfo> ReadGroupMap(TFile& input)
{
    TTree* tree = nullptr;
    input.GetObject("group_map", tree);
    if (tree == nullptr) {
        throw std::runtime_error("projection file has no group_map tree");
    }
    RequireBranch(tree, "group_index");
    RequireBranch(tree, "jpc");
    RequireBranch(tree, "label");
    int group_index = -1;
    char jpc[16] = {0};
    char label[32] = {0};
    tree->SetBranchAddress("group_index", &group_index);
    tree->SetBranchAddress("jpc", jpc);
    tree->SetBranchAddress("label", label);
    std::vector<GroupInfo> result;
    for (Long64_t row = 0; row < tree->GetEntries(); ++row) {
        tree->GetEntry(row);
        result.push_back({group_index, jpc, label});
    }
    std::sort(
        result.begin(), result.end(),
        [](const GroupInfo& first, const GroupInfo& second) {
            return first.index < second.index;
        });
    return result;
}

inline void ValidateProjectionContract(TFile& input)
{
    TTree* metadata = nullptr;
    input.GetObject("metadata", metadata);
    if (metadata == nullptr || metadata->GetEntries() != 1) {
        throw std::runtime_error(
            "projection file has no single-row metadata tree");
    }
    RequireBranch(metadata, "schema_version");
    RequireBranch(metadata, "n_background_samples");
    int schema_version = 0;
    int number_background_samples = 0;
    metadata->SetBranchAddress("schema_version", &schema_version);
    metadata->SetBranchAddress(
        "n_background_samples", &number_background_samples);
    metadata->GetEntry(0);
    if (schema_version != 2) {
        throw std::runtime_error(
            "unsupported projection schema version "
            + std::to_string(schema_version));
    }

    TTree* background_map = nullptr;
    input.GetObject("background_map", background_map);
    if (background_map == nullptr) {
        throw std::runtime_error("projection file has no background_map tree");
    }
    const char* required[] = {
        "background_index", "label", "n_events",
        "likelihood_coefficient", "projection_weight"};
    for (const char* name : required) RequireBranch(background_map, name);
    if (background_map->GetEntries() != number_background_samples) {
        throw std::runtime_error(
            "projection background_map size is inconsistent with metadata");
    }
}

inline TH1D* NewHistogram(
    const std::string& prefix,
    const VariableSpec& specification,
    int serial)
{
    TH1D* histogram = new TH1D(
        Form("%s_%d", prefix.c_str(), serial),
        "",
        specification.bins,
        specification.lower,
        specification.upper);
    histogram->SetDirectory(nullptr);
    histogram->Sumw2();
    return histogram;
}

struct PanelHistograms {
    TH1D* data = nullptr;
    TH1D* background = nullptr;
    TH1D* signal = nullptr;
    TH1D* total = nullptr;
    std::vector<TH1D*> groups;
    std::vector<TH1D*> components;
};

inline PanelHistograms BuildPanel(
    TTree* data_tree,
    TTree* mc_tree,
    TTree* background_tree,
    const VariableSpec& specification,
    int serial,
    const std::vector<GroupInfo>& groups,
    const std::vector<ComponentInfo>& components,
    bool fill_components)
{
    PanelHistograms result;
    result.data = NewHistogram("gvv_data", specification, serial);
    result.background = NewHistogram("gvv_bg", specification, serial);
    result.signal = NewHistogram("gvv_signal", specification, serial);
    for (std::size_t index = 0; index < groups.size(); ++index) {
        result.groups.push_back(NewHistogram(
            "gvv_group_" + std::to_string(index), specification, serial));
    }
    if (fill_components) {
        for (std::size_t index = 0; index < components.size(); ++index) {
            result.components.push_back(NewHistogram(
                "gvv_component_" + std::to_string(index),
                specification,
                serial));
        }
    }
    Branches data_values;
    data_values.Bind(data_tree, false, false);
    for (Long64_t event = 0; event < data_tree->GetEntries(); ++event) {
        data_tree->GetEntry(event);
        FillObservable(result.data, data_values, specification.variable, 1.0);
    }
    Branches background_values;
    background_values.Bind(background_tree, false, true);
    for (Long64_t event = 0; event < background_tree->GetEntries(); ++event) {
        background_tree->GetEntry(event);
        FillObservable(
            result.background,
            background_values,
            specification.variable,
            background_values.weight_bg);
    }
    Branches mc_values;
    mc_values.Bind(mc_tree, true, false);
    for (Long64_t event = 0; event < mc_tree->GetEntries(); ++event) {
        mc_tree->GetEntry(event);
        FillObservable(
            result.signal, mc_values, specification.variable, mc_values.weight);
        if (mc_values.weight_group == nullptr
            || mc_values.weight_group->size() != groups.size()) {
            throw std::runtime_error(
                "projection group-weight layout is inconsistent");
        }
        for (std::size_t group = 0; group < groups.size(); ++group) {
            FillObservable(
                result.groups[group], mc_values, specification.variable,
                mc_values.weight_group->at(group));
        }
        if (fill_components) {
            for (std::size_t component = 0;
                 component < components.size();
                 ++component) {
                const int index = components[component].index;
                const std::size_t matrix_size = components.size();
                if (mc_values.weight_component == nullptr
                    || mc_values.weight_component->size()
                           != matrix_size * matrix_size
                    || static_cast<std::size_t>(index) >= matrix_size) {
                    throw std::runtime_error(
                        "projection component-weight layout is inconsistent");
                }
                FillObservable(
                    result.components[component], mc_values,
                    specification.variable,
                    mc_values.weight_component->at(
                        static_cast<std::size_t>(index) * matrix_size
                        + index));
            }
        }
    }
    result.total = dynamic_cast<TH1D*>(
        result.signal->Clone(Form("gvv_total_%d", serial)));
    result.total->SetDirectory(nullptr);
    result.total->Add(result.background);
    return result;
}

inline std::pair<double, int> PearsonChiSquare(
    const TH1D* data,
    const TH1D* expectation)
{
    double chi_square = 0.0;
    int number_bins = 0;
    for (int bin = 1; bin <= data->GetNbinsX(); ++bin) {
        const double expected = expectation->GetBinContent(bin);
        const double observed = data->GetBinContent(bin);
        if (!(expected > 0.0) || !std::isfinite(expected)
            || !std::isfinite(observed)) continue;
        chi_square += (observed - expected) * (observed - expected) / expected;
        ++number_bins;
    }
    return std::make_pair(chi_square, number_bins);
}

inline void FormatAxes(
    TH1D* data,
    const VariableSpec& specification,
    const TH1D* total)
{
    const double width =
        (specification.upper - specification.lower) / specification.bins;
    data->GetXaxis()->SetTitle(specification.x_title);
    if (specification.mass_axis) {
        data->GetYaxis()->SetTitle(
            Form("Events / (%.1f MeV/#font[12]{c}^{2})", 1000.0 * width));
    } else {
        data->GetYaxis()->SetTitle(Form("Events / %.3g", width));
    }
    data->GetXaxis()->CenterTitle(kTRUE);
    data->GetYaxis()->CenterTitle(kTRUE);
    data->GetXaxis()->SetNdivisions(505);
    data->GetYaxis()->SetNdivisions(505);
    data->SetMarkerStyle(8);
    data->SetMarkerSize(0.55);
    data->SetLineColor(kBlack);
    data->SetLineWidth(1);
    const double maximum = std::max(data->GetMaximum(), total->GetMaximum());
    const double minimum = std::min(0.0, total->GetMinimum());
    data->GetYaxis()->SetRangeUser(
        minimum < 0.0 ? 1.25 * minimum : 0.0,
        maximum > 0.0 ? 1.45 * maximum : 1.0);
}

inline void StylePanel(PanelHistograms& panel)
{
    panel.background->SetFillStyle(3004);
    panel.background->SetFillColor(kBlue);
    panel.background->SetLineColor(kBlue);
    panel.total->SetLineColor(kBlue + 1);
    panel.total->SetLineWidth(2);
    const int styles[] = {2, 7, 9, 3, 5};
    const int colors[] = {kRed + 1, kGreen + 2, kMagenta + 1, kOrange + 7,
                          kCyan + 2};
    for (std::size_t group = 0; group < panel.groups.size(); ++group) {
        panel.groups[group]->SetLineColor(colors[group % 5]);
        panel.groups[group]->SetLineStyle(styles[group % 5]);
        panel.groups[group]->SetLineWidth(2);
    }
}

inline void DrawProjection(
    const char* input_file,
    const char* output_prefix,
    bool detailed,
    bool show_components)
{
    SetBESIIIStyle();
    std::unique_ptr<TFile> input(TFile::Open(input_file, "READ"));
    if (!input || input->IsZombie()) {
        throw std::runtime_error(
            std::string("cannot open projection file ") + input_file);
    }
    ValidateProjectionContract(*input);
    TTree* data = nullptr;
    TTree* mc = nullptr;
    TTree* background = nullptr;
    input->GetObject("data", data);
    input->GetObject("MC", mc);
    input->GetObject("bg", background);
    if (data == nullptr || mc == nullptr || background == nullptr) {
        throw std::runtime_error("projection file is missing data/MC/bg tree");
    }
    const std::vector<ComponentInfo> components = ReadComponentMap(*input);
    const std::vector<GroupInfo> groups = ReadGroupMap(*input);
    std::vector<VariableSpec> variables = MainVariables();
    if (detailed) variables.push_back(OmegaMassVariable());
    const int columns = detailed || show_components ? 4 : 3;
    TCanvas* canvas = new TCanvas(
        Form("gvv_projection_%d_%d", detailed, show_components),
        "GVV projections", columns == 4 ? 1320 : 1080, 720);
    canvas->Divide(columns, 2, 0.002, 0.002);
    const std::array<int, 7> colors = {
        TColor::GetColor("#08306B"),  // dark blue
        TColor::GetColor("#2171B5"),  // blue
        TColor::GetColor("#6BAED6"),  // baby blue
        TColor::GetColor("#9ECAE1"),  // light blue
        TColor::GetColor("#41AB5D"),  // green
        TColor::GetColor("#A1D76A"),  // yellow green
        TColor::GetColor("#FDE725")   // yellow
    };
    std::vector<PanelHistograms> panels;
    for (std::size_t variable = 0; variable < variables.size(); ++variable) {
        canvas->cd(static_cast<int>(variable) + 1);
        PanelHistograms panel = BuildPanel(
            data, mc, background, variables[variable],
            static_cast<int>(variable), groups, components, show_components);
        StylePanel(panel);
        FormatAxes(panel.data, variables[variable], panel.total);
        panel.data->Draw("E1");
        panel.background->Draw("HIST SAME");
        if (show_components) {
            for (std::size_t component = 0;
                 component < panel.components.size(); ++component) {
                panel.components[component]->SetLineColor(
                    colors[component % colors.size()]);
                panel.components[component]->SetLineStyle(1);
                panel.components[component]->SetLineWidth(1);
                panel.components[component]->SetMarkerStyle(0);
                panel.components[component]->SetMarkerSize(0);
                panel.components[component]->SetFillStyle(0);
                panel.components[component]->Draw("HIST C SAME");
            }
        } else {
            for (TH1D* group : panel.groups) group->Draw("HIST SAME");
        }
        panel.total->Draw("HIST SAME");
        panel.data->Draw("E1 SAME");
        const std::pair<double, int> chi_square =
            PearsonChiSquare(panel.data, panel.total);
        TLatex label;
        label.SetNDC();
        label.SetTextFont(22);
        label.SetTextSize(0.047);
        const char panel_letter = static_cast<char>('a' + variable);
        label.DrawLatex(
            0.18, 0.84,
            Form("(%c) #chi^{2}/N_{bin}=%.1f/%d", panel_letter,
                 chi_square.first, chi_square.second));
        std::cout << variables[variable].x_title
                  << "  chi2/Nbin=" << chi_square.first
                  << '/' << chi_square.second << '\n';
        panels.push_back(panel);
    }
    const int legend_pad = detailed ? 8 : (show_components ? 7 : 1);
    canvas->cd(legend_pad);
    TLegend* legend = show_components
                          ? new TLegend(0.10, 0.08, 0.94, 0.92)
                          : new TLegend(0.54, 0.60, 0.93, 0.86);
    legend->SetBorderSize(0);
    legend->SetFillStyle(0);
    legend->SetTextFont(22);
    legend->SetTextSize(show_components ? 0.055 : 0.050);
    legend->AddEntry(panels[0].data, "Data", "lep");
    legend->AddEntry(panels[0].background, "Background", "f");
    legend->AddEntry(panels[0].total, "Total fit", "l");
    if (show_components) {
        for (std::size_t component = 0;
             component < components.size(); ++component) {
            legend->AddEntry(
                panels[0].components[component],
                RootLabel(components[component].label).c_str(), "l");
        }
    } else {
        for (std::size_t group = 0; group < groups.size(); ++group) {
            const std::string label = "coherent " + groups[group].label;
            legend->AddEntry(panels[0].groups[group], label.c_str(), "l");
        }
    }
    legend->Draw();
    canvas->Print((std::string(output_prefix) + ".pdf").c_str());
    canvas->Print((std::string(output_prefix) + ".eps").c_str());
}

} // namespace gvvplot

#endif // GVV_PLOT_UTILS_H

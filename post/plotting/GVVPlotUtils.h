// Shared Projection I/O and histogram-building utilities.
// Plot-specific observables, binning, axes, colors, canvas layout, legends,
// annotations, and output names belong in each ROOT macro under macros/.
#ifndef GVV_PLOT_UTILS_H
#define GVV_PLOT_UTILS_H

#include "TFile.h"
#include "TH1D.h"
#include "TROOT.h"
#include "TString.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TTree.h"

#include <algorithm>
#include <cmath>
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
    kPhiOmega = 4,
    kMassOmega = 5,
    kCosThetaDecayPlaneOmega = 6,
    kPhiDecayPlaneOmega = 7,
    kDeltaPhiDecayPlanes = 8,
    kCosThetaPipOmega = 9,
    kCosThetaPimOmega = 10,
    kCosThetaPi0Omega = 11,
    kMassPipPimOmega = 12,
    kMassPipPi0Omega = 13,
    kMassPimPi0Omega = 14
};

// A macro supplies these values explicitly so its axes and binning can be
// reviewed and changed without editing a shared plotting header.
struct VariableSpec {
    Variable variable;
    int bins;
    double lower;
    double upper;
    const char* x_title;
    bool mass_axis;
};

// Common ROOT defaults shared by every gVV figure. Individual macros remain
// free to override any setting after this function returns.
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

// Resolve a plotting-macro path. Absolute paths are preserved; relative paths
// are interpreted from the project root found three levels above macros/.
// This makes the user-facing defaults and runtime overrides independent of the
// directory from which ROOT is launched.
inline std::string ResolveProjectPath(
    const char* path,
    const char* macro_file)
{
    if (path == nullptr || path[0] == '\0') {
        throw std::runtime_error("plot path must not be empty");
    }
    if (gSystem->IsAbsoluteFileName(path)) return path;

    TString source = macro_file == nullptr ? "" : macro_file;
    if (!gSystem->IsAbsoluteFileName(source.Data())) {
        source = TString::Format(
            "%s/%s", gSystem->WorkingDirectory(), source.Data());
    }
    const TString macro_directory = gSystem->DirName(source.Data());
    TString project_root = TString::Format(
        "%s/../../..", macro_directory.Data());
    gSystem->ExpandPathName(project_root);
    TString result = TString::Format(
        "%s/%s", project_root.Data(), path);
    gSystem->ExpandPathName(result);
    return result.Data();
}

inline void RequireBranch(TTree* tree, const char* name)
{
    if (tree->GetBranch(name) == nullptr) {
        throw std::runtime_error(
            std::string("missing branch '") + name + "' in tree "
            + tree->GetName());
    }
}

inline double WrapAzimuth(double angle)
{
    constexpr double pi = 3.14159265358979323846;
    constexpr double two_pi = 2.0 * pi;
    while (angle <= -pi) angle += two_pi;
    while (angle > pi) angle -= two_pi;
    return angle;
}

struct Branches {
    double m_omega1 = 0.0;
    double m_omega2 = 0.0;
    double m_omegaomega = 0.0;
    double m_gammaomega1 = 0.0;
    double m_gammaomega2 = 0.0;
    double m_pip1_pim1 = 0.0;
    double m_pip1_pi01 = 0.0;
    double m_pim1_pi01 = 0.0;
    double m_pip2_pim2 = 0.0;
    double m_pip2_pi02 = 0.0;
    double m_pim2_pi02 = 0.0;
    double cos_theta_gamma = 0.0;
    double cos_theta_omega1 = 0.0;
    double phi_omega1 = 0.0;
    double cos_theta_decay_plane_omega1 = 0.0;
    double phi_decay_plane_omega1 = 0.0;
    double cos_theta_decay_plane_omega2 = 0.0;
    double phi_decay_plane_omega2 = 0.0;
    double delta_phi_decay_planes = 0.0;
    double cos_theta_pip_omega1 = 0.0;
    double cos_theta_pim_omega1 = 0.0;
    double cos_theta_pi0_omega1 = 0.0;
    double cos_theta_pip_omega2 = 0.0;
    double cos_theta_pim_omega2 = 0.0;
    double cos_theta_pi0_omega2 = 0.0;
    double decay_plane_normal_magnitude_omega1 = 0.0;
    double decay_plane_normal_magnitude_omega2 = 0.0;
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
            "m_pip1_pim1", "m_pip1_pi01", "m_pim1_pi01",
            "m_pip2_pim2", "m_pip2_pi02", "m_pim2_pi02",
            "cos_theta_gamma", "cos_theta_omega1", "phi_omega1",
            "cos_theta_decay_plane_omega1", "phi_decay_plane_omega1",
            "cos_theta_decay_plane_omega2", "phi_decay_plane_omega2",
            "delta_phi_decay_planes",
            "cos_theta_pip_omega1", "cos_theta_pim_omega1",
            "cos_theta_pi0_omega1", "cos_theta_pip_omega2",
            "cos_theta_pim_omega2", "cos_theta_pi0_omega2",
            "decay_plane_normal_magnitude_omega1",
            "decay_plane_normal_magnitude_omega2"};
        for (const char* name : required) RequireBranch(tree, name);
        tree->SetBranchAddress("m_omega1", &m_omega1);
        tree->SetBranchAddress("m_omega2", &m_omega2);
        tree->SetBranchAddress("m_omegaomega", &m_omegaomega);
        tree->SetBranchAddress("m_gammaomega1", &m_gammaomega1);
        tree->SetBranchAddress("m_gammaomega2", &m_gammaomega2);
        tree->SetBranchAddress("m_pip1_pim1", &m_pip1_pim1);
        tree->SetBranchAddress("m_pip1_pi01", &m_pip1_pi01);
        tree->SetBranchAddress("m_pim1_pi01", &m_pim1_pi01);
        tree->SetBranchAddress("m_pip2_pim2", &m_pip2_pim2);
        tree->SetBranchAddress("m_pip2_pi02", &m_pip2_pi02);
        tree->SetBranchAddress("m_pim2_pi02", &m_pim2_pi02);
        tree->SetBranchAddress("cos_theta_gamma", &cos_theta_gamma);
        tree->SetBranchAddress("cos_theta_omega1", &cos_theta_omega1);
        tree->SetBranchAddress("phi_omega1", &phi_omega1);
        tree->SetBranchAddress(
            "cos_theta_decay_plane_omega1",
            &cos_theta_decay_plane_omega1);
        tree->SetBranchAddress(
            "phi_decay_plane_omega1", &phi_decay_plane_omega1);
        tree->SetBranchAddress(
            "cos_theta_decay_plane_omega2",
            &cos_theta_decay_plane_omega2);
        tree->SetBranchAddress(
            "phi_decay_plane_omega2", &phi_decay_plane_omega2);
        tree->SetBranchAddress(
            "delta_phi_decay_planes", &delta_phi_decay_planes);
        tree->SetBranchAddress(
            "cos_theta_pip_omega1", &cos_theta_pip_omega1);
        tree->SetBranchAddress(
            "cos_theta_pim_omega1", &cos_theta_pim_omega1);
        tree->SetBranchAddress(
            "cos_theta_pi0_omega1", &cos_theta_pi0_omega1);
        tree->SetBranchAddress(
            "cos_theta_pip_omega2", &cos_theta_pip_omega2);
        tree->SetBranchAddress(
            "cos_theta_pim_omega2", &cos_theta_pim_omega2);
        tree->SetBranchAddress(
            "cos_theta_pi0_omega2", &cos_theta_pi0_omega2);
        tree->SetBranchAddress(
            "decay_plane_normal_magnitude_omega1",
            &decay_plane_normal_magnitude_omega1);
        tree->SetBranchAddress(
            "decay_plane_normal_magnitude_omega2",
            &decay_plane_normal_magnitude_omega2);
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

// Identical-omega observables use the project's exchange-symmetric fill
// convention. The odd-moment diagnostic intentionally does not use it.
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
        histogram->Fill(values.cos_theta_omega1, 0.5 * weight);
        histogram->Fill(-values.cos_theta_omega1, 0.5 * weight);
    } else if (variable == kPhiOmega) {
        histogram->Fill(values.phi_omega1, 0.5 * weight);
        histogram->Fill(
            WrapAzimuth(values.phi_omega1 + 3.14159265358979323846),
            0.5 * weight);
    } else if (variable == kMassOmega) {
        histogram->Fill(values.m_omega1, 0.5 * weight);
        histogram->Fill(values.m_omega2, 0.5 * weight);
    } else if (variable == kCosThetaDecayPlaneOmega) {
        histogram->Fill(
            values.cos_theta_decay_plane_omega1, 0.5 * weight);
        histogram->Fill(
            values.cos_theta_decay_plane_omega2, 0.5 * weight);
    } else if (variable == kPhiDecayPlaneOmega) {
        histogram->Fill(values.phi_decay_plane_omega1, 0.5 * weight);
        histogram->Fill(values.phi_decay_plane_omega2, 0.5 * weight);
    } else if (variable == kDeltaPhiDecayPlanes) {
        histogram->Fill(values.delta_phi_decay_planes, 0.5 * weight);
        histogram->Fill(-values.delta_phi_decay_planes, 0.5 * weight);
    } else if (variable == kCosThetaPipOmega) {
        histogram->Fill(values.cos_theta_pip_omega1, 0.5 * weight);
        histogram->Fill(values.cos_theta_pip_omega2, 0.5 * weight);
    } else if (variable == kCosThetaPimOmega) {
        histogram->Fill(values.cos_theta_pim_omega1, 0.5 * weight);
        histogram->Fill(values.cos_theta_pim_omega2, 0.5 * weight);
    } else if (variable == kCosThetaPi0Omega) {
        histogram->Fill(values.cos_theta_pi0_omega1, 0.5 * weight);
        histogram->Fill(values.cos_theta_pi0_omega2, 0.5 * weight);
    } else if (variable == kMassPipPimOmega) {
        histogram->Fill(values.m_pip1_pim1, 0.5 * weight);
        histogram->Fill(values.m_pip2_pim2, 0.5 * weight);
    } else if (variable == kMassPipPi0Omega) {
        histogram->Fill(values.m_pip1_pi01, 0.5 * weight);
        histogram->Fill(values.m_pip2_pi02, 0.5 * weight);
    } else if (variable == kMassPimPi0Omega) {
        histogram->Fill(values.m_pim1_pi01, 0.5 * weight);
        histogram->Fill(values.m_pim2_pi02, 0.5 * weight);
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
            throw std::runtime_error(
                "invalid component index in component_map");
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
    if (schema_version != 3) {
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

// Own the input file together with the trees and dynamic maps used by every
// plotting macro. The file must outlive the returned TTree pointers.
struct ProjectionInput {
    std::unique_ptr<TFile> file;
    TTree* data = nullptr;
    TTree* mc = nullptr;
    TTree* background = nullptr;
    std::vector<ComponentInfo> components;
    std::vector<GroupInfo> groups;
};

inline ProjectionInput LoadProjection(const char* input_file)
{
    ProjectionInput result;
    result.file.reset(TFile::Open(input_file, "READ"));
    if (!result.file || result.file->IsZombie()) {
        throw std::runtime_error(
            std::string("cannot open projection file ") + input_file);
    }
    ValidateProjectionContract(*result.file);
    result.file->GetObject("data", result.data);
    result.file->GetObject("MC", result.mc);
    result.file->GetObject("bg", result.background);
    if (result.data == nullptr || result.mc == nullptr
        || result.background == nullptr) {
        throw std::runtime_error(
            "projection file is missing data/MC/bg tree");
    }
    result.components = ReadComponentMap(*result.file);
    result.groups = ReadGroupMap(*result.file);
    return result;
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

// Convert one Projection observable into unstyled histograms. The caller owns
// axis formatting, curve styling, draw order, annotations, and legends.
inline PanelHistograms BuildPanel(
    const ProjectionInput& input,
    const VariableSpec& specification,
    int serial,
    bool fill_components)
{
    PanelHistograms result;
    result.data = NewHistogram("gvv_data", specification, serial);
    result.background = NewHistogram("gvv_bg", specification, serial);
    result.signal = NewHistogram("gvv_signal", specification, serial);
    for (std::size_t index = 0; index < input.groups.size(); ++index) {
        result.groups.push_back(NewHistogram(
            "gvv_group_" + std::to_string(index), specification, serial));
    }
    if (fill_components) {
        for (std::size_t index = 0; index < input.components.size(); ++index) {
            result.components.push_back(NewHistogram(
                "gvv_component_" + std::to_string(index),
                specification,
                serial));
        }
    }

    Branches data_values;
    data_values.Bind(input.data, false, false);
    for (Long64_t event = 0; event < input.data->GetEntries(); ++event) {
        input.data->GetEntry(event);
        FillObservable(result.data, data_values, specification.variable, 1.0);
    }

    Branches background_values;
    background_values.Bind(input.background, false, true);
    for (Long64_t event = 0;
         event < input.background->GetEntries();
         ++event) {
        input.background->GetEntry(event);
        FillObservable(
            result.background,
            background_values,
            specification.variable,
            background_values.weight_bg);
    }

    Branches mc_values;
    mc_values.Bind(input.mc, true, false);
    for (Long64_t event = 0; event < input.mc->GetEntries(); ++event) {
        input.mc->GetEntry(event);
        FillObservable(
            result.signal,
            mc_values,
            specification.variable,
            mc_values.weight);
        if (mc_values.weight_group == nullptr
            || mc_values.weight_group->size() != input.groups.size()) {
            throw std::runtime_error(
                "projection group-weight layout is inconsistent");
        }
        for (std::size_t group = 0; group < input.groups.size(); ++group) {
            FillObservable(
                result.groups[group],
                mc_values,
                specification.variable,
                mc_values.weight_group->at(group));
        }
        if (fill_components) {
            for (std::size_t component = 0;
                 component < input.components.size();
                 ++component) {
                const int index = input.components[component].index;
                const std::size_t matrix_size = input.components.size();
                if (mc_values.weight_component == nullptr
                    || mc_values.weight_component->size()
                           != matrix_size * matrix_size
                    || static_cast<std::size_t>(index) >= matrix_size) {
                    throw std::runtime_error(
                        "projection component-weight layout is inconsistent");
                }
                FillObservable(
                    result.components[component],
                    mc_values,
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

} // namespace gvvplot

#endif // GVV_PLOT_UTILS_H
